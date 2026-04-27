import os
import json
import torch
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import pandas as pd
from scipy.signal import welch
from layers.vlinear_arch import OrthTransform 

class aiops:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options.get('seed', 1)
        self.num_vars = options.get('num_vars', 200)
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options.get('shuffle', False)

        # The sub-categories defined in your directory tree
        """
            service	--> throughput, latency, error_rate (The symptoms).
            container	--> cpu_usage, memory_working_set, network_receive_bytes.
            jvm	--> gc_pause_time, heap_usage, thread_count
            node -->	load_1m, disk_io_time, mem_available (The infrastructure).
            istio -->	request_duration, tcp_sent_bytes (The communication).
        """
        self.metric_types = ['container', 'istio', 'jvm', 'node', 'service']
        #self.essential_keywords = [
        #    'cpu_system', 'cpu_usage', 'mem_usage', 'heap_used', 
        #    'latency', 'error_count', 'throughput', 'gc_time',
        #    'load_1m', 'network_receive', 'request_duration'
        #]

    def predict_best_sampling_rate(self, df, current_interval_sec=60):
        """
        Analyzes the wide table to suggest an optimal sampling interval.
        
        Args:
            df: The wide table (full_df) from get_wide_table()
            current_interval_sec: The existing time delta between rows (default 60s)
            
        Returns:
            Suggested interval in seconds and a rationale.
        """
        # 1. Variance Check: Identify "Fast" vs "Slow" columns
        # We look at the Power Spectral Density (PSD)
        sampling_recommendations = []
        
        # We sample a subset of columns to save time
        sample_cols = df.columns[np.random.choice(len(df.columns), min(20, len(df.columns)))]
        
        for col in sample_cols:
            series = df[col].values
            # Compute the frequency components
            freqs, psd = welch(series, fs=1/current_interval_sec)
            
            # Find the frequency below which 90% of the power resides
            cumulative_psd = np.cumsum(psd)
            total_power = cumulative_psd[-1]
            if total_power == 0: continue
            
            # Nyquist frequency needed for this specific metric
            idx_90 = np.where(cumulative_psd >= 0.90 * total_power)[0][0]
            f_max = freqs[idx_90]
            
            # Sampling interval = 1 / (2 * f_max)
            if f_max > 0:
                sampling_recommendations.append(1 / (2 * f_max))

        if not sampling_recommendations:
            return current_interval_sec, "Insufficient variance to determine rate."

        avg_suggested_interval = np.median(sampling_recommendations)
        
        # Logic-based clamping
        if avg_suggested_interval < current_interval_sec:
            reason = "High-frequency noise/spikes detected. Consider higher resolution if available."
        elif avg_suggested_interval > current_interval_sec * 2:
            reason = "Data is redundant. You can downsample to save memory."
        else:
            reason = "Current sampling rate is optimal for the signal-to-noise ratio."

        return round(avg_suggested_interval), reason

    # Integration snippet:
    # suggested_rate, reason = predict_best_sampling_rate(full_df)
    # print(f"Suggested Interval: {suggested_rate}s | Reason: {reason}")

    def get_wide_table(self):
        """
        Flattens the entire directory tree into a single Wide Table.
        """
        # Find all date directories (e.g., 2022-05-01)
        date_dirs = sorted([d for d in glob(os.path.join(self.data_dir, "2022-*")) if os.path.isdir(d)])
        
        all_date_frames = []

        for d_dir in tqdm(date_dirs, desc="Processing Date Folders"):
            # Path: root/2022-05-01/cloudbed/metric/
            metric_base = os.path.join(d_dir, "cloudbed", "metric")
            day_frames = []

            for m_type in self.metric_types:
                type_path = os.path.join(metric_base, m_type)
                if not os.path.exists(type_path):
                    continue

                csv_files = glob(os.path.join(type_path, "*.csv"))
                for f in csv_files:
                    
                    # e.g., 'container_cpu_system'
                    metric_name = os.path.basename(f).replace(".csv", "").replace("kpi_", "")
                    # DIMENSIONALITY REDUCTION: Only keep essential metrics
                    #if not any(key in metric_name for key in self.essential_keywords):
                    #    continue
                    df = pd.read_csv(f)
                    
                    # Keep the first occurrence of each (timestamp, cmdb_id) pair
                    df = df.drop_duplicates(keep='first')

                    # Core Logic: Pivot the data
                    # Original: [timestamp, cmdb_id, value]
                    # Pivoted: Index=timestamp, Columns=cmdb_id_metric_name
                    try:
                        pivoted = df.pivot_table(
                            index='timestamp', 
                            columns='cmdb_id', 
                            values='value', 
                            aggfunc='max' 
                        )
                        pivoted.columns = [f"{col}_{metric_name}" for col in pivoted.columns]
                        day_frames.append(pivoted)
                    except Exception as e:
                        print(f"Error processing {f}: {e}")

            if day_frames:
                # Merge all metrics for this specific day
                day_wide = pd.concat(day_frames, axis=1)
                all_date_frames.append(day_wide)

        # Final Vertical Concatenation of all days
        print("Finalizing global alignment...")
        full_df = pd.concat(all_date_frames, axis=0).sort_index()
        
        # Topology-Agnostic Handling: 
        # If a metric didn't exist at a certain time (e.g., a container wasn't scaled up), fill with 0.
        full_df = full_df.fillna(0)
        
        return full_df

    def parse_json_groundtruth(self, full_df):
        """
        Maps JSON event points to the Wide-Table binary label matrix.
        Assumes fault lasts 10 minutes (600s) if duration is not provided.
        """
        label_matrix = np.zeros(full_df.shape)
        gt_path = os.path.join(self.data_dir, "groundtruth", "*.json")
        gt_files = glob(gt_path)

        for f in gt_files:
            with open(f, 'r') as jfile:
                gt_data = json.load(jfile)
            
            # Iterating through the JSON lists provided in your snippet
            for i in range(len(gt_data['timestamp'])):
                onset = gt_data['timestamp'][i]
                cmdb_id = gt_data['cmdb_id'][i]
                
                # Window: Label as abnormal for 10 minutes starting from the timestamp
                time_mask = (full_df.index >= onset) & (full_df.index <= onset + 600)
                
                # Find all columns associated with this cmdb_id that survived filtering
                faulty_indices = [idx for idx, col in enumerate(full_df.columns) 
                                 if col.startswith(cmdb_id)]
                
                for idx in faulty_indices:
                    label_matrix[time_mask, idx] = 1
                    
        return label_matrix
    
    def get_binary_flags(self, df):
        """
        Returns a numpy array of flags (1 for binary/static, 0 for continuous).
        A column is 'binary' if it has 2 or fewer unique values.
        """
        flags = []
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count <= 2:
                flags.append(1)
            else:
                flags.append(0)
        return np.array(flags)

    def generate_example(self):
        df = self.get_wide_table()
        avg_suggested_interval, reason = self.predict_best_sampling_rate(df)  # Optional: Get sampling rate suggestions based on the data
        
        ###df = df.iloc[::(avg_suggested_interval/2) // 60, :]  # Resample the data based on the suggested interval (assuming original is 1 minute)
        print(f"Data resampled to every {avg_suggested_interval} seconds. Reason: {reason}")
        # 1. Drop Zero-Variance
        df = df.loc[:, (df.std() > 1e-6)]
        
        # 2. Top-K Volatility (Coefficient of Variation)
        # 1. Calculate volatility as usual
        volatility = df.std() / (df.mean() + 1e-6)
        
                # 2. Identify which columns are binary/categorical (2 or fewer unique values)
        """
        unique_metrics = []
        seen_roots = set()
        for col in volatility.sort_values(ascending=False).index:
            # Create a 'root' name by stripping units
            root = col.replace('_MB', '').replace('_bytes', '').replace('_total', '')
            if root not in seen_roots:
                unique_metrics.append(col)
                seen_roots.add(root)
            if len(unique_metrics) >= 100: # Get a large enough pool
                break

        is_binary = df.apply(lambda x: x.nunique() <= 2)
        unique_volatility = volatility[unique_metrics]
        # 3. Split the pools
        continuous_vol = unique_volatility[~is_binary]
        binary_vol = unique_volatility[is_binary]

        # 4. Set a Quota (e.g., 25 Continuous, 5 Binary)
        # Adjust these numbers to ensure you keep your 84% coverage!
        num_continuous = 25 
        num_binary = self.num_vars - num_continuous

        # 5. Combine the best of both worlds
        important_cols = []
        important_cols.extend(continuous_vol.sort_values(ascending=False).head(num_continuous).index)
        important_cols.extend(binary_vol.sort_values(ascending=False).head(num_binary).index)
        """
        print(f"Total metrics after filtering: {volatility.sort_values(ascending=False).head(20)}")
        important_cols = volatility.sort_values(ascending=False).head(self.num_vars).index

        df_subset = df[important_cols]
        print(f"Total metrics after filtering: {important_cols}", len(important_cols))

        print(f"Final wide table shape: {df_subset.shape}")
        
        # 3. Label Processing
        label_matrix = self.parse_json_groundtruth(df_subset)
        
        data = df_subset.values
        split = int(len(data) * 0.8)
        
        # Scaling
        #scaler = StandardScaler()
        #train_data_raw = data[:split]
        #test_data_raw = data[split:]
        #test_labels_raw = label_matrix[split:]
        # 1. Log Transform metrics that can have massive spikes (Latency/Throughput)
        # Identify columns that aren't binary
        #non_binary_cols = [c for c in range(data.shape[1]) if self.binary_flags[c] == 0]
        #data[:, non_binary_cols] = np.log1p(data[:, non_binary_cols])

        ## 2. Identify Binary vs Continuous 
        ## Do this BEFORE any transformations
        #self.binary_flags = self.get_binary_flags(df_subset)
        #print(f"Binary flags (1 for binary/static, 0 for continuous): {self.binary_flags}")
#
        ## 3. Apply Log Transform to Continuous Variables
        ## This prevents 'Massive Spikes' from drowning out the causal signal
        #data = df_subset.values
        #continuous_cols = np.where(self.binary_flags == 0)[0]
        #
        ## We only log metrics that have a significant range (> 1.0)
        ## to avoid squashing metrics that are already small.
        #for col_idx in continuous_cols:
        #    if data[:, col_idx].max() > 1.0:
        #        data[:, col_idx] = np.sqrt(data[:, col_idx])

        # 2. Use RobustScaler instead of StandardScaler
        scaler = RobustScaler() 
        train_data_raw = data[:split]
        test_data_raw = data[split:]
        test_labels_raw = label_matrix[split:]
        
        scaler.fit(train_data_raw)
        train_scaled = scaler.transform(train_data_raw)
        test_scaled = scaler.transform(test_data_raw)

        # 4. Temporal Chunking (Normal/Train)
        # Using your suggested chunk_size and step
        x_n_list = []
        step = 1
        chunk_size = 1 * self.window_size # e.g., 300 timestamps per sample
        
        for i in range(0, len(train_scaled) - chunk_size, step):
            x_n_list.append(train_scaled[i : i + chunk_size])
            
        # 5. Temporal Windowing (Abnormal/Test)
        # For testing, we typically use a sliding window to get a prediction for each step
        x_ab_list = []
        y_ab_list = []
        
        # We use chunk_size here to match the training shape
        for i in range(0, len(test_scaled) - chunk_size, step):
            x_ab_list.append(test_scaled[i : i + chunk_size])
            y_ab_list.append(test_labels_raw[i : i + chunk_size])

        # Convert to Numpy Arrays for the OrthTransform
        self.data_dict['x_n_list'] = np.array(x_n_list)
        self.data_dict['x_ab_list'] = np.array(x_ab_list)
        self.data_dict['label_list'] = np.array(y_ab_list)

        # 6. Binary Flags (Required for your OrthTransform logic)
        # Identifies columns that are strictly binary (0 or 1)
        self.binary_flags = np.array([
            1 if train_data_raw[:, c].max() - train_data_raw[:, c].min() == 1 
            and np.unique(train_data_raw[:, c]).size <= 2 else 0 
            for c in range(train_data_raw.shape[1])
        ])
        
        print(f"Dataset generated: x_n {self.data_dict['x_n_list'].shape}, "
              f"x_ab {self.data_dict['x_ab_list'].shape}")
        
        self.save_data()

    def save_data(self):
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
        for key in ['x_n_list', 'x_ab_list', 'label_list']:
            np.save(os.path.join(self.data_dir, f'{key}.npy'), self.data_dict[key])
        print(f"Flattened AIOps matrices saved to {self.data_dir}")

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
        orth_matrix_dir = os.path.join(self.data_dir, 'orth_transform_meta')
        return self.apply_orthogonal_transform(save_path=orth_matrix_dir, device='cpu')

    def apply_orthogonal_transform(self, save_path, device='cpu'):
        # ... (same as your SWaT implementation)
        """
        This is where you prove your efficiency.
        By transforming the Wide-Table (V >> 100) in O(V),
        you beat GVAR's O(V^2) causality computation.
        """
        os.makedirs(save_path, exist_ok=True)
        self.orth_transformer = OrthTransform(
            dataset_obj=self, 
            time_lag=self.window_size,
            save_path=save_path, 
            device=device
        )
        x_n_tensor = torch.from_numpy(self.data_dict['x_n_list']).float().to(device)
        with torch.no_grad():
            self.data_dict['x_n_orth'] = self.orth_transformer(x_n_tensor).cpu().numpy()
        x_ab_tensor = torch.from_numpy(self.data_dict['x_ab_list']).float().to(device)
        with torch.no_grad():
            self.data_dict['x_ab_orth'] = self.orth_transformer(x_ab_tensor).cpu().numpy()
        return self.orth_transformer

if __name__ == "__main__":
    options = {
        'data_dir': '/home/db2003/Desktop/Amr/Tests/Medicine/dataset/aiops22-pre/初赛评分数据',
        'seed': 1,
        'window_size': 10,
        'shuffle': True
    }
    dataset = aiops(options)
    dataset.generate_example()