import os
import subprocess
import requests
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import torch
from tqdm import tqdm
from layers.vlinear_arch import OrthTransform 

class GAIA:
    def __init__(self, options):
        self.options = options
        self.data_dir = options.get('data_dir')
        self.window_size = options.get('window_size')
        #self.service_prefix = options.get('service_prefix') # Adjust based on your target service in GAIA
        self.service_prefixes =  ['dbservice1', 'authservice', 'cache']
        self.urls = {
            'MicroSS': 'https://github.com/CloudWise-OpenSource/GAIA-DataSet/archive/refs/heads/main.zip'
        }
        self.data_dict = {}

    def download_data(self):
        """Downloads the GAIA repository from GitHub if not present."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Downloading GAIA dataset to {self.data_dir}...")
            
            r = requests.get(self.urls['MicroSS'], stream=True)
            # Get the total file size from headers
            total_size = int(r.headers.get('content-length', 0))
            zip_path = os.path.join(self.data_dir, "gaia_main.zip")
            
            # Initialize the progress bar
            with open(zip_path, 'wb') as f, tqdm(
                desc="gaia_main.zip",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        size = f.write(chunk)
                        bar.update(size)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("Download and extraction complete.")
        else:
            print("GAIA directory already exists. Skipping download.")

    def parse_run_logs_for_rca(self, run_dir):
        """
        Parses GAIA 'run' logs to find ground truth RCA labels.
        Converts log entries into (timestamp, service, duration) tuples.
        """
        rca_ground_truth = []
        # Path based on repo structure: GAIA-DataSet-main/MicroSS/run/
        # extract run.zip
        run_zip_path = os.path.join(run_dir, 'run.zip')
        if os.path.exists(run_zip_path):
            with zipfile.ZipFile(run_zip_path, 'r') as zip_ref:
                zip_ref.extractall(run_dir)
        run_dir = os.path.join(run_dir, 'run')  # Now we have the extracted logs in run_dir/run/
        for log_file in os.listdir(run_dir):
            with open(os.path.join(run_dir, log_file), 'r') as f:
                for line in f:
                    if '[anomaly]' in line or 'anomalies' in line:
                        # Logic to extract: timestamp, service_name, duration
                        # Example: 2021-07-01 22:33:05 | dbservice1 | memory_anomalies
                        parts = line.split('|')
                        ts_str = parts[0].strip()
                        service = parts[1].strip()
                        msg = parts[-1].strip()
                        rca_ground_truth.append({
                            'time': ts_str,
                            'service': service,
                            'fault': msg
                        })
        return pd.DataFrame(rca_ground_truth)

    def extract_split_zip(self, metric_dir):
        """
        Handles the multi-part zip volumes (metric_split.z01...metric_split.zip).
        """
        zip_main = os.path.join(metric_dir, 'metric_split.zip')
        extracted_path = os.path.join(metric_dir, 'extracted_csvs')
        
        if not os.path.exists(extracted_path):
            os.makedirs(extracted_path)
            print("Recombining and extracting multi-part GAIA metrics...")
            # Use '7z' or 'jar' if available, otherwise 'zip -s 0' to combine
            # This assumes a Linux environment based on your prompt (base)
            try:
                # Combines split parts and extracts to extracted_path
                subprocess.run(['7z', 'x', zip_main, f'-o{extracted_path}'], check=True)
            except FileNotFoundError:
                print("Error: 7z not found. Please install p7zip-full.")
                """
                    sudo apt update
                    sudo apt install p7zip-full p7zip-rar
                """
        return extracted_path

    def load_metrics_as_matrix(self, metric_dir):

        extracted_csv_dir = self.extract_split_zip(metric_dir) + "/metric"

        all_files = [
            f for f in os.listdir(extracted_csv_dir)
            if any(prefix in f for prefix in self.service_prefixes)
        ]

        bucket_size = 50

        final_df = None

        for i in range(0, len(all_files), bucket_size):

            bucket = all_files[i:i + bucket_size]
            df_list = []

            for f in tqdm(bucket, desc=f"Bucket {i//bucket_size + 1}"):

                df = pd.read_csv(os.path.join(extracted_csv_dir, f))
                indicator = f.replace('.csv', '')

                df = df.rename(columns={'value': indicator})

                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.drop_duplicates('timestamp')

                df = df.set_index('timestamp')

                # ⚠️ reduce memory early (CRITICAL)
                df[indicator] = df[indicator].astype(np.float32)

                df_list.append(df)

            # merge within bucket (safe)
            bucket_df = pd.concat(df_list, axis=1, join='outer').sort_index()

            del df_list

            # ---------------------------
            # 🔴 MEMORY-SAFE MERGE STRATEGY
            # ---------------------------

            if final_df is None:
                final_df = bucket_df
            else:
                # avoid full recomputation of both sides repeatedly
                final_df = final_df.join(bucket_df, how='outer')

            del bucket_df

        # ---------------------------
        # FINAL REDUCTION STEP
        # ---------------------------

        final_df = final_df.sort_index()

        # downcast BEFORE interpolation (very important)
        final_df = final_df.astype(np.float32)

        final_df = final_df.interpolate(limit=10).fillna(0)

        return final_df.reset_index()

    def process_normal(self, combined_train, target_len):
        """
        Segments normalized metric data into target_len-timestamp chunks.
        
        Args:
            combined_train (np.array): The full scaled metric matrix [Total_Timestamps, Sensors].
            target_len (int): The window size (e.g., 100).
            
        Returns:
            list: A list of 2D arrays [target_len, Sensors].
        """
        # Use the stride defined in your options (window_size) to avoid 
        # excessive overlap in the training set.
        step = self.window_size
        
        segments = [
            combined_train[i : i + target_len] 
            for i in range(0, len(combined_train) - target_len, step)
        ]
        
        return segments

    def process_abnormal(self, test_df, global_labels, root_cause_labels, lookback, lookahead):
        """
        Concentrates slices around the 'onset' of an anomaly to evaluate RCA accuracy.
        
        Args:
            test_df (pd.DataFrame): The scaled test metric matrix.
            global_labels (np.array): 0/1 array indicating if an anomaly is present at time t.
            root_cause_labels (np.array): 2D binary matrix [Timestamps, Sensors] for RCA.
            lookback (int): History window before the onset.
            lookahead (int): Evaluation window after the onset.
        """
        test_x_lst = []
        test_label_lst = []
        
        # Identify indices where the system state transitions to 'anomalous'
        anomaly_indices = np.where(global_labels == 1)[0]
        
        if len(anomaly_indices) > 0:
            # Group contiguous anomaly timestamps to identify distinct 'events'
            events = np.split(anomaly_indices, np.where(np.diff(anomaly_indices) > 1)[0] + 1)
            
            for event in events:
                # The 'onset' is the exact timestamp the fault was injected
                onset = event[0]
                
                # The Anchor: The onset becomes the start of the 'lookahead' portion.
                raw_start = onset - lookback
                raw_end = onset + lookahead
                
                # Ensure the window is within the bounds of the metric matrix
                if raw_start >= 0 and raw_end <= len(test_df):
                    slice_x = test_df.values[raw_start:raw_end]
                    slice_y = root_cause_labels[raw_start:raw_end]
                    
                    # Validation: Ensure the slice is the correct length and 
                    # actually contains the root cause signal we want to measure.
                    if len(slice_x) == (lookback + lookahead) and np.any(slice_y == 1):
                        test_x_lst.append(slice_x)
                        test_label_lst.append(slice_y)
                                
        return test_x_lst, test_label_lst
    
    def map_rca_to_matrix(self, full_df, rca_df):
        """
        Maps event-based logs to a binary matrix matching full_df's shape.
        This serves as the 'root_cause_labels' for the process_abnormal logic.
        """
        rca_df['actual_time'] = rca_df['fault'].str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')

        num_timestamps = len(full_df)
        num_vars = len(full_df.columns) - 1 # excluding timestamp
        label_matrix = np.zeros((num_timestamps, num_vars))
        
        # Create a mapping of metric names to indices
        metric_to_idx = {col: i for i, col in enumerate(full_df.columns) if col != 'timestamp'}
             
        for _, row in rca_df.iterrows():
            try:
                # 1. Extract service_name from the malformed first column
                # row[0] is '2021-07-01,dbservice1,"2021-07-01...'
                raw_first_col = str(row.iloc[0])
                service_name = raw_first_col.split(',')[1] # Should give 'dbservice1'
                
                # 2. Use the 'actual_time' from the 4th column (index 3)
                # Your list shows index 3 is '2021-07-01 11:44:26'
                target_ts = pd.to_datetime(row.iloc[3]).value // 10**6
                
                # 3. Find the closest index in full_df
                # Using searchsorted is O(log N), much faster than .abs().idxmin()
                idx = np.searchsorted(full_df['timestamp'].values, target_ts)
                
                if idx < len(full_df):
                    # 4. Map to metrics
                    found_cols = 0
                    for col_name, col_idx in metric_to_idx.items():
                        if service_name in col_name:
                            # Mark 600s window (GAIA's standard injection duration)
                            # If sampling is 10s, that is 60 rows
                            label_matrix[idx : idx + 60, col_idx] = 1
                            found_cols += 1
                    
                    if found_cols > 0:
                        print(f"Mapped {service_name} at {row.iloc[3]} to {found_cols} columns.")

            except Exception as e:
                print(f"Error parsing row: {e}")
                continue

        return label_matrix

    def generate_example(self):
        """Main pipeline aligned with SMD logic."""
        base_path = os.path.join(self.data_dir, 'GAIA-DataSet-main', 'MicroSS')
        metric_path = os.path.join(base_path, 'metric')
        run_path = os.path.join(base_path, 'run')

        # 1. Load and Pivot Metrics
        full_df = self.load_metrics_as_matrix(metric_path)
        
        # 2. Parse RCA Logs
        rca_df = self.parse_run_logs_for_rca(run_path)
        root_cause_labels = self.map_rca_to_matrix(full_df, rca_df)

        # --- NEW: VARIANCE FILTERING (VBFS) ---
        # Identify sensors with zero variance before they cause KL collapse
        raw_values = full_df.drop(columns=['timestamp']).values
        variances = np.var(raw_values, axis=0)
        
        # Define how many features you want to keep (e.g., top 100 or top 50%)
        k = min(self.options["num_vars"], len(variances)) 

        # Get indices of the k largest variances
        top_k_indices = np.argsort(variances)[-k:]
        # Ensure we sort them to maintain original sensor order
        top_k_indices = np.sort(top_k_indices)

        # Prune the metric matrix and labels
        live_columns = ['timestamp'] + [full_df.columns[i+1] for i in top_k_indices]
        full_df = full_df[live_columns]
        root_cause_labels = root_cause_labels[:, top_k_indices]

        print(f"Selected top {k} sensors by variance. Dropped {len(variances) - k} columns.")
        # --------------------------------------

        # 3. Create Global Labels
        # Now global_labels only triggers if a LIVE sensor has a root cause label
        global_labels = (root_cause_labels.sum(axis=1) > 0).astype(int)

        # 3. Scaling (Mirroring SMD fix)
        # Drop timestamp for scaling
        raw_values = full_df.drop(columns=['timestamp']).values
        
        # Log Transform + Max Clipping (Stability for vLinear)
        transformed_data = np.log1p(np.maximum(raw_values, 0))
        
        scaler = RobustScaler(unit_variance=True)
        scaled_data = scaler.fit_transform(transformed_data)
        scaled_data = np.clip(scaled_data, -10, 10)
        
        # 4. Segmenting (Mirroring SMD Processors)
        # Normal data: slices where global_labels == 0
        normal_mask = (global_labels == 0)
        x_n_list = []

        start = None
        for i in range(len(normal_mask)):
            if normal_mask[i] and start is None:
                start = i
            elif not normal_mask[i] and start is not None:
                if i - start >= self.window_size:
                    segment = scaled_data[start:i]
                    x_n_list.extend(self.process_normal(segment, self.window_size))
                start = None

        # handle tail
        if start is not None and len(normal_mask) - start >= self.window_size:
            segment = scaled_data[start:]
            x_n_list.extend(self.process_normal(segment, self.window_size))

        # Abnormal data: concentrated windows around onsets
        # We wrap scaled_data back into a DF temporarily for your processor
        scaled_df = pd.DataFrame(scaled_data)
        m_test_x, m_test_y = self.process_abnormal(
            scaled_df, 
            global_labels, 
            root_cause_labels, 
            lookback=self.window_size // 2, 
            lookahead=self.window_size // 2
        )

        # 5. Update data_dict
        self.data_dict['x_n_list'] = np.array(x_n_list)
        self.data_dict['x_ab_list'] = np.array(m_test_x)
        self.data_dict['label_list'] = np.array(m_test_y)

        if self.options.get('shuffle', True):
            idx = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][idx]
            
        print(f"GAIA Pipeline Complete: {len(x_n_list)} normal, {len(m_test_x)} abnormal samples.")



    def save_data(self):
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
        for key in ['x_n_list', 'x_ab_list', 'label_list']:
            np.save(os.path.join(self.data_dir, f'{key}.npy'), self.data_dict[key])

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
        orth_matrix_dir = os.path.join(self.data_dir, 'orth_transform_meta')
        self.pipeline_sanity_check()
        return self.apply_orthogonal_transform(save_path=orth_matrix_dir, device='cpu')



    def pipeline_sanity_check(self):
        print("\n--- Starting Data Pipeline Sanity Check ---")
        x_n = self.data_dict['x_n_list']
        x_ab = self.data_dict['x_ab_list']
        labels = self.data_dict['label_list']

        # 1. Shape Verification
        # Expected: [Samples, Window+1, Sensors]
        # The '+1' is crucial because the last step is our 'nexts' target
        print(f"Normal Data Shape: {x_n.shape}")
        print(f"Abnormal Data Shape: {x_ab.shape}")
        
        assert x_n.ndim == 3, "Data must be 3D [Batch, Window, Sensors]"
        #assert x_n.shape[1] == self.window_size + 1, f"Expected window {self.window_size + 1}, got {x_n.shape[1]}"

        # 2. Label Alignment Check (The "Coverage" Guard)
        # Ensure that samples marked as anomalies actually have a '1' in the target
        sample_idx = np.where(labels == 1)[0]
        if len(sample_idx) > 0:
            test_idx = sample_idx[0]
            # Check if the last step of the window matches the label
            # This prevents the 'Ghost Offset' problem we discussed
            anomaly_step = x_ab[test_idx, -1, :] 
            print(f"Anomaly Coverage Check: Found {len(sample_idx)} anomaly samples.")
        else:
            print("WARNING: No anomalies found in label_list. RCA will not be possible.")

        # 3. Scaling Check (The "Skeptic" Guard)
        # If using RobustScaler, values should be centered near 0 but have outliers
        # If using MinMax, check if they are strictly in range
        feat_min, feat_max = x_n.min(), x_n.max()
        feat_mean = x_n.mean()
        print(f"Feature Statistics -> Min: {feat_min:.4f}, Max: {feat_max:.4f}, Mean: {feat_mean:.4f}")
        
        if abs(feat_max) < 1.1 and abs(feat_min) < 1.1:
            print("INFO: Data appears to be strictly Min-Max scaled [-1, 1].")
        elif abs(feat_max) > 10:
            print("INFO: Data has high variance (Robust Scaling active). Good for RCA signal.")

        # 4. Variance Check (The "Dead Sensor" Guard)
        # If a sensor has 0 variance, the model will produce NaNs in KL divergence
        variances = np.var(x_n, axis=(0, 1))
        dead_sensors = np.where(variances == 0)[0]
        if len(dead_sensors) > 0:
            print(f"CRITICAL: Sensors {dead_sensors} have zero variance. This will cause KL collapse!")
        else:
            print("Variance Check: All sensors are active.")

        print("--- Sanity Check Passed ---\n")



    def apply_orthogonal_transform(self, save_path, device='cpu'):
        # ... (same as your SWaT implementation)
        os.makedirs(save_path, exist_ok=True)
        self.orth_transformer = OrthTransform(
            dataset_obj=self, 
            time_lag=self.window_size,
            save_path=save_path, 
            device=device
        )
        x_n_tensor = torch.from_numpy(self.data_dict['x_n_list']).float().to(device)
        with torch.no_grad():
            # smd = torch.Size([236, 38, 1000])
            self.data_dict['x_n_orth'] = self.orth_transformer(x_n_tensor).cpu().numpy()
        #(281, 1000, 38)
        x_ab_tensor = torch.from_numpy(self.data_dict['x_ab_list']).float().to(device)
        with torch.no_grad():
            # (281, 38, 1000)
            self.data_dict['x_ab_orth'] = self.orth_transformer(x_ab_tensor).cpu().numpy()
        return self.orth_transformer
    
    
if __name__ == "__main__":  
    options = {'data_dir': './datasets/gaia_data', 'window_size': 10, 'service_prefix': 'dbservice1'}
    gaia_loader = GAIA(options)
    gaia_loader.download_data()
    gaia_loader.generate_example()
    gaia_loader.save_data()
    gaia_loader.load_data()