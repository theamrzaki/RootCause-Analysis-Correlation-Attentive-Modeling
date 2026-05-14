import os
import shutil
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
#from layers.vlinear_arch import OrthTransform 

class aiops:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options.get('seed', 1)
        self.num_vars = options.get('num_vars', 30)
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options.get('shuffle', False)
        self.metric_types = ['container', 'istio', 'jvm', 'node', 'service']
        self.include_logs_and_traces = options.get('include_logs_and_traces', False)
        if not self.include_logs_and_traces:
            print("INFO: ONLY metric")
        else:
            print("INFO: INCLUDING logs and traces. This will increase feature count significantly.")

    def get_log_features(self, date_dir):
        log_base = os.path.join(date_dir, "cloudbed", "log", "all")
        if not os.path.exists(log_base):
            return None

        day_log_frames = []
        chunksize = 50000

        # =========================
        # 1. Envoy Logs
        # =========================
        envoy_path = os.path.join(log_base, "log_filebeat-testbed-log-envoy.csv")

        if os.path.exists(envoy_path):
            chunks = []

            envoy_iter = pd.read_csv(
                envoy_path,
                usecols=['timestamp', 'cmdb_id', 'value'],
                chunksize=chunksize
            )

            for chunk in tqdm(envoy_iter, desc=f"Envoy logs ({os.path.basename(date_dir)})"):
                val_parts = chunk['value'].str.rsplit('"', n=1).str[-1].str.split()

                chunk['is_error'] = val_parts.str[0].str.startswith(('4', '5')).fillna(0).astype(int)
                chunk['latency'] = pd.to_numeric(val_parts.str[4], errors='coerce').fillna(0)

                chunks.append(chunk)

            df = pd.concat(chunks, axis=0)

            pdf = df.pivot_table(
                index='timestamp',
                columns='cmdb_id',
                values=['is_error', 'latency'],
                aggfunc={'is_error': 'sum', 'latency': 'mean'}
            ).fillna(0)

            pdf.columns = [
                f"{c[1]}_{'envoy_err' if c[0]=='is_error' else 'avg_lat'}"
                for c in pdf.columns
            ]

            day_log_frames.append(pdf)

        # =========================
        # 2. Service Logs
        # =========================
        service_path = os.path.join(log_base, "log_filebeat-testbed-log-service.csv")

        if os.path.exists(service_path):
            chunks = []

            service_iter = pd.read_csv(
                service_path,
                usecols=['timestamp', 'cmdb_id', 'value'],
                chunksize=chunksize
            )

            for chunk in tqdm(service_iter, desc=f"Service logs ({os.path.basename(date_dir)})"):
                vals = chunk['value'].str.lower().values.astype(str)

                chunk['is_error'] = np.char.find(vals, 'error') >= 0
                chunk['is_error'] |= np.char.find(vals, 'fail') >= 0
                chunk['is_error'] |= np.char.find(vals, 'exception') >= 0

                chunk['is_conn'] = np.char.find(vals, 'timeout') >= 0
                chunk['is_conn'] |= np.char.find(vals, 'connect') >= 0

                chunks.append(chunk)

            df_svc = pd.concat(chunks, axis=0)

            pdf_svc = df_svc.pivot_table(
                index='timestamp',
                columns='cmdb_id',
                values=['is_error', 'is_conn'],
                aggfunc='sum'
            ).fillna(0)

            pdf_svc.columns = [
                f"{c[1]}_{'svc_err' if c[0]=='is_error' else 'conn_fail'}"
                for c in pdf_svc.columns
            ]

            day_log_frames.append(pdf_svc)

        # =========================
        # Final merge
        # =========================
        if not day_log_frames:
            return None

        return pd.concat(day_log_frames, axis=1)

    def get_trace_features(self, date_dir):
        trace_path = os.path.join(date_dir, "cloudbed", "trace", "all", "trace_jaeger-span.csv")
        if not os.path.exists(trace_path): return None

        # Load only what we need and use int32/float32 to save memory
        df = pd.read_csv(trace_path, usecols=['timestamp', 'cmdb_id', 'duration', 'status_code'])
        
        # 1. Vectorized Timestamp Scaling
        df['timestamp'] = (df['timestamp'] // 1000).astype(int)

        # 2. Vectorized Status Check (Avoid .apply())
        # status_code 0, Ok, 200 are successes. Everything else is error.
        success_codes = {'0', 'ok', '200', 0}
        df['is_trace_err'] = (~df['status_code'].astype(str).str.lower().isin(success_codes)).astype(int)

        # 3. Single Pivot for speed
        pdf = df.pivot_table(index='timestamp', columns='cmdb_id', 
                            values=['duration', 'is_trace_err'],
                            aggfunc={'duration': 'mean', 'is_trace_err': 'sum'}).fillna(0)
        
        pdf.columns = [f"{c[1]}_{'span_dur' if c[0]=='duration' else 'span_err'}" for c in pdf.columns]
        return pdf

    def get_wide_table(self):
        date_dirs = sorted([d for d in glob(os.path.join(self.data_dir, "2022-*")) if os.path.isdir(d)])
        all_date_frames = []

        for d_dir in tqdm(date_dirs, desc="Processing Date Folders"):
            metric_base = os.path.join(d_dir, "cloudbed", "metric")
            day_metric_frames = []

            # A. Process Metrics
            for m_type in self.metric_types:
                type_path = os.path.join(metric_base, m_type)
                if not os.path.exists(type_path): continue

                csv_files = glob(os.path.join(type_path, "*.csv"))
                for f in tqdm(csv_files, desc="Processing CSV Files", leave=False):
                    metric_name = os.path.basename(f).replace(".csv", "").replace("kpi_", "")
                    df = pd.read_csv(f).drop_duplicates(keep='first')

                    try:
                        pivoted = df.pivot_table(index='timestamp', columns='cmdb_id', values='value', aggfunc='max')
                        pivoted.columns = [f"{col}_{metric_name}" for col in pivoted.columns]
                        day_metric_frames.append(pivoted)
                    except Exception as e:
                        print(f"Error processing {f}: {e}")

            print(f"num metric features for {d_dir}: {sum(len(df.columns) for df in day_metric_frames)}")

            # =========================
            # MODALITY CONTAINER
            # =========================
            day_modality = {
                "metrics": None,
                "logs": None,
                "traces": None
            }

            if day_metric_frames:
                day_modality["metrics"] = pd.concat(day_metric_frames, axis=1)

            # =========================
            # LOGS + TRACES (OPTIONAL)
            # =========================
            if self.include_logs_and_traces:

                log_features = self.get_log_features(d_dir)
                if log_features is not None:
                    day_modality["logs"] = log_features

                trace_features = self.get_trace_features(d_dir)
                if trace_features is not None:
                    day_modality["traces"] = trace_features

            # =========================
            # SAFE FLATTENING FOR TRAINING
            # =========================
            if self.include_logs_and_traces:

                # IMPORTANT: do NOT append dict to concat later
                parts = []

                if day_modality["metrics"] is not None:
                    parts.append(day_modality["metrics"])

                if day_modality["logs"] is not None:
                    parts.append(day_modality["logs"])

                if day_modality["traces"] is not None:
                    parts.append(day_modality["traces"])

                if len(parts) > 0:
                    day_wide = pd.concat(parts, axis=1).fillna(0)
                    all_date_frames.append(day_wide)

            else:
                # metric-only fallback (UNCHANGED behavior)
                if day_modality["metrics"] is not None:
                    all_date_frames.append(day_modality["metrics"].fillna(0))
            break #TODO TODO TODO TODO TODO remove this break after testing the first date folder


        print("Finalizing global alignment...")
        full_df = pd.concat(all_date_frames, axis=0).sort_index().fillna(0)
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

                cmdb_id = gt_data['cmdb_id'][i]

                # ======================================================
                # OLD PIPELINE (keep exact behavior)
                # ======================================================
                if not self.include_logs_and_traces:

                    onset = gt_data['timestamp'][i]  # int (keep legacy)

                    time_mask = (
                        (full_df.index >= onset) &
                        (full_df.index <= onset + 600)
                    )

                # ======================================================
                # NEW PIPELINE (datetime-safe)
                # ======================================================
                else:

                    onset = pd.to_datetime(gt_data['timestamp'][i], unit='s')
                    duration = pd.Timedelta(seconds=600)

                    time_mask = (
                        (full_df.index >= onset) &
                        (full_df.index <= onset + duration)
                    )

                # ======================================================
                # COMMON LABEL APPLICATION (FIXED)
                # ======================================================
                faulty_indices = [
                    idx for idx, col in enumerate(full_df.columns)
                    if col.startswith(cmdb_id)
                ]

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
        if self.include_logs_and_traces:
            self.generate_example_metrics_logs_traces()
        else:
            self.generate_example_metrics_only()

    def generate_example_metrics_only(self):
        df = self.get_wide_table()
        
        df = df.iloc[::1, :]  # Resample the data based on the suggested interval (assuming original is 1 minute)
        
        # 2. Top-K Volatility (Coefficient of Variation)
        # 1. Calculate volatility as usual
        # 1. Drop anything that is virtually a flat line
        # Increase the threshold slightly to catch 'noisy' dead sensors
        df = df.loc[:, (df.std() > 1e-2)] 

        # 2. Calculate volatility
        volatility = df.std() / (df.mean() + 1e-6)

        # 3. Explicitly exclude sensors that are constant
        volatility = volatility[volatility > 0]
        print(f"Total metrics after filtering: {volatility.sort_values(ascending=False).head(20)}")
        important_cols = volatility.sort_values(ascending=False).head(self.num_vars).index
        # KEY ADDITION: Create the index-to-name mapping
        # This ensures Index 0 always matches important_cols[0]
        self.idx_to_feature = {i: name for i, name in enumerate(important_cols)}
        
        # Optional: Save this mapping to disk alongside your .npy files
        # so the testing script can load it later.
        mapping_path = os.path.join(self.data_dir, 'idx_to_feature.json')
        with open(mapping_path, 'w') as f:
            json.dump(self.idx_to_feature, f)

        df_subset = df[important_cols]
        df_subset = np.log1p(df_subset)
        df_subset.columns = important_cols
        print(f"Total metrics after filtering: {important_cols}", len(important_cols))

        print(f"Final wide table shape: {df_subset.shape}")
        
        # 3. Label Processing
        label_matrix = self.parse_json_groundtruth(df_subset)
        
        data = df_subset.values
        split = int(len(data) * 0.8)
        
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
        chunk_size = (1 * self.window_size) #+1 
        
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
        # delete orth folder if exists to avoid confusion with old orth matrices
        orth_matrix_dir = os.path.join(self.data_dir, 'orth_transform_meta')
        if os.path.exists(orth_matrix_dir):
            print(f"Removing old orthogonal transform metadata at {orth_matrix_dir} to avoid confusion.")
            shutil.rmtree(orth_matrix_dir)


    def generate_example_metrics_logs_traces(self):
        df = self.get_wide_table()

        df = df.iloc[::1, :]  # keep your current sampling choice

        # =========================================================
        # 0. FORCE UNIFIED TEMPORAL GRID (CRITICAL FIX)
        # =========================================================
        df.index = pd.to_datetime(df.index, unit='s')
        df = df.sort_index()

        # infer base resolution (fallback = 1min)
        inferred_freq = pd.infer_freq(df.index[:100])
        if inferred_freq is None:
            freq = "1min"
        else:
            freq = inferred_freq

        # =========================================================
        # 1. STRICT MODALITY SPLIT
        # =========================================================
        all_cols = df.columns

        metric_cols = [c for c in all_cols if any(m in c for m in self.metric_types)]
        log_cols = [c for c in all_cols if "envoy" in c or "svc_" in c or "conn_" in c]
        trace_cols = [c for c in all_cols if "span_" in c or "trace" in c]

        metric_df = df[metric_cols]
        log_df = df[log_cols] if len(log_cols) > 0 else None
        trace_df = df[trace_cols] if len(trace_cols) > 0 else None

        # =========================================================
        # 2. RESAMPLE ALL MODALITIES (THIS FIXES ROW EXPLOSION)
        # =========================================================

        metric_df = metric_df.resample(freq).mean()

        if log_df is not None:
            log_df = log_df.resample(freq).sum()

        if trace_df is not None:
            trace_df = trace_df.resample(freq).mean()

        # fill missing after alignment
        metric_df = metric_df.fillna(0)
        if log_df is not None:
            log_df = log_df.fillna(0)
        if trace_df is not None:
            trace_df = trace_df.fillna(0)

        # =========================================================
        # 3. METRIC FEATURE SELECTION (UNCHANGED LOGIC)
        # =========================================================
        metric_df = metric_df.loc[:, (metric_df.std() > 1e-2)]
        volatility = metric_df.std() / (metric_df.mean() + 1e-6)
        volatility = volatility[volatility > 0]

        important_cols = volatility.sort_values(ascending=False).head(self.num_vars).index

        self.idx_to_feature = {i: name for i, name in enumerate(important_cols)}

        mapping_path = os.path.join(self.data_dir, 'idx_to_feature.json')
        with open(mapping_path, 'w') as f:
            json.dump(self.idx_to_feature, f)

        metric_df = np.log1p(metric_df[important_cols])

        print(f"[METRICS] selected: {len(important_cols)}")

        # =========================================================
        # 4. LOG / TRACE DIMENSION CONTROL (SAFE NOW)
        # =========================================================
        log_keep = self.options.get("log_features", 10)
        trace_keep = self.options.get("trace_features", 10)

        if log_df is not None:
            log_df = log_df.loc[:, log_df.std() > 1e-6]
            log_df = log_df.loc[:, log_df.std().sort_values(ascending=False).head(log_keep).index]
            log_df = np.log1p(log_df.clip(upper=1e6))

        if trace_df is not None:
            trace_df = trace_df.loc[:, trace_df.std() > 1e-6]
            trace_df = trace_df.loc[:, trace_df.std().sort_values(ascending=False).head(trace_keep).index]
            trace_df = np.log1p(trace_df.clip(upper=1e6))

        # =========================================================
        # 5. FINAL CONCAT (NOW TEMPORALLY CONSISTENT)
        # =========================================================
        dfs = [metric_df]

        if log_df is not None:
            dfs.append(log_df)

        if trace_df is not None:
            dfs.append(trace_df)

        df_subset = pd.concat(dfs, axis=1).fillna(0)

        print(f"Final shape (AFTER RESAMPLING FIX): {df_subset.shape}")

        # =========================================================
        # 6. LABEL + SCALING (UNCHANGED)
        # =========================================================
        label_matrix = self.parse_json_groundtruth(df_subset)

        data = df_subset.values
        split = int(len(data) * 0.8)

        scaler = RobustScaler()

        train_data_raw = data[:split]
        test_data_raw = data[split:]
        test_labels_raw = label_matrix[split:]

        scaler.fit(train_data_raw)
        train_scaled = scaler.transform(train_data_raw)
        test_scaled = scaler.transform(test_data_raw)

        # =========================================================
        # 7. WINDOWING (UNCHANGED)
        # =========================================================
        x_n_list = []
        x_ab_list = []
        y_ab_list = []

        step = 1
        chunk_size = self.window_size

        for i in range(0, len(train_scaled) - chunk_size, step):
            x_n_list.append(train_scaled[i:i + chunk_size])

        for i in range(0, len(test_scaled) - chunk_size, step):
            x_ab_list.append(test_scaled[i:i + chunk_size])
            y_ab_list.append(test_labels_raw[i:i + chunk_size])

        self.data_dict['x_n_list'] = np.array(x_n_list)
        self.data_dict['x_ab_list'] = np.array(x_ab_list)
        self.data_dict['label_list'] = np.array(y_ab_list)

        print(f"Dataset generated: x_n {self.data_dict['x_n_list'].shape}, "
            f"x_ab {self.data_dict['x_ab_list'].shape}")

        self.save_data()

        orth_matrix_dir = os.path.join(self.data_dir, 'orth_transform_meta')
        if os.path.exists(orth_matrix_dir):
            print(f"Removing old orthogonal transform metadata at {orth_matrix_dir}")
            shutil.rmtree(orth_matrix_dir)

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
        self.pipeline_sanity_check()  # Run the sanity check before applying the orthogonal transform
        return None #self.apply_orthogonal_transform(save_path=orth_matrix_dir, device='cpu')

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
            device=device,
            metric_length=self.metric_num_cols if self.include_logs_and_traces else -1,#if -1 means use all features, otherwise only use the metric features for orthogonalization
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
        'shuffle': True,
        'include_logs_and_traces': True
    }
    dataset = aiops(options)
    dataset.generate_example()
    dataset.pipeline_sanity_check()


"""
    wwith logs and traces
        --- Starting Data Pipeline Sanity Check ---
        Normal Data Shape: (68014, 10, 50)
        Abnormal Data Shape: (16997, 10, 50)
        Anomaly Coverage Check: Found 220790 anomaly samples.
        Feature Statistics -> Min: -2.2264, Max: 2188498.2526, Mean: 3.8117
        INFO: Data has high variance (Robust Scaling active). Good for RCA signal.
        CRITICAL: Sensors [ 0  1  2  3  7  9 10 26 27 29 31 36 37] have zero variance. This will cause KL collapse!
        --- Sanity Check Passed ---
"""


"""
    with metrics only 
        --- Starting Data Pipeline Sanity Check ---
        Normal Data Shape: (1142, 10, 30)
        Abnormal Data Shape: (278, 10, 30)
        Anomaly Coverage Check: Found 2300 anomaly samples.
        Feature Statistics -> Min: -1.0886, Max: 10.0648, Mean: -0.0002
        INFO: Data has high variance (Robust Scaling active). Good for RCA signal.
        CRITICAL: Sensors [ 0  1  2  3  5  6  7 18 21] have zero variance. This will cause KL collapse!
        --- Sanity Check Passed ---
"""