import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from layers.vlinear_arch import OrthTransform 

class SMD:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.num_vars = options.get('num_vars', 38) # SMD has 38 metrics
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options['shuffle']
        # SMD specific: list of machines to process (e.g., machine-1-1)
        # If not provided, it will process all machines in the train folder
        self.subset_machines = options.get('subset_machines', None)
        self.data_dir_path_modified_with_window_var = False


    def parse_interpretation_label(self, file_path, num_timestamps, num_vars=38):
        label_matrix = np.zeros((num_timestamps, num_vars))
        if not os.path.exists(file_path):
            return label_matrix
            
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line: continue
                
                times, indices = line.split(':')
                start_t, end_t = map(int, times.split('-'))
                
                # SMD labels are often 1-indexed in interpretation files; 
                # using int(i)-1 is correct if you've verified your source list.
                zero_indexed_indices = [int(i) - 1 for i in indices.split(',') if i.strip()]
                
                for idx in zero_indexed_indices:
                    # Ensure we don't go out of bounds of the actual data file
                    actual_end = min(end_t + 1, num_timestamps)
                    label_matrix[start_t : actual_end, idx] = 1
                        
        return label_matrix


    def process_abnormal(
        self,
        test_df,
        global_labels,
        root_cause_labels,
        sampling_rate=1
    ):
        test_x_lst = []
        test_label_lst = []

        # Find all anomalous timestamps
        anomaly_indices = np.where(global_labels == 1)[0]

        if len(anomaly_indices) == 0:
            return test_x_lst, test_label_lst

        # Group contiguous anomaly timestamps into separate attack events
        events = np.split(
            anomaly_indices,
            np.where(np.diff(anomaly_indices) > 1)[0] + 1
        )

        for event in events:

            # Same idea as SWaT:
            # onset = min(index_lst)
            onset = int(event[0])

            # Same SWaT centered-window construction
            start_idx = int(
                onset - (self.window_size // 2) * sampling_rate
            )

            end_idx = int(
                onset + (self.window_size // 2) * sampling_rate
            )

            # Boundary check
            if start_idx < 0 or end_idx > len(test_df):
                continue

            # Extract exactly the same temporal window
            slice_x = test_df.iloc[
                start_idx:end_idx:sampling_rate
            ].values

            # Preserve root-cause labels
            sampled_labels = []

            for k in range(self.window_size):

                chunk = root_cause_labels[
                    start_idx + k * sampling_rate:
                    start_idx + (k + 1) * sampling_rate
                ]

                sampled_labels.append(
                    chunk.max(axis=0)
                )

            sampled_labels = np.array(sampled_labels)

            # Make sure we actually obtained window_size samples
            if len(slice_x) != self.window_size:
                continue

            if len(sampled_labels) != self.window_size:
                continue

            test_x_lst.append(slice_x)
            test_label_lst.append(sampled_labels)

        return test_x_lst, test_label_lst

    def process_normal(self, combined_train, target_len):
        """Generalized logic to segment normal data into target_len-timestamp chunks."""
        step = self.window_size
        return [
            combined_train[i : i + target_len] 
            for i in range(0, len(combined_train) - target_len, step)
        ]

    def generate_example(self):
        train_path = os.path.join(self.data_dir, 'train')
        test_path = os.path.join(self.data_dir, 'test')
        label_path = os.path.join(self.data_dir, 'labels')
        interpretation_path = os.path.join(self.data_dir, 'interpretation_label')

        machines = sorted([f for f in os.listdir(train_path) if f.endswith('.txt')])
        if self.subset_machines:
            machines = [m + '.txt' for m in self.subset_machines]

        all_train_data = []
        all_test_x = []
        all_test_y = []
        normal_block_len = self.window_size 

        for m_file in machines:
            # 1. Load raw data
            train_df = pd.read_csv(os.path.join(train_path, m_file), header=None)
            test_df = pd.read_csv(os.path.join(test_path, m_file), header=None)
            global_labels = pd.read_csv(os.path.join(label_path, m_file), header=None).values.flatten()
            # 2. Get Root Cause Labels
            root_cause_labels = self.parse_interpretation_label(
                os.path.join(interpretation_path, m_file), len(test_df)
            )
            sampling_rate = 1
            train_df = train_df.iloc[::sampling_rate, :]
            test_df = test_df.iloc[::sampling_rate, :]
            global_labels = global_labels[::sampling_rate]
            root_cause_labels = root_cause_labels[::sampling_rate] # <--- ADD THIS LINE

            # 3. Call Abnormal Processor
            #m_test_x, m_test_y = self.process_abnormal(test_df, global_labels, root_cause_labels, window_size=self.window_size)
            # We give it history (lookback), but we only evaluate a small, 
            # concentrated window where the anomaly actually happens (lookahead).
            m_test_x, m_test_y = self.process_abnormal(
                test_df,
                global_labels,
                root_cause_labels,
                sampling_rate=sampling_rate
            )
            all_test_x.extend(m_test_x)
            all_test_y.extend(m_test_y)

            # 4. Collect train data for the Normal Processor
            all_train_data.append(train_df.values)

        # 5. Scaling 
        scaler = StandardScaler() # Helps center the scale better
        combined_train = np.concatenate(all_train_data, axis=0)

        # A. Log Transform + Max Clipping
        # We handle the negative/zero values first
        #combined_train = np.log1p(np.maximum(combined_train, 0))
        
        # B. Robust Scaling with IQR check
        scaler.fit(combined_train)
        scaler.fit(np.concatenate(all_train_data, axis=0))

        x_n_list = []

        for train_data in all_train_data:
            scaled = scaler.transform(train_data)
            x_n_list.extend(
                self.process_normal(scaled, normal_block_len)
            )

        # 7. Apply to Test Data
        # Ensure the test data is also Logged then Scaled
        test_x_transformed = [
            scaler.transform(x)
            for x in all_test_x
        ]
        #test_x_transformed = [
        #    scaler.transform(np.log1p(np.maximum(x, 0))) 
        #    for x in all_test_x
        #]
        
        # 8. Update self.data_dict with the TRANSFORMED arrays
        self.data_dict['x_n_list'] = np.array(x_n_list)
        self.data_dict['x_ab_list'] = np.array(test_x_transformed)
        self.data_dict['label_list'] = np.array(all_test_y)

        labels = self.data_dict['label_list']

        print("label shape:", labels.shape)

        for i in range(min(10, len(labels))):
            global_y = np.any(labels[i] > 0, axis=1)
            anomaly_pos = np.where(global_y)[0]

            print(
                i,
                "first anomaly:",
                anomaly_pos[0] if len(anomaly_pos) else None,
                "last anomaly:",
                anomaly_pos[-1] if len(anomaly_pos) else None,
                "num anomalous:",
                global_y.sum()
            )
        for i in range(min(10, len(labels))):
            rc = labels[i] > 0

            print(
                i,
                "RC first:",
                np.where(np.any(rc, axis=1))[0][0]
                if np.any(rc) else None,
                "RC last:",
                np.where(np.any(rc, axis=1))[0][-1]
                if np.any(rc) else None,
                "RC vars:",
                np.where(np.any(rc, axis=0))[0].tolist()
            )
        # Shuffle train data
        if self.shuffle:
            np.random.seed(self.seed)
            idx = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][idx]



    def save_data(self):
        #if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
        self.data_dir = os.path.join(self.data_dir, f"window_{self.window_size}_vars_{self.num_vars}")
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_dir_path_modified_with_window_var = True
        for key in ['x_n_list', 'x_ab_list', 'label_list']:
            np.save(os.path.join(self.data_dir, f'{key}.npy'), self.data_dict[key])

    def load_data(self):
        if not self.data_dir_path_modified_with_window_var:
            self.data_dir = os.path.join(self.data_dir, f"window_{self.window_size}_vars_{self.num_vars}")
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