import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from layers.vlinear_arch import OrthTransform

# Assuming OrthTransform is defined in your framework codebase
# from transforms import OrthTransform


class BATADAL:
    def __init__(self, options):
        """
        Initialize the BATADAL dataset processing class with the given options.

        Parameters:
        - options (dict): A dictionary containing keys such as 'seed', 'num_vars',
                          'data_dir', 'window_size', and 'shuffle'.
        """
        self.options = options
        self.data_dict = {}
        self.seed = options.get('seed', 42)
        self.num_vars = options.get('num_vars', 43)
        self.data_dir = options['data_dir']
        self.window_size = options.get('window_size', 100)
        self.shuffle = options.get('shuffle', False)
        self.data_dir_path_modified_with_window_var = False

    def _get_attack_metadata(self):
        """
        Constructs ground truth attack metadata based on BATADAL Table 2 definitions.
        Maps attack periods to specific targeted sensors/actuators.
        """
        attacks = [
            {
                'id': 1,
                'start': '2016-09-13 23:00',
                'end': '2016-09-16 00:00',
                'targets': ['L_T7', 'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11']
            },
            {
                'id': 2,
                'start': '2016-09-26 11:00',
                'end': '2016-09-27 10:00',
                'targets': ['L_T7', 'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11']
            },
            {
                'id': 3,
                'start': '2016-10-09 09:00',
                'end': '2016-10-11 20:00',
                'targets': ['L_T1', 'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2']
            },
            {
                'id': 4,
                'start': '2016-10-29 19:00',
                'end': '2016-11-02 16:00',
                'targets': ['L_T1', 'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2', 'P_J280']
            },
            {
                'id': 5,
                'start': '2016-11-26 17:00',
                'end': '2016-11-29 04:00',
                'targets': ['F_PU7', 'S_PU7', 'L_T4']
            },
            {
                'id': 6,
                'start': '2016-12-06 07:00',
                'end': '2016-12-10 04:00',
                'targets': ['F_PU7', 'S_PU7', 'L_T4']
            },
            {
                'id': 7,
                'start': '2016-12-14 15:00',
                'end': '2016-12-19 04:00',
                'targets': ['L_T1', 'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2', 'F_PU7', 'S_PU7', 'L_T4']
            }
        ]
        return pd.DataFrame(attacks)

    def generate_example(self):
        """
        Generate examples by loading, cleaning, and processing BATADAL dataset files.
        Stores output arrays in self.data_dict.
        """
        # ----------------------------
        # Load Normal and Abnormal Data
        # ----------------------------
        normal_csv = os.path.join(self.data_dir, 'BATADAL_dataset03.csv')
        abnormal_csv = os.path.join(self.data_dir, 'BATADAL_dataset04.csv')

        df_normal = pd.read_csv(normal_csv, header=0)
        df_abnormal = pd.read_csv(abnormal_csv, header=0)

        # Strip whitespace from column names
        df_normal.columns = df_normal.columns.str.strip()
        df_abnormal.columns = df_abnormal.columns.str.strip()

        # Extract telemetry feature column names (excluding DATETIME and ATT_FLAG)
        feature_cols = [c for c in df_normal.columns if c not in ['DATETIME', 'ATT_FLAG']]
        col_dic = {col: i for i, col in enumerate(feature_cols)}

        # Set binary flags (1 for binary operational states S_*, 0 for continuous)
        self.binary_flags = np.array([1 if df_abnormal[col].nunique() <= 2 else 0 for col in feature_cols])

        # ----------------------------
        # Clean Normal Data
        # ----------------------------
        df_normal_features = df_normal[feature_cols].copy()

        # ----------------------------
        # Prepare Abnormal Data & Root Cause Labels
        # ----------------------------
        df_abnormal['Adjusted Timestamp'] = pd.to_datetime(df_abnormal['DATETIME'], format='%d/%m/%y %H')
        df_abnormal_features = df_abnormal[feature_cols].copy()

        # Initialize full 2D label matrix [TimeSteps, Sensors]
        labels = np.zeros(df_abnormal_features.values.shape)

        df_label_clean = self._get_attack_metadata()

        for i in range(len(df_label_clean)):
            lower = pd.to_datetime(df_label_clean.iloc[i]['start'])
            upper = pd.to_datetime(df_label_clean.iloc[i]['end'])
            attack_targets = df_label_clean.iloc[i]['targets']

            attack_indices = df_abnormal[
                (df_abnormal['Adjusted Timestamp'] >= lower) &
                (df_abnormal['Adjusted Timestamp'] <= upper)
            ].index.values

            if len(attack_indices) > 0:
                for target in attack_targets:
                    if target in col_dic:
                        labels[attack_indices, col_dic[target]] = 1

        # ----------------------------
        # Process Each Attack Event into Window Samples
        # ----------------------------
        test_x_lst = []
        test_label_lst = []

        sampling_rate = 1

        for i in range(len(df_label_clean)):
            lower = pd.to_datetime(df_label_clean.iloc[i]['start'])
            upper = pd.to_datetime(df_label_clean.iloc[i]['end'])

            index_lst = np.array(df_abnormal[
                (df_abnormal['Adjusted Timestamp'] >= lower) &
                (df_abnormal['Adjusted Timestamp'] <= upper)
            ].index.values)

            if len(index_lst) > 0:
                onset = min(index_lst)
                start_idx = int(onset - (self.window_size // 2) * sampling_rate)
                end_idx = int(onset + (self.window_size // 2) * sampling_rate)

                # Boundary checking
                if start_idx >= 0 and end_idx <= len(df_abnormal_features):
                    test_x_lst.append(
                        df_abnormal_features.iloc[start_idx:end_idx:sampling_rate].values
                    )

                    sampled_labels = []
                    for k in range(self.window_size):
                        chunk = labels[start_idx + k * sampling_rate : start_idx + (k + 1) * sampling_rate]
                        sampled_labels.append(chunk.max(axis=0))

                    test_label_lst.append(np.array(sampled_labels))

        # ----------------------------
        # Process Normal Data: Segment and Scale
        # ----------------------------
        x_n_list = [
            df_normal_features.iloc[i : i + self.window_size].values
            for i in range(0, len(df_normal_features), self.window_size)
            if i + self.window_size <= len(df_normal_features)
        ]

        scaler = StandardScaler()
        scaler.fit(np.concatenate(x_n_list, axis=0))

        x_n_list = [scaler.transform(segment) for segment in x_n_list]
        test_x_lst = [scaler.transform(example) for example in test_x_lst]

        # ----------------------------
        # Store Processed Data
        # ----------------------------
        self.data_dict['x_n_list'] = np.array(x_n_list)
        self.data_dict['x_ab_list'] = np.array(test_x_lst)
        self.data_dict['label_list'] = np.array(test_label_lst)

        if self.shuffle:
            np.random.seed(self.seed)
            indices = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][indices]

    def save_data(self):
        """Save the processed data arrays to .npy files in the data directory."""
        self.data_dir = os.path.join(self.data_dir, f"window_{self.window_size}_vars_{self.num_vars}")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.data_dir_path_modified_with_window_var = True

        np.save(os.path.join(self.data_dir, 'x_n_list'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'label_list'), self.data_dict['label_list'])

    def apply_orthogonal_transform(self, save_path, device='cpu'):
        """Projects windowed data into the orthogonal domain using the Q matrix."""
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
        
        print(f"Orthogonal transformation complete. Shape: {self.data_dict['x_n_orth'].shape}")
        return self.orth_transformer

    def load_data(self):
        """Loads saved .npy files and applies OrthTransform."""
        if not self.data_dir_path_modified_with_window_var:
            self.data_dir = os.path.join(self.data_dir, f"window_{self.window_size}_vars_{self.num_vars}")
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))

        orth_matrix_dir = os.path.join(self.data_dir, 'orth_transform_meta')
        device = 'cpu'
        self.pipeline_sanity_check()
        if self.options.get('disable_orth_proj', False):
            print("Orthogonal projection disabled. Skipping transformation.")
            return None
        return self.apply_orthogonal_transform(save_path=orth_matrix_dir, device=device)

    def pipeline_sanity_check(self):
        print("\n--- Starting BATADAL Data Pipeline Sanity Check ---")

        x_n = self.data_dict['x_n_list']
        x_ab = self.data_dict['x_ab_list']
        labels = self.data_dict['label_list']

        print(f"Normal Data Shape:   {x_n.shape}")
        print(f"Abnormal Data Shape: {x_ab.shape}")
        print(f"Label Shape:         {labels.shape}")

        assert x_n.ndim == 3, "Normal data must be [Batch, Window, Sensors]"
        assert x_ab.ndim == 3, "Abnormal data must be [Batch, Window, Sensors]"
        assert labels.ndim == 3, "Labels must be [Batch, Window, Sensors]"
        assert x_ab.shape == labels.shape, f"Abnormal data and labels mismatch: {x_ab.shape} vs {labels.shape}"
        assert x_n.shape[2] == x_ab.shape[2], "Normal and abnormal sensor dimensions do not match"
        assert x_n.shape[1] == self.window_size, f"Normal window mismatch: expected {self.window_size}, got {x_n.shape[1]}"
        assert x_ab.shape[1] == self.window_size, f"Abnormal window mismatch: expected {self.window_size}, got {x_ab.shape[1]}"

        anomaly_samples = np.sum(labels)
        print(f"Total root-cause labels: {int(anomaly_samples)}")

        if anomaly_samples == 0:
            print("WARNING: No root cause labels detected. RCA evaluation will not be possible.")
        else:
            abnormal_windows = np.where(np.any(labels == 1, axis=(1, 2)))[0]
            print(f"Abnormal windows with root causes: {len(abnormal_windows)}/{len(labels)}")

        feat_min, feat_max = x_n.min(), x_n.max()
        feat_mean, feat_std = x_n.mean(), x_n.std()
        print(f"Normal Feature Statistics -> Min: {feat_min:.4f}, Max: {feat_max:.4f}, Mean: {feat_mean:.4f}, Std: {feat_std:.4f}")

        variances = np.var(x_n, axis=(0, 1))
        dead_sensors = np.where(variances == 0)[0]
        if len(dead_sensors) > 0:
            print(f"CRITICAL: Sensors with zero variance: {dead_sensors}")
        else:
            print("Variance Check: All sensors are active.")

        print("--- BATADAL Sanity Check Passed ---\n")