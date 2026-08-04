import os
from matplotlib.pyplot import step
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from layers.vlinear_arch import OrthTransform 

class WADI:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.num_vars = options.get('num_vars', 127) # WADI typically has 127 sensors
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options['shuffle']
        self.data_dir_path_modified_with_window_var = False

    def clean_column_names(self, df):
        """Removes the long system paths from WADI column headers."""
        new_cols = []
        for col in df.columns:
            if '\\' in col:
                new_cols.append(col.split('\\')[-1])
            else:
                new_cols.append(col.strip())
        df.columns = new_cols
        return df

    def generate_example(self):
        # ----------------------------
        # 1. Paths
        # ----------------------------
        normal_path = os.path.join(self.data_dir, 'WADI_14days.csv')
        attack_path = os.path.join(self.data_dir, 'WADI_attackdata.csv')
        label_info_path = os.path.join(self.data_dir, 'attack_description.xlsx')

        # ----------------------------
        # 2. Load Normal Data
        # ----------------------------
        temp_headers = pd.read_csv(normal_path, skiprows=3, nrows=0).columns.tolist()
        df_normal = pd.read_csv(normal_path, skiprows=1000, names=temp_headers)
        df_normal = self.clean_column_names(df_normal)

        df_normal.fillna(0, inplace=True)
        df_normal.drop(columns=[c for c in ['Row', 'Date', 'Time'] if c in df_normal.columns], inplace=True)
        df_normal.ffill(inplace=True)
        df_normal.bfill(inplace=True)

        print(f"Normal shape: {df_normal.shape}")

        # ----------------------------
        # 3. Load Attack Data
        # ----------------------------
        df_attack = pd.read_csv(attack_path)
        df_attack = self.clean_column_names(df_attack)

        df_attack.fillna(0, inplace=True)
        df_attack.ffill(inplace=True)
        df_attack.bfill(inplace=True)

        df_attack['Datetime'] = pd.to_datetime(df_attack['Date'] + ' ' + df_attack['Time'])

        print(f"Attack shape: {df_attack.shape}")

        # ----------------------------
        # 4. Align Features
        # ----------------------------
        common_sensors = df_normal.columns.tolist()
        df_attack_features = df_attack[common_sensors].copy()

        # ----------------------------
        # 5. Labeling (FULL RESOLUTION) ✅
        # ----------------------------
        labels = np.zeros((len(df_attack), len(common_sensors)))

        if os.path.exists(label_info_path):
            df_label_meta = pd.read_excel(label_info_path, skiprows=4)
            df_label_meta = df_label_meta.dropna(subset=['Start Time', 'End Time'])

            col_map = {col.upper(): i for i, col in enumerate(common_sensors)}

            for _, row in df_label_meta.iterrows():
                date_val = str(row['Date']).split(' ')[0]
                start_dt = pd.to_datetime(date_val + ' ' + str(row['Start Time']))
                end_dt   = pd.to_datetime(date_val + ' ' + str(row['End Time']))

                points = str(row['Attack Point (s)']) \
                            .replace('\n', ',') \
                            .replace('and', ',') \
                            .split(',')

                mask = (df_attack['Datetime'] >= start_dt) & (df_attack['Datetime'] <= end_dt)
                idx_range = np.where(mask)[0]

                if len(idx_range) > 0:
                    for p in points:
                        p_clean = p.strip().upper()
                        matched_col = next(
                            (c_idx for name, c_idx in col_map.items()
                            if p_clean in name.replace('_', '')),
                            None
                        )
                        if matched_col is not None:
                            labels[idx_range, matched_col] = 1

        # ----------------------------
        # 6. Downsampling (ONCE ONLY) ✅
        # ----------------------------
        sample_rate = 20

        df_normal = df_normal.iloc[::sample_rate].reset_index(drop=True)
        df_attack_features = df_attack_features.iloc[::sample_rate].reset_index(drop=True)
        df_attack = df_attack.iloc[::sample_rate].reset_index(drop=True)
        labels = labels[::sample_rate]

        # ----------------------------
        # 7. Scaling
        # ----------------------------
        scaler = StandardScaler()
        #df_normal = df_normal + np.random.normal(0, 1e-6, df_normal.shape)

        scaler.fit(df_normal.values)
        scaler.scale_[scaler.scale_ < 1e-4] = 1.0

        scaled_normal = scaler.transform(df_normal.values)
        scaled_attack = scaler.transform(df_attack_features.values)

        scaled_normal = np.clip(scaled_normal, -15, 15)
        scaled_attack = np.clip(scaled_attack, -15, 15)

        print(f"Max scaled: {np.max(scaled_attack)} | Min scaled: {np.min(scaled_attack)}")

        # ----------------------------
        # 8. Segment Normal Data
        # ----------------------------
        x_n_list = [
            scaled_normal[i:i+self.window_size]
            for i in range(
                0,
                len(scaled_normal)-self.window_size,
                self.window_size
            )
        ]
        self.data_dict['x_n_list'] = np.array(x_n_list)
        

        # ----------------------------
        # 9. Segment Attack Data (SWaT-style) ✅
        # ----------------------------
        global_labels = (labels.sum(axis=1) > 0).astype(int)
        anomaly_starts = np.where(np.diff(global_labels) > 0)[0] + 1  # fix alignment

        test_x_lst = []
        test_y_lst = []

        #for start_idx in anomaly_starts:
        #    s = int(start_idx - 2 * self.window_size)
        #    e = int(start_idx + 1 * self.window_size)
#
        #    if s >= 0 and e <= len(scaled_attack):
        #        test_x_lst.append(scaled_attack[s:e])
        #        test_y_lst.append(labels[s:e])
        for onset in anomaly_starts:
            start_idx = int(onset - self.window_size//2)
            end_idx = int(onset + self.window_size//2)
            if start_idx >=0 and end_idx <= len(scaled_attack):
                test_x_lst.append(
                    scaled_attack[start_idx:end_idx]
                )
                test_y_lst.append(
                    labels[start_idx:end_idx]
                )
        self.data_dict['x_ab_list'] = np.array(test_x_lst)
        self.data_dict['label_list'] = np.array(test_y_lst)

        # ----------------------------
        # 10. Metadata
        # ----------------------------
        self.num_vars = df_normal.shape[1]

        self.binary_flags = np.array([
            1 if df_normal[col].nunique() <= 2 else 0
            for col in common_sensors
        ])

        # ----------------------------
        # 11. Shuffle
        # ----------------------------
        if self.shuffle:
            np.random.seed(self.seed)
            idx = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][idx]

    def save_data(self):
        """
        Save the processed data arrays to .npy files in the data directory.
        Matches the implementation in swat.py and smd.py.
        """
        self.data_dir = os.path.join(self.data_dir, f"window_{self.window_size}_vars_{self.num_vars}")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.data_dir_path_modified_with_window_var = True
        
        # Save the primary data lists
        for key in ['x_n_list', 'x_ab_list', 'label_list']:
            if key in self.data_dict:
                np.save(os.path.join(self.data_dir, f'{key}.npy'), self.data_dict[key])
        
        # Save binary flags as they are needed for the OrthTransform initialization
        if hasattr(self, 'binary_flags'):
            np.save(os.path.join(self.data_dir, 'binary_flags.npy'), self.binary_flags)

    def load_data(self):
        """
        Loads saved .npy files and automatically applies OrthTransform.
        Follows the 'Silent Operator' pattern to ensure model readiness.
        """
        # 1. Load standard lists
        if not self.data_dir_path_modified_with_window_var:
            self.data_dir = os.path.join(self.data_dir, f"window_{self.window_size}_vars_{self.num_vars}")
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
        
        # 2. Load metadata needed for OrthTransform
        if os.path.exists(os.path.join(self.data_dir, 'binary_flags.npy')):
            self.binary_flags = np.load(os.path.join(self.data_dir, 'binary_flags.npy'))

        # 3. Define path for the Q matrix (orthogonal metadata)
        orth_matrix_dir = os.path.join(self.data_dir, 'orth_transform_meta')
        
        # 4. Immediately apply/re-initialize the OrthTransform
        # This matches the load_data behavior in your smd.py and swat.py
        device = 'cpu'
        self.pipeline_sanity_check()
        return self.apply_orthogonal_transform(save_path=orth_matrix_dir, device=device)
    

    def apply_orthogonal_transform(self, save_path, device='cpu'):
        os.makedirs(save_path, exist_ok=True)
        self.orth_transformer = OrthTransform(
            dataset_obj=self, 
            time_lag=self.window_size,
            save_path=save_path, 
            device=device
        )
        x_n_tensor = torch.from_numpy(self.data_dict['x_n_list']).float().to(device)
        x_ab_tensor = torch.from_numpy(self.data_dict['x_ab_list']).float().to(device)
        
        with torch.no_grad():
            self.data_dict['x_n_orth'] = self.orth_transformer(x_n_tensor).cpu().numpy()
            self.data_dict['x_ab_orth'] = self.orth_transformer(x_ab_tensor).cpu().numpy()
        
        return self.orth_transformer
    

    def pipeline_sanity_check(self):
        print("\n--- Starting SWaT Data Pipeline Sanity Check ---")

        x_n = self.data_dict['x_n_list']
        x_ab = self.data_dict['x_ab_list']
        labels = self.data_dict['label_list']

        # --------------------------------
        # 1. Shape Verification
        # --------------------------------
        print(f"Normal Data Shape:   {x_n.shape}")
        print(f"Abnormal Data Shape: {x_ab.shape}")
        print(f"Label Shape:         {labels.shape}")
        
        print(
            f"Sensor count: {x_n.shape[-1]}"
        )
        assert x_n.ndim == 3, "Normal data must be [Batch, Window, Sensors]"
        assert x_ab.ndim == 3, "Abnormal data must be [Batch, Window, Sensors]"
        assert labels.ndim == 3, "Labels must be [Batch, Window, Sensors]"

        assert x_ab.shape == labels.shape, \
            f"Abnormal data and labels mismatch: {x_ab.shape} vs {labels.shape}"

        assert x_n.shape[2] == x_ab.shape[2], \
            "Normal and abnormal sensor dimensions do not match"

        assert x_n.shape[1] == self.window_size, \
            f"Normal window mismatch: expected {self.window_size}, got {x_n.shape[1]}"

        assert x_ab.shape[1] == self.window_size, \
            f"Abnormal window mismatch: expected {self.window_size}, got {x_ab.shape[1]}"


        # --------------------------------
        # 2. Root Cause Label Coverage
        # --------------------------------
        anomaly_samples = np.sum(labels)

        print(f"Total root-cause labels: {int(anomaly_samples)}")

        if anomaly_samples == 0:
            print(
                "WARNING: No root cause labels detected. "
                "RCA evaluation will not be possible."
            )
        else:
            abnormal_windows = np.where(
                np.any(labels == 1, axis=(1,2))
            )[0]

            print(
                f"Abnormal windows with root causes: "
                f"{len(abnormal_windows)}/{len(labels)}"
            )


        # --------------------------------
        # 3. Feature Statistics
        # --------------------------------
        feat_min = x_n.min()
        feat_max = x_n.max()
        feat_mean = x_n.mean()
        feat_std = x_n.std()

        print(
            f"Normal Feature Statistics -> "
            f"Min: {feat_min:.4f}, "
            f"Max: {feat_max:.4f}, "
            f"Mean: {feat_mean:.4f}, "
            f"Std: {feat_std:.4f}"
        )

        if abs(feat_max) > 10 or abs(feat_min) > 10:
            print(
                "INFO: Large values detected. "
                "Check scaling/clipping."
            )


        # --------------------------------
        # 4. Sensor Variance Check
        # --------------------------------
        variances = np.var(
            x_n,
            axis=(0,1)
        )

        dead_sensors = np.where(
            variances == 0
        )[0]

        if len(dead_sensors) > 0:
            print(
                f"CRITICAL: Sensors with zero variance: "
                f"{dead_sensors}"
            )
        else:
            print(
                "Variance Check: All sensors are active."
            )


        # --------------------------------
        # 5. Window Length Distribution
        # --------------------------------
        normal_lengths = np.unique(
            [x.shape[0] for x in x_n]
        )

        abnormal_lengths = np.unique(
            [x.shape[0] for x in x_ab]
        )

        print(
            f"Normal window lengths: {normal_lengths}"
        )

        print(
            f"Abnormal window lengths: {abnormal_lengths}"
        )



        print("--- WADI Data Pipeline Sanity Check Passed ---\n")

