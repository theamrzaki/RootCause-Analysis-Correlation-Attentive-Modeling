from torch import nn
import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from layers.vlinear_arch import OrthTransform # Assuming you save the previous code there

class SWaT:
    def __init__(self, options):
        """
        Initialize the SWaT dataset processing class with the given options.

        Parameters:
        - options (dict): A dictionary containing keys such as 'seed', 'num_vars',
                          'data_dir', 'window_size', and 'shuffle'.
        """
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.num_vars = options['num_vars']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options['shuffle']
        self.data_dir_path_modified_with_window_var = False

    def generate_example(self):
        """
        Generate examples by loading, cleaning, and processing the SWaT dataset.
        This method loads the label, normal, and abnormal data files, performs
        necessary cleaning, scaling, and window slicing operations, and stores
        the processed arrays in self.data_dict.
        """
        # ----------------------------
        # Load Attack Label Data
        # ----------------------------
        label_file = os.path.join(self.data_dir, 'List_of_attacks_Final.xlsx')
        df_label = pd.read_excel(label_file, header=0, index_col=0)


        # ----------------------------
        # Load Normal and Abnormal Data
        # ----------------------------
        normal_csv = os.path.join(self.data_dir, 'SWaT_Normal.csv')
        abnormal_csv = os.path.join(self.data_dir, 'SWaT_Abnormal.csv')

        if os.path.exists(normal_csv) and os.path.exists(abnormal_csv):
            df_normal = pd.read_csv(normal_csv, header=0, index_col=0)
            df_abnormal = pd.read_csv(abnormal_csv, header=0, index_col=0)
        else:
            normal_excel = os.path.join(self.data_dir, 'SWaT_Dataset_Normal_v1.xlsx')
            abnormal_excel = os.path.join(self.data_dir, 'SWaT_Dataset_Attack_v0.xlsx')
            df_normal = pd.read_excel(normal_excel, header=1)
            df_normal.to_csv(normal_csv)
            df_abnormal = pd.read_excel(abnormal_excel, header=1)
            df_abnormal.to_csv(abnormal_csv)

        # ----------------------------
        # Clean Label Data
        # ----------------------------
        # Drop rows where 'Start Time' or 'End Time' is missing
        df_label_clean = df_label.dropna(subset=['Start Time', 'End Time'], how='any').copy()
        # Remove columns not needed for further processing
        df_label_clean.drop(columns=['Start State', 'Attack', 'Expected Impact or attacker intent',
                                      'Unexpected Outcome', 'Actual Change'], inplace=True)
        # Convert 'Start Time' and 'End Time' to datetime for processing
        df_label_clean['Start Time'] = pd.to_datetime(df_label_clean['Start Time'])
        # Construct 'Adjusted End Time' by combining the date from 'Start Time' with the time from 'End Time'
        df_label_clean['Adjusted End Time'] = df_label_clean.apply(
            lambda row: pd.to_datetime(
            row['Start Time'].strftime('%Y-%m-%d') + ' ' + row['End Time'].strftime('%H:%M:%S')), axis=1)
        # Save cleaned label data to CSV
        df_label_clean.to_csv(os.path.join(self.data_dir, 'SWaT_label.csv'))

        # ----------------------------
        # Clean Normal Data
        # ----------------------------
        # Select only rows marked as 'Normal'
        df_normal = df_normal.loc[df_normal['Normal/Attack'] == 'Normal']
        # Drop unnecessary columns and downsample by taking every 10th row
        df_normal.drop(columns=[' Timestamp', 'Normal/Attack'], inplace=True)
        df_normal = df_normal[::3].reset_index(drop=True)

        # ----------------------------
        # Clean Abnormal Data
        # ----------------------------
        # Remove any rows with missing values and reset index
        df_abnormal.dropna(how='any', inplace=True)
        df_abnormal.reset_index(drop=True, inplace=True)
        # Initialize label matrix with zeros; columns from 1 to second-last column are used
        labels = np.zeros(df_abnormal.values[:, 1:-1].shape)
        # Convert the timestamp column to datetime using the given format, then standardize its format
        df_abnormal['Adjusted Timestamp'] = pd.to_datetime(
            df_abnormal[' Timestamp'], format=' %d/%m/%Y %I:%M:%S %p'
        ).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_abnormal['Adjusted Timestamp'] = pd.to_datetime(df_abnormal['Adjusted Timestamp'])

        # ----------------------------
        # Create Column Dictionary for Abnormal Data
        # ----------------------------
        # Create a mapping from cleaned column names (without leading spaces) to their index
        col_dic = {}
        for i in df_abnormal.columns.values[1:-2]:
            col_dic[i.lstrip()] = len(col_dic)


        # --- Add binary/continuous flags ---
        # 1 if binary (e.g., ON/OFF pumps), 0 if continuous
        self.binary_flags = np.array([1 if df_abnormal[col].nunique() == 2 else 0 for col in df_abnormal.columns.values[1:-2]])

        # ----------------------------
        # Process Each Attack Event for Abnormal Data
        # ----------------------------
        test_x_lst = []
        test_label_lst = []

        for i in range(len(df_label_clean)):
            # Define the lower and upper time bounds for the attack event
            lower = df_label_clean.iloc[i]['Start Time']
            upper = df_label_clean.iloc[i]['Adjusted End Time']
            # Extract the list of attack points (column names) from the label data
            attack_lst = df_label_clean.iloc[i]['Attack Point'].split(",")
            # Map attack points to their corresponding column indices
            attack_lst_ind = [col_dic[j.replace('-', '').lstrip().upper()] for j in attack_lst]
            # Find indices in abnormal data where the timestamp is within the attack interval and marked as 'Attack'
            index_lst = np.array(df_abnormal.loc[
                (df_abnormal['Adjusted Timestamp'] >= lower) &
                (df_abnormal['Adjusted Timestamp'] <= upper) &
                (df_abnormal['Normal/Attack'] == 'Attack')
            ].index.values)
            if len(index_lst) > 0:
                # Mark the corresponding attack points in the label matrix as 1 for these indices
                for j in attack_lst_ind:
                    labels[index_lst, j] = 1
                # AERCA had test data function in windows size, while normal not 
                # Define the window for the example based on the minimum index in the attack interval
                #start_idx = int(min(index_lst) - 2 * 10 * self.window_size)
                #end_idx = int(min(index_lst) + 1 * 10 * self.window_size)
                ## Slice the abnormal data and label arrays with a step of 10
                #test_x_lst.append(
                #    df_abnormal.iloc[start_idx:end_idx:10, 1:-2].values
                #)
                #test_label_lst.append(
                #    labels[start_idx:end_idx:10]
                #)

                # Preserve AERCA's 10-step downsampling while making
                # abnormal windows consistent with training window_size.
                sampling_rate = 10
                onset = min(index_lst)

                start_idx = int(onset - (self.window_size//2) * sampling_rate)
                end_idx = int(onset + (self.window_size//2) * sampling_rate)
                if start_idx >= 0 and end_idx <= len(df_abnormal):
                    test_x_lst.append(
                        df_abnormal.iloc[
                            start_idx:end_idx:sampling_rate,
                            1:-2
                        ].values
                    )
                    #to preserve the label information
                    sampled_labels = []
                    for k in range(self.window_size):
                        chunk = labels[
                            start_idx+k*sampling_rate:
                            start_idx+(k+1)*sampling_rate
                        ]

                        sampled_labels.append(
                            chunk.max(axis=0)
                        )

                    sampled_labels = np.array(sampled_labels)
                    test_label_lst.append(sampled_labels)

        # ----------------------------
        # Process Normal Data: Split and Scale
        # ----------------------------
        #fixed segements from AERCA
        # Split normal data into segments of 1000 rows each, ensuring each segment has exactly 1000 rows
        #x_n_list = [
        #    df_normal.iloc[i:i + 1000].values
        #    for i in range(0, len(df_normal), 1000)
        #    if i + 1000 < len(df_normal)
        #]
        x_n_list = [
            df_normal.iloc[i:i + self.window_size].values
            for i in range(0, len(df_normal), self.window_size)
            if i + self.window_size < len(df_normal)
        ]
        # Initialize and fit the StandardScaler on the concatenated normal data segments
        scaler = StandardScaler()
        scaler.fit(np.concatenate(x_n_list, axis=0))
        # Transform each segment of normal data
        x_n_list = [scaler.transform(segment) for segment in x_n_list]
        # Transform each abnormal example using the same scaler
        test_x_lst = [scaler.transform(example) for example in test_x_lst]

        # ----------------------------
        # Store Processed Data in data_dict
        # ----------------------------
        self.data_dict['x_n_list'] = np.array(x_n_list)
        if self.shuffle:
            np.random.seed(self.seed)
            indices = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list']
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][indices]
        #(20, 3, 51)
        self.data_dict['x_ab_list'] = np.array(test_x_lst)
        #(20, 3, 51)
        self.data_dict['label_list'] = np.array(test_label_lst)

    def save_data(self):
        """
        Save the processed data arrays to .npy files in the data directory.
        """
        self.data_dir = os.path.join(self.data_dir, f"window_{self.window_size}_vars_{self.num_vars}")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.data_dir_path_modified_with_window_var = True

        np.save(os.path.join(self.data_dir, 'x_n_list'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'label_list'), self.data_dict['label_list'])

    def apply_orthogonal_transform(self, save_path, device='cpu'):
        """
        Projects windowed data into the orthogonal domain using the Q matrix.
        """
        # Ensure the save directory for the matrix exists
        os.makedirs(save_path, exist_ok=True)

        # 1. Initialize the Transform 
        # It will use self.data_dict['x_n_list'] to compute Q if not saved
        self.orth_transformer = OrthTransform(
            dataset_obj=self, 
            time_lag=self.window_size,
            save_path=save_path, 
            device=device
        )
        
        # 2. Transform Normal Data
        x_n_tensor = torch.from_numpy(self.data_dict['x_n_list']).float().to(device)
        with torch.no_grad():
            #(164, 51, 1000)
            self.data_dict['x_n_orth'] = self.orth_transformer(x_n_tensor).cpu().numpy()
        
        # 3. Transform Abnormal (Attack) Data
        # (20, 3, 51)
        # label = (20, 3, 51)
        x_ab_tensor = torch.from_numpy(self.data_dict['x_ab_list']).float().to(device)
        with torch.no_grad():
            #(20, 51, 3)
            self.data_dict['x_ab_orth'] = self.orth_transformer(x_ab_tensor).cpu().numpy()
        
        print(f"Orthogonal transformation complete. Shape: {self.data_dict['x_n_orth'].shape}")
        return self.orth_transformer

    def load_data(self):
        """
        Loads saved .npy files and immediately applies OrthTransform.
        """
        # Load standard lists
        if not self.data_dir_path_modified_with_window_var:
            self.data_dir = os.path.join(self.data_dir, f"window_{self.window_size}_vars_{self.num_vars}")
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))

        # Define path for the Q matrix specifically
        orth_matrix_dir = os.path.join(self.data_dir, 'orth_transform_meta')
        
        device = 'cpu'
        self.pipeline_sanity_check()
        return self.apply_orthogonal_transform(save_path=orth_matrix_dir, device=device)

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


        print("--- SWaT Sanity Check Passed ---\n")


        