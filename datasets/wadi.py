import os
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
        # 1. Path definitions
        normal_path = os.path.join(self.data_dir, 'WADI_14days.csv')
        attack_path = os.path.join(self.data_dir, 'WADI_attackdata.csv')
        label_info_path = os.path.join(self.data_dir, 'attack_description.xlsx') # Excel containing start/end times

        # 1. FIND THE HEADERS: WADI headers are typically on line 1, 
        # but data starts much later. We read line 0 to get the names.
        temp_headers = pd.read_csv(normal_path,skiprows=3, nrows=0).columns.tolist()

        # 2. LOAD DATA: Skip the metadata lines but use the headers we found.
        # WADI data often has ~1000 lines of junk. 
        df_normal = pd.read_csv(normal_path, skiprows=1000, names=temp_headers)
        df_normal = self.clean_column_names(df_normal)
        
        # 1. Drop columns that are completely empty (fixes WADI trailing column issues)
        df_normal.fillna(0, inplace=True) #df_normal.dropna(axis=1, how='all', inplace=True)
        
        # 2. Drop non-sensor columns
        drop_cols = ['Row', 'Date', 'Time']
        df_normal.drop(columns=[c for c in drop_cols if c in df_normal.columns], inplace=True)
        
        # 3. Fill minor gaps instead of dropping everything
        # WADI sensors sometimes have intermittent missing values
        df_normal.ffill(inplace=True) # Forward fill
        df_normal.bfill(inplace=True) # Backward fill for the very first row if needed
        
        # Now check shape; it should no longer be (0, 130)
        print(f"Shape after cleaning: {df_normal.shape}")

        # 3. Load and Clean Attack Data
        # Load attack data
        df_attack = pd.read_csv(attack_path)
        df_attack = self.clean_column_names(df_attack)

        # 1. Drop columns that are entirely NaN (WADI has many 'Unnamed' empty columns)
        df_attack.fillna(0, inplace=True)#df_attack.dropna(axis=1, how='all', inplace=True)

        # 2. Fill missing values rather than dropping rows
        # During attacks, some sensors might drop packets. ffill ensures we keep the last known state.
        df_attack.ffill(inplace=True)
        df_attack.bfill(inplace=True)

        # 3. Synchronize Timestamps (Crucial for WADI)
        # WADI attack data uses 'Date' and 'Time' columns that must be merged for labeling
        df_attack['Datetime'] = pd.to_datetime(df_attack['Date'] + ' ' + df_attack['Time'])
        print(f"Attack data shape after cleaning: {df_attack.shape}")

        # 4. Labeling Logic
        # Initialize zero matrix for root cause labels (indices of specific sensors)
        #sensor_cols = [col for col in df_attack.columns if col not in ['Row', 'Date', 'Time', 'Datetime']]
        #labels = np.zeros((len(df_attack), len(sensor_cols)))

#        4. Labeling Logic
        # CRITICAL: Only use sensors present in the normal (training) data

        # --- THE COMPRESSOR: Manual Downsampling ---

        common_sensors = df_normal.columns.tolist()
        df_attack_features = df_attack[common_sensors].copy()
        
        # Initialize labels based on the training feature count
        labels = np.zeros((len(df_attack), len(common_sensors)))

        # WADI 1s data is too 'slow'. 10s or 30s is the sweet spot.
        sample_rate = 20#0 # 10 seconds

        df_normal = df_normal.iloc[::sample_rate, :].reset_index(drop=True)
        df_attack_features = df_attack_features.iloc[::sample_rate, :].reset_index(drop=True)
        labels = labels[::sample_rate]
        # IMPORTANT: keep timestamps aligned for labeling/debugging
        df_attack = df_attack.iloc[::sample_rate, :].reset_index(drop=True)

        # Important: You must also downsample your labels to match the new length!
        #labels = labels[::sample_rate]


        # If you have an external Excel for WADI labels (similar to SWaT):
        if os.path.exists(label_info_path):
            # Load the sheet. We skip the very top summary rows.
            df_label_meta = pd.read_excel(label_info_path, skiprows=4)
            
            # 1. Clean the Metadata: Keep only rows with a valid S.No or Start Time
            # This ignores the "Attack sequence..." text rows
            df_label_meta = df_label_meta.dropna(subset=['Start Time', 'End Time'])
            
            # 2. Fix the Columns: Ensure we map correctly to the image headers
            # From image: 'Start Time', 'End Time', 'Attack Point(s)'
            #sensor_cols = [col for col in df_attack.columns if col not in ['Date', 'Time', 'Datetime']]
            #col_map = {col.upper(): i for i, col in enumerate(sensor_cols)}
            # Map sensor names to indices using the synchronized common_sensors list
            col_map = {col.upper(): i for i, col in enumerate(common_sensors)}
            for _, row in df_label_meta.iterrows():
                # Parse Date and Time. The image shows Date as M-D-YY
                date_val = str(row['Date']).split(' ')[0] 
                start_dt = pd.to_datetime(date_val + ' ' + str(row['Start Time']))
                end_dt = pd.to_datetime(date_val + ' ' + str(row['End Time']))
                # 3. Handle multiple sensors in one row (e.g., 2MCV101, 2MCV201)
                # We replace newlines and 'and' with commas for splitting
                points = str(row['Attack Point (s)']).replace('\n', ',').replace('and', ',').split(',')
                
                mask = (df_attack['Datetime'] >= start_dt) & (df_attack['Datetime'] <= end_dt)
                idx_range = np.where(mask)[0]
                
                if len(idx_range) > 0:
                    for p in points:
                        p_clean = p.strip().upper()
                        # Remove underscores from the column names during the search to find a match
                        matched_col = next((c_idx for name, c_idx in col_map.items() 
                                        if p_clean in name.replace('_', '')), None)
                        
                        if matched_col is not None:
                            labels[idx_range, matched_col] = 1


        # remove "row", "date", "time" columns from attack data if they exist, since they are not part of the features
        # Now drop the metadata columns before scaling
        drop_cols = ['Row', 'Date', 'Time','Datetime']
        df_attack.drop(columns=[c for c in drop_cols if c in df_attack.columns], inplace=True)
        # 5. Scaling
        scaler = StandardScaler()
        df_normal = df_normal + np.random.normal(0, 1e-6, df_normal.shape)
        scaler.fit(df_normal.values)
        # --- THE SHIELD: Stabilize Constant Sensors ---
        # If std is too small (< 1e-4), force the scale to 1.0. 
        # This prevents the 54,000+ explosion.
        scaler.scale_[scaler.scale_ < 1e-4] = 1.0
        
        scaled_normal = scaler.transform(df_normal.values)
        scaled_attack = scaler.transform(df_attack_features.values)

        # --- THE SAFETY RAIL: Clip Extreme Outliers ---
        # No legitimate anomaly needs to be 54,000 standard deviations away.
        # Clipping to 15 or 20 keeps it as a "Strong Anomaly" but makes POT happy.
        scaled_normal = np.clip(scaled_normal, -15, 15)
        scaled_attack = np.clip(scaled_attack, -15, 15)

        print(f"Max scaled value: {np.max(scaled_attack)}")
        print(f"Min scaled value: {np.min(scaled_attack)}")

        # 6. Segmenting (Normal)
        normal_block_len = 1000
        self.data_dict['x_n_list'] = np.array([
            scaled_normal[i : i + normal_block_len]
            for i in range(0, len(scaled_normal), normal_block_len)
            if i + normal_block_len <= len(scaled_normal)
        ])

        # 7. Segmenting (Abnormal/Attack)
        # Using the SWaT windowing logic: find start of attack, take history + lookahead
        # This assumes global labels are derived from the 'labels' matrix sum
        global_labels = (labels.sum(axis=1) > 0).astype(int)
        anomaly_starts = np.where(np.diff(global_labels) > 0)[0]
        
        test_x_lst = []
        test_y_lst = []
        
        # Match the logic used in SMD (lookback=window_size*2, lookahead=5)
        # This creates the '7' or '3' middle dimension seen in your successful tests
        lookback = self.window_size * 20 # If window_size=1, this is 2
        lookahead = 200                  # Change to 1 for SWaT-style '3' total length
        
        test_x_lst = []
        test_y_lst = []

        for start_idx in anomaly_starts:
            s = start_idx - lookback
            e = start_idx + lookahead
            
            if s >= 0 and e <= len(scaled_attack):
                test_x_lst.append(scaled_attack[s:e])
                test_y_lst.append(labels[s:e])

        self.data_dict['x_ab_list'] = np.array(test_x_lst)
        self.data_dict['label_list'] = np.array(test_y_lst)

        # 8. Metadata for Orthogonal Transform
        # Force synchronization with the actual dataframe shape (123)
        self.num_vars = df_normal.shape[1]
        
        # Match the binary flag logic using common_sensors to avoid indexing errors
        self.binary_flags = np.array([
            1 if df_normal[col].nunique() <= 2 else 0 
            for col in common_sensors
        ])

        if self.shuffle:
            np.random.seed(self.seed)
            idx = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][idx]

    def save_data(self):
        """
        Save the processed data arrays to .npy files in the data directory.
        Matches the implementation in swat.py and smd.py.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
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