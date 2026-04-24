import os
import glob
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from layers.vlinear_arch import OrthTransform 

class TrainTicket:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options['shuffle']
        
        # Step 1: Pre-scan for Global Schema (Essential for Train-Ticket)
        self.global_cols, self.num_vars = self._get_global_schema()
        print(f"Detected {self.num_vars} unique metrics across all scenarios.")

    def _get_global_schema(self):
        """Finds the union of all columns across all data.csv files."""
        all_cols = set()
        csv_files = glob.glob(os.path.join(self.data_dir, "*/*/simple_data.csv"))
        for f in csv_files:
            df_cols = pd.read_csv(f, nrows=0).columns.tolist()
            all_cols.update(df_cols)
        if 'time' in all_cols: all_cols.remove('time')
        sorted_cols = sorted(list(all_cols))
        return sorted_cols, len(sorted_cols)

    def _align_and_extract(self, df):
        """Matches DF to global columns, filling missing ones with 0."""
        aligned = df.drop(columns=['time'], errors='ignore').reindex(
            columns=self.global_cols, fill_value=0
        )
        return aligned.values

    def get_rc_indices(self, service, fault):
        fault_map = {'delay': 'latency', 'loss': 'latency', 'cpu': 'cpu', 'mem': 'mem', 'disk': 'disk'}
        target_fault = fault_map.get(fault, fault)
        return [i for i, col in enumerate(self.global_cols) if service in col and target_fault in col]

    #def process_abnormal(self, full_values, onset_idx, service, fault):
    #    """Follows your SMD 'Anchor' logic."""
    #    rc_idxs = self.get_rc_indices(service, fault)
    #    
    #    # Define the testing anchor
    #    raw_start = onset_idx - self.window_size
    #    raw_end = onset_idx + 5  # Small lookahead for precision
    #    
    #    if raw_start >= 0 and raw_end <= len(full_values):
    #        label_matrix = np.zeros_like(full_values)
    #        for idx in rc_idxs:
    #            label_matrix[onset_idx:, idx] = 1
    #        
    #        return full_values[raw_start:raw_end], label_matrix[raw_start:raw_end]
    #    return None, None

    #def process_abnormal(self, full_values, onset_idx, service, fault):
    #    rc_idxs = self.get_rc_indices(service, fault)
    #    
    #    # 1. Start the window exactly 'window_size' before the fault
    #    # This ensures the model has the full history leading up to the event
    #    raw_start = onset_idx - self.window_size
    #    
    #    # 2. Define how many prediction steps you want to evaluate (lookahead)
    #    # We need enough data so that the sliding window (size W+1) can move
    #    # across the data. To get 'n' predictions, you need (W+1) + (n-1) steps.
    #    num_predictions = 5 
    #    raw_end = onset_idx + (self.window_size + 1) + (num_predictions - 1)
    #    
    #    # 3. Boundary Check
    #    if raw_start >= 0 and raw_end <= len(full_values):
    #        # Create a label matrix of zeros with the same shape as the full data
    #        label_matrix = np.zeros_like(full_values)
    #        
    #        # 4. Inject Ground Truth
    #        # Root cause starts exactly at onset_idx and persists
    #        for idx in rc_idxs:
    #            label_matrix[onset_idx : raw_end, idx] = 1
    #        
    #        # 5. Extract the slices
    #        # Features: [window_size + window_size + lookahead, num_vars]
    #        # Labels:   [window_size + window_size + lookahead, num_vars]
    #        return full_values[raw_start:raw_end], label_matrix[raw_start:raw_end]
    #        
    #    return None, None
    
    def process_abnormal(self, full_values, onset_idx, service, fault):
        rc_idxs = self.get_rc_indices(service, fault)
        
        # history = window_size
        raw_start = onset_idx - self.window_size
        # runway = enough for the sliding window to produce 5 outputs
        raw_end = onset_idx + self.window_size + 5 
        
        if raw_start >= 0 and raw_end <= len(full_values):
            # Create label matrix for the slice
            label_matrix = np.zeros((raw_end - raw_start, full_values.shape[1]))
            
            # CRITICAL: The local index of the fault in the slice is exactly window_size
            local_fault_start = self.window_size 
            
            for idx in rc_idxs:
                if idx < full_values.shape[1]:
                    label_matrix[local_fault_start:, idx] = 1
            
            return full_values[raw_start:raw_end], label_matrix
        return None, None

    def generate_example(self):
        all_train_chunks = []
        all_test_x = []
        all_test_y = []
        
        fault_folders = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]
        
        for f_folder in fault_folders:
            parts = f_folder.split('_')
            if len(parts) < 2: continue
            service, fault = parts[0], parts[1]
            
            scenario_path = os.path.join(self.data_dir, f_folder)
            scenarios = [s for s in os.listdir(scenario_path) if os.path.isdir(os.path.join(scenario_path, s))]
            
            for s_id in scenarios:
                path = os.path.join(scenario_path, s_id)
                data_file = os.path.join(path, 'simple_data.csv')
                if not os.path.exists(data_file): continue
                
                df = pd.read_csv(data_file)
                with open(os.path.join(path, 'inject_time.txt'), 'r') as f:
                    inject_time = int(f.read().strip())

                # Align to global schema
                full_values = self._align_and_extract(df)
                
                normal_mask = df['time'] < inject_time
                normal_values = full_values[normal_mask]
                
                onset_idx = df[df['time'] >= inject_time].index[0]

                # 1. Normal Processing (Sliding Window)
                block_size = self.window_size# * 10  # large block to capture long-term patterns
                step = self.window_size // 2 if self.window_size > 1 else 1
                for i in range(0, len(normal_values)- block_size + 1, step):
                    chunk = normal_values[i : i + block_size]
                    if chunk.shape[0] == block_size:
                        all_train_chunks.append(chunk)

                # 2. Abnormal Processing
                tx, ty = self.process_abnormal(full_values, onset_idx, service, fault)
                if tx is not None:
                    all_test_x.append(tx)
                    all_test_y.append(ty)

        # 3. Scaling (Global across all 1526 columns)
        combined_train = np.concatenate(all_train_chunks, axis=0) # [Total_Time, 1526]
        stds = np.std(combined_train, axis=0)
        active_mask = stds > 1e-9
        
        scaler = StandardScaler()
        if np.any(active_mask):
            scaler.fit(combined_train[:, active_mask])

        def finalize(window):
            """
            Processes window to match OrthTransform expectations:
            Input window shape: (Time, Vars)
            Output window shape: (Time, Vars)  <-- No more .T here
            """
            scaled = np.zeros_like(window)
            if np.any(active_mask):
                scaled[:, active_mask] = scaler.transform(window[:, active_mask])
            return scaled # Shape: [Window, 1526]

        # Final storage: [Batch, Window, Vars]
        self.data_dict['x_n_list'] = np.array([finalize(w) for w in all_train_chunks])
        self.data_dict['x_ab_list'] = np.array([finalize(x) for x in all_test_x])
        
        # Labels are kept as [Window, Vars] to match the time-steps
        self.data_dict['label_list'] = np.array(all_test_y)
        
        # Binary flags for OrthTransform
        self.binary_flags = np.array([
            1 if (combined_train[:, c].max() - combined_train[:, c].min() == 1) 
            and (np.unique(combined_train[:, c]).size <= 2) else 0 
            for c in range(self.num_vars)
        ])

    def save_data(self):
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
        for key in ['x_n_list', 'x_ab_list', 'label_list']:
            np.save(os.path.join(self.data_dir, f'{key}.npy'), self.data_dict[key])
        # Also save binary flags and num_vars (needed for model init)
        np.save(os.path.join(self.data_dir, 'binary_flags.npy'), self.binary_flags)

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
        self.binary_flags = np.load(os.path.join(self.data_dir, 'binary_flags.npy'))
        orth_matrix_dir = os.path.join(self.data_dir, 'orth_transform_meta')
        return None#self.apply_orthogonal_transform(save_path=orth_matrix_dir, device='cpu')

    def apply_orthogonal_transform(self, save_path, device='cpu'):
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