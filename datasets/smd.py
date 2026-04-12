import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
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



    def process_abnormal(self, test_df, global_labels, root_cause_labels, target_len):
        test_x_lst = []
        test_label_lst = []
        
        # Stride is 10, so to get target_len timestamps, we need (target_len * 10) raw rows
        stride = 10
        raw_width = target_len * stride # 10 * 10 = 100
        
        anomaly_indices = np.where(global_labels == 1)[0]
        if len(anomaly_indices) > 0:
            events = np.split(anomaly_indices, np.where(np.diff(anomaly_indices) > 1)[0] + 1)
            
            for event in events:
                start_idx_event = event[0]
                
                # Center the window on the start of the event
                # 50 rows before, 50 rows after = 100 total raw rows
                raw_start = int(start_idx_event - (raw_width // 2))
                raw_end = int(start_idx_event + (raw_width // 2))
                
                if raw_start >= 0 and raw_end <= len(test_df):
                    slice_x = test_df.values[raw_start:raw_end:stride]
                    slice_y = root_cause_labels[raw_start:raw_end:stride]
                    
                    if len(slice_x) == target_len:
                        test_x_lst.append(slice_x)
                        test_label_lst.append(slice_y)
        return test_x_lst, test_label_lst

    def process_normal(self, combined_train, target_len):
        """Generalized logic to segment normal data into target_len-timestamp chunks."""
        # Use a step equal to target_len to ensure non-overlapping windows like SWaT
        return [
            combined_train[i : i + target_len] 
            for i in range(0, len(combined_train), target_len) 
            if i + target_len <= len(combined_train)
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
        target_len = 10 

        for m_file in machines:
            # 1. Load raw data
            train_df = pd.read_csv(os.path.join(train_path, m_file), header=None)
            test_df = pd.read_csv(os.path.join(test_path, m_file), header=None)
            global_labels = pd.read_csv(os.path.join(label_path, m_file), header=None).values.flatten()
            
            # 2. Get Root Cause Labels
            root_cause_labels = self.parse_interpretation_label(
                os.path.join(interpretation_path, m_file), len(test_df)
            )

            # 3. Call Abnormal Processor
            m_test_x, m_test_y = self.process_abnormal(test_df, global_labels, root_cause_labels, target_len)
            all_test_x.extend(m_test_x)
            all_test_y.extend(m_test_y)

            # 4. Collect train data for the Normal Processor
            all_train_data.append(train_df.values)

        # 5. Scaling
        scaler = StandardScaler()
        combined_train = np.concatenate(all_train_data, axis=0)
        scaler.fit(combined_train)
        
        # 6. Call Normal Processor
        # We transform the combined data first, then segment it
        scaled_train = scaler.transform(combined_train)
        x_n_list = self.process_normal(scaled_train, target_len)

        # 7. Final Storage
        self.data_dict['x_n_list'] = np.array(x_n_list)
        test_x_transformed = [scaler.transform(x) for x in all_test_x]
        self.data_dict['x_ab_list'] = np.array(test_x_transformed)
        self.data_dict['label_list'] = np.array(all_test_y)
        
        # Shuffle train data
        if self.shuffle:
            np.random.seed(self.seed)
            idx = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][idx]

        # Flags for binary features (used in Orthogonal logic)
        self.binary_flags = np.array([
            1 if combined_train[:, c].max() - combined_train[:, c].min() == 1 
            and np.unique(combined_train[:, c]).size <= 2 else 0 
            for c in range(combined_train.shape[1])
        ])




    def save_data(self):
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
        for key in ['x_n_list', 'x_ab_list', 'label_list']:
            np.save(os.path.join(self.data_dir, f'{key}.npy'), self.data_dict[key])

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
        orth_matrix_dir = os.path.join(self.data_dir, 'orth_transform_meta')
        return None#self.apply_orthogonal_transform(save_path=orth_matrix_dir, device='cpu')

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
            self.data_dict['x_n_orth'] = self.orth_transformer(x_n_tensor).cpu().numpy()
        x_ab_tensor = torch.from_numpy(self.data_dict['x_ab_list']).float().to(device)
        with torch.no_grad():
            self.data_dict['x_ab_orth'] = self.orth_transformer(x_ab_tensor).cpu().numpy()
        return self.orth_transformer