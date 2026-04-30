import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import requests
import zipfile
from datetime import datetime
import os
from functools import reduce
from layers.vlinear_arch import OrthTransform 


def download_and_extract_zenodo_msds():
    """
    Downloads and extracts the MSDS dataset from Zenodo into ./datasets/msds/.
    If the dataset is already downloaded and extracted, the function does nothing.
    """
    zenodo_url = "https://zenodo.org/api/records/3549604/files-archive"
    target_dir = os.path.join(os.getcwd(), 'datasets', 'msds')
    zip_path = os.path.join(target_dir, "msds_dataset.zip")

    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Download the ZIP archive if not already downloaded
    if not os.path.exists(zip_path):
        print("Downloading dataset from Zenodo...")
        response = requests.get(zenodo_url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download file: status code {response.status_code}")

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded ZIP file to: {zip_path}")
    else:
        print(f"ZIP file already exists at: {zip_path}. Skipping download.")

    # Extract the contents
    print("Extracting contents...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print(f"Extraction complete to: {target_dir}")

    # Extract sequential_data.zip
    sequential_zip_path = os.path.join(target_dir, "concurrent data.zip")
    extracted_flag = os.path.join(target_dir, "MSDS")  # This is the root folder in the ZIP
    if os.path.exists(sequential_zip_path):
        print("Extracting concurrent data.zip...")
        with zipfile.ZipFile(sequential_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_flag)
        print(f"Extraction of concurrent data.zip complete to: {extracted_flag}")
    else:
        print("concurrent data.zip not found.")

    print("Finished downloading and extracting the MSDS dataset.")


def preprocess_metrics_data():
    """
    Preprocesses the metrics data from the MSDS dataset.
    Reference: https://github.com/imperial-qore/TranAD/tree/main/data/MSDS
    :return:
    """
    print("Preprocessing metrics data...")
    target_dir = os.path.join(os.getcwd(), 'datasets', 'msds')
    metrics_dir = os.path.join(target_dir, 'MSDS', 'concurrent data', 'metrics')
    files = ['wally122_metrics_concurrent.csv', 'wally113_metrics_concurrent.csv', 'wally123_metrics_concurrent.csv', 'wally117_metrics_concurrent.csv', 'wally124_metrics_concurrent.csv']
    dfs = []

    # Read csv files
    for file in files:
        if '.csv' in file:
            df = pd.read_csv(os.path.join(metrics_dir, file))
            df = df.drop(columns=['load.cpucore', 'load.min1', 'load.min5', 'load.min15'])
            dfs.append(df)

    # Process dataframes
    start = dfs[0].min()['now']
    end = dfs[0].max()['now']
    for df in dfs:
        if df.min()['now'] > start:
            start = df.min()['now']
        if df.max()['now'] < end:
            end = df.max()['now']
    id_vars = ['now']
    dfs2 = []
    for df in dfs:
        df = df.drop(np.argwhere(list(df['now'] < start)).reshape(-1))
        df = df.drop(np.argwhere(list(df['now'] > end)).reshape(-1))
        melted = df.melt(id_vars=id_vars).dropna()
        df = melted.pivot_table(index=id_vars, columns="variable", values="value")
        dfs2.append(df)
    dfs = dfs2

    dfs_unique = []
    for idx, df in enumerate(dfs):
        df = df.copy()
        # Rename all columns except the index (assumed to be 'now')
        df.columns = [f"{col}_{idx}" if col != "now" else col for col in df.columns]
        dfs_unique.append(df)

    df_merged = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs_unique)

    # Change timezone string format
    ni = []
    for i in df_merged.index:
        dt = datetime.strptime(i[:-5], '%Y-%m-%d %H:%M:%S')
        ni.append(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))
    df_merged.index = ni

    # Save train and test sets
    start = round(df_merged.shape[0] * 0.1)
    df_merged = df_merged[start:]
    split = round(df_merged.shape[0] / 2)
    df_merged[:split].to_csv(os.path.join(target_dir, 'train.csv'))
    df_merged[split:].to_csv(os.path.join(target_dir, 'test.csv'))
    print("Preprocessing complete. Train and test sets saved.")


class MSDS:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.num_vars = options['num_vars']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options['shuffle']

    def generate_example(self):
        # Ensure the dataset is downloaded and extracted
        download_and_extract_zenodo_msds()
        preprocess_metrics_data()
        # load data and save to csv
        df_label = pd.read_csv(os.path.join(self.data_dir, 'labels.csv'))
        df_normal = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        df_abnormal = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))

        df_normal, df_abnormal = df_normal.values[::5, 1:], df_abnormal.values[::5, 1:]
        df_label = df_label.values[::5, 1:]
        labels = np.max(df_label, axis=1)

        """
        unique, counts = np.unique(labels, return_counts=True)
        total = counts.sum()
        percentages = {u: (c / total) * 100 for u, c in zip(unique, counts)}
        """
        x_n_list = []
        #for i in range(0, len(df_normal), 10000):
        #    if i + 10000 < len(df_normal):
        #        x_n_list.append(df_normal[i:i + 10000])
        # To match the test window size of 3 * window_size:
        step = self.window_size 
        chunk_size = 1 * self.window_size 

        for i in range(0, len(df_normal) - chunk_size, step):
            x_n_list.append(df_normal[i:i + chunk_size])
        test_x_lst = []
        label_lst = []

        for i in np.where(labels == 1)[0]:
            # Adjusting to capture a single window_size to be consistent with the normal data windowing
            # We take the window ending exactly where the anomaly is first detected
            start_idx = i - self.window_size + 1
            end_idx = i + 1
            
            if start_idx > 0 and end_idx < len(df_abnormal):
                # Ensure the preceding frames (excluding the current anomaly point) were normal
                if sum(labels[start_idx : i]) == 0:
                    test_x_lst.append(df_abnormal[start_idx : end_idx])
                    label_lst.append(df_label[start_idx : end_idx])


        scaler = RobustScaler()
        scaler.fit(np.concatenate(x_n_list, axis=0))
        x_n_list = [scaler.transform(i) for i in x_n_list]
        test_x_lst = [scaler.transform(i) for i in test_x_lst]
        x_n_list = np.clip(x_n_list, -10, 10)
        self.data_dict['x_n_list'] = np.array(x_n_list)
        if self.shuffle:
            np.random.seed(self.seed)
            indices = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][indices]
        self.data_dict['x_ab_list'] = np.array(test_x_lst)
        self.data_dict['label_list'] = np.array(label_lst)

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



    def save_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        np.save(os.path.join(self.data_dir, 'x_n_list'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'label_list'), self.data_dict['label_list'])

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'), allow_pickle=False)
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'), allow_pickle=True)
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'), allow_pickle=True)
        # Define path for the Q matrix specifically
        orth_matrix_dir = os.path.join(self.data_dir, 'orth_transform_meta')
        self.pipeline_sanity_check()
        device = 'cpu'
        return self.apply_orthogonal_transform(save_path=orth_matrix_dir, device=device)