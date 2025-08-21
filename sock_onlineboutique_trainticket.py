import os
import pandas as pd
import tqdm
import numpy as np
from os.path import basename, dirname

BASE_PATH = "/home/db2003/Desktop/Amr/amocrca/data/combined"
DATASETS = ["sock-shop", "online-boutique", "train-ticket"]
TRAIN_RATIO = 0.7  # 70% train, 30% test

def get_metric_folders(dataset_name):
    """Return all metric folder names under any root folder (baro, rcd, rca-eval, etc.) for a dataset."""
    dataset_path = os.path.join(BASE_PATH, dataset_name)
    root_folders = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    metric_names = set()
    for folder in root_folders:
        for metric_name in os.listdir(folder):
            metric_path = os.path.join(folder, metric_name)
            if os.path.isdir(metric_path):
                metric_names.add(metric_name)
    return sorted(list(metric_names))

def process_dataset(dataset_name):
    dataset_path = os.path.join(BASE_PATH, dataset_name)
    
    all_train, all_test = [], []
    all_train_labels, all_test_labels = [], []

    allowed_columns = get_metric_folders(dataset_name)
    metric_to_idx = {m: i for i, m in enumerate(allowed_columns)}
    print(f"{dataset_name}: Found {len(allowed_columns)} allowed metric columns.")

    root_folders = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    for folder in tqdm.tqdm(root_folders, desc=f"Processing root folders for {dataset_name}"):
        services = [os.path.join(folder, s) for s in os.listdir(folder) if os.path.isdir(os.path.join(folder, s))]
        for service_path in tqdm.tqdm(services, desc=f"Processing services", leave=False):
            metrics = [os.path.join(service_path, m) for m in os.listdir(service_path) if os.path.isdir(os.path.join(service_path, m))]
            
            for metric_path in metrics:
                data_file = os.path.join(metric_path, "simple_data.csv")
                label_file = os.path.join(metric_path, "inject_time.txt")
                if not os.path.exists(data_file):
                    continue

                df = pd.read_csv(data_file)
                if 'time' not in df.columns:
                    raise ValueError(f"'time' column not found in {data_file}")
                
                # Keep only allowed metric columns + 'time'
                metric_name = basename(dirname(metric_path))
                if metric_name not in allowed_columns:
                    continue
                df = df[['time'] + [c for c in df.columns if c == metric_name]]

                # Initialize full label matrix (rows=time steps, cols=all allowed metrics)
                full_labels = np.zeros((len(df), len(allowed_columns)), dtype=int)

                # Load anomaly times (assume CSV uses same timestamps as df['time'])
                anomaly_times = []
                if os.path.exists(label_file):
                    with open(label_file) as f:
                        anomaly_times = [int(x) for x in f.read().strip().split(",") if x.strip()]

                # Mark anomalies in the column corresponding to this metric
                col_idx = metric_to_idx[metric_name]
                for t in anomaly_times:
                    if 0 <= t < len(df):
                        full_labels[t, col_idx] = 1

                # Split into train/test
                normal_rows = np.where(full_labels.sum(axis=1) == 0)[0]
                train_len = int(len(normal_rows) * TRAIN_RATIO)
                train_rows = normal_rows[:train_len]
                test_rows = np.setdiff1d(np.arange(len(df)), train_rows)

                train_df = df.iloc[train_rows].copy()
                test_df = df.iloc[test_rows].copy()
                train_labels = full_labels[train_rows]
                test_labels = full_labels[test_rows]

                all_train.append(train_df)
                all_test.append(test_df)
                all_train_labels.append(train_labels)
                all_test_labels.append(test_labels)

    # Combine all runs for dataset
    if all_train:
        train_df = pd.concat(all_train).reset_index(drop=True)
        test_df = pd.concat(all_test).reset_index(drop=True)
        train_labels_df = pd.DataFrame(np.vstack(all_train_labels), columns=allowed_columns)
        test_labels_df = pd.DataFrame(np.vstack(all_test_labels), columns=allowed_columns)

        # Add index column
        train_df.insert(0, 'index', range(len(train_df)))
        test_df.insert(0, 'index', range(len(test_df)))
        train_labels_df.insert(0, 'index', range(len(train_labels_df)))
        test_labels_df.insert(0, 'index', range(len(test_labels_df)))

        os.makedirs(f"datasets/{dataset_name}", exist_ok=True)
        train_df.to_csv(f"datasets/{dataset_name}/train.csv", index=False)
        test_df.to_csv(f"datasets/{dataset_name}/test.csv", index=False)
        train_labels_df.to_csv(f"datasets/{dataset_name}/train_labels.csv", index=False)
        test_labels_df.to_csv(f"datasets/{dataset_name}/test_labels.csv", index=False)

        print(f"{dataset_name}: Saved train/test + train_labels/test_labels")
    else:
        print(f"{dataset_name}: No data found to process!")

# Process all datasets
for dataset in DATASETS:
    process_dataset(dataset)
