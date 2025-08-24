import os
import pandas as pd

def build_sockshop_dataset_dynamic(root_dir, output_dir="datasets/sock-shop"):
    os.makedirs(output_dir, exist_ok=True)

    train_dfs, test_dfs = [], []
    all_columns = set()

    # Step 1: Discover all columns from normal and anomalous CSVs
    for service in os.listdir(root_dir):
        service_path = os.path.join(root_dir, service)
        if not os.path.isdir(service_path):
            continue
        for run in os.listdir(service_path):
            run_path = os.path.join(service_path, run)
            if not os.path.isdir(run_path):
                continue
            for fname in ["normal.csv", "anomalous.csv"]:
                fpath = os.path.join(run_path, fname)
                if os.path.exists(fpath):
                    df = pd.read_csv(fpath)
                    all_columns.update(df.columns)
    
    # Sort columns to have consistent order
    # remove "-" and "_" from column names
    all_columns = [col.replace("-", "").replace("_", "") for col in all_columns]
    all_columns = sorted(list(all_columns))
    column_map = {col: idx for idx, col in enumerate(all_columns)}
    print("Column map:", column_map)

    # Step 2: Combine all normals for train
    for service in os.listdir(root_dir):
        service_path = os.path.join(root_dir, service)
        if not os.path.isdir(service_path):
            continue
        for run in os.listdir(service_path):
            run_path = os.path.join(service_path, run)
            if not os.path.isdir(run_path):
                continue
            normal_file = os.path.join(run_path, "normal.csv")
            if os.path.exists(normal_file):
                df_normal = pd.read_csv(normal_file)
                train_dfs.append(df_normal)

    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)

    # Step 3: Combine all anomalies for test and build labels
    labels_list = []
    for service in os.listdir(root_dir):
        service_path = os.path.join(root_dir, service)
        if not os.path.isdir(service_path):
            continue
        for run in os.listdir(service_path):
            run_path = os.path.join(service_path, run)
            if not os.path.isdir(run_path):
                continue
            anomalous_file = os.path.join(run_path, "anomalous.csv")
            if os.path.exists(anomalous_file):
                df_anom = pd.read_csv(anomalous_file)
                test_dfs.append(df_anom)

                # Initialize zero labels
                label_matrix = pd.DataFrame(0, index=range(len(df_anom)), columns=all_columns)

                # remove "-" and "_" from service name to match column names
                service = service.replace("-", "").replace("_", "")
                # Set 1 only for the column corresponding to this service folder
                if service in all_columns:
                    label_matrix[service] = 1
                else:
                    # Optional: if service not in columns, you can create it
                    label_matrix[service] = 1
                    if service not in all_columns:
                        all_columns.append(service)

                labels_list.append(label_matrix)


    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()

    labels_df = pd.concat(labels_list, ignore_index=True) if labels_list else pd.DataFrame()
    labels_df.insert(0, 'index', range(len(labels_df)))
    labels_df = labels_df[["index"] + all_columns]
    # Assertions
    assert len(test_df) == len(labels_df), f"Mismatch: test={len(test_df)} vs labels={len(labels_df)}"
    #assert list(labels_df.columns) == all_columns+1, "Label columns do not match expected columns"

    # Save test and labels
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    labels_df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

    print("✅ Done!")
    print(f"Train: {train_df.shape}, Test: {test_df.shape}, Labels: {labels_df.shape}")


BASE_PATH = "/home/db2003/Desktop/Amr/amocrca/data/combined/sock-shop/sock-shop-rcd"
build_sockshop_dataset_dynamic(BASE_PATH)



