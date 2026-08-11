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


    """
        since the WADI attack label dataset contains wrong dates (year 2047, and july instead of Oct)
        we redefine the attacks from attack_description.xlsx in a dictionary to enable reproducibility and correct labeling of the attack data.
    """
    def generate_example(self):
        # ============================================================
        # 1. Paths
        # ============================================================

        normal_path = os.path.join(
            self.data_dir,
            "WADI_14days.csv"
        )

        attack_path = os.path.join(
            self.data_dir,
            "WADI_attackdata.csv"
        )


        # ============================================================
        # 2. Load Normal Data
        # ============================================================

        temp_headers = pd.read_csv(
            normal_path,
            skiprows=3,
            nrows=0
        ).columns.tolist()

        df_normal = pd.read_csv(
            normal_path,
            skiprows=1000,
            names=temp_headers
        )

        df_normal = self.clean_column_names(
            df_normal
        )

        df_normal.fillna(0, inplace=True)

        df_normal.drop(
            columns=[
                c for c in ["Row", "Date", "Time"]
                if c in df_normal.columns
            ],
            inplace=True
        )

        df_normal.ffill(inplace=True)
        df_normal.bfill(inplace=True)

        print(
            f"Normal shape: {df_normal.shape}"
        )


        # ============================================================
        # 3. Load Attack Data
        # ============================================================

        df_attack = pd.read_csv(
            attack_path
        )

        df_attack = self.clean_column_names(
            df_attack
        )

        df_attack.fillna(0, inplace=True)

        df_attack.ffill(inplace=True)
        df_attack.bfill(inplace=True)

        df_attack["Datetime"] = pd.to_datetime(
            df_attack["Date"] + " " + df_attack["Time"]
        )

        print(
            f"Attack shape: {df_attack.shape}"
        )


        # ============================================================
        # 4. Verify Attack Data Range
        # ============================================================

        print(
            "\n========== WADI ATTACK DATA RANGE =========="
        )

        print(
            "Start:",
            df_attack["Datetime"].min()
        )

        print(
            "End:  ",
            df_attack["Datetime"].max()
        )

        print(
            "Dates:",
            sorted(
                df_attack["Datetime"].dt.date.unique()
            )
        )

        print(
            "============================================\n"
        )


        # ============================================================
        # 5. Align Features
        # ============================================================

        common_sensors = df_normal.columns.tolist()

        df_attack_features = (
            df_attack[common_sensors]
            .copy()
        )


        # ============================================================
        # 6. Reproducible WADI Attack Specification
        # ============================================================

        WADI_ATTACKS = {

            1: {
                "date": "2017-10-09",
                "start": "19:25:00",
                "end": "19:50:16",
                "sensors": ["1MV001"],
            },

            2: {
                "date": "2017-10-10",
                "start": "10:24:10",
                "end": "10:34:00",
                "sensors": ["1FIT001"],
            },

            3: {
                "date": "2017-10-10",
                "start": "10:55:00",
                "end": "11:24:00",
                "sensors": ["2LIT002"],
            },

            4: {
                "date": "2017-10-10",
                "start": "11:07:46",
                "end": "11:12:15",
                "sensors": ["1AIT001"],
            },

            5: {
                "date": "2017-10-10",
                "start": "11:30:40",
                "end": "11:44:50",
                "sensors": [
                    "2MCV101",
                    "2MCV201",
                    "2MCV301",
                    "2MCV401",
                    "2MCV501",
                    "2MCV601",
                ],
            },

            6: {
                "date": "2017-10-10",
                "start": "13:39:30",
                "end": "13:50:40",
                "sensors": [
                    "2MCV101",
                    "2MCV201",
                ],
            },

            7: {
                "date": "2017-10-10",
                "sub_events": [
                    {
                        "start": "14:48:17",
                        "end": "14:59:55",
                        "sensors": ["1AIT002"],
                    },
                    {
                        "start": "14:53:44",
                        "end": "15:00:32",
                        "sensors": ["2MV003"],
                    },
                ],
            },

            8: {
                "date": "2017-10-10",
                "start": "17:40:00",
                "end": "17:49:40",
                "sensors": ["2MCV007"],
            },

            9: {
                "date": "2017-10-11",
                "start": "10:55:00",
                "end": "10:56:27",
                "sensors": [
                    "1-P-005",
                    "1-P-006",
                ],
            },

            10: {
                "date": "2017-10-11",
                "start": "11:17:54",
                "end": "11:31:20",
                "sensors": ["1MV001"],
            },

            11: {
                "date": "2017-10-11",
                "start": "11:36:31",
                "end": "11:47:00",
                "sensors": ["2MCV007"],
            },

            12: {
                "date": "2017-10-11",
                "start": "11:59:00",
                "end": "12:05:00",
                "sensors": ["2MCV007"],
            },

            13: {
                "date": "2017-10-11",
                "start": "12:07:30",
                "end": "12:10:52",
                "sensors": ["2PIC003"],
            },

            14: {
                "date": "2017-10-11",
                "start": "12:16:00",
                "end": "12:25:36",
                "sensors": [
                    "1P001",
                    "1P003",
                ],
            },

            15: {
                "date": "2017-10-11",
                "start": "15:26:30",
                "end": "15:37:00",
                "sensors": ["2LIT002"],
            },
        }


        # ============================================================
        # 7. Explicit WADI Sensor Mapping
        # ============================================================

        WADI_SENSOR_MAP = {

            "1AIT001": "1_AIT_001_PV",
            "1AIT002": "1_AIT_002_PV",
            "1FIT001": "1_FIT_001_PV",
            "1MV001": "1_MV_001_STATUS",

            "1P001": "1_P_001_STATUS",
            "1P003": "1_P_003_STATUS",
            "1-P-005": "1_P_005_STATUS",
            "1-P-006": "1_P_006_STATUS",

            "2LIT002": "2_LT_002_PV",

            "2MCV007": "2_MCV_007_CO",
            "2MCV101": "2_MCV_101_CO",
            "2MCV201": "2_MCV_201_CO",
            "2MCV301": "2_MCV_301_CO",
            "2MCV401": "2_MCV_401_CO",
            "2MCV501": "2_MCV_501_CO",
            "2MCV601": "2_MCV_601_CO",

            "2MV003": "2_MV_003_STATUS",

            "2PIC003": "2_PIC_003_PV",
        }


        # ============================================================
        # 8. Validate Sensor Mapping
        # ============================================================

        sensor_to_idx = {
            sensor: idx
            for idx, sensor in enumerate(common_sensors)
        }

        for attack_sensor, csv_sensor in WADI_SENSOR_MAP.items():

            if csv_sensor not in sensor_to_idx:

                raise ValueError(
                    f"WADI sensor mapping invalid: "
                    f"{attack_sensor} -> {csv_sensor}"
                )


        # ============================================================
        # 9. Generate Root-Cause Labels
        # ============================================================

        labels = np.zeros(
            (
                len(df_attack),
                len(common_sensors)
            ),
            dtype=np.float32
        )


        for attack_id, attack in WADI_ATTACKS.items():

            date = attack["date"]

            if "sub_events" in attack:

                events = attack["sub_events"]

            else:

                events = [{
                    "start": attack["start"],
                    "end": attack["end"],
                    "sensors": attack["sensors"],
                }]


            for event in events:

                start_dt = pd.Timestamp(
                    f"{date} {event['start']}"
                )

                end_dt = pd.Timestamp(
                    f"{date} {event['end']}"
                )

                mask = (
                    (df_attack["Datetime"] >= start_dt)
                    &
                    (df_attack["Datetime"] <= end_dt)
                )

                idx_range = np.flatnonzero(mask)

                if len(idx_range) == 0:

                    raise ValueError(
                        f"Attack {attack_id} has no "
                        f"matching samples: "
                        f"{start_dt} -> {end_dt}"
                    )


                for attack_sensor in event["sensors"]:

                    csv_sensor = (
                        WADI_SENSOR_MAP[
                            attack_sensor
                        ]
                    )

                    sensor_idx = (
                        sensor_to_idx[csv_sensor]
                    )

                    labels[
                        idx_range,
                        sensor_idx
                    ] = 1.0


        # ============================================================
        # 10. Label Diagnostics
        # ============================================================

        print(
            "\n========== WADI LABELING SUMMARY =========="
        )

        print(
            f"Attack definitions: {len(WADI_ATTACKS)}"
        )

        print(
            f"Sensors:            {len(common_sensors)}"
        )

        print(
            f"Positive labels:    {int(labels.sum())}"
        )

        print(
            "\nAttack-specific labels:"
        )

        for attack_id, attack in WADI_ATTACKS.items():

            if "sub_events" in attack:

                sensors = [
                    sensor
                    for event in attack["sub_events"]
                    for sensor in event["sensors"]
                ]

            else:

                sensors = attack["sensors"]

            print(
                f"Attack {attack_id:2d}: "
                f"{', '.join(sensors)}"
            )

        print(
            "============================================\n"
        )


        # ============================================================
        # 11. Downsampling
        # ============================================================

        sample_rate = 1

        df_normal = (
            df_normal
            .iloc[::sample_rate]
            .reset_index(drop=True)
        )

        df_attack_features = (
            df_attack_features
            .iloc[::sample_rate]
            .reset_index(drop=True)
        )

        df_attack = (
            df_attack
            .iloc[::sample_rate]
            .reset_index(drop=True)
        )

        labels = labels[::sample_rate]


        # ============================================================
        # 12. Scaling
        # ============================================================

        scaler = StandardScaler()

        scaler.fit(
            df_normal.values
        )

        scaled_normal = scaler.transform(
            df_normal.values
        )

        scaled_attack = scaler.transform(
            df_attack_features.values
        )

        print(
            f"Max scaled: {np.max(scaled_attack)} | "
            f"Min scaled: {np.min(scaled_attack)}"
        )

        print("\n========== SCALER DIAGNOSTICS ==========")
        for i, sensor in enumerate(common_sensors):
            print(
                f"{i:3d} {sensor:30s} "
                f"mean={scaler.mean_[i]:12.6f} "
                f"std={scaler.scale_[i]:12.6f}"
            )
        small_scale = np.where(scaler.scale_ < 1e-6)[0]
        print("Near-zero scale sensors:")
        for i in small_scale:
            print(i, common_sensors[i], scaler.scale_[i])

        # ============================================================
        # 13. Normal Windows
        # ============================================================

        x_n_list = [
            scaled_normal[i:i + self.window_size]
            for i in range(
                0,
                len(scaled_normal) - self.window_size,
                self.window_size
            )
        ]

        self.data_dict["x_n_list"] = np.asarray(
            x_n_list,
            dtype=np.float32
        )


        # ============================================================
        # 14. Attack Windows
        # ============================================================

        test_x_lst = []
        test_y_lst = []

        half_window = self.window_size // 2


        for attack_id, attack in WADI_ATTACKS.items():

            date = attack["date"]

            if "sub_events" in attack:

                events = attack["sub_events"]

            else:

                events = [{
                    "start": attack["start"],
                    "end": attack["end"],
                    "sensors": attack["sensors"],
                }]


            # Earliest event onset
            attack_start = min(
                pd.Timestamp(
                    f"{date} {event['start']}"
                )
                for event in events
            )


            onset_idx = np.searchsorted(
                df_attack["Datetime"].values,
                attack_start.to_datetime64()
            )


            start_idx = (
                onset_idx - half_window
            )

            end_idx = (
                onset_idx + half_window
            )


            if (
                start_idx < 0
                or
                end_idx > len(scaled_attack)
            ):

                raise ValueError(
                    f"Attack {attack_id} window "
                    f"outside data range"
                )


            test_x_lst.append(
                scaled_attack[
                    start_idx:end_idx
                ]
            )

            test_y_lst.append(
                labels[
                    start_idx:end_idx
                ]
            )


            print(
                f"Attack {attack_id:2d}: "
                f"onset={attack_start} | "
                f"indices={start_idx}:{end_idx}"
            )


        self.data_dict["x_ab_list"] = np.asarray(
            test_x_lst,
            dtype=np.float32
        )

        self.data_dict["label_list"] = np.asarray(
            test_y_lst,
            dtype=np.float32
        )


        # ============================================================
        # 15. Metadata
        # ============================================================

        self.num_vars = df_normal.shape[1]

        self.binary_flags = np.array([
            1 if df_normal[col].nunique() <= 2 else 0
            for col in common_sensors
        ])


        # ============================================================
        # 16. Shuffle Normal Data Only
        # ============================================================

        if self.shuffle:

            np.random.seed(self.seed)

            idx = np.random.permutation(
                len(self.data_dict["x_n_list"])
            )

            self.data_dict["x_n_list"] = (
                self.data_dict["x_n_list"][idx]
            )

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

