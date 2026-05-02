import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import QuantileTransformer, RobustScaler
import inspect
from tqdm import tqdm

from utils.utils import (topk, topk_at_step)
class StatisticalRCA:
    @staticmethod
    def _run_pyrca_loop(xs, labels, analyzer_class, config_class, **config_kwargs):
        """Unified loop for PyRCA-based analyzers to ensure consistent Top-K evaluation."""
        k_all = []
        k_at_step_all = []
        num_vars = xs[0].shape[1]

        for i in tqdm(range(len(xs)), desc="PyRCA Evaluation"):
            # Convert window to DataFrame
            df_x = pd.DataFrame(xs[i], columns=[f"var_{j}" for j in range(num_vars)])
            
            # Initialize and train
            model = analyzer_class(config=config_class(**config_kwargs))

            # Find root causes (most PyRCA methods take df_x as abnormal input)
            # For RCD, it usually takes (normal_df, abnormal_df). If only one is provided, 
            # we use df_x for both or a zeroed-baseline.
            try:
                if analyzer_class.__name__ == 'RCD':
                    #print("Using RCD with df_x as both normal and abnormal input (not ideal, but consistent with your setup)")
                    results_raw = model.find_root_causes(df_x, df_x)
                else:
                    results_raw = model.find_root_causes(df_x)

                # Convert to z_score vector
                z_scores = np.zeros(num_vars)
                for var_name, _ in results_raw.root_cause_nodes:
                    idx = int(var_name.replace("var_", ""))
                    z_scores[idx] = 1.0
                
                # Consistency Fix: Align with your new label maxing logic
                current_labels = np.max(labels[i], axis=0, keepdims=True)
                z_scores_exp = np.expand_dims(z_scores, axis=0) # [1, num_vars]

                k_all.append(topk(z_scores_exp, current_labels, threshold=0.5))
                k_at_step_all.append(topk_at_step(z_scores_exp, current_labels))
            except Exception as e:
                print(f"Method failed for window {i}: {e}")
                continue

        return StatisticalRCA._summarize(k_all, k_at_step_all)

    @staticmethod
    def _summarize(k_all, k_at_step_all):
        """Aggregates results into the format expected by your logging."""
        if not k_all:
            return None
        return {
            "avg_k_all": np.array(k_all).mean(axis=0),
            "avg_k_at_step": np.array(k_at_step_all).mean(axis=0)
        }

    @staticmethod
    def evaluate_rcd(xs, labels, bins=None, gamma=5):
        from pyrca.analyzers.rcd import RCD, RCDConfig
        return StatisticalRCA._run_pyrca_loop(xs, labels, RCD, RCDConfig, bins=bins, gamma=gamma)

    @staticmethod
    def evaluate_nsigma(xs, labels, n=3):
        """Classical statistical baseline: flags any sensor exceeding N standard deviations."""
        k_all = []
        k_at_step_all = []
        num_vars = xs[0].shape[1]

        for i in range(len(xs)):
            x = xs[i]
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0) + 1e-9
            
            # Simple z-score: how far is the last point from the window mean
            z_scores = np.abs((x[-1] - mean) / std)
            z_scores = np.expand_dims(z_scores, axis=0)
            
            current_labels = np.max(labels[i], axis=0, keepdims=True)
            k_all.append(topk(z_scores, current_labels, threshold=n))
            k_at_step_all.append(topk_at_step(z_scores, current_labels))

        return StatisticalRCA._summarize(k_all, k_at_step_all)

    @staticmethod
    def evaluate_baro(xs, labels, scalar_type="Robust", seq_len=1):
        """
        Implementation using the RCAEval logic.
        Treats the beginning of the window as 'normal' and the end as 'anomalous'.
        """
        k_all = []
        k_at_step_all = []
        num_vars = xs[0].shape[1]
        
        for i in range(len(xs)):
            window_data = xs[i] # [Window_Size, Num_Vars]
            
            # RCAEval BARO logic: split window into normal (history) and abnormal (current)
            # We use the first 70% of the window as 'normal' context for the scaler
            split_idx = int(0.7 * len(window_data))
            normal_part = window_data[:split_idx]
            anomal_part = window_data[split_idx:]
            
            ranks = []
            for col_idx in range(num_vars):
                a = normal_part[:, col_idx]
                b = anomal_part[:, col_idx]

                # Sequence aggregation logic from RCAEval
                if seq_len > 1:
                    a = a[: (len(a) // seq_len) * seq_len].reshape(-1, seq_len).mean(axis=1)
                    b = b[: (len(b) // seq_len) * seq_len].reshape(-1, seq_len).mean(axis=1)

                # Fit Scaler based on scalar_type (Default: Robust)
                if scalar_type == "Robust":
                    scaler = RobustScaler().fit(a.reshape(-1, 1))
                elif scalar_type == "Standard":
                    scaler = StandardScaler().fit(a.reshape(-1, 1))
                elif scalar_type == "Quantile":
                    scaler = QuantileTransformer(output_distribution="normal").fit(a.reshape(-1, 1))
                
                # Transform and get max z-score as the anomaly rank
                zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
                ranks.append(np.max(zscores))

            # Convert ranks to a 2D score array [1, Num_Vars] for topk functions
            z_scores_final = np.array(ranks).reshape(1, -1)
            
            # Align with your label maxing logic
            current_labels = np.max(labels[i], axis=0, keepdims=True)
            
            k_all.append(topk(z_scores_final, current_labels, threshold=0.5))
            k_at_step_all.append(topk_at_step(z_scores_final, current_labels))

        return StatisticalRCA._summarize(k_all, k_at_step_all)