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

        all_scores = []
        all_labels = []
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
                if np.sum(current_labels) == 0:
                    continue
                z_scores_exp = np.expand_dims(z_scores, axis=0) # [1, num_vars]
                
                k_all.append(topk(z_scores_exp, current_labels, threshold=0.5))
                k_at_step_all.append(topk_at_step(z_scores_exp, current_labels))
                all_scores.append(z_scores_exp)
                all_labels.append(current_labels)
            except Exception as e:
                print(f"Method failed for window {i}: {e}")
                continue

        return StatisticalRCA._summarize(k_all, k_at_step_all, all_scores=all_scores, all_labels=all_labels)

    @staticmethod
    def _summarize(k_all, k_at_step_all, all_scores=None, all_labels=None):
        """Aggregates results into the format expected by your logging."""
        if not k_all:
            return None
        return {
            "avg_k_all": np.array(k_all).mean(axis=0),
            "avg_k_at_step": np.array(k_at_step_all).mean(axis=0),
            "scores": all_scores,
            "labels": all_labels,
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

        all_scores = []
        all_labels = []
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
            all_scores.append(z_scores)
            all_labels.append(current_labels)
        return StatisticalRCA._summarize(k_all, k_at_step_all, all_scores=all_scores, all_labels=all_labels)

    @staticmethod
    def evaluate_baro(xs, labels, scalar_type="Robust", seq_len=1):
        """
        Implementation using the RCAEval logic.
        Treats the beginning of the window as 'normal' and the end as 'anomalous'.
        """
        k_all = []
        k_at_step_all = []

        all_scores = []
        all_labels = []
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
            if np.sum(current_labels) == 0:
                continue
            k_all.append(topk(z_scores_final, current_labels, threshold=0.5))
            k_at_step_all.append(topk_at_step(z_scores_final, current_labels))
            all_scores.append(z_scores_final)
            all_labels.append(current_labels)

        return StatisticalRCA._summarize(k_all, k_at_step_all, all_scores=all_scores, all_labels=all_labels)  
                                         
    @staticmethod
    def evaluate_torai(xs, labels, n_components=None):
        from sklearn.mixture import GaussianMixture
        from pyrca.analyzers.rcd import RCD, RCDConfig
        
        k_all = []
        k_at_step_all = []

        all_scores = []
        all_labels = []
        num_vars = xs[0].shape[1]
        
        for i in tqdm(range(len(xs)), desc="Faithful TORAI Eval"):
            window_data = xs[i]
            split_idx = int(0.7 * len(window_data))
            
            # Step 1: Metric Z-Score Ranking (Coarse Stage)
            metric_ranks = []
            normal_part = window_data[:split_idx]
            anomal_part = window_data[split_idx:]
            
            for col_idx in range(num_vars):
                a = normal_part[:, col_idx]
                b = anomal_part[:, col_idx]
                scaler = StandardScaler().fit(a.reshape(-1, 1))
                zscores = np.abs(scaler.transform(b.reshape(-1, 1)))
                metric_ranks.append(np.max(zscores))
            
            # Normalize scores (Faithful to: x[1] / sum(...) logic)
            total_sum = sum(metric_ranks) + 1e-9
            norm_ranks = [s / total_sum for s in metric_ranks]
            
            # Step 2: Prepare GMM Feature Matrix (Faithful to: X_train = m.to_numpy())
            X_train = np.array(norm_ranks).reshape(-1, 1)

            # Step 3: Optimal Cluster Selection (Faithful to: bic_score_all logic)
            if n_components is None:
                bic_scores = []
                n_range = range(1, min(num_vars, 10) + 1)
                for n in n_range:
                    gmm = GaussianMixture(n_components=n, max_iter=50, random_state=0).fit(X_train)
                    bic_scores.append(gmm.bic(X_train))
                n_comp_opt = n_range[np.argmin(bic_scores)]
            else:
                n_comp_opt = n_components

            # Step 4: Clustering & Cluster Ranking (Faithful to: cluster_rank.sort)
            estimator = GaussianMixture(n_components=n_comp_opt, max_iter=50, random_state=0).fit(X_train)
            y_pred = estimator.predict(X_train)
            
            cluster_info = []
            for c_idx in range(n_comp_opt):
                indices = np.where(y_pred == c_idx)[0]
                if len(indices) > 0:
                    cluster_score = X_train[indices].mean()
                    cluster_info.append((c_idx, cluster_score))
            cluster_info.sort(key=lambda x: x[1], reverse=True)

            # Step 5: Final Ranking (Faithful to: service_ranks_rcd logic)
            final_ordered_indices = []
            for c_idx, _ in cluster_info:
                indices_in_cluster = np.where(y_pred == c_idx)[0]
                
                if len(indices_in_cluster) == 1:
                    final_ordered_indices.append(indices_in_cluster[0])
                else:
                    # Sort internally by score first (Faithful to: aa.sort logic)
                    internal_scores = [(idx, norm_ranks[idx]) for idx in indices_in_cluster]
                    internal_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Causal Refinement (Faithful to: _rcd_multimodal call)
                    try:
                        # Convert to DF for PyRCA RCD
                        vars_in_cluster = [f"var_{idx}" for idx in indices_in_cluster]
                        norm_df = pd.DataFrame(normal_part[:, indices_in_cluster], columns=vars_in_cluster)
                        anom_df = pd.DataFrame(anomal_part[:, indices_in_cluster], columns=vars_in_cluster)
                        
                        rcd_model = RCD(config=RCDConfig(bins=5, gamma=5))
                        rc_results = rcd_model.find_root_causes(norm_df, anom_df)
                        
                        rcd_ordered = [int(node[0].replace("var_", "")) for node in rc_results.root_cause_nodes]
                        # Append RCD results, then fill in remaining cluster members
                        final_ordered_indices.extend(rcd_ordered)
                        for idx, _ in internal_scores:
                            if idx not in final_ordered_indices:
                                final_ordered_indices.append(idx)
                    except:
                        # Fallback to internal sorted scores
                        final_ordered_indices.extend([x[0] for x in internal_scores])

            # Step 6: Format Output for your Top-K
            z_scores_final = np.zeros(num_vars)
            for rank, idx in enumerate(final_ordered_indices):
                # Higher score = higher rank (earlier in list)
                z_scores_final[idx] = num_vars - rank 
            
            z_scores_exp = np.expand_dims(z_scores_final, axis=0)
            current_labels = np.max(labels[i], axis=0, keepdims=True)
            if np.sum(current_labels) == 0:
                continue
            k_all.append(topk(z_scores_exp, current_labels, threshold=0.5))
            k_at_step_all.append(topk_at_step(z_scores_exp, current_labels))
            all_scores.append(z_scores_exp)
            all_labels.append(current_labels)

        return StatisticalRCA._summarize(k_all, k_at_step_all, all_scores=all_scores, all_labels=all_labels)