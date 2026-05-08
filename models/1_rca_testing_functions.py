
    def _testing_root_cause(
        self,
        xs,
        labels,
        alpha: float = 0.5,
        use_attention_fusion: bool = False,
        eval_mode: str = "window"
    ):
        import time
        import psutil
        import threading
        import os
        import torch

        coeff_architecture = self.options["coeff_architecture"]

        # =========================================================
        # 0. Baseline Models
        # =========================================================
        if coeff_architecture in ["rcd", "baro", "nsigma", "torai"]:
            eval_fn = getattr(StatisticalRCA, f"evaluate_{coeff_architecture}")
            res = eval_fn(xs, labels)

            if res:
                k_at_step_all = res["avg_k_at_step"]

                self._log_and_print("=== Baseline RCA ===")
                self._log_and_print("AC@1: {:.5f}", k_at_step_all[0])
                self._log_and_print("AC@3: {:.5f}", k_at_step_all[2])
                self._log_and_print("AC@5: {:.5f}", k_at_step_all[4])
                self._log_and_print("AC@10: {:.5f}", k_at_step_all[9])

                write_results(
                    self.options,
                    self.local_model_name,
                    [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]],
                    k_at_step_all,
                    0,
                    self.options.get("results_csv"),
                )
            return res

        # =========================================================
        # 1. Load Model + Stats
        # =========================================================
        self.load_state_dict(
            torch.load(os.path.join(self.save_dir, f"{self.model_name}.pt"),
                    map_location=self.device)
        )
        self.eval()

        self.us_mean_encoder = np.load(
            os.path.join(self.save_dir, f"{self.model_name}_us_mean_encoder.npy")
        )
        self.us_std_encoder = np.load(
            os.path.join(self.save_dir, f"{self.model_name}_us_std_encoder.npy")
        )

        eps = 1e-8

        # =========================================================
        # 🔹 MEMORY TRACKING (UNIFIED CPU + GPU)
        # =========================================================
        use_cuda = torch.cuda.is_available() and self.device != "cpu"

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        process = psutil.Process(os.getpid())

        peak_mem_bytes = {"value": 0}
        stop_event = threading.Event()

        def memory_poller():
            """CPU memory polling for true peak tracking."""
            while not stop_event.is_set():
                mem = process.memory_info().rss
                if mem > peak_mem_bytes["value"]:
                    peak_mem_bytes["value"] = mem
                time.sleep(0.01)  # 10ms resolution

        monitor_thread = None
        if not use_cuda:
            monitor_thread = threading.Thread(target=memory_poller)
            monitor_thread.start()

        model_mem_mb = self.total_params * 4 / (1024 ** 2)

        # =========================================================
        # 2. Inference Phase
        # =========================================================
        us_all = []
        us_samples = []
        attn_samples = []

        with torch.no_grad():
            for i in tqdm(range(len(xs)), desc="Inference"):
                x = xs[i]
                label = labels[i]

                _, _, _, _, _, _, _, us, attn_weights = self._testing_step(
                    x, label, add_u=False
                )

                u_np = us.cpu().numpy()

                us_all.append(u_np)
                us_samples.append(u_np)

                if use_attention_fusion:
                    attn_samples.append(attn_weights.mean(dim=0).cpu().numpy())

        us_all = np.concatenate(us_all, axis=0)

        # =========================================================
        # 🔹 STOP MEMORY TRACKING
        # =========================================================
        if not use_cuda:
            stop_event.set()
            monitor_thread.join()

            # final correction sample
            final_mem = process.memory_info().rss
            peak_mem_mb = max(peak_mem_bytes["value"], final_mem) / (1024 ** 2)
        else:
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # =========================================================
        # 3. POT Thresholds
        # =========================================================
        z_all = -(us_all - self.us_mean_encoder) / (self.us_std_encoder + eps)

        pot_thresholds = np.zeros(self.num_vars)

        for j in range(self.num_vars):
            col = z_all[:, j]
            col = col[np.isfinite(col)]

            if len(col) == 0:
                pot_thresholds[j] = 0.0
                continue

            try:
                pot_val, _ = pot(col, self.risk, self.initial_level, self.num_candidates)
            except:
                pot_val = np.mean(col) + 3 * np.std(col)

            pot_thresholds[j] = pot_val

        # =========================================================
        # 4. Evaluation Loop
        # =========================================================
        k_all, k_at_step_all = [], []
        mrr_list = []
        hr1_list, hr3_list, hr5_list, hr10_list = [], [], [], []
        inference_times = []
        dropped = 0

        t_eval = self.window_size // 2

        for i in tqdm(range(len(xs)), desc="Evaluation"):
            start_time = time.time()

            z_scores = -(us_samples[i] - self.us_mean_encoder) / (self.us_std_encoder + eps)

            if eval_mode == "onset":
                current_labels = labels[i][t_eval:t_eval+1]
            elif eval_mode == "window":
                current_labels = np.max(labels[i], axis=0, keepdims=True)
            else:
                raise ValueError("eval_mode must be 'onset' or 'window'")

            if not np.any(current_labels):
                dropped += 1
                continue

            if use_attention_fusion:
                attn = attn_samples[i].mean(axis=2).mean(axis=0)
                attn = np.expand_dims(attn, axis=0)
                z_scores = alpha * z_scores + (1 - alpha) * attn

            try:
                k_lst = topk(z_scores, current_labels, pot_thresholds)
                k_at = topk_at_step(z_scores, current_labels)

                k_all.append(k_lst)
                k_at_step_all.append(k_at)

                ranking = np.argsort(-z_scores[0])
                true_idx = np.where(current_labels[0] == 1)[0]

                rr = 0.0
                for rank, idx in enumerate(ranking, start=1):
                    if idx in true_idx:
                        rr = 1.0 / rank
                        break
                mrr_list.append(rr)

                def hit(k):
                    return int(any(idx in ranking[:k] for idx in true_idx))

                hr1_list.append(hit(1))
                hr3_list.append(hit(3))
                hr5_list.append(hit(5))
                hr10_list.append(hit(10))

            except:
                dropped += 1
                continue

            inference_times.append(time.time() - start_time)

        # =========================================================
        # 5. Aggregation
        # =========================================================
        valid = len(k_all)
        total = len(xs)
        coverage = valid / total if total > 0 else 0.0

        self._log_and_print(
            "Coverage: {}/{} ({:.2f}%) | Dropped: {}",
            valid, total, 100 * coverage, dropped
        )

        if valid == 0:
            return None

        k_all = np.mean(np.array(k_all), axis=0)
        k_at_step_all = np.mean(np.array(k_at_step_all), axis=0)

        mrr = np.mean(mrr_list)
        hr1, hr3, hr5, hr10 = map(np.mean, [hr1_list, hr3_list, hr5_list, hr10_list])

        auc_k = np.mean(k_at_step_all[:10])
        std_ac = np.std(np.array(k_at_step_all))

        avg_time = np.mean(inference_times)
        throughput = 1.0 / avg_time if avg_time > 0 else 0.0

        # =========================================================
        # 6. Logging (ESEM-ready)
        # =========================================================
        self._log_and_print("=== RCA Performance ===")
        self._log_and_print("AC@1: {:.5f}", k_at_step_all[0])
        self._log_and_print("AC@3: {:.5f}", k_at_step_all[2])
        self._log_and_print("AC@5: {:.5f}", k_at_step_all[4])
        self._log_and_print("AC@10: {:.5f}", k_at_step_all[9])

        self._log_and_print("MRR: {:.5f}", mrr)
        self._log_and_print("HR@1/3/5/10: {:.5f} {:.5f} {:.5f} {:.5f}",
                            hr1, hr3, hr5, hr10)

        self._log_and_print("=== Efficiency ===")
        self._log_and_print("Params: {}", self.total_params)
        self._log_and_print("Model Memory (MB): {:.2f}", model_mem_mb)
        self._log_and_print("Peak Memory (MB): {:.2f}", peak_mem_mb)
        self._log_and_print("Avg time: {:.6f}s", avg_time)
        self._log_and_print("Throughput: {:.2f} samples/s", throughput)

        # =========================================================
        # 7. Save
        # =========================================================
        write_results(
            self.options,
            self.local_model_name,
            [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]],
            k_at_step_all,
            self.total_params,
            self.options.get("results_csv"),
            extra_metrics={
                "mrr": mrr,
                "hr@1": hr1,
                "hr@3": hr3,
                "hr@5": hr5,
                "hr@10": hr10,
                "auc@10": auc_k,
                "std_ac": std_ac,
                "coverage": coverage,
                "avg_time": avg_time,
                "throughput": throughput,
                "model_mem_mb": model_mem_mb,
                "peak_mem_mb": peak_mem_mb,
            }
        )

        return {
            "avg_k_at_step": k_at_step_all,
            "mrr": mrr,
            "hr@1": hr1,
            "hr@3": hr3,
            "hr@5": hr5,
            "hr@10": hr10,
            "auc@10": auc_k,
            "std_ac": std_ac,
            "coverage": coverage,
            "avg_time": avg_time,
            "throughput": throughput,
            "model_mem_mb": model_mem_mb,
            "peak_mem_mb": peak_mem_mb,
        }

    def _testing_root_cause_services_metrics(self, xs, labels, alpha: float = 0.5, use_attention_fusion: bool = False, eval_mode: str = "window"):
        # 0. Feature Mapping Setup
        mapping_path = '/home/db2003/Desktop/Amr/Tests/Medicine/dataset/aiops22-pre/初赛评分数据/idx_to_feature.json'
        with open(mapping_path, 'r') as f:
            self.idx_to_feature = json.load(f)
        feature_names = [self.idx_to_feature[str(i)] for i in range(self.num_vars)]

        coeff_architecture = self.options["coeff_architecture"]
        # 1. Baseline check
        if coeff_architecture in ["rcd", "baro", "nsigma", "torai"]:
            if coeff_architecture == "rcd":
                res = StatisticalRCA.evaluate_rcd(xs, labels)
            elif coeff_architecture == "baro":
                res = StatisticalRCA.evaluate_baro(xs, labels)
            elif coeff_architecture == "nsigma":
                res = StatisticalRCA.evaluate_nsigma(xs, labels)
            elif coeff_architecture == "torai":
                res = StatisticalRCA.evaluate_torai(xs, labels)
            if res:
                k_at_step_all = res["avg_k_at_step"]
                self._log_and_print('Root cause analysis AC@1: {:.5f}', k_at_step_all[0])
                self._log_and_print('Root cause analysis AC@3: {:.5f}', k_at_step_all[2])
                self._log_and_print('Root cause analysis AC@10: {:.5f}', k_at_step_all[9])
                
                # Write results for the RQ tables
                write_results(self.options, self.local_model_name, 
                              [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]], 
                              k_at_step_all, 0, self.options.get("results_csv"))
            return res


        # 2. Model Loading & Setup
        self.load_state_dict(torch.load(os.path.join(self.save_dir, f'{self.model_name}.pt'), map_location=self.device))
        self.eval()
        
        self.us_mean_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'))
        self.us_std_encoder = np.load(os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'))

        us_list = []        
        us_sample_list = [] 
        attn_list = []
        
        # 3. Inference Loop
        with torch.no_grad():
            for i in tqdm(range(len(xs)), desc="Inference"):
                x = xs[i]
                label = labels[i]
                _, _, _, _, _, _, _, us, attn_weights = self._testing_step(x, label, add_u=False)
                u_numpy = us.cpu().numpy() 
                us_sample_list.append(u_numpy)
                us_list.append(u_numpy)
                if use_attention_fusion:
                    attn_mean = attn_weights.mean(dim=0).cpu().numpy()
                    attn_list.append(attn_mean)

        # 4. Global POT Threshold Calculation
        us_all = np.concatenate(us_list, axis=0) 
        us_all_z_score = (-(us_all - self.us_mean_encoder) / self.us_std_encoder)
        
        us_all_z_score_pot = []
        for i in range(self.num_vars):
            col_data = us_all_z_score[:, i]
            col_data = col_data[np.isfinite(col_data)]
            if col_data.size == 0:
                us_all_z_score_pot.append(0.0)
                continue
            try:
                pot_val, _ = pot(col_data, self.risk, self.initial_level, self.num_candidates)
            except:
                pot_val = np.mean(col_data) + 3 * np.std(col_data)
            us_all_z_score_pot.append(pot_val)
        us_all_z_score_pot = np.array(us_all_z_score_pot)

        # 5. Top-K Evaluation (Faithful to Original Loop)
        k_all = []
        k_at_step_all = []
        
        # Sub-level tracking
        results = {
            "service": {"top1": 0, "top3": 0, "top5": 0, "top10": 0},
            "metric": {"top1": 0, "top3": 0, "top5": 0, "top10": 0},
            "node": {"top1": 0, "top3": 0, "top5": 0, "top10": 0}
        }
        dropped = 0
        valid_samples = 0
        for i in tqdm(range(len(xs)), desc="Top-K Evaluation"):
            us_sample = us_sample_list[i]
            z_scores = (-(us_sample - self.us_mean_encoder) / self.us_std_encoder)
            
            if use_attention_fusion:
                attn_per_lag = attn_list[i].mean(axis=2)
                attn_importance = attn_per_lag.mean(axis=0)
                attn_importance = np.expand_dims(attn_importance, axis=0).repeat(z_scores.shape[0], axis=0)
                z_scores = alpha * z_scores + (1 - alpha) * attn_importance
      
            if eval_mode == "onset":
                t_eval = self.window_size // 2
                current_labels = labels[i][t_eval:t_eval+1]
            elif eval_mode == "window":
                current_labels = np.max(labels[i], axis=0, keepdims=True)
            else:
                raise ValueError("eval_mode must be 'onset' or 'window'")
            current_labels = current_labels.astype(bool)  # safety normalization
            # Ground Truth Check for valid_samples count
            if not np.any(current_labels):
                dropped += 1
                continue
            
            valid_samples += 1

            try:
                # Original Top-K Logic (Faithful)
                k_lst = topk(z_scores, current_labels, us_all_z_score_pot)
                k_at_step = topk_at_step(z_scores, current_labels)
                k_all.append(k_lst)
                k_at_step_all.append(k_at_step)

                # --- Faithfully Integrated Multi-Level Logic ---
                gt_indices = np.where(current_labels[0] > 0)[0]
                gt_completes = [feature_names[idx] for idx in gt_indices]

                # Parsing helper based on: node.service-id-metric
                def parse(name):
                    node = name.split('.')[0]
                    service = name.split('.')[1].split("-")[0]

                    # preserve fault keyword explicitly
                    lower = name.lower()

                    if "cpu" in lower:
                        metric = "cpu"
                    elif "mem" in lower:
                        metric = "mem"
                    elif "disk" in lower or "io" in lower:
                        metric = "disk"
                    elif "socket" in lower:
                        metric = "socket"
                    elif "lat" in lower or "delay" in lower:
                        metric = "delay"
                    elif "loss" in lower:
                        metric = "loss"
                    else:
                        metric = "unknown"

                    return node, service, metric

                gt_nodes = set(parse(m)[0] for m in gt_completes)
                gt_services = set(parse(m)[1] for m in gt_completes)
                gt_metrics = set(parse(m)[2] for m in gt_completes)

                sorted_indices = np.argsort(z_scores[0])[::-1]
                ranked_completes = [feature_names[idx] for idx in sorted_indices]

                # Ranked Sub-lists
                seen_n, r_nodes = set(), []
                seen_s, r_services = set(), []
                seen_m, r_metrics = set(), []

                for m in ranked_completes:
                    n, s, met = parse(m)
                    if n not in seen_n: r_nodes.append(n); seen_n.add(n)
                    if s not in seen_s: r_services.append(s); seen_s.add(s)
                    if met not in seen_m: r_metrics.append(met); seen_m.add(met)

                for k in [1, 3, 5, 10]:
                    if any(n in gt_nodes for n in r_nodes[:k]): results["node"][f"top{k}"] += 1
                    if any(s in gt_services for s in r_services[:k]): results["service"][f"top{k}"] += 1
                    if any(m in gt_metrics for m in r_metrics[:k]): results["metric"][f"top{k}"] += 1

            except Exception as e:
                self._log_and_print(f"Error for sample {i}: {str(e)}")
                continue
        onset_coverage = valid_samples / len(xs)
        # 6. Result Aggregation (Faithful Output)
        self._log_and_print("RCA Coverage: {}/{} ({:.2f}%)", valid_samples, len(xs), (valid_samples/len(xs))*100)
        
        if valid_samples > 0:
            k_at_step_all = np.array(k_at_step_all).mean(axis=0)
            
            # 6a. Original Logs
            self._log_and_print('--- COMPLETE LEVEL RCA ---')
            self._log_and_print('Root cause analysis AC@1: {:.5f}', k_at_step_all[0])
            self._log_and_print('Root cause analysis AC@3: {:.5f}', k_at_step_all[2])
            self._log_and_print('Root cause analysis AC@5: {:.5f}', k_at_step_all[4])
            self._log_and_print('Root cause analysis AC@10: {:.5f}', k_at_step_all[9])

            # 6b. New Sub-Level Logs
            for track in ["node", "service", "metric"]:
                self._log_and_print(f'\n--- {track.upper()} LEVEL RCA ---')
                for k in [1, 3, 5, 10]:
                    acc = results[track][f"top{k}"] / valid_samples
                    self._log_and_print(f'AC@{k}: {acc:.5f}')
            
            write_results(self.options, self.local_model_name, [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]], 
                          k_at_step_all, self.total_params, self.options.get("results_csv")+"_(micro service breakdown)",
                          metric_results={k: v / valid_samples for k, v in results["metric"].items()},
                          node_results={k: v / valid_samples for k, v in results["node"].items()},
                          service_results={k: v / valid_samples for k, v in results["service"].items()},
                          RCA_coverage=(valid_samples/len(xs))*100)
        else:
            self._log_and_print("Zero valid samples found.")

