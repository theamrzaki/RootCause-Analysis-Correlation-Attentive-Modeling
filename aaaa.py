
    def _testing_root_cause_multimodal_fast(self, xs, labels):
        """
        Multimodal RCA evaluation (fast + consistent + metric-safe).
        Assumes POD-style RCA over variables/hidden dimensions.
        """

        import os, time
        import numpy as np
        import torch
        import psutil
        from tqdm import tqdm

        # -------------------------------------------------
        # Load model + stats
        # -------------------------------------------------
        self.load_state_dict(
            torch.load(
                os.path.join(self.save_dir, f"{self.model_name}.pt"),
                map_location=self.device
            )
        )
        self.eval()

        self.us_mean_encoder = np.load(
            os.path.join(self.save_dir, f"{self.model_name}_us_mean_encoder.npy")
        )
        self.us_std_encoder = np.load(
            os.path.join(self.save_dir, f"{self.model_name}_us_std_encoder.npy")
        )

        use_cuda = torch.cuda.is_available() and self.device != "cpu"
        process = psutil.Process(os.getpid())
        peak_mem = 0

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        # -------------------------------------------------
        # metrics
        # -------------------------------------------------
        valid_k_curves = []

        mrr_list = []
        hr1_list, hr3_list, hr5_list, hr10_list = [], [], [], []
        inference_times = []

        # -------------------------------------------------
        # label normalization (robust POD-safe)
        # -------------------------------------------------
        def normalize_label(label):
            label = np.asarray(label)

            # scalar
            if label.ndim == 0:
                return np.zeros(1, dtype=np.int32)

            # time x vars → collapse time (POD assumption)
            if label.ndim == 2:
                return (label.max(axis=0) > 0).astype(np.int32)

            # already vector
            if label.ndim == 1:
                return (label > 0).astype(np.int32)

            raise ValueError(f"Unsupported label shape: {label.shape}")

        # -------------------------------------------------
        # safe score collapse (VERY important for multimodal us)
        # -------------------------------------------------
        def collapse_us(us):
            us = np.asarray(us)

            if us.ndim == 1:
                return us

            # (hidden, vars) OR (modalities, vars) OR (T, vars)
            return us.mean(axis=tuple(range(us.ndim - 1)))

        # -------------------------------------------------
        # inference
        # -------------------------------------------------
        with torch.no_grad():
            for i in tqdm(range(len(xs)), desc="Multimodal RCA FAST"):

                start = time.time()

                x = xs[i]
                label = labels[i]

                _, _, _, _, _, _, _, us, attn = self._testing_step(
                    x, label, add_u=False
                )

                us = us.detach().cpu().numpy()
                z_scores = (-(us - self.us_mean_encoder) / self.us_std_encoder)

                # collapse to variable-level scores
                score = collapse_us(z_scores)

                current_labels = normalize_label(label)
                true_idx = np.where(current_labels == 1)[0]

                # -------------------------------------------------
                # skip samples with no ground truth anomalies
                # (important: keeps metric consistency)
                # -------------------------------------------------
                if len(true_idx) == 0:
                    inference_times.append(time.time() - start)
                    if not use_cuda:
                        peak_mem = max(peak_mem, process.memory_info().rss)
                    continue

                # -------------------------------------------------
                # ranking
                # -------------------------------------------------
                ranking = np.argsort(-score)

                rr = 0.0
                for rank, idx in enumerate(ranking, start=1):
                    if idx in true_idx:
                        rr = 1.0 / rank
                        break

                mrr_list.append(rr)

                hr1_list.append(int(any(idx in ranking[:1] for idx in true_idx)))
                hr3_list.append(int(any(idx in ranking[:3] for idx in true_idx)))
                hr5_list.append(int(any(idx in ranking[:5] for idx in true_idx)))
                hr10_list.append(int(any(idx in ranking[:10] for idx in true_idx)))

                # -------------------------------------------------
                # top-k curve (ONLY valid samples)
                # -------------------------------------------------
                try:
                    valid_k_curves.append(
                        topk_at_step_multimodality(
                            z_scores, current_labels
                        )
                    )
                except Exception:
                    continue

                inference_times.append(time.time() - start)

                if not use_cuda:
                    peak_mem = max(peak_mem, process.memory_info().rss)

        # -------------------------------------------------
        # safety check
        # -------------------------------------------------
        if len(valid_k_curves) == 0:
            self._log_and_print("No valid RCA samples found.")
            return None

        # -------------------------------------------------
        # aggregation
        # -------------------------------------------------
        k_at_step_all = np.mean(np.array(valid_k_curves), axis=0)

        mrr = np.mean(mrr_list)
        hr1 = np.mean(hr1_list)
        hr3 = np.mean(hr3_list)
        hr5 = np.mean(hr5_list)
        hr10 = np.mean(hr10_list)

        auc_k = np.mean(k_at_step_all[:10])
        std_ac = np.std(k_at_step_all)

        avg_time = np.mean(inference_times)
        throughput = 1.0 / avg_time if avg_time > 0 else 0.0

        peak_mem_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if use_cuda else peak_mem / (1024 ** 2)
        )

        coverage = len(valid_k_curves) / len(xs)

        # -------------------------------------------------
        # logs
        # -------------------------------------------------
        self._log_and_print("FAST Multimodal RCA (clean)")
        self._log_and_print(
            "AC@1/3/5/10: {:.5f} {:.5f} {:.5f} {:.5f}",
            k_at_step_all[0], k_at_step_all[2],
            k_at_step_all[4], k_at_step_all[9]
        )

        self._log_and_print("MRR: {:.5f}", mrr)
        self._log_and_print("HR@1/3/5/10: {:.5f} {:.5f} {:.5f} {:.5f}",
                            hr1, hr3, hr5, hr10)

        self._log_and_print("Time: {:.6f}s | Throughput: {:.2f}",
                            avg_time, throughput)

        # -------------------------------------------------
        # write results
        # -------------------------------------------------
        write_results(
            self.options,
            self.local_model_name,
            [
                k_at_step_all[0],
                k_at_step_all[2],
                k_at_step_all[4],
                k_at_step_all[9],
            ],
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
                "model_mem_mb": peak_mem_mb,
            },
        )

        return {
            "k_at_step": k_at_step_all,
            "mrr": mrr,
            "hr": (hr1, hr3, hr5, hr10),
            "auc@10": auc_k,
            "std_ac": std_ac,
            "avg_time": avg_time,
            "throughput": throughput,
            "coverage": coverage,
            "peak_mem_mb": peak_mem_mb,
        }



    def _testing_root_cause_multimodal_fast_to_be_deleted(
        self,
        xs,
        labels,
        alpha: float = 0.5
    ):

        self.load_state_dict(
            torch.load(
                os.path.join(self.save_dir, f'{self.model_name}.pt'),
                map_location=self.device
            )
        )
        self.eval()

        self.us_mean_encoder = np.load(
            os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy')
        )
        self.us_std_encoder = np.load(
            os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy')
        )

        use_cuda = torch.cuda.is_available() and self.device != "cpu"

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        process = psutil.Process(os.getpid())
        peak_mem = 0

        # metrics
        k_at_step_all = []
        mrr_list = []
        hr1_list, hr3_list, hr5_list, hr10_list = [], [], [], []
        inference_times = []

        self.eval()

        with torch.no_grad():
            for i in tqdm(range(len(xs)), desc="Multimodal RCA (fast)"):

                start = time.time()

                x = xs[i]
                label = labels[i]

                _, _, _, _, _, _, _, us, attn = self._testing_step(
                    x, label, add_u=False
                )

                # -----------------------------
                # move once
                # -----------------------------
                us = us.detach().cpu().numpy()

                z_scores = (-(us - self.us_mean_encoder) / self.us_std_encoder)

                current_labels = (labels[i] > 0).astype(np.int32)
                # -----------------------------
                # Top-K + metrics (streaming)
                # -----------------------------
                try:
                    k_at_step = topk_at_step_multimodality(z_scores, current_labels)
                    k_at_step_all.append(k_at_step)

                    ranking = np.argsort(-z_scores[0])
                    true_idx = np.where(current_labels[0] == 1)[0]
                    # skip samples with no ground-truth faults
                    if len(true_idx) == 0:
                        continue

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

                    inference_times.append(time.time() - start)

                except Exception as e:
                    logging.error(f"Error occurred while processing sample {i}: {e}")       
                    continue

                # lightweight memory tracking
                if not use_cuda:
                    peak_mem = max(peak_mem, process.memory_info().rss)

        # -----------------------------
        # aggregation
        # -----------------------------
        k_at_step_all = np.array(k_at_step_all).mean(axis=0)

        mrr = np.mean(mrr_list)
        hr1, hr3, hr5, hr10 = map(np.mean, [hr1_list, hr3_list, hr5_list, hr10_list])

        auc_k = np.mean(k_at_step_all[:10])
        std_ac = np.std(k_at_step_all)

        avg_time = np.mean(inference_times)
        throughput = 1.0 / avg_time if avg_time > 0 else 0.0

        peak_mem_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if use_cuda else peak_mem / (1024 ** 2)
        )

        self._log_and_print("FAST Multimodal RCA Results")
        self._log_and_print("MRR: {:.5f}", mrr)
        self._log_and_print("HR@1/3/5/10: {:.5f} {:.5f} {:.5f} {:.5f}",
                            hr1, hr3, hr5, hr10)
        self._log_and_print("Avg time: {:.6f}s | Throughput: {:.2f}",
                            avg_time, throughput)

        return {
            "k_at_step": k_at_step_all,
            "mrr": mrr,
            "hr": (hr1, hr3, hr5, hr10),
            "auc@10": auc_k,
            "std_ac": std_ac,
            "avg_time": avg_time,
            "throughput": throughput,
            "peak_mem_mb": peak_mem_mb
        }


    def _testing_root_cause_multimodal_fast(
        self,
        xs,
        labels
    ):
        # -------------------------------------------------
        # load model
        # -------------------------------------------------
        self.load_state_dict(
            torch.load(
                os.path.join(self.save_dir, f"{self.model_name}.pt"),
                map_location=self.device
            )
        )
        self.eval()

        self.us_mean_encoder = np.load(
            os.path.join(self.save_dir, f"{self.model_name}_us_mean_encoder.npy")
        )
        self.us_std_encoder = np.load(
            os.path.join(self.save_dir, f"{self.model_name}_us_std_encoder.npy")
        )

        use_cuda = torch.cuda.is_available() and self.device != "cpu"
        process = psutil.Process(os.getpid())
        peak_mem = 0

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        # -------------------------------------------------
        # metrics
        # -------------------------------------------------
        k_at_step_all = []
        mrr_list, hr1_list, hr3_list, hr5_list, hr10_list = [], [], [], [], []
        inference_times = []

        # -------------------------------------------------
        # inference
        # -------------------------------------------------
        self.eval()
    
        with torch.no_grad():
            for i in tqdm(range(len(xs)), desc="Multimodal RCA FAST v2"):

                start = time.time()

                x = xs[i]
                label = labels[i]

                _, _, _, _, _, _, _, us, attn = self._testing_step(
                    x, label, add_u=False
                )

                us = us.detach().cpu().numpy()
                z_scores = (-(us - self.us_mean_encoder) / self.us_std_encoder)

                # -------------------------------------------------
                # SAFE MULTIMODAL LABEL HANDLING
                # -------------------------------------------------
                current_labels = (np.asarray(label) > 0).astype(np.int32)

                # -------------------------------------------------
                # top-k curve (always computed)
                # -------------------------------------------------
                try:
                    k_at_step_all.append(
                        topk_at_step_multimodality(z_scores, current_labels)
                    )
                except Exception:
                    continue

                # -------------------------------------------------
                # ranking metrics (skip empty anomaly cases only here)
                # -------------------------------------------------
                true_idx = np.where(current_labels[0] == 1)[0]

                if len(true_idx) > 0:

                    ranking = np.argsort(-z_scores[0])

                    rr = 0.0
                    for rank, idx in enumerate(ranking, start=1):
                        if idx in true_idx:
                            rr = 1.0 / rank
                            break
                    mrr_list.append(rr)

                    hr1_list.append(int(any(idx in ranking[:1] for idx in true_idx)))
                    hr3_list.append(int(any(idx in ranking[:3] for idx in true_idx)))
                    hr5_list.append(int(any(idx in ranking[:5] for idx in true_idx)))
                    hr10_list.append(int(any(idx in ranking[:10] for idx in true_idx)))

                inference_times.append(time.time() - start)

                if not use_cuda:
                    peak_mem = max(peak_mem, process.memory_info().rss)

        # -------------------------------------------------
        # aggregation
        # -------------------------------------------------
        if len(k_at_step_all) == 0:
            self._log_and_print("No valid samples for RCA.")
            return None

        k_at_step_all = np.mean(np.array(k_at_step_all), axis=0)

        mrr = np.mean(mrr_list) if len(mrr_list) > 0 else 0.0
        hr1 = np.mean(hr1_list) if len(hr1_list) > 0 else 0.0
        hr3 = np.mean(hr3_list) if len(hr3_list) > 0 else 0.0
        hr5 = np.mean(hr5_list) if len(hr5_list) > 0 else 0.0
        hr10 = np.mean(hr10_list) if len(hr10_list) > 0 else 0.0

        auc_k = np.mean(k_at_step_all[:10])
        std_ac = np.std(k_at_step_all)

        avg_time = np.mean(inference_times)
        throughput = 1.0 / avg_time if avg_time > 0 else 0.0

        peak_mem_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if use_cuda else peak_mem / (1024 ** 2)
        )

        # -------------------------------------------------
        # logs
        # -------------------------------------------------
        self._log_and_print("FAST Multimodal RCA v2")
        self._log_and_print("AC@1/3/5/10: {:.5f} {:.5f} {:.5f} {:.5f}",
                            k_at_step_all[0], k_at_step_all[2],
                            k_at_step_all[4], k_at_step_all[9])

        self._log_and_print("MRR: {:.5f}", mrr)
        self._log_and_print("HR@1/3/5/10: {:.5f} {:.5f} {:.5f} {:.5f}",
                            hr1, hr3, hr5, hr10)

        self._log_and_print("Time: {:.6f}s | Throughput: {:.2f}",
                            avg_time, throughput)

        # -------------------------------------------------
        # write results (IMPORTANT FIX)
        # -------------------------------------------------
        write_results(
            self.options,
            self.local_model_name,
            [
                k_at_step_all[0],
                k_at_step_all[2],
                k_at_step_all[4],
                k_at_step_all[9],
            ],
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
                "coverage": len(k_at_step_all) / len(xs),
                "avg_time": avg_time,
                "throughput": throughput,
                "model_mem_mb": peak_mem_mb,
            },
        )

        return {
            "k_at_step": k_at_step_all,
            "mrr": mrr,
            "hr": (hr1, hr3, hr5, hr10),
            "auc@10": auc_k,
            "std_ac": std_ac,
            "avg_time": avg_time,
            "throughput": throughput,
            "peak_mem_mb": peak_mem_mb,
        }


