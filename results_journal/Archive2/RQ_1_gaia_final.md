Running:  arch=CUTS_PLUS | dataset=gaia | seed=2 | window_size=10 | lr=1e-4 | main_model=aerca_based | preprocessing_data=0
Selected dataset: gaia
Set seed: 2, lr: 0.0001 window_size: 10 encoder_alpha: 1.0 decoder_alpha: 1.0 encoder_gamma: 0.2 decoder_gamma: 0.2 encoder_lambda: 0.5 decoder_lambda: 0.5 beta: 0.01
shrinkage :0.5
Loading existing data...

--- Starting Data Pipeline Sanity Check ---
Normal Data Shape: (119720, 10, 50)
Abnormal Data Shape: (350, 10, 50)
Anomaly Coverage Check: Found 18521 anomaly samples.
Feature Statistics -> Min: -1.2503, Max: 10.0000, Mean: 1.2317
Variance Check: All sensors are active.
--- Sanity Check Passed ---

Loading precomputed Q matrix from /home/db2003/Desktop/Amr/RootCause-Analysis-Correlation-Attentive-Modeling/datasets/gaia_data/orth_transform_meta/swat_q_matrix_lag10.npy
Number of parameters in encoder: 274738
----------------------------------
Total number of parameters in AERCA: 274738
----------------------------------
Initializing weights...
Start training AERCA model...
Epoch: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [5:33:39<00:00, 100.10s/it]
Training complete
Avg Epoch Time: 94.6048s
Throughput: 1012.38 samples/s
Peak Memory: 2723.30 MB