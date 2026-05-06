#!/bin/bash
# chmod +x grid_search.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval
#!/bin/bash
# High-Impact Ablation for ASE 2026

# 1. FIXED PARAMETERS (The Winning Base)
SEEDS=(7) # Expand to (7 10 42) for final results

# 2. ABLATION GRIDS
WINDOW_SIZES=(6 20 30 10)
BETA_VALS=(0.01 0.05) # Small beta was the winner
LAMBDA_VALS=(0.5 1.0) # Testing higher sparsity for cleaner HitRate@1
GAMMA_VALS=(0.2 0.5)
DATASET="smd" # Switching to your current successful dataset
coeff_architecture=("vlinear")
RESULTS_FILE="grid_search_results.csv"

for win in "${WINDOW_SIZES[@]}"; do
        first_in_window=true
        for g in "${GAMMA_VALS[@]}"; do
        for b in "${BETA_VALS[@]}"; do
        for l in "${LAMBDA_VALS[@]}"; do
            for coeff in "${coeff_architecture[@]}"; do
            for s in "${SEEDS[@]}"; do
                
                # Skip the 5 runs already completed for Window 6
                if [ "$win" == "6" ]; then
                    if [ "$g" == "0.2" ]; then continue; fi # Finished all Gamma 0.2
                    if [ "$g" == "0.5" ] && [ "$b" == "0.01" ] && [ "$l" == "0.5" ]; then continue; fi # Finished this specific one
                fi

                # Determine if preprocessing is needed
                if [ "$first_in_window" = true ]; then
                    preprocessing=1
                    first_in_window=false
                    echo "--- Preprocessing Window Size $win ---"
                else
                    preprocessing=0
                fi

                echo "Running: Win=$win, Gamma=$g, Lambda=$l, Seed=$s"

                python main.py \
                    --window_size "$win" \
                    --dataset "$DATASET" \
                    --encoder_gamma "$g" --decoder_gamma "$g" \
                    --encoder_lambda "$l" --decoder_lambda "$l" \
                    --seed "$s" \
                    --preprocessing_data "$preprocessing" \
                    --results_csv "$RESULTS_FILE" \
                    --correlated_KL 0 --beta "$b" \
                    --main_model aerca_based --coeff_architecture "$coeff" \
                    --time_freq_representation vlinear --lr 1e-4 \
                    --epochs 200 --early_stopping 0 \
                    --attention_dim 256 --num_attention_heads 2 \
                    --outer_heads_num 2 --outer_hidden_dim 256
            done
            done
        done
        done
        done
done