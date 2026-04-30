#!/bin/bash
# chmod +x grid_search.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval
#!/bin/bash
# High-Impact Ablation for ASE 2026

# 1. FIXED PARAMETERS (The Winning Base)
SEEDS=(7) # Expand to (7 10 42) for final results
BETA=0.0

# 2. ABLATION GRIDS
WINDOW_SIZES=(6 8 10 20 30)
GAMMA_VALS=(0.2)
LAMBDA_VALS=(0.5)
DATASET="smd"
coeff_architecture=("vlinear")
RESULTS_FILE="grid_search_results.csv"

for win in "${WINDOW_SIZES[@]}"; do
    for g in "${GAMMA_VALS[@]}"; do
        for l in "${LAMBDA_VALS[@]}"; do
            for coeff in "${coeff_architecture[@]}"; do
            for s in "${SEEDS[@]}"; do
                
                echo "Running: Win=$win, Gamma=$g, Lambda=$l, Seed=$s"

                python main.py \
                    --window_size "$win" \
                    --dataset "$DATASET" \
                    --encoder_gamma "$g" --decoder_gamma "$g" \
                    --encoder_lambda "$l" --decoder_lambda "$l" \
                    --seed "$s" \
                    --preprocessing_data 1 \
                    --results_csv "$RESULTS_FILE" \
                    --correlated_KL 0 --beta "$BETA" \
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