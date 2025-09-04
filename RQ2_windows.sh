#!/bin/bash

# --- Configurations ---
seeds=(1 2 3)
coeff_architecture=("TemporalGNN_Attention")
dataset=("msds")
lrs=("5e-4")
attention_dims=(128)
num_heads=(4)
corelated_list=(0)
window_size=(2 3 4 5)
outer_heads_num=4
outer_hidden_dim=256

# --- Helper function to run experiments ---
run_experiment_msds() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for window_size_item in "${window_size[@]}"; do
        for arch in "${coeff_architecture[@]}"; do
            for lr in "${lrs[@]}"; do
                for seed in "${seeds[@]}"; do
                    for att_dim in "${attention_dims[@]}"; do
                        for heads in "${num_heads[@]}"; do
                            echo "Running: dataset=$dataset | seed=$seed | arch=$arch | lr=$lr | att_dim=$att_dim | heads=$heads | AMOC=$use_amoc"

                                cmd="python3 main.py \
                                    --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                    --lr=$lr \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --coeff_architecture=$arch \
                                    --window_size=$window_size_item \
                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --attention_dim=$att_dim \
                                    --num_attention_heads=$heads \
                                    --outer_heads_num=$outer_heads_num \
                                    --outer_hidden_dim=$outer_hidden_dim \
                                    --results_csv=results_RQ3_windows.csv"

                            eval $cmd
                        done
                    done
                done
            done
        done
    done
}
# --- Run experiments ---
# 2. With AMOC
#run_experiment_msds 1






# --- Configurations ---
seeds=(1 2 3)
coeff_architecture=("TemporalGNN_Attention")
dataset=("swat")
lrs=("1e-4")
attention_dims=(256)
num_heads=(2)
corelated_list=(0)
window_size=(2 3 4 5)
outer_heads_num=4
outer_hidden_dim=256

# --- Helper function to run experiments ---
run_experiment_swat() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for window_size_item in "${window_size[@]}"; do
        for arch in "${coeff_architecture[@]}"; do
            for lr in "${lrs[@]}"; do
                for seed in "${seeds[@]}"; do
                    for att_dim in "${attention_dims[@]}"; do
                        for heads in "${num_heads[@]}"; do
                            echo "Running: dataset=$dataset | seed=$seed | arch=$arch | lr=$lr | att_dim=$att_dim | heads=$heads | AMOC=$use_amoc"

                                cmd="python3 main.py \
                                    --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                    --lr=$lr \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --coeff_architecture=$arch \
                                    --window_size=$window_size_item \
                                    --training_aerca=1 \
                                    --epochs=100 \
                                    --early_stopping=0 \
                                    --attention_dim=$att_dim \
                                    --num_attention_heads=$heads \
                                    --outer_heads_num=$outer_heads_num \
                                    --outer_hidden_dim=$outer_hidden_dim \
                                    --results_csv=results_RQ3_windows.csv"

                            eval $cmd
                        done
                    done
                done
            done
        done
    done
}
# --- Run experiments ---
# 2. With AMOC
#run_experiment_swat 1


seeds=(1 2 3)
coeff_architecture=("deep_mlp")
dataset=("swat" "msds")
lrs=("1e-6")
att_dim=256
heads=2
corelated_list=(0)
window_size=(2 3)
outer_heads=(4)
outer_hidden_dim=(256)

# --- Helper function to run experiments ---
run_experiment_deepmlp() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
            for data in "${dataset[@]}"; do
                for arch in "${coeff_architecture[@]}"; do
                    for lr in "${lrs[@]}"; do
                        for outer_att_dim_val in "${outer_hidden_dim[@]}"; do
                            for outer_heads_val in "${outer_heads[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size_item | lr=$lr"

                                cmd="python3 main.py \
                                    --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                    --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                    --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \
                                    --lr=$lr \
                                    --seed=$seed \
                                    --dataset=$data \
                                    --coeff_architecture=$arch \
                                    --window_size=$window_size_item \
                                    --training_aerca=1 \
                                    --epochs=5000 \
                                    --early_stopping=1 \
                                    --results_csv=results_RQ3_windows.csv"

                                eval $cmd
                            done
                        done
                    done
                done
            done
        done
    done
}
# --- Run experiments ---
# 2. With AMOC
run_experiment_deepmlp 1


