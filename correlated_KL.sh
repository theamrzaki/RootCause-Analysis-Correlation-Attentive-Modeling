#!/bin/bash

# --- Configurations ---
seeds=(1 2 3)
coeff_architecture=("TemporalGNN_Attention")
dataset="msds"
lrs=("5e-4" "1e-4")
attention_dims=(128 256)
num_heads=(1 2 4)
corelated_list=(0)
window_size=1

# --- Helper function to run experiments ---
run_experiment() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for correlated_KL in "${corelated_list[@]}"; do
        for arch in "${coeff_architecture[@]}"; do
            for seed in "${seeds[@]}"; do
                for lr in "${lrs[@]}"; do
                    for att_dim in "${attention_dims[@]}"; do
                        for heads in "${num_heads[@]}"; do
                            echo "Running: dataset=$dataset | seed=$seed | arch=$arch | lr=$lr | att_dim=$att_dim | heads=$heads | AMOC=$use_amoc"

                            cmd="python3 main.py \
                                --correlated_KL=$correlated_KL \
                                --seed=$seed \
                                --dataset=$dataset \
                                --coeff_architecture=$arch \
                                --lr=$lr \
                                --window_size=$window_size \
                                --attention_dim=$att_dim \
                                --num_attention_heads=$heads"

                            if [ "$use_amoc" -eq 1 ]; then
                                cmd="$cmd --AMOC_Loss=1 --mean_std_recon_loss=1"
                            fi

                            eval $cmd
                        done
                    done
                done
            done
        done
    done
}

# --- Run experiments ---
# 1. Without AMOC
run_experiment 0

# 2. With AMOC
run_experiment 1
