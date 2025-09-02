#!/bin/bash

# --- Configurations ---
seeds=(1 2 3)
coeff_architecture=("rcd" "epsilon_diagnosis")
dataset=("swat" "msds")
lrs=("1e-4")
att_dim=256
heads=2
corelated_list=(0)
window_size=(1)
outer_heads=(4)
outer_hidden_dim=(256)

# --- Helper function to run experiments ---
run_experiment1() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for window_size_item in "${window_size[@]}"; do
        for data in "${dataset[@]}"; do
            for arch in "${coeff_architecture[@]}"; do
                for lr in "${lrs[@]}"; do
                    for seed in "${seeds[@]}"; do
                        for outer_att_dim_val in "${outer_hidden_dim[@]}"; do
                            for outer_heads_val in "${outer_heads[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size_item"

                                cmd="python3 main.py \
                                    --seed=$seed \
                                    --dataset=$data \
                                    --coeff_architecture=$arch \
                                    --window_size=$window_size_item \
                                    --training_aerca=0 \
                                    --results_csv=results_RQ2_baselines.csv"

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
run_experiment1 1
