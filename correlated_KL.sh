#!/bin/bash

# --- Configurations ---
seeds=(1 2 3)
coeff_architecture=("TemporalGNN_Attention")
dataset=("swat")
lrs=("1e-4" "1e-5" "5e-5")
attention_dims=(64 128 256)
num_heads=(1 2 4)
corelated_list=(0)
window_size=(1)

# --- Helper function to run experiments ---
run_experiment1() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for window_size_item in "${window_size[@]}"; do
        for arch in "${coeff_architecture[@]}"; do
            for lr in "${lrs[@]}"; do
                for seed in "${seeds[@]}"; do
                    for att_dim in "${attention_dims[@]}"; do
                        for heads in "${num_heads[@]}"; do
                            echo "Running: dataset=$dataset | seed=$seed | arch=$arch | lr=$lr | att_dim=$att_dim | heads=$heads | AMOC=$use_amoc"

                            cmd="python3 main.py \
                                --correlated_KL=0 \
                                --seed=$seed \
                                --dataset=$dataset \
                                --coeff_architecture=$arch \
                                --lr=$lr \
                                --window_size=$window_size_item \
                                --attention_dim=$att_dim \
                                --num_attention_heads=$heads"
                            if [ "$use_amoc" -eq 1 ]; then
                                cmd="$cmd --AMOC_Loss=0 --mean_std_recon_loss=0"
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
# 2. With AMOC
run_experiment1 1


