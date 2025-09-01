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
                                --num_attention_heads=$heads \
                                --outer_heads_num=$outer_heads_num \
                                --outer_hidden_dim=$outer_hidden_dim"
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
