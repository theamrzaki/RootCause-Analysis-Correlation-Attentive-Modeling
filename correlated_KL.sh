#!/bin/bash

# --- Configurations ---
seeds=(1 2 3)
coeff_architecture=("deep_mlp")
dataset=("swat")
lrs=("1e-6")
attention_dims=(128)
num_heads=(2)
corelated_list=(0)
window_size=(2 3 4)

# --- Helper function to run experiments ---
run_experiment1() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for window_size_item in "${window_size[@]}"; do
        for arch in "${coeff_architecture[@]}"; do
            for seed in "${seeds[@]}"; do
                for lr in "${lrs[@]}"; do
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
                                --early_stopping=1 \
                                --epochs=5000"
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
# 2. With AMOC
run_experiment1 1





seeds=(1 2 3)
coeff_architecture=("TemporalGNN_Attention")
dataset=("msds")
lrs=("5e-4")
attention_dims=(128)
num_heads=(2)
corelated_list=(0)
window_size=(2 3 4)

# --- Helper function to run experiments ---
run_experiment2() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for window_size_item in "${window_size[@]}"; do
        for arch in "${coeff_architecture[@]}"; do
            for seed in "${seeds[@]}"; do
                for lr in "${lrs[@]}"; do
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
# 2. With AMOC
run_experiment2 1




# --- Configurations ---
seeds=(1 2 3)
coeff_architecture=("deep_mlp")
dataset=("msds")
lrs=("1e-6")
attention_dims=(128)
num_heads=(2)
corelated_list=(0)
window_size=(2 3 4)

# --- Helper function to run experiments ---
run_experiment3() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for window_size_item in "${window_size[@]}"; do
        for arch in "${coeff_architecture[@]}"; do
            for seed in "${seeds[@]}"; do
                for lr in "${lrs[@]}"; do
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
                                --early_stopping=1 \
                                --epochs=5000"
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
# 2. With AMOC
run_experiment3 1