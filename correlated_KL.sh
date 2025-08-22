#!/bin/bash

seeds=(4 5 6 7 8 9)
coeff_architecture=(TemporalGNN_Attention)
dataset="swat"
lrs=("1e-4")   # two different learning rates
attention_dims=(128)  # different attention dimensions
num_heads=(2)
corelated_list=(0)  # whether to use correlated KL or not

for correlated_KL in "${corelated_list[@]}"; do
  for window_size in 1; do
    for arch in "${coeff_architecture[@]}"; do
      for seed in "${seeds[@]}"; do
        for lr in "${lrs[@]}"; do
          for att_dim in "${attention_dims[@]}"; do
            for heads in "${num_heads[@]}"; do
              echo "Running for $dataset | seed=$seed | arch=$arch | lr=$lr | att_dim=$att_dim | heads=$heads"

              python3 main.py \
                --correlated_KL="$correlated_KL" \
                --seed="$seed" \
                --dataset="$dataset" \
                --coeff_architecture="$arch" \
                --lr="$lr" \
                --window_size="$window_size" \
                --attention_dim="$att_dim" \
                --num_attention_heads="$heads"
            done
          done
        done
      done
    done
  done
done




#!/bin/bash

seeds=(4 5 6 7 8 9)
coeff_architecture=(deep_mlp)
dataset="swat"
lrs=("1e-6")   # two different learning rates
attention_dims=(128)  # different attention dimensions
num_heads=(2)
corelated_list=(0)  # whether to use correlated KL or not

for correlated_KL in "${corelated_list[@]}"; do
  for window_size in 1; do
    for arch in "${coeff_architecture[@]}"; do
      for seed in "${seeds[@]}"; do
        for lr in "${lrs[@]}"; do
          for att_dim in "${attention_dims[@]}"; do
            for heads in "${num_heads[@]}"; do
              echo "Running for $dataset | seed=$seed | arch=$arch | lr=$lr | att_dim=$att_dim | heads=$heads"

              python3 main.py \
                --correlated_KL="$correlated_KL" \
                --seed="$seed" \
                --dataset="$dataset" \
                --coeff_architecture="$arch" \
                --lr="$lr" \
                --window_size="$window_size" \
                --attention_dim="$att_dim" \
                --num_attention_heads="$heads" \
                --early_stopping=1 \
                --epochs=5000
            done
          done
        done
      done
    done
  done
done




seeds=(4 5 6 7 8 9)
coeff_architecture=(TemporalGNN_Attention)
dataset="swat"
lrs=("1e-4")   # two different learning rates
attention_dims=(128)  # different attention dimensions
num_heads=(2)
corelated_list=(0)  # whether to use correlated KL or not

for correlated_KL in "${corelated_list[@]}"; do
  for window_size in 1; do
    for arch in "${coeff_architecture[@]}"; do
      for seed in "${seeds[@]}"; do
        for lr in "${lrs[@]}"; do
          for att_dim in "${attention_dims[@]}"; do
            for heads in "${num_heads[@]}"; do
              echo "Running for $dataset | seed=$seed | arch=$arch | lr=$lr | att_dim=$att_dim | heads=$heads"

              python3 main.py \
                --correlated_KL="$correlated_KL" \
                --seed="$seed" \
                --dataset="$dataset" \
                --coeff_architecture="$arch" \
                --lr="$lr" \
                --window_size="$window_size" \
                --attention_dim="$att_dim" \
                --num_attention_heads="$heads" \
                --mean_std_recon_loss=1 \
                --AMOC_Loss=1
            done
          done
        done
      done
    done
  done
done