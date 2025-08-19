#!/bin/bash

seeds=(2 3 4 11 12 13)
coeff_architecture=(TemporalGNN_Attention)
dataset="msds"
lrs=("1e-4" "5e-4")   # two different learning rates
attention_dims=(32 64 128 256)  # different attention dimensions
num_heads=(2 4)

for window_size in 1; do
  for arch in "${coeff_architecture[@]}"; do
    for seed in "${seeds[@]}"; do
      for lr in "${lrs[@]}"; do
        for att_dim in "${attention_dims[@]}"; do
          for heads in "${num_heads[@]}"; do
            echo "Running for $dataset | seed=$seed | arch=$arch | lr=$lr | att_dim=$att_dim | heads=$heads"

            python3 main.py \
              --correlated_KL=0 \
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
