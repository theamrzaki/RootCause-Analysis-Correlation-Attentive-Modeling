#!/bin/bash

seeds=(13 12 11 4 3 2)
coeff_architecture=(TemporalGNN_Attention)
dataset="msds"
lrs=("5e-4")   # two different learning rates
attention_dims=(128)  # different attention dimensions
num_heads=(2)
window_size_list=(1)  # different window sizes
for window_size in "${window_size_list[@]}"; do
  for arch in "${coeff_architecture[@]}"; do
    for seed in "${seeds[@]}"; do
      for lr in "${lrs[@]}"; do
        for att_dim in "${attention_dims[@]}"; do
          for heads in "${num_heads[@]}"; do
            echo "Running for $dataset | seed=$seed | arch=$arch | lr=$lr | att_dim=$att_dim | heads=$heads | window=$window_size"

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
