#!/bin/bash

seeds=(4)
attention_modes=("none")

for correlated_KL in 1; do
  for attention_mode in "${attention_modes[@]}"; do
    for seed in "${seeds[@]}"; do
      echo "Running with correlated_KL=$correlated_KL, seed=$seed, global_attention_over_all_lag=$attention_mode"
      python3 main.py \
        --correlated_KL="$correlated_KL" \
        --seed="$seed" \
        --dataset="swat" \
        --lambda_indep=0.5 \
        --lambda_corr=1.0 \
        --global_attention_over_all_lag="$attention_mode"
    done
  done
done