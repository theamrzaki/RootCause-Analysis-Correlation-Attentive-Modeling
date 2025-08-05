#!/bin/bash

seeds=(1 2 3)
global_attention_over_all_lag=(0)

for correlated_KL in 0 1; do
  for global_attention_over_all_lag in "${global_attention_over_all_lag[@]}"; do
    for seed in "${seeds[@]}"; do
      echo "Running with correlated_KL=$correlated_KL, seed=$seed, global_attention_over_all_lag=$global_attention_over_all_lag"
      python3 main.py \
        --correlated_KL="$correlated_KL" \
        --seed="$seed" \
        --dataset="swat" \
        --lambda_indep=1.0 \
        --lambda_corr=1.0 \
        --global_attention_over_all_lag=$global_attention_over_all_lag
    done
  done
done
