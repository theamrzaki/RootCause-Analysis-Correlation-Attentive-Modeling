#!/bin/bash

seeds=(4)
coeff_architecture=(AttentionCoeffGNN_multihead)
dataset="swat"

for correlated_KL in 0; do
  for arch in "${coeff_architecture[@]}"; do
    for seed in "${seeds[@]}"; do
      echo "Running for $dataset on seed=$seed with (correlated_KL=$correlated_KL,  architecture=$arch)"
      python3 main.py \
        --correlated_KL="$correlated_KL" \
        --seed="$seed" \
        --dataset="$dataset" \
        --coeff_architecture="$arch"
    done
  done
done