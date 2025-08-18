#!/bin/bash

seeds=(1 2 3 4 11 12 13)
coeff_architecture=(deep_mlp TemporalGNN_Attention)
dataset="msds"

for correlated_KL in 0 1; do
  for arch in "${coeff_architecture[@]}"; do
    for seed in "${seeds[@]}"; do
      # set lr based on arch
      if [ "$arch" == "deep_mlp" ]; then
        lr="1e-6"
      else
        lr="1e-4"
      fi

      echo "Running for $dataset on seed=$seed with (correlated_KL=$correlated_KL, architecture=$arch, lr=$lr)"
      python3 main.py \
        --correlated_KL="$correlated_KL" \
        --seed="$seed" \
        --dataset="$dataset" \
        --coeff_architecture="$arch" \
        --lr="$lr"
    done
  done
done
