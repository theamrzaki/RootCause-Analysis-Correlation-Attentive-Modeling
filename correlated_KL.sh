#!/bin/bash

seeds=(2 3)
gloabl_attention_over_all_lag =(0 1)

for correlated_KL in 1; do
    for gloabl_attention_over_all_lag in "${gloabl_attention_over_all_lag[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "Running with correlated_KL=$correlated_KL, seed=$seed, gloabl_attention_over_all_lag=$gloabl_attention_over_all_lag"
            python3 main.py --correlated_KL="$correlated_KL" --seed="$seed" --dataset="msds" --lambda_indep=1.0 --lambda_corr=0.8 --gloabl_attention_over_all_lag="$gloabl_attention_over_all_lag" 
        done
    done
done
