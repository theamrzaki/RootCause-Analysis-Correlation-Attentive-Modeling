source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval


python main.py --correlated_KL 0 --mean_std_recon_loss 0 --AMOC_Loss 0 --encoder_alpha 0.01 --decoder_alpha 0.01 --encoder_gamma 0.01 --decoder_gamma 0.01 --encoder_lambda 0.01 --decoder_lambda 0.01 --beta 0.01 --main_model aerca_based --coeff_architecture vlinear --time_freq_representation vlinear --lr 1e-4 --seed 7 --dataset aiops --window_size 25 --training_aerca 1 --epochs 200 --early_stopping 0 --preprocessing_data 1 --results_csv RQ_1_aiops.csv --attention_dim 256 --num_attention_heads 2 --outer_heads_num 2 --outer_hidden_dim 256


#running using AERCA standard loss