# --- Configurations ---
seeds=(1 2 3)
coeff_architecture=("TemporalGNN_Attention_crossattn")
dataset=("lotka_volterra")
main_model=("aerca_based")
lrs=("1e-4")
att_dim=64
heads=2
corelated_list=(0)
window_size=(7)
outer_heads=(2)
outer_hidden_dim=(64)
num_vars=(60 20 30 40 50 )

# --- Helper function to run experiments --- (still mag phase)
run_experiment_CrGSTA_spatial() {
    for var in "${num_vars[@]}"; do
        for seed in "${seeds[@]}"; do
                                    echo "Running: dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size_item | lr=$lr"
                                    # only first time make preprocessing = 1 seed=1 and coeff_architecture="deep_mlp" and window_size=1
                                    if [ "$seed" == "1" ]; then
                                        preprocessing_data=1
                                    else
                                        preprocessing_data=0
                                    fi
                                    echo "preprocessing_data=$preprocessing_data"      

                                    cmd="python3 main.py \
                                                    --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                        --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                        --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \
                                        --main_model=$main_model \
                                        --lr=$lrs \
                                        --preprocessing_data=$preprocessing_data \
                                        --seed=$seed \
                                        --dataset="$dataset"\
                                        --coeff_architecture=$coeff_architecture \
                                        --window_size=$window_size \
                                        --training_aerca=1 \
                                        --epochs=100 \
                                        --early_stopping=0 \
                                        --time_freq_representation=mag_phase \
                                        --attention_dim=$att_dim \
                                        --num_attention_heads=$heads \
                                        --outer_heads_num=$outer_heads \
                                        --outer_hidden_dim=$outer_hidden_dim \
                                        --num_vars="$var" \
                                        --results_csv=RQ_2_results_lorenz_spatial_correct.csv"

                                    eval $cmd
                                
                            
                        
                    
                
            
        done
    done
}
# --- Run experiments ---
# 2. With AMOC
run_experiment_CrGSTA_spatial





seeds=(1 2 3)
dataset=("lotka_volterra")
main_model=("FEDformer" "iTransformer")
lrs=("1e-4")
att_dim=64
heads=2
corelated_list=(0)
window_size=(7)
outer_heads=(2)
outer_hidden_dim=(64)
num_vars=(60 20 30 40 50)

# --- Helper function to run experiments ---
run_experiment_CrGSTA_spatial_lotka() {
    for var in "${num_vars[@]}"; do
        for seed in "${seeds[@]}"; do
            for arch in "${main_model[@]}"; do
                    echo "Running: dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size | lr=$lrs"
                    # only first time make preprocessing = 1 seed=1 and coeff_architecture="deep_mlp" and window_size=1
                    if [ "$seed" == "1" ]; then
                        preprocessing_data=1
                    else
                        preprocessing_data=0
                    fi
                            echo "preprocessing_data=$preprocessing_data"      

                        cmd="python3 main.py \
                            --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                            --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                            --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \
                            --lr=$lrs \
                            --main_model=$arch \
                            --attention_dim=$outer_hidden_dim \
                            --num_attention_heads=$outer_heads \
                            --seed=$seed \
                            --dataset=$dataset \
                            --window_size=$window_size \
                            --training_aerca=1 \
                            --epochs=100 \
                            --early_stopping=0 \
                            --num_vars="$var" \
                            --results_csv=RQ_2_results_lorenz_spatial_correct.csv"

                        eval $cmd
            done
        done
    done
}
# --- Run experiments ---
# 2. With AMOC
run_experiment_CrGSTA_spatial_lotka





