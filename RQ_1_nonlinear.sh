

# --- Configurations ---
seeds=(7 8 9 10 11 12)
dataset=("nonlinear")
coeff_architecture="TemporalGNN_Attention_crossattn"
window_size=(1 3 5 7 10)
main_model=("aerca_based")
att_dim=64
heads=2
outer_heads=2
outer_hidden_dim=64

# --- Helper function to run experiments ---
run_experiment_nonlinear_CrGSTA() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                        for main_model_item in "${main_model[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model_item"

                                #seed 1 and window size 12 for preprocessing
                                if [ "$seed" == "1" ] && [ "$window_size_item" == "1" ]; then
                                    preprocessing_data=1
                                else
                                    preprocessing_data=0
                                fi

                                cmd="python3 main.py \
                                    --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                    --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                    --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \

                                    --main_model=$main_model \
                                    --coeff_architecture="$coeff_architecture" \
                                    --preprocessing_data=$preprocessing_data \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --window_size=$window_size_item \
                                    --num_vars=15 \
                                    --training_aerca=1 \
                                    --results_csv=RQ_1_lorenz96_new.csv \
                                    
                                    --epochs=100 \
                                    --time_freq_representation=mag_phase \
                                    --attention_dim="$att_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads" \
                                    --outer_hidden_dim="$outer_hidden_dim" \

                                    "

                                eval $cmd
                        done
            
        done
    done
}
# --- Run experiments ---
run_experiment_nonlinear_CrGSTA








# --- Configurations ---
seeds=(7 8 9 10 11 12)
dataset=("nonlinear")
coeff_architecture="deep_mlp"
window_size=(1 3 5 7 10)
main_model=("aerca_based")
att_dim=64
heads=2
outer_heads=2
outer_hidden_dim=64

# --- Helper function to run experiments ---
run_experiment_nonlinear_CrGSTA_deep() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                        for main_model_item in "${main_model[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model_item"

                                #seed 1 and window size 12 for preprocessing
                                #if [ "$seed" == "1" ] && [ "$window_size_item" == "1" ]; then
                                #    preprocessing_data=1
                                #else
                                #    preprocessing_data=0
                                #fi

                                cmd="python3 main.py \
                                    --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                    --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                    --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \

                                    --main_model=$main_model \
                                    --coeff_architecture="$coeff_architecture" \
                                    --preprocessing_data=0 \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --window_size=$window_size_item \
                                    --num_vars=15 \
                                    --training_aerca=1 \
                                    --results_csv=RQ_1_lorenz96_new.csv \
                                    
                                    --epochs=100 \
                                    --time_freq_representation=mag_phase \
                                    --attention_dim="$att_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads" \
                                    --outer_hidden_dim="$outer_hidden_dim" \

                                    "

                                eval $cmd
                        done
            
        done
    done
}
# --- Run experiments ---
run_experiment_nonlinear_CrGSTA_deep





# --- Configurations ---
seeds=(7 8 9 10 11 12)
dataset=("nonlinear")
coeff_architecture="deep_mlp"
window_size=(1 3 5 7 10)
main_model=("FEDformer")
att_dim=64
heads=2
outer_heads=2
outer_hidden_dim=64

# --- Helper function to run experiments ---
run_experiment_nonlinear_CrGSTA_fedformer() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                        for main_model_item in "${main_model[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model_item"

                                #seed 1 and window size 12 for preprocessing
                                #if [ "$seed" == "1" ] && [ "$window_size_item" == "1" ]; then
                                #    preprocessing_data=1
                                #else
                                #    preprocessing_data=0
                                #fi

                                cmd="python3 main.py \
                                    --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                    --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                    --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \

                                    --main_model=$main_model \
                                    --coeff_architecture="$coeff_architecture" \
                                    --preprocessing_data=0 \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --window_size=$window_size_item \
                                    --num_vars=15 \
                                    --training_aerca=1 \
                                    --results_csv=RQ_1_lorenz96_new.csv \
                                    
                                    --epochs=100 \
                                    --time_freq_representation=mag_phase \
                                    --attention_dim="$att_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads" \
                                    --outer_hidden_dim="$outer_hidden_dim" \

                                    "

                                eval $cmd
                        done
            
        done
    done
}
# --- Run experiments ---
run_experiment_nonlinear_CrGSTA_fedformer






# --- Configurations ---
seeds=(7 8 9 10 11 12)
dataset=("nonlinear")
coeff_architecture="deep_mlp"
window_size=(1 3 5 7 10)
main_model=("iTransformer")
att_dim=64
heads=2
outer_heads=2
outer_hidden_dim=64

# --- Helper function to run experiments ---
run_experiment_nonlinear_CrGSTA_iTransformer() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                        for main_model_item in "${main_model[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model_item"

                                #seed 1 and window size 12 for preprocessing
                                #if [ "$seed" == "1" ] && [ "$window_size_item" == "1" ]; then
                                #    preprocessing_data=1
                                #else
                                #    preprocessing_data=0
                                #fi

                                cmd="python3 main.py \
                                    --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                    --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                    --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \

                                    --main_model=$main_model \
                                    --coeff_architecture="$coeff_architecture" \
                                    --preprocessing_data=0 \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --window_size=$window_size_item \
                                    --num_vars=15 \
                                    --training_aerca=1 \
                                    --results_csv=RQ_1_lorenz96_new.csv \
                                    
                                    --epochs=100 \
                                    --time_freq_representation=mag_phase \
                                    --attention_dim="$att_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads" \
                                    --outer_hidden_dim="$outer_hidden_dim" \

                                    "

                                eval $cmd
                        done
            
        done
    done
}
# --- Run experiments ---
run_experiment_nonlinear_CrGSTA_iTransformer





# --- Configurations ---
seeds=(7 8 9 10 11 12)
dataset=("nonlinear")
coeff_architecture=("rcd" "epsilon_diagnosis")
window_size=(1 3 5 7 10)
main_model=("aerca_based")
att_dim=64
heads=2
outer_heads=2
outer_hidden_dim=64

# --- Helper function to run experiments ---
run_experiment_nonlinear_CrGSTA_baselines() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                        for main_model_item in "${main_model[@]}"; do
                            for coeff_architecture_item in "${coeff_architecture[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model_item"

                                #seed 1 and window size 12 for preprocessing
                                #if [ "$seed" == "1" ] && [ "$window_size_item" == "1" ]; then
                                #    preprocessing_data=1
                                #else
                                #    preprocessing_data=0
                                #fi

                                cmd="python3 main.py \
                                    --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                    --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                    --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \

                                    --main_model=$main_model_item \
                                    --coeff_architecture=$coeff_architecture_item \
                                    --preprocessing_data=0 \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --window_size=$window_size_item \
                                    --num_vars=15 \
                                    --training_aerca=0 \
                                    --results_csv=RQ_1_lorenz96_new.csv \
                                    
                                    --epochs=100 \
                                    --time_freq_representation=mag_phase \
                                    --attention_dim="$att_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads" \
                                    --outer_hidden_dim="$outer_hidden_dim" \

                                    "

                                eval $cmd
                        
                done
            done
        done
    done
}
# --- Run experiments ---
run_experiment_nonlinear_CrGSTA_baselines