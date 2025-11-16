
# --- Configurations ---
seeds=(7)
dataset=("msds")
lrs=("1e-4")
window_size=(1)
arch="TemporalGNN_Attention_crossattn"
main_model="aerca_based"
attention_dim=16
heads=2
outer_att_dim_val=16
outer_heads_val=2
# --- Helper function to run experiments ---
run_experiment_msds_casestudy() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model"

                                cmd="python3 main.py \
                                        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                        --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                        --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \

                                    --main_model=$main_model \
                                    --coeff_architecture="$arch" \
                                    --time_freq_representation="mag_phase" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --plot_case_study=0 \
                                    --plot_case_study_heatmap=0 \
                                    --plot_latent_clusters=1 \
                                    --plot_latent_reduction="TSNE" \

                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --preprocessing_data=0 \
                                    --combine_method="attention" \
                                    --results_csv="RQ_1_msds_casestudy.csv" \

                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            
        done
    done
}
# --- Run experiments ---
# 2. With AMOC
#run_experiment_msds_casestudy





seeds=(7)
window_size=(1)
coeff_architecture="deep_mlp"
dataset="msds"
main_model="aerca_based"
lrs="1e-6"
att_dim=256

run_experiment_deepmlp() {
    for window_size_item in "${window_size[@]}"; do
        for seed in "${seeds[@]}"; do
                    echo "Running: dataset=$dataset | seed=$seed | arch=$coeff_architecture | window_size=$window_size_item | lr=$lrs"

                    cmd="python3 main.py \
                        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                        --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                        --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \
                        --main_model=$main_model \
                        --lr=$lrs \
                        --seed=$seed \
                        --dataset=$dataset \
                        --plot_case_study=1 \
                        --coeff_architecture=$coeff_architecture \
                        --window_size=$window_size_item \
                        --results_csv="RQ_1_msds_casestudy.csv" \

                        --plot_case_study=0 \
                        --plot_case_study_heatmap=0 \
                        --plot_latent_clusters=1 \
                        --plot_latent_reduction="TSNE" \

                        --preprocessing_data=0 \
                        --training_aerca=1 \
                        --epochs=5000 \
                        --early_stopping=1 "

                    eval $cmd
        done
    done
}

#run_experiment_deepmlp






#-------------Lotka-Volterra Case Study -------------#


# --- Configurations ---
seeds=(1)
dataset=("lotka_volterra")
coeff_architecture="TemporalGNN_Attention_crossattn"
window_size=(1)
main_model=("aerca_based")
att_dim=64
heads=2
outer_heads=2
outer_hidden_dim=64

# --- Helper function to run experiments ---
run_experiment_Lotka_Volterra_CrGSTA() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                        for main_model_item in "${main_model[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model_item"

                                if [ "$seed" == "1" ]; then
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
                                    --num_vars=20 \
                                    --training_aerca=1 \

                                    --plot_case_study=0 \
                                    --plot_case_study_heatmap=0 \
                                    --plot_latent_clusters=1 \
                                    --plot_latent_reduction="TSNE" \

                                    --epochs=100 \
                                    --time_freq_representation=mag_phase \
                                    --attention_dim="$att_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads" \
                                    --outer_hidden_dim="$outer_hidden_dim" \

                                    --results_csv=RQ_1_lotka_volterra_new.csv"

                                eval $cmd
                        done
            
        done
    done
}
# --- Run experiments ---
run_experiment_Lotka_Volterra_CrGSTA