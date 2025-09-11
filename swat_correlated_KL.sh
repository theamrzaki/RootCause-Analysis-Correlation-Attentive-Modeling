# --- Configurations ---
seeds=(1 2 3 4 5 6)
coeff_architecture=("TemporalGNN_Attention" "TemporalGNN_Attention_fourier" "TemporalGNN_Attention_crossattn")
dataset=("swat")
lrs=("1e-4")
att_dim=256
heads=2
corelated_list=(0)
window_size=(10)
outer_heads=(2)
outer_hidden_dim=(256)

# --- Helper function to run experiments ---
run_experiment_RoGSTA_SWAT() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for window_size_item in "${window_size[@]}"; do
        for data in "${dataset[@]}"; do
            for arch in "${coeff_architecture[@]}"; do
                for lr in "${lrs[@]}"; do
                    for seed in "${seeds[@]}"; do
                        for outer_att_dim_val in "${outer_hidden_dim[@]}"; do
                            for outer_heads_val in "${outer_heads[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size_item | lr=$lr"

                                cmd="python3 main.py \
                                                --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                    --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                    --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \
                                    --lr=$lr \
                                    --seed=$seed \
                                    --dataset=$data \
                                    --coeff_architecture=$arch \
                                    --window_size=$window_size_item \
                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --attention_dim=$att_dim \
                                    --num_attention_heads=$heads \
                                    --outer_heads_num=$outer_heads_val \
                                    --outer_hidden_dim=$outer_att_dim_val \
                                    --results_csv=results_fouriers.csv"

                                eval $cmd
                            done
                        done
                    done
                done
            done
        done
    done
}
# --- Run experiments ---
# 2. With AMOC
run_experiment_RoGSTA_SWAT 1













# --- Configurations ---
seeds=(1 2 3 4 5 6)
coeff_architecture=("TemporalGNN_Attention" "TemporalGNN_Attention_fourier" "TemporalGNN_Attention_crossattn")
dataset=("msds")
lrs=("5e-4")
att_dim=128
heads=4
corelated_list=(0)
window_size=(10)
outer_heads=(4)
outer_hidden_dim=(256)

# --- Helper function to run experiments ---
run_experiment_RoGSTA_MSDS() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for window_size_item in "${window_size[@]}"; do
        for data in "${dataset[@]}"; do
            for arch in "${coeff_architecture[@]}"; do
                for lr in "${lrs[@]}"; do
                    for seed in "${seeds[@]}"; do
                        for outer_att_dim_val in "${outer_hidden_dim[@]}"; do
                            for outer_heads_val in "${outer_heads[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size_item | lr=$lr"

                                cmd="python3 main.py \
                                                --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                    --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                    --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \
                                    --lr=$lr \
                                    --seed=$seed \
                                    --dataset=$data \
                                    --coeff_architecture=$arch \
                                    --window_size=$window_size_item \
                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --attention_dim=$att_dim \
                                    --num_attention_heads=$heads \
                                    --outer_heads_num=$outer_heads_val \
                                    --outer_hidden_dim=$outer_att_dim_val \
                                    --results_csv=results_fouriers.csv"

                                eval $cmd
                            done
                        done
                    done
                done
            done
        done
    done
}
# --- Run experiments ---
# 2. With AMOC
#run_experiment_RoGSTA_MSDS 1