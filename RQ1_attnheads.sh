


# --- Configurations ---
seeds=(1 2 3)
coeff_architecture=("TemporalGNN_Attention")
dataset=("swat")
lrs=("1e-4")
att_dim=(128 256)
heads=(1 2 4)
corelated_list=(0)
window_size=(1)
outer_heads=4
outer_hidden_dim=256

# --- Helper function to run experiments ---
run_RQ1_attnheads_SWAT() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for window_size_item in "${window_size[@]}"; do
        for data in "${dataset[@]}"; do
            for arch in "${coeff_architecture[@]}"; do
                for lr in "${lrs[@]}"; do
                    for seed in "${seeds[@]}"; do
                        for att_dim_val in "${att_dim[@]}"; do
                            for heads_val in "${heads[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size_item | lr=$lr"

                                cmd="python3 main.py \
                                    --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                    --lr=$lr \
                                    --seed=$seed \
                                    --dataset=$data \
                                    --coeff_architecture=$arch \
                                    --window_size=$window_size_item \
                                    --training_aerca=1 \
                                    --epochs=100 \
                                    --early_stopping=0 \
                                    --attention_dim=$att_dim_val \
                                    --num_attention_heads=$heads_val \
                                    --outer_heads_num=$outer_heads \
                                    --outer_hidden_dim=$outer_hidden_dim \
                                    --results_csv=results_RQ1_swat_attnheads.csv"

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
run_RQ1_attnheads_SWAT 1







# --- Configurations ---
seeds=(1 2 3)
coeff_architecture=("TemporalGNN_Attention")
dataset=("msds")
lrs=("1e-4")
att_dim=(128 256)
heads=(1 2 4)
corelated_list=(0)
window_size=(1)
outer_heads=4
outer_hidden_dim=256

# --- Helper function to run experiments ---
run_RQ1_attnheads_msds() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for window_size_item in "${window_size[@]}"; do
        for data in "${dataset[@]}"; do
            for arch in "${coeff_architecture[@]}"; do
                for lr in "${lrs[@]}"; do
                    for seed in "${seeds[@]}"; do
                        for att_dim_val in "${att_dim[@]}"; do
                            for heads_val in "${heads[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size_item | lr=$lr"

                                cmd="python3 main.py \
                                    --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                    --lr=$lr \
                                    --seed=$seed \
                                    --dataset=$data \
                                    --coeff_architecture=$arch \
                                    --window_size=$window_size_item \
                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --attention_dim=$att_dim_val \
                                    --num_attention_heads=$heads_val \
                                    --outer_heads_num=$outer_heads \
                                    --outer_hidden_dim=$outer_hidden_dim \
                                    --results_csv=results_RQ1_msds_attnheads.csv"

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
run_RQ1_attnheads_msds 1