# --- Configurations ---
seeds=(1 2)

# Architectures
main_architecture="TemporalGNN_Attention_fourier"
baseline_architecture="TemporalGNN_Attention"
main_model=("aerca_based")

# Common settings
dataset="lotka_volterra"
lr="1e-4"
att_dim=64
heads=2
window_sizes=(7)
outer_heads=2
outer_hidden_dim=64

# Ablation settings
time_freq_representations=("normal" "mag_phase")
combine_methods=("freq_only" "gated" "sum" "concat")

# --- Helper function to run a single experiment ---
run_single() {
    local arch=$1
    local seed=$2
    local ws=$3
    local tf=$4
    local combine=$5

    echo "Running: arch=$arch | dataset=$dataset | seed=$seed | ws=$ws | tf=$tf | combine=$combine"

     Preprocessing only for first time TemporalGNN_Attention
    if [ "$arch" == "TemporalGNN_Attention" ]; then
        preprocessing_data=1
    else
        preprocessing_data=0
    fi
    
    python3 main.py \
        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
        --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
        --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \
        --coeff_architecture="$arch" \
        --preprocessing_data=0 \
        --main_model=$main_model \
        ${tf:+--time_freq_representation="$tf"} \
        ${combine:+--combine_method="$combine"} \
        --lr="$lr" \
        --seed="$seed" \
        --dataset="$dataset" \
        --window_size="$ws" \
        --training_aerca=1 \
        --epochs=100 \
        --early_stopping=0 \
        --attention_dim="$att_dim" \
        --num_attention_heads="$heads" \
        --outer_heads_num="$outer_heads" \
        --outer_hidden_dim="$outer_hidden_dim" \
        --num_vars=40 \
        --results_csv=RQ_3_lotka.csv
}

# --- Run experiments ---
for seed in "${seeds[@]}"; do
    for ws in "${window_sizes[@]}"; do
        
        # --- Baseline (no ablations, no tf/combine flags) ---
        run_single "$baseline_architecture" "$seed" "$ws"

        # --- Main architecture with ablations ---
        for tf in "${time_freq_representations[@]}"; do
            for combine in "${combine_methods[@]}"; do
                run_single "$main_architecture" "$seed" "$ws" "$tf" "$combine"
            done
        done
    done
done









# --- Configurations ---
seeds=(1 2 3 4 5 6)

# Architectures
main_architecture="TemporalGNN_Attention_crossattn"

# Common settings
dataset="lotka_volterra"
lr="1e-4"
att_dim=32
heads=2
window_sizes=(7)
outer_heads=2
outer_hidden_dim=32

# Ablation settings
time_freq_representations=("normal" "mag_phase")

# --- Helper function to run a single experiment ---
run_single() {
    local arch=$1
    local seed=$2
    local ws=$3
    local tf=$4

    echo "Running: arch=$arch | dataset=$dataset | seed=$seed | ws=$ws | tf=$tf"
    
    # Preprocessing already done for first time TemporalGNN_Attention

    python3 main.py \
        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
        --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
        --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \
        --coeff_architecture="$arch" \
        --preprocessing_data=0 \
        ${tf:+--time_freq_representation="$tf"} \
        --lr="$lr" \
        --seed="$seed" \
        --dataset="$dataset" \
        --window_size="$ws" \
        --training_aerca=1 \
        --epochs=100 \
        --early_stopping=0 \
        --attention_dim="$att_dim" \
        --num_attention_heads="$heads" \
        --outer_heads_num="$outer_heads" \
        --outer_hidden_dim="$outer_hidden_dim" \
        --num_vars=40 \
        --results_csv=RQ_3_results_all_attention_lotka.csv
}

# --- Run experiments ---
#for seed in "${seeds[@]}"; do
#    for ws in "${window_sizes[@]}"; do
#        # --- Main architecture with ablations ---
#        for tf in "${time_freq_representations[@]}"; do
#            run_single "$main_architecture" "$seed" "$ws" "$tf"
#        done
#    done
#done
