# --- Configurations ---
seeds=(1 2 3)
coeff_architecture=("TemporalGNN_Attention_crossattn")
dataset=("lotka_volterra")
lrs=("1e-4")
att_dim=32
heads=2
window_size=7       # fixed
num_vars=40         # fixed

# The parameter to sweep (example: alpha_lv)
sweep_param_name="alpha_lv"
sweep_param_values=(0.1 0.2 0.3 0.4 0.5)

# --- Helper function to run experiments ---
run_experiment_CrGSTA_sweep() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for seed in "${seeds[@]}"; do
        for sweep_val in "${sweep_param_values[@]}"; do
            for arch in "${coeff_architecture[@]}"; do
                for data in "${dataset[@]}"; do

                    # Preprocessing only for first time deep_mlp
                    if [ "$arch" == "deep_mlp" ]; then
                        preprocessing_data=1
                    else
                        preprocessing_data=0
                    fi

                    echo "Running: dataset=$data | seed=$seed | arch=$arch | sweep_param=$sweep_param_name | value=$sweep_val"

                    cmd="python3 main.py \
                        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                        --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                        --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \
                        --lr=${lrs[0]} \
                        --preprocessing_data=$preprocessing_data \
                        --seed=$seed \
                        --dataset=$data \
                        --coeff_architecture=$arch \
                        --window_size=$window_size \
                        --training_aerca=1 \
                        --epochs=100 \
                        --early_stopping=0 \
                        --attention_dim=$att_dim \
                        --num_attention_heads=$heads \
                        --outer_heads_num=2 \
                        --outer_hidden_dim=32 \
                        --num_vars=$num_vars \
                        --$sweep_param_name=$sweep_val \
                        --results_csv=results_lorenz_sweep_alpha_lv.csv"

                    eval $cmd
                done
            done
        done
    done
}

# --- Run experiments with AMOC ---
run_experiment_CrGSTA_sweep 1
