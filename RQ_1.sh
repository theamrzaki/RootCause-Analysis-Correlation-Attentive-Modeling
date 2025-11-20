# --- Configurations ---
seeds=(1 2 3 4 5 6)
dataset=("swat")
lrs=("1e-4")
window_size=(5 1 7 10 12 15)
main_model=("FEDformer")
attention_dim=256
heads=2
# --- Helper function to run experiments ---
run_experiment_SWAT_Fedformer() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                        for main_model_item in "${main_model[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model_item"

                                cmd="python3 main.py \
                                                --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                    --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                    --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \
                                    --lr=$lrs \
                                    --main_model=$main_model_item \
                                    --attention_dim=$attention_dim \
                                    --num_attention_heads=$heads \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --window_size=$window_size_item \
                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --results_csv=RQ_1_swat.csv"

                                eval $cmd
                        done
            
        done
    done
}
# --- Run experiments ---
# 2. With AMOC
#run_experiment_SWAT_Fedformer 1



# --- Configurations ---
seeds=(1 2 3 4 5 6)
dataset=("swat")
lrs=("1e-4")
window_size=(5 1 7 10 12)
main_model=("iTransformer")
attention_dim=256
heads=2
# --- Helper function to run experiments ---
run_experiment_SWAT_iTransformer() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                        for main_model_item in "${main_model[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model_item"

                                cmd="python3 main.py \
                                                --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                    --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                    --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \
                                    --lr=$lrs \
                                    --main_model=$main_model_item \
                                    --attention_dim=$attention_dim \
                                    --num_attention_heads=$heads \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --window_size=$window_size_item \
                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --results_csv=RQ_1_swat.csv"

                                eval $cmd
                        done
            
        done
    done
}
# --- Run experiments ---
# 2. With AMOC
#run_experiment_SWAT_iTransformer 1



# --- Configurations ---
seeds=(1 2 3 4 5 6)
dataset=("swat")
lrs=("1e-4")
window_size=(5 1 7 10 12)
arch="TemporalGNN_Attention_crossattn"
main_model="aerca_based"
attention_dim=256
heads=1
outer_att_dim_val=256
outer_heads_val=1
# --- Helper function to run experiments ---
run_experiment_SWAT_CrGSTA() {
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

                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --preprocessing_data=0 \
                                    --combine_method="attention" \
                                    --results_csv="RQ_1_swat.csv" \

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
run_experiment_SWAT_CrGSTA 1


seeds=(7 8 9 10 11 12)
window_size=(1 2 3 4 5)
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
                        --coeff_architecture=$coeff_architecture \
                        --window_size=$window_size_item \
                        --preprocessing_data=0 \
                        --training_aerca=1 \
                        --epochs=5000 \
                        --early_stopping=1 \
                        --results_csv=RQ_1_swat.csv"

                    eval $cmd
        done
    done
}

#run_experiment_deepmlp





# --- Configurations ---
seeds=(1 2 3 4 5 6)
dataset=("swat")
lrs=("1e-4")
window_size=(5 1 7 10 12)
arch="GVAR"
main_model="aerca_based"
attention_dim=256
heads=2
outer_att_dim_val=256
outer_heads_val=2
# --- Helper function to run experiments ---
run_experiment_SWAT_GVAR() {
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

                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --preprocessing_data=0 \
                                    --combine_method="attention" \
                                    --results_csv="RQ_1_swat.csv" \

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
#run_experiment_SWAT_GVAR 1






# --- Configurations ---
seeds=(1 2 3 4 5 6)
dataset=("swat")
lrs=("1e-4")
window_size=(5 1 7 10 12)
arch="causalrca"
main_model="aerca_based"
attention_dim=256
heads=2
outer_att_dim_val=256
outer_heads_val=2
# --- Helper function to run experiments ---
run_experiment_SWAT_causalrca() {
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

                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --preprocessing_data=0 \
                                    --combine_method="attention" \
                                    --results_csv="RQ_1_swat.csv" \

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
#run_experiment_SWAT_causalrca 1

#-------------------------------------------------------------------
#------------------------Lotka Volterra-----------------------------
#-------------------------------------------------------------------


seeds=(1 2 3)
coeff_architecture=("rcd" "epsilon_diagnosis")
data=("lotka_volterra")
lrs=("1e-4")
att_dim=256
heads=2
corelated_list=(0)
window_size=(1 5 7 10 12)
outer_heads=(4)
outer_hidden_dim=(256)
main_model=("aerca_based")
# --- Helper function to run experiments ---
run_experiment_baselines() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                for arch in "${coeff_architecture[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size_item"
                                if [ "$seed" == "1" ]; then
                                    preprocessing_data=1
                                else
                                    preprocessing_data=0
                                fi

                                cmd="python3 main.py \
                                    --seed=$seed \
                                    --dataset=$data \
                                    --coeff_architecture=$arch \
                                    --preprocessing_data=$preprocessing_data \
                                    --window_size=$window_size_item \
                                    --main_model=$main_model \
                                    --training_aerca=0 \                                    
                                    --results_csv=RQ_1_lotka_volterra_new.csv"

                                eval $cmd
            done
        done
    done
}
#run_experiment_baselines










# --- Configurations ---
seeds=(1 2 3)
dataset=("lotka_volterra")
coeff_architecture="deep_mlp"
window_size=(1 5 7 10 12)
main_model=("aerca_based")

# --- Helper function to run experiments ---
run_experiment_Lotka_Volterra_deep_mlp() {
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
                                    --main_model=$main_model \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --preprocessing_data=$preprocessing_data \
                                    --window_size=$window_size_item \
                                    --attention_dim=$attention_dim \
                                    --num_attention_heads=$heads \
                                    --num_vars=40 \
                                    --training_aerca=1 \

                                    --early_stopping=1 \
                                    --results_csv=RQ_1_lotka_volterra_new.csv"

                                eval $cmd
                        done
            
        done
    done
}
# --- Run experiments ---
# 2. With AMOC
#run_experiment_Lotka_Volterra_deep_mlp 1





# --- Configurations ---
seeds=(2 3)
dataset=("lotka_volterra")
coeff_architecture="TemporalGNN_Attention_crossattn"
window_size=(12 1 5 7 10)
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
                                    --num_vars=40 \
                                    --training_aerca=1 \

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
#run_experiment_Lotka_Volterra_CrGSTA





# --- Configurations ---
seeds=(1 2 3)
dataset=("lotka_volterra")
lrs=("1e-4")
window_size=(1 5 7 10 12)
main_model=("FEDformer")
attention_dim=64
heads=2
# --- Helper function to run experiments ---
run_experiment_Lotka_Volterra_Fedformer() {
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
                                    --main_model=$main_model \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --preprocessing_data=$preprocessing_data \
                                    --window_size=$window_size_item \
                                    --attention_dim=$attention_dim \
                                    --num_attention_heads=$heads \
                                    --num_vars=40 \
                                    --training_aerca=1 \
                                    --epochs=100 \
                                    --results_csv=RQ_1_lotka_volterra_new.csv"

                                eval $cmd
                        done
            
        done
    done
}
# --- Run experiments ---
# 2. With AMOC
#run_experiment_Lotka_Volterra_Fedformer 1




# --- Configurations ---
seeds=(1 2 3)
dataset=("lotka_volterra")
lrs=("1e-4")
window_size=(1 5 7 10 12)
main_model=("iTransformer")
attention_dim=64
heads=2
# --- Helper function to run experiments ---
run_experiment_Lotka_Volterra_Fedformer() {
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
                                    --main_model=$main_model \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --preprocessing_data=$preprocessing_data \
                                    --window_size=$window_size_item \
                                    --attention_dim=$attention_dim \
                                    --num_attention_heads=$heads \
                                    --num_vars=40 \
                                    --training_aerca=1 \
                                    --epochs=100 \
                                    --results_csv=RQ_1_lotka_volterra_new.csv"

                                eval $cmd
                        done
            
        done
    done
}
# --- Run experiments ---
# 2. With AMOC
#run_experiment_Lotka_Volterra_Fedformer 1




# --- Configurations ---
seeds=(1 2 3)
dataset=("lotka_volterra")
lrs=("1e-4")
window_size=(1 5 7 10 12)
coeff_architecture="GVAR"
main_model=("aerca_based")
attention_dim=64
heads=2
# --- Helper function to run experiments ---
run_experiment_Lotka_Volterra_GVAR() {
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
                                    --main_model=$main_model \
                                    --coeff_architecture="$coeff_architecture" \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --preprocessing_data=$preprocessing_data \
                                    --window_size=$window_size_item \
                                    --attention_dim=$attention_dim \
                                    --num_attention_heads=$heads \
                                    --num_vars=40 \
                                    --training_aerca=1 \
                                    --epochs=100 \
                                    --results_csv=RQ_1_lotka_volterra_new.csv"

                                eval $cmd
                        done
            
        done
    done
}
# --- Run experiments ---
# 2. With AMOC
#run_experiment_Lotka_Volterra_GVAR 1





# --- Configurations ---
seeds=(1 2 3)
dataset=("lotka_volterra")
lrs=("1e-4")
window_size=(1 5 7 10 12)
coeff_architecture="causalrca"
main_model=("aerca_based")
attention_dim=256
heads=2
# --- Helper function to run experiments ---
run_experiment_Lotka_Volterra_causalrca() {
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
                                    --main_model=$main_model \
                                    --coeff_architecture="$coeff_architecture" \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --preprocessing_data=$preprocessing_data \
                                    --window_size=$window_size_item \
                                    --attention_dim=$attention_dim \
                                    --num_attention_heads=$heads \
                                    --num_vars=40 \
                                    --training_aerca=1 \
                                    --epochs=100 \
                                    --results_csv=RQ_1_lotka_volterra_new.csv"

                                eval $cmd
                        done
            
        done
    done
}
# --- Run experiments ---
# 2. With AMOC
#run_experiment_Lotka_Volterra_causalrca 1