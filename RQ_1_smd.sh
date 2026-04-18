seeds=(1)
dataset="smd"
window_size=(10)
lrs=("1e-4")
#to run this script, first use chmod +x RQ_1_smd.sh

#-------------------------------------------------------------------
#-----------------------------vlinear-------------------------------
#-------------------------------------------------------------------

arch="vlinear"
main_model="aerca_based"
attention_dim=256
heads=1
outer_att_dim_val=256
outer_heads_val=1

run_SMD_CrGSTA() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model"

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
                                    --coeff_architecture="$arch" \
                                    --time_freq_representation="mag_phase" \

                                    --preprocessing_data="$preprocessing_data" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --preprocessing_data=0 \
                                    --combine_method="attention" \
                                    --results_csv="RQ_1_smd.csv" \

                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            
        done
    done
}


#run_SMD_CrGSTA 1


#-------------------------------------------------------------------
#------------------------------Deep MLP-----------------------------
#-------------------------------------------------------------------
coeff_architecture="deep_mlp"
main_model="aerca_based"
att_dim=256

run_deepmlp() {
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
                        --epochs=1000 \
                        --early_stopping=1 \
                        --results_csv=RQ_1_smd.csv"

                    eval $cmd
        done
    done
}
#run_deepmlp



#-------------------------------------------------------------------
#-----------------------------Fedformer-----------------------------
#-------------------------------------------------------------------

lrs=("1e-4")
main_model=("FEDformer")
attention_dim=256
heads=2

run_SMD_Fedformer() {
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
                                    --preprocessing_data=0 \
                                    --window_size=$window_size_item \
                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --results_csv=RQ_1_smd.csv"

                                eval $cmd
                        done
            
        done
    done
}
#run_SMD_Fedformer 1


#-------------------------------------------------------------------
#-----------------------------iTransformer--------------------------
#-------------------------------------------------------------------

lrs=("1e-4")
main_model=("iTransformer")
attention_dim=256
heads=2

run_SMD_iTransformer() {
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
                                    --preprocessing_data=0 \
                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --results_csv=RQ_1_smd.csv"

                                eval $cmd
                        done
            
        done
    done
}


#run_SMD_iTransformer 1







#-------------------------------------------------------------------
#-----------------------------GVAR----------------------------------
#-------------------------------------------------------------------

lrs=("1e-4")
arch="GVAR"
main_model="aerca_based"
attention_dim=256
heads=2
outer_att_dim_val=256
outer_heads_val=2

run_SMD_GVAR() {
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
                                    --results_csv="RQ_1_smd.csv" \

                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            
        done
    done
}
run_SMD_GVAR 1



seeds=(1)
dataset="wadi"
window_size=(5 1 7 10)
lrs=("1e-4")
arch="GVAR"
main_model="aerca_based"
attention_dim=256
heads=2
outer_att_dim_val=256
outer_heads_val=2

run_WADI_GVAR() {
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
                                    --results_csv="RQ_1_wadi.csv" \

                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            
        done
    done
}
run_WADI_GVAR 1



seeds=(1)
dataset="swat"
window_size=(5 1 7 10)
lrs=("1e-4")
arch="GVAR"
main_model="aerca_based"
attention_dim=256
heads=2
outer_att_dim_val=256
outer_heads_val=2

run_swat_GVAR() {
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
run_swat_GVAR 1



#-------------------------------------------------------------------
#-----------------------------causalrca-----------------------------
#-------------------------------------------------------------------

arch="causalrca"
main_model="aerca_based"
attention_dim=256
heads=2
outer_att_dim_val=256
outer_heads_val=2

run_SMD_causalrca() {
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
                                    --results_csv="RQ_1_smd.csv" \

                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            
        done
    done
}

#run_SMD_causalrca 1

#-------------------------------------------------------------------
#-----------------------------rcd & epsilon diagnosis---------------
#-------------------------------------------------------------------


coeff_architecture=("rcd" "epsilon_diagnosis")
att_dim=256
heads=2
corelated_list=(0)
outer_heads=(4)
outer_hidden_dim=(256)
main_model=("aerca_based")

run_experiment_baselines() {
    local use_amoc=$1  # 0 = no AMOC, 1 = AMOC
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                for arch in "${coeff_architecture[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size_item"
                                

                                cmd="python3 main.py \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --coeff_architecture=$arch \
                                    --preprocessing_data=0 \
                                    --window_size=$window_size_item \
                                    --main_model=$main_model \
                                    --training_aerca=0 \                                    
                                    --results_csv=RQ_1_smd.csv"

                                eval $cmd
            done
        done
    done
}
run_experiment_baselines


