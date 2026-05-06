source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

seeds=(1)
dataset="aiops"
window_size=(6 10 20 30)
lrs=("1e-4")

BETA_VAL=0.01
LAMBDA_VAL=0.5
GAMMA_VAL=0.2

#-------------------------------------------------------------------
#-----------------------------vlinear-------------------------------
#-------------------------------------------------------------------



run_vlinear() {
    local preprocessing_data=$1
    local seed=$2
    local window_size_item=$3

    local arch="vlinear"
    local main_model="aerca_based"
    local attention_dim=16
    local heads=2
    local outer_att_dim_val=16
    local outer_heads_val=2

                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model"


                                cmd="python3 main.py \
                                        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                        --encoder_gamma=$GAMMA_VAL --decoder_gamma=$GAMMA_VAL \
                                        --encoder_lambda=$LAMBDA_VAL --decoder_lambda=$LAMBDA_VAL --beta=$BETA_VAL \

                                    --main_model=$main_model \
                                    --coeff_architecture="$arch" \
                                    --time_freq_representation="mag_phase" \

                                    --preprocessing_data="$preprocessing_data" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --training_aerca=1 \
                                    --epochs=200 \
                                    --early_stopping=0 \
                                    --combine_method="attention" \
                                    --results_csv="RQ_1_aiops_allmodels.csv" \

                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            
}



#-------------------------------------------------------------------
#-----------------------------GVAR----------------------------------
#-------------------------------------------------------------------



run_GVAR() {
    local preprocessing_data=$1
    local seed=$2
    local window_size_item=$3

    local lrs=("1e-4")
    local arch="GVAR"
    local main_model="aerca_based"
    local attention_dim=256
    local heads=2
    local outer_att_dim_val=256
    local outer_heads_val=2

                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model"

                                cmd="python3 main.py \
                                        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                        --encoder_gamma=$GAMMA_VAL --decoder_gamma=$GAMMA_VAL \
                                        --encoder_lambda=$LAMBDA_VAL --decoder_lambda=$LAMBDA_VAL --beta=$BETA_VAL \

                                    --main_model=$main_model \
                                    --coeff_architecture="$arch" \
                                    --time_freq_representation="mag_phase" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --training_aerca=1 \
                                    --epochs=200 \
                                    --early_stopping=0 \
                                    --preprocessing_data=$preprocessing_data \
                                    --combine_method="attention" \
                                    --results_csv="RQ_1_aiops_allmodels.csv" \

                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            
}


#-------------------------------------------------------------------
#-----------------------------GVAR----------------------------------
#-------------------------------------------------------------------



run_cLSTM() {
    local preprocessing_data=$1
    local seed=$2
    local window_size_item=$3

    local lrs=("1e-4")
    local arch="cLSTM"
    local main_model="aerca_based"
    local attention_dim=256
    local heads=2
    local outer_att_dim_val=256
    local outer_heads_val=2

                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model"

                                cmd="python3 main.py \
                                        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                        --encoder_gamma=$GAMMA_VAL --decoder_gamma=$GAMMA_VAL \
                                        --encoder_lambda=$LAMBDA_VAL --decoder_lambda=$LAMBDA_VAL --beta=$BETA_VAL \

                                    --main_model=$main_model \
                                    --coeff_architecture="$arch" \
                                    --time_freq_representation="mag_phase" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --training_aerca=1 \
                                    --epochs=200 \
                                    --early_stopping=0 \
                                    --preprocessing_data=$preprocessing_data \
                                    --combine_method="attention" \
                                    --results_csv="RQ_1_aiops_allmodels.csv" \

                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            
}

#-------------------------------------------------------------------
#-----------------------------causalrca-----------------------------
#-------------------------------------------------------------------



run_causalrca() {

    local preprocessing_data=$1
    local seed=$2
    local window_size_item=$3

    local arch="causalrca"
    local main_model="aerca_based"
    local attention_dim=256
    local heads=2
    local outer_att_dim_val=256
    local outer_heads_val=2

                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model"

                                cmd="python3 main.py \
                                        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                        --encoder_gamma=$GAMMA_VAL --decoder_gamma=$GAMMA_VAL \
                                        --encoder_lambda=$LAMBDA_VAL --decoder_lambda=$LAMBDA_VAL --beta=$BETA_VAL \

                                    --main_model=$main_model \
                                    --coeff_architecture="$arch" \
                                    --time_freq_representation="mag_phase" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --training_aerca=1 \
                                    --epochs=200 \
                                    --early_stopping=0 \
                                    --preprocessing_data=$preprocessing_data \
                                    --combine_method="attention" \
                                    --results_csv="RQ_1_aiops_allmodels.csv" \

                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            
}

#-------------------------------------------------------------------
#-----------------------------rcd & epsilon diagnosis---------------
#-------------------------------------------------------------------



run_experiment_baselines() {
    local preprocessing_data=$1
    local seed=$2
    local window_size_item=$3

    local coeff_architecture=("torai")
    local main_model="aerca_based" # Changed from array to string

    for arch in "${coeff_architecture[@]}"; do
        echo "Running: dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size_item"

        cmd="python3 main.py \
            --seed=$seed \
            --dataset=$dataset \
            --coeff_architecture=$arch \
            --preprocessing_data=$preprocessing_data \
            --window_size=$window_size_item \
            --main_model=$main_model \
            --training_aerca=0 \
            --results_csv=RQ_1_aiops_allmodels.csv" # Removed the comment/backslash error here

        eval $cmd
    done
}



#-------------------------------------------------------------------
#-----------------------------GAIA experiments-------------------------
#-------------------------------------------------------------------

for window_size_item in "${window_size[@]}"; do
    for seed in "${seeds[@]}"; do
        if [ $seed -eq 1 ]; then
            preprocessing_data=1
        else
            preprocessing_data=0
        fi
        #run_vlinear $preprocessing_data $seed $window_size_item
        run_experiment_baselines $preprocessing_data $seed $window_size_item
        if [ $preprocessing_data -eq 1 ]; then
            preprocessing_data=0
        fi
        #run_GVAR $preprocessing_data $seed $window_size_item
        #
        #run_cLSTM $preprocessing_data $seed $window_size_item
    done
done