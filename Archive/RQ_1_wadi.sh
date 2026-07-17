source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

seeds=(1)
dataset="wadi"
window_size=(1 5 7 10 12)
lrs=("1e-4")

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
                                    --combine_method="attention" \
                                    --results_csv="RQ_1_wadi_window_correct.csv" \

                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            
}


#-------------------------------------------------------------------
#------------------------------Deep MLP-----------------------------
#-------------------------------------------------------------------


run_deepmlp() {
    local preprocessing_data=$1
    local seed=$2
    local window_size_item=$3

    local coeff_architecture="deep_mlp"
    local main_model="aerca_based"
    local att_dim=256
    
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
                        --preprocessing_data=$preprocessing_data \
                        --training_aerca=1 \
                        --epochs=1000 \
                        --early_stopping=1 \
                        --results_csv=RQ_1_wadi_window_correct.csv"

                    eval $cmd
}



#-------------------------------------------------------------------
#-----------------------------Fedformer-----------------------------
#-------------------------------------------------------------------


run_Fedformer() {
    local preprocessing_data=$1
    local seed=$2
    local window_size_item=$3

    local lrs=("1e-4")
    local main_model=("FEDformer")
    local attention_dim=256
    local heads=2

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
                                    --preprocessing_data=$preprocessing_data \
                                    --window_size=$window_size_item \
                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --results_csv=RQ_1_wadi_window_correct.csv"

                                eval $cmd
                        done
            
}


#-------------------------------------------------------------------
#-----------------------------iTransformer--------------------------
#-------------------------------------------------------------------


run_iTransformer() {
    local preprocessing_data=$1
    local seed=$2
    local window_size_item=$3

    local lrs=("1e-4")
    local main_model=("iTransformer")
    local attention_dim=256
    local heads=2

                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model"

                                cmd="python3 main.py \
                                                --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                    --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                    --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \
                                    --lr=$lrs \
                                    --main_model=$main_model \
                                    --attention_dim=$attention_dim \
                                    --num_attention_heads=$heads \
                                    --seed=$seed \
                                    --dataset=$dataset \
                                    --window_size=$window_size_item \
                                    --preprocessing_data=$preprocessing_data \
                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --results_csv=RQ_1_wadi_window_correct.csv"

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
                                    --preprocessing_data=$preprocessing_data \
                                    --combine_method="attention" \
                                    --results_csv="RQ_1_wadi_window_correct.csv" \

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
                                    --preprocessing_data=$preprocessing_data \
                                    --combine_method="attention" \
                                    --results_csv="RQ_1_wadi_window_correct.csv" \

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

    local coeff_architecture=("rcd" "epsilon_diagnosis")
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
            --results_csv=RQ_1_wadi_window_correct.csv" # Removed the comment/backslash error here

        eval $cmd
    done
}



#-------------------------------------------------------------------
#-----------------------------SWAT experiments-------------------------
#-------------------------------------------------------------------

for window_size_item in "${window_size[@]}"; do
    for seed in "${seeds[@]}"; do
        if [ $seed -eq 1 ]; then
            preprocessing_data=1
        else
            preprocessing_data=0
        fi
        run_vlinear $preprocessing_data $seed $window_size_item
        if [ $preprocessing_data -eq 1 ]; then
            preprocessing_data=0
        fi
        run_deepmlp $preprocessing_data $seed $window_size_item
        run_Fedformer $preprocessing_data $seed $window_size_item
        run_iTransformer $preprocessing_data $seed $window_size_item
        run_GVAR $preprocessing_data $seed $window_size_item
        run_causalrca $preprocessing_data $seed $window_size_item
        run_experiment_baselines $preprocessing_data $seed $window_size_item
    done
done