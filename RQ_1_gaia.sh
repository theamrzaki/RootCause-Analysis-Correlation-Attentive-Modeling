source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

seeds=(2)
dataset="gaia"
window_size=(8 10 12 14)
lrs=("1e-4")
results_csv="RQ_1_gaia_final.csv"

BETA_VAL=0.01
LAMBDA_VAL=0.5
GAMMA_VAL=0.2


#-------------------------------------------------------------------
#-----------------------------deep models---------------------------
#-------------------------------------------------------------------



run_deep_models() {
    local preprocessing_data=$1
    local seed=$2
    local window_size_item=$3
    local arch=$4
    local main_model="aerca_based"
    local hidden_layer_size=256

                                echo "Running:  arch=$arch | dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model | preprocessing_data=$preprocessing_data"


                                cmd="python3 main.py \
                                        --encoder_gamma=$GAMMA_VAL --decoder_gamma=$GAMMA_VAL \
                                        --encoder_lambda=$LAMBDA_VAL --decoder_lambda=$LAMBDA_VAL --beta=$BETA_VAL \

                                    --main_model=$main_model \
                                    --coeff_architecture="$arch" \
                                    --temporal_mixer=0 \

                                    --preprocessing_data="$preprocessing_data" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --training_aerca=1 \
                                    --epochs=200 \
                                    --results_csv="$results_csv" \

                                    --hidden_layer_size="$hidden_layer_size" \
                                "

                                eval $cmd
                        
            
}

#-------------------------------------------------------------------
#-----------------------------experiment_baselines------------------
#-------------------------------------------------------------------	


run_experiment_baselines() {
    local preprocessing_data=$1
    local seed=$2
    local window_size_item=$3

    local coeff_architecture=("torai" "baro" "rcd")
    local main_model="aerca_based"

    for arch in "${coeff_architecture[@]}"; do
        echo "Running: arch=$arch | dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size_item |  preprocessing_data=$preprocessing_data"
        # only run preprocessing for the first coeff_architecture to avoid redundant preprocessing
        if [ $arch == "torai" ]; then
            preprocessing_flag=$preprocessing_data
        else
            preprocessing_flag=0
        fi
        cmd="python3 main.py \
            --seed=$seed \
            --dataset=$dataset \
            --coeff_architecture=$arch \
            --preprocessing_data=$preprocessing_flag \
            --window_size=$window_size_item \
            --main_model=$main_model \
            --training_aerca=0 \
            --results_csv=$results_csv" 

        eval $cmd
    done
}




#---------------------------------------------------------------------------
#-----------------------------run experiments-------------------------------
#---------------------------------------------------------------------------
seeds=(3)
window_size=(8 10 12 14)
for seed in "${seeds[@]}"; do
    for window_size_item in "${window_size[@]}"; do
        preprocessing_data=1
        run_deep_models $preprocessing_data $seed $window_size_item "vlinear"
        preprocessing_data=0
    done
done



seeds=(1 3)
window_size=(8 10 12 14)

for seed in "${seeds[@]}"; do
    for window_size_item in "${window_size[@]}"; do
        preprocessing_data=1
        if [ $seed -eq 1 ]; then
            # Seed 1 missing
            if [ $window_size_item -eq 8 ] || [ $window_size_item -eq 10 ]; then
                run_deep_models $preprocessing_data $seed $window_size_item "GVAR"
                run_deep_models $preprocessing_data $seed $window_size_item "cLSTM"
            fi
        elif [ $seed -eq 3 ]; then
            # Seed 3 missing
            run_deep_models $preprocessing_data $seed $window_size_item "vlinear"
        fi
    done
done