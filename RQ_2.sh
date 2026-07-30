source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

seeds=(1)
dataset="aiops"
window_size_item=20
num_vars=(100 150 200 250)
lrs=("1e-4")
results_csv="RQ_2_aiops_TSE_var.csv"

BETA_VAL=0.01
LAMBDA_VAL=0.5
GAMMA_VAL=0.2

#-------------------------------------------------------------------
#-----------------------------deep models-------------------------------
#-------------------------------------------------------------------



run_deep_models() {
    local preprocessing_data=$1
    local seed=$2
    local window_size_item=$3
    local var=$4
    local arch=$5
    local main_model="aerca_based"
    local hidden_layer_size=256

                                echo "Running:  arch=$arch | dataset=$dataset | seed=$seed | window_size=$window_size_item | num_vars=$var | main_model=$main_model | preprocessing_data=$preprocessing_data"


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
                                    --num_vars="$var" \

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
    local var=$4

    local coeff_architecture=("torai" "baro" "rcd")
    local main_model="aerca_based"

    for arch in "${coeff_architecture[@]}"; do
        echo "Running: arch=$arch | dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size_item | num_vars=$var |  preprocessing_data=$preprocessing_data"
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
            --num_vars=$var \
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
num_vars=(100)
for seed in "${seeds[@]}"; do
    for var in "${num_vars[@]}"; do
        #preprocessing_data=1 
        #run_deep_models $preprocessing_data $seed $window_size_item $var "vlinear"
        preprocessing_data=0
        run_deep_models $preprocessing_data $seed $window_size_item $var "GVAR"
        run_experiment_baselines $preprocessing_data $seed $window_size_item $var
        run_deep_models $preprocessing_data $seed $window_size_item $var "cLSTM"
        run_deep_models $preprocessing_data $seed $window_size_item $var "CUTS_PLUS"
    done
done


num_vars=(150 250)
for seed in "${seeds[@]}"; do
    for var in "${num_vars[@]}"; do
        preprocessing_data=1 
        run_deep_models $preprocessing_data $seed $window_size_item $var "vlinear"
        preprocessing_data=0
        run_deep_models $preprocessing_data $seed $window_size_item $var "GVAR"
        run_experiment_baselines $preprocessing_data $seed $window_size_item $var
        run_deep_models $preprocessing_data $seed $window_size_item $var "cLSTM"
        run_deep_models $preprocessing_data $seed $window_size_item $var "CUTS_PLUS"
    done
done



