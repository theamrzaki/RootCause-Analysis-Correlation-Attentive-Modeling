source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

seeds=(2)
dataset="gaia"
window_size=(8 10 12 14)
lrs=("1e-4")
results_csv="RQ_4_gaia_ablations.csv"

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

                                    --disable_orth_proj="$disable_orth_proj" \
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


seeds=(1)
window_size=(12)

for seed in "${seeds[@]}"; do
    for window_size_item in "${window_size[@]}"; do
        preprocessing_data=1 
        disable_orth_proj=1
        echo "Running no orthogonal projection: preprocessing_data=$preprocessing_data, disable_orth_proj=$disable_orth_proj"
        ###run_deep_models $preprocessing_data $seed $window_size_item "vlinear" $disable_orth_proj


        #no KL
        preprocessing_data=0
        BETA_VAL=0 #KL
        LAMBDA_VAL=0.5 #sparse
        GAMMA_VAL=0.2 #smooth
        disable_orth_proj=0 
        echo "Running no KL: BETA_VAL=$BETA_VAL, LAMBDA_VAL=$LAMBDA_VAL, GAMMA_VAL=$GAMMA_VAL, disable_orth_proj=$disable_orth_proj"
        run_deep_models $preprocessing_data $seed $window_size_item "vlinear" $disable_orth_proj


        #no sparse
        preprocessing_data=0
        BETA_VAL=0.01 #KL
        LAMBDA_VAL=0 #sparse
        GAMMA_VAL=0.2 #smooth
        disable_orth_proj=0 
        echo "Running no sparse: BETA_VAL=$BETA_VAL, LAMBDA_VAL=$LAMBDA_VAL, GAMMA_VAL=$GAMMA_VAL, disable_orth_proj=$disable_orth_proj"
        run_deep_models $preprocessing_data $seed $window_size_item "vlinear" $disable_orth_proj



        #no smooth
        preprocessing_data=0
        BETA_VAL=0.01 #KL
        LAMBDA_VAL=0.5 #sparse
        GAMMA_VAL=0 #smooth
        disable_orth_proj=0 
        echo "Running no smooth: BETA_VAL=$BETA_VAL, LAMBDA_VAL=$LAMBDA_VAL, GAMMA_VAL=$GAMMA_VAL, disable_orth_proj=$disable_orth_proj"
        run_deep_models $preprocessing_data $seed $window_size_item "vlinear" $disable_orth_proj



        #normal 
        preprocessing_data=0
        BETA_VAL=0.01 #KL
        LAMBDA_VAL=0.5 #sparse
        GAMMA_VAL=0.2 #smooth
        disable_orth_proj=0 
        echo "Running normal: BETA_VAL=$BETA_VAL, LAMBDA_VAL=$LAMBDA_VAL, GAMMA_VAL=$GAMMA_VAL, disable_orth_proj=$disable_orth_proj"
        run_deep_models $preprocessing_data $seed $window_size_item "vlinear" $disable_orth_proj
    done
done
