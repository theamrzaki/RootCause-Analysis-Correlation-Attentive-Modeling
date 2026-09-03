source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

lrs=("1e-4")



#-------------------------------------------------------------------
#-----------------------------vlinear-------------------------------
#-------------------------------------------------------------------

run_deep_models() {
    echo "Received arguments: $@"
    local preprocessing_data=$1
    local seed=$2
    local window_size_item=$3
    local arch=$4
    local latent_mode=$5
    local context=$6
    local pool=$7

    local coeff_mode=$8
    local predictor=$9
    local temporal_mixer=${10}

    local BETA_VAL=${11}
    local LAMBDA_VAL=${12}
    local GAMMA_VAL=${13}

    local batch_size=${14}
    local main_model="aerca_based"
    local hidden_layer_size=${15}
    local epochs=${16}

    local exp_name=${17}
    echo "Running arch=$arch | latent=$latent_mode | context=$context | pool=$pool | seed=$seed | window=$window_size_item | coeff_mode=$coeff_mode | predictor=$predictor | temporal_mixer=$temporal_mixer"


    cmd="python3 main.py \
        --encoder_gamma=$GAMMA_VAL --decoder_gamma=$GAMMA_VAL \
        --encoder_lambda=$LAMBDA_VAL --decoder_lambda=$LAMBDA_VAL \
        --beta=$BETA_VAL \
        
        --main_model=$main_model \
        --coeff_architecture=$arch \
        --batch_size=$batch_size \
        
        --latent_mode=$latent_mode \
        --context=$context \
        --pool=$pool \
        --coeff_mode=$coeff_mode \
        --predictor=$predictor \
        --temporal_mixer=$temporal_mixer \

        --use_MoM=0 \
        --preprocessing_data=$preprocessing_data \

        --lr=$lrs \
        --seed=$seed \
        --dataset=$dataset \
        --window_size=$window_size_item \

        --training_aerca=1 \
        --epochs=$epochs \
        --results_csv=$results_csv \
        
        --exp_name=$exp_name \
        
        --hidden_layer_size=$hidden_layer_size"


    eval $cmd
}

exp_name="Ablations_windows"
#-------------------Datasets (SWaT, WADI)-----------------------


dataset="wadi"
results_csv="Ablations_wadi_windows_new.csv"

seeds=()
window_size=(4 8 12 16)

latent_mode="mul"
context="gate"
pool="max"
coeff_mode="symmetric"
predictor="linear"
temporal_mixer=1

# Data has already been preprocessed.
preprocessing_data=0
batch_size=512
hidden_layer_size=128
epochs=50
for seed in "${seeds[@]}"; do
    for window_size_item in "${window_size[@]}"; do

            beta=0.005
            gamma=0.5
            lambda=0.5

            echo "================================================"
            echo "window_size_item: $window_size_item"
            echo "================================================"

            run_deep_models \
                $preprocessing_data \
                $seed \
                $window_size_item \
                "vlinear" \
                $latent_mode \
                $context \
                $pool \
                $coeff_mode \
                $predictor \
                $temporal_mixer \
                $beta \
                $lambda \
                $gamma \
                $batch_size \
                $hidden_layer_size \
                $epochs \
                $exp_name

        
    done
done





dataset="swat"
results_csv="Ablations_swat_windows_new.csv"

seeds=()
window_size=(4 8 12 16) #4 8 16

latent_mode="mul"
context="linear_attn"
pool="max"
coeff_mode="symmetric"
predictor="mlp"
temporal_mixer=1

# Data has already been preprocessed.
preprocessing_data=0
batch_size=512
hidden_layer_size=128
epochs=200
for seed in "${seeds[@]}"; do
    for window_size_item in "${window_size[@]}"; do

            beta=0.005
            gamma=0.5
            lambda=0.5

            echo "================================================"
            echo "window_size_item: $window_size_item"
            echo "================================================"

            run_deep_models \
                $preprocessing_data \
                $seed \
                $window_size_item \
                "vlinear" \
                $latent_mode \
                $context \
                $pool \
                $coeff_mode \
                $predictor \
                $temporal_mixer \
                $beta \
                $lambda \
                $gamma \
                $batch_size \
                $hidden_layer_size \
                $epochs \
                $exp_name

        
    done
done














dataset="BATADAL"
results_csv="Ablations_batadal_windows_256_256.csv"

seeds=(1 2 3)
window_size=(4 8 16 24) #4 16)

latent_mode="mul"
context="linear_attn"
pool="max"
coeff_mode="symmetric"
predictor="linear"
temporal_mixer=1

# Data has already been preprocessed.
preprocessing_data=0
batch_size=256
hidden_layer_size=256
epochs=200
for seed in "${seeds[@]}"; do
    for window_size_item in "${window_size[@]}"; do

            beta=0.0
            gamma=0.5
            lambda=0.5

            echo "================================================"
            echo "window_size_item: $window_size_item"
            echo "================================================"

            run_deep_models \
                $preprocessing_data \
                $seed \
                $window_size_item \
                "vlinear" \
                $latent_mode \
                $context \
                $pool \
                $coeff_mode \
                $predictor \
                $temporal_mixer \
                $beta \
                $lambda \
                $gamma \
                $batch_size \
                $hidden_layer_size \
                $epochs\
                $exp_name

        
    done
done