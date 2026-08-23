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
    local hidden_layer_size=128

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
        --epochs=200 \
        --results_csv=$results_csv \

        --hidden_layer_size=$hidden_layer_size"


    eval $cmd
}


#-------------------Datasets (SWaT, WADI)-----------------------

seed=1

dataset="batadal"
results_csv="z_Ablations_batadal_loss_window16.csv"

seeds=(1)
window_size=(16)

# ============================================================
# FIXED ARCHITECTURE
# ============================================================

           # "context": "gate",
           # "latent_mode": "mul",
           # "pool": "max",
           # "coeff_mode": "symmetric",
           # "predictor": "linear",

latent_mode="mul"
context="linear_attn"
pool="max"
coeff_mode="symmetric"
predictor="linear"
temporal_mixer=0

# Data has already been preprocessed.
preprocessing_data=0

batch_size=512


# ============================================================
# 1. KL LOSS / BETA SENSITIVITY
#
# Current reference:
#   beta   = 0.005
#   lambda = 0.5
#   gamma  = 0.5
#
# Focus around the current value rather than a broad sweep.
# ============================================================

BETAs=(0.005 0.01 0.02 0.5) #0 0.001 0.02 ||| 0 0.005 0.01 0.02

#for seed in "${seeds[@]}"; do
#    for window_size_item in "${window_size[@]}"; do
#
#        for beta in "${BETAs[@]}"; do
#
#            lambda=0.5
#            gamma=0.5
#
#            echo "================================================"
#            echo "KL sensitivity"
#            echo "seed=$seed"
#            echo "window=$window_size_item"
#            echo "beta=$beta"
#            echo "lambda=$lambda"
#            echo "gamma=$gamma"
#            echo "================================================"
#
#            run_deep_models \
#                $preprocessing_data \
#                $seed \
#                $window_size_item \
#                "vlinear" \
#                $latent_mode \
#                $context \
#                $pool \
#                $coeff_mode \
#                $predictor \
#                $temporal_mixer \
#                $beta \
#                $lambda \
#                $gamma \
#                $batch_size
#
#        done
#    done
#done


# ============================================================
# 2. SPARSITY LOSS / LAMBDA SENSITIVITY
#
# Fixed:
#   beta  = 0.005
#   gamma = 0.5
#
# Current reference:
#   lambda = 0.5
# ============================================================

Lambdas=(0 0.001 0.1 0.2 0.35 0.5)

for seed in "${seeds[@]}"; do
    for window_size_item in "${window_size[@]}"; do

        for lambda in "${Lambdas[@]}"; do

            beta=0.005
            gamma=0.5

            echo "================================================"
            echo "Sparsity sensitivity"
            echo "seed=$seed"
            echo "window=$window_size_item"
            echo "beta=$beta"
            echo "lambda=$lambda"
            echo "gamma=$gamma"
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
                $batch_size

        done
    done
done


# ============================================================
# 3. SMOOTHNESS LOSS / GAMMA SENSITIVITY
#
# Fixed:
#   beta   = 0.005
#   lambda = 0.5
#
# Current reference:
#   gamma = 0.5
# ============================================================

Gammas=(0 0.001 0.1 0.25 0.3 0.35 0.4 0.5)

#for seed in "${seeds[@]}"; do
#    for window_size_item in "${window_size[@]}"; do
#
#        for gamma in "${Gammas[@]}"; do
#
#            beta=0.005
#            lambda=0.35
#
#            echo "================================================"
#            echo "Smoothness sensitivity"
#            echo "seed=$seed"
#            echo "window=$window_size_item"
#            echo "beta=$beta"
#            echo "lambda=$lambda"
#            echo "gamma=$gamma"
#            echo "================================================"
#
#            run_deep_models \
#                $preprocessing_data \
#                $seed \
#                $window_size_item \
#                "vlinear" \
#                $latent_mode \
#                $context \
#                $pool \
#                $coeff_mode \
#                $predictor \
#                $temporal_mixer \
#                $beta \
#                $lambda \
#                $gamma \
#                $batch_size
#
#        done
#    done
#done
