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

    local main_model="aerca_based"
    local hidden_layer_size=128

    echo "Running arch=$arch | latent=$latent_mode | context=$context | pool=$pool | seed=$seed | window=$window_size_item | coeff_mode=$coeff_mode | predictor=$predictor | temporal_mixer=$temporal_mixer"


    cmd="python3 main.py \
        --encoder_gamma=$GAMMA_VAL --decoder_gamma=$GAMMA_VAL \
        --encoder_lambda=$LAMBDA_VAL --decoder_lambda=$LAMBDA_VAL \
        --beta=$BETA_VAL \

        --main_model=$main_model \
        --coeff_architecture=$arch \

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


#-------------------SMD-----------------------

dataset="swat"
results_csv="Ablations_SWAT_loss.csv"
seed=1
window_size_item=(64)


latent_mode="mul"
context="linear_attn"
pool="max"
coeff_mode="symmetric"
predictor="linear"
preprocessing_data=1

temporal_mixer=1
# ============================================================
# 1. KL LOSS (beta) SENSITIVITY
#    lambda = 0.5
#    gamma  = 0.5
# ============================================================

#BETAs=(0.001 0.0025 0.005 0.02 0.05)
#
#for beta in "${BETAs[@]}"; do
#
#    lambda=0.5
#    gamma=0.5
#
#    echo "Running KL: Beta=$beta, Lambda=$lambda, Gamma=$gamma"
#
#    run_deep_models \
#        $preprocessing_data \
#        $seed \
#        $window_size_item \
#        "vlinear" \
#        $latent_mode \
#        $context \
#        $pool \
#        $coeff_mode \
#        $predictor \
#        $temporal_mixer \
#        $beta \
#        $lambda \
#        $gamma
#
#done

# ============================================================
# 2. SPARSITY LOSS (lambda) SENSITIVITY
#
# beta  = 0.005  <-- best current beta for AC@1
# gamma = 0.5
# ============================================================

Lambdas=(0.2 0.3 0.4 0.5 0.6 0.7)

for lambda in "${Lambdas[@]}"; do

    beta=0.005
    gamma=0.5

    echo "Running Sparsity: Beta=$beta, Lambda=$lambda, Gamma=$gamma"

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
        $gamma

done


# ============================================================
# 3. SMOOTHNESS LOSS (gamma) SENSITIVITY
#    beta   = 0.01
#    lambda = 0.5
# ============================================================

Gammas=(0.05 0.25 0.35 0.45 0.5 0.55 0.65 0.75)
for gamma in "${Gammas[@]}"; do

    beta=0.005
    lambda=0.5

    echo "Running Smoothness: Beta=$beta, Lambda=$lambda, Gamma=$gamma"

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
        $gamma

done