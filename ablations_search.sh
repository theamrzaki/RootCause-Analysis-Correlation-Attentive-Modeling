source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

lrs=("1e-4")

BETA_VAL=0.005
LAMBDA_VAL=0.5
GAMMA_VAL=0.5

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

    local batch_size=${11}
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
        --batch_size=$batch_size \

        --lr=$lrs \
        --seed=$seed \
        --dataset=$dataset \
        --window_size=$window_size_item \

        --training_aerca=1\
        --epochs=200 \
        --results_csv=$results_csv \

        --hidden_layer_size=$hidden_layer_size"


    eval $cmd
}


#-------------------SMD-----------------------

dataset="swat"
results_csv="Ablations_SWAT_window8.csv"
seeds=(1)
window_size=(8)


dataset="swat"
results_csv="Ablations_SWAT_window8.csv"
seeds=(1)
window_size=(8)

batch_size=512

# Focused ablation:
#
# Baseline:
# mul + linear_attn + max + symmetric + linear + TM=1
#
# Latent:
# gate, add
#
# Context:
# gate
#
# Pool:
# split_diff, split_max
#
# Predictor:
# mlp

latent_modes=("mul" "gate" "add")
contexts=("linear_attn" "gate")
pools=("max" "split_diff" "split_max")
coeff_mode_list=("symmetric")
predictor_list=("linear" "mlp")
temporal_mixer_list=(1)

for seed in "${seeds[@]}"; do
    for window_size_item in "${window_size[@]}"; do

        # --------------------------------------------------
        # Baseline
        # --------------------------------------------------
        run_deep_models \
            0 \
            $seed \
            $window_size_item \
            "vlinear" \
            "mul" \
            "linear_attn" \
            "max" \
            "symmetric" \
            "linear" \
            1 \
            $batch_size

        # --------------------------------------------------
        # Latent construction ablation
        # --------------------------------------------------
        for latent_mode in gate add; do

            run_deep_models \
                0 \
                $seed \
                $window_size_item \
                "vlinear" \
                $latent_mode \
                "linear_attn" \
                "max" \
                "symmetric" \
                "linear" \
                1 \
                $batch_size

        done

        # --------------------------------------------------
        # Context ablation
        # --------------------------------------------------
        run_deep_models \
            0 \
            $seed \
            $window_size_item \
            "vlinear" \
            "mul" \
            "gate" \
            "max" \
            "symmetric" \
            "linear" \
            1 \
            $batch_size

        # --------------------------------------------------
        # Pooling ablation
        # --------------------------------------------------
        for pool in split_diff split_max; do

            run_deep_models \
                0 \
                $seed \
                $window_size_item \
                "vlinear" \
                "mul" \
                "linear_attn" \
                $pool \
                "symmetric" \
                "linear" \
                1 \
                $batch_size

        done

        # --------------------------------------------------
        # Prediction head ablation
        # --------------------------------------------------
        run_deep_models \
            0 \
            $seed \
            $window_size_item \
            "vlinear" \
            "mul" \
            "linear_attn" \
            "max" \
            "symmetric" \
            "mlp" \
            1 \
            $batch_size

    done
done