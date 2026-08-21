#!/bin/bash

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
    local batch_size=${11}
    local disable_orthogonal_projection=${12}
    local epochs=${13}

    local main_model="aerca_based"
    local hidden_layer_size=${14}
    local exp_name=${15}

    echo "Running arch=$arch | latent=$latent_mode | context=$context | pool=$pool | seed=$seed | window=$window_size_item | coeff_mode=$coeff_mode | predictor=$predictor | temporal_mixer=$temporal_mixer"

    cmd="python3 main.py \
        --encoder_gamma=$GAMMA_VAL --decoder_gamma=$GAMMA_VAL \
        --encoder_lambda=$LAMBDA_VAL --decoder_lambda=$LAMBDA_VAL \
        --beta=$BETA_VAL \

        --exp_name=$exp_name \

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

        --training_aerca=1 \
        --epochs=$epochs \
        --results_csv=$results_csv \
        --disable_orth_proj=$disable_orthogonal_projection \
        --hidden_layer_size=$hidden_layer_size"

    eval $cmd
}

datasets=("swat") # "swat"
results_csv_wadi="3_Ablations_WADI_window8_MLP.csv" 
results_csv_swat="3_Ablations_SWAT_window8.csv"
results_csv_batadal="3_Ablations_BATADAL_window8_samelosses.csv"
seeds=(2 3)

exp_name="Ablations"
for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        if [[ "$dataset" == "wadi" ]]; then
            echo "Running experiments for WADI dataset"
            results_csv=$results_csv_wadi
            beta_default=0.005
            lambda_default=0.5
            context_default="linear_attn"
            latent_default="mul"
            epochs=50
            window_size_item=8
            batch_size=512
            hidden_layer_size=128
        elif [[ "$dataset" == "swat" ]]; then
            echo "Running experiments for SWAT dataset"
            results_csv=$results_csv_swat
            beta_default=0.005
            lambda_default=0.5
            context_default="linear_attn"
            latent_default="mul"
            epochs=200
            window_size_item=8
            batch_size=512
            hidden_layer_size=128
        else
            echo "Running experiments for BATADAL dataset"
            results_csv=$results_csv_batadal
            beta_default=0.005
            lambda_default=0.5
            context_default="linear_attn"
            latent_default="mul"
            epochs=200
            window_size_item=16
            batch_size=256
            hidden_layer_size=256
        fi

        BETA_VAL=$beta_default
        LAMBDA_VAL=$lambda_default
        GAMMA_VAL=0.5

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
            $batch_size \
            0 \
            $epochs \
            $hidden_layer_size \
            $exp_name

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
                $batch_size \
                0 \
                $epochs \
                $hidden_layer_size \
                $exp_name
        done
        
        # --------------------------------------------------
        # Context ablation
        # --------------------------------------------------
        run_deep_models \
            0 \
            $seed \
            $window_size_item \
            "vlinear" \
            $latent_default \
            "gate" \
            "max" \
            "symmetric" \
            "linear" \
            1 \
            $batch_size \
            0 \
            $epochs \  
            $hidden_layer_size \ 
            $exp_name

        # --------------------------------------------------
        # Context ablation (with no temporal mixer)
        # --------------------------------------------------
        run_deep_models \
            0 \
            $seed \
            $window_size_item \
            "vlinear" \
            $latent_default \
            $context_default \
            "max" \
            "symmetric" \
            "linear" \
            0 \
            $batch_size \
            0 \
            $epochs \   
            $hidden_layer_size \
            $exp_name

        # --------------------------------------------------
        # Context ablation (with no orthogonal projection)
        # --------------------------------------------------
        run_deep_models \
            0 \
            $seed \
            $window_size_item \
            "vlinear" \
            $latent_default \
            $context_default \
            "max" \
            "symmetric" \
            "linear" \
            1 \
            $batch_size \
            1 \
            $epochs \
            $hidden_layer_size \
            $exp_name

        # --------------------------------------------------
        # Prediction head ablation
        # --------------------------------------------------
        run_deep_models \
            0 \
            $seed \
            $window_size_item \
            "vlinear" \
            $latent_default \
            $context_default \
            "max" \
            "symmetric" \
            "mlp" \
            1 \
            $batch_size \
            0 \
            $epochs \
            $hidden_layer_size \
            $exp_name

        # --------------------------------------------------
        # Ablation of the loss function components
        # --------------------------------------------------
        # No Beta
        BETA_VAL=0
        LAMBDA_VAL=0.5
        GAMMA_VAL=0.5

        run_deep_models \
            0 \
            $seed \
            $window_size_item \
            "vlinear" \
            "mul" \
            $context_default \
            "max" \
            "symmetric" \
            "linear" \
            1 \
            $batch_size \
            0 \
            $epochs \
            $hidden_layer_size \
            $exp_name

        # No Lambda
        BETA_VAL=$beta_default
        LAMBDA_VAL=0
        GAMMA_VAL=0.5

        run_deep_models \
            0 \
            $seed \
            $window_size_item \
            "vlinear" \
            "mul" \
            $context_default \
            "max" \
            "symmetric" \
            "linear" \
            1 \
            $batch_size \
            0 \
            $epochs \
            $hidden_layer_size \
            $exp_name

        # No Gamma
        BETA_VAL=$beta_default
        LAMBDA_VAL=0.5
        GAMMA_VAL=0

        run_deep_models \
            0 \
            $seed \
            $window_size_item \
            "vlinear" \
            "mul" \
            $context_default \
            "max" \
            "symmetric" \
            "linear" \
            1 \
            $batch_size \
            0 \
            $epochs \
            $hidden_layer_size \
            $exp_name

    done
done