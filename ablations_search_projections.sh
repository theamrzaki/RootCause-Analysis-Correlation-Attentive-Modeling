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
    local transformation=${16}
    local plot_latents=${17}

    echo "Running seed: $seed | dataset: $dataset | transformation: $transformation"

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
        --hidden_layer_size=$hidden_layer_size\

        --transformation=$transformation\
        --plot_latents=$plot_latents"

    eval $cmd
}

datasets=("wadi") # "swat" "wadi") # "swat" "wadi"
results_csv_wadi="4_Ablations_WADI_projections.csv" 
results_csv_swat="4_Ablations_SWAT_projections.csv"
results_csv_batadal="4_Ablations_BATADAL_projections.csv"
seeds=(1)

transformations=("orthogonal")
# "learned" "legendre" "laguerre" "chebyshev" "hermite" "fourier"
exp_name="Ablations_Projections"
plot_latents=1
for seed in "${seeds[@]}"; do
    for transformation in "${transformations[@]}"; do
        for dataset in "${datasets[@]}"; do
            if [[ "$dataset" == "wadi" ]]; then
                echo "Running experiments for WADI dataset"
                results_csv=$results_csv_wadi
                beta_default=0.005
                lambda_default=0.5
                epochs=50
                window_size_item=8
                batch_size=512
                hidden_layer_size=128

                latent_mode=("mul")
                context=("gate") #<--main diff
                pool=("max")
                coeff_mode=("symmetric") 
                predictor=("linear") 
                temporal_mixer=(1) 

            elif [[ "$dataset" == "swat" ]]; then
                echo "Running experiments for SWAT dataset"
                results_csv=$results_csv_swat
                beta_default=0.005
                lambda_default=0.5
                epochs=200
                window_size_item=8
                batch_size=512
                hidden_layer_size=128

                latent_mode=("mul") 
                context=("linear_attn") 
                pool=("max") 
                coeff_mode=("symmetric") 
                predictor=("mlp") #<--main diff
                temporal_mixer=(1) 
            else
                echo "Running experiments for BATADAL dataset"
                results_csv=$results_csv_batadal
                beta_default=0 #<--main diff
                lambda_default=0.5
                epochs=200
                window_size_item=16
                batch_size=256
                hidden_layer_size=256

                latent_mode=("mul")
                context=("linear_attn") 
                pool=("max")
                coeff_mode=("symmetric")
                predictor=("linear")
                temporal_mixer=(1)
            fi

            BETA_VAL=$beta_default
            LAMBDA_VAL=$lambda_default
            GAMMA_VAL=0.5

            #if transformation is "none", we can disable orthogonal projection
            if [[ "$transformation" == "none" ]]; then
                disable_orthogonal_projection=1
            else
                disable_orthogonal_projection=0
            fi

            run_deep_models \
                0 \
                $seed \
                $window_size_item \
                "vlinear" \
                "mul" \
                $context \
                "max" \
                "symmetric" \
                $predictor\
                1 \
                $batch_size \
                $disable_orthogonal_projection \
                $epochs \
                $hidden_layer_size \
                $exp_name \
                $transformation \
                $plot_latents

        done
    done
done

datasets=("swat" "wadi") # "swat" "wadi") # "swat" "wadi"
results_csv_wadi="4_Ablations_WADI_projections.csv" 
results_csv_swat="4_Ablations_SWAT_projections.csv"
results_csv_batadal="4_Ablations_BATADAL_projections.csv"
seeds=(1)

transformations=("learned" "legendre" "laguerre" "chebyshev" "hermite" "fourier" )  #"none" "orthogonal"
# "learned" "legendre" "laguerre" "chebyshev" "hermite" "fourier"
exp_name="Ablations_Projections"
plot_latents=1
for seed in "${seeds[@]}"; do
    for transformation in "${transformations[@]}"; do
        for dataset in "${datasets[@]}"; do
            if [[ "$dataset" == "wadi" ]]; then
                echo "Running experiments for WADI dataset"
                results_csv=$results_csv_wadi
                beta_default=0.005
                lambda_default=0.5
                epochs=50
                window_size_item=8
                batch_size=512
                hidden_layer_size=128

                latent_mode=("mul")
                context=("gate") #<--main diff
                pool=("max")
                coeff_mode=("symmetric") 
                predictor=("linear") 
                temporal_mixer=(1) 

            elif [[ "$dataset" == "swat" ]]; then
                echo "Running experiments for SWAT dataset"
                results_csv=$results_csv_swat
                beta_default=0.005
                lambda_default=0.5
                epochs=200
                window_size_item=8
                batch_size=512
                hidden_layer_size=128

                latent_mode=("mul") 
                context=("linear_attn") 
                pool=("max") 
                coeff_mode=("symmetric") 
                predictor=("mlp") #<--main diff
                temporal_mixer=(1) 
            else
                echo "Running experiments for BATADAL dataset"
                results_csv=$results_csv_batadal
                beta_default=0 #<--main diff
                lambda_default=0.5
                epochs=200
                window_size_item=16
                batch_size=256
                hidden_layer_size=256

                latent_mode=("mul")
                context=("linear_attn") 
                pool=("max")
                coeff_mode=("symmetric")
                predictor=("linear")
                temporal_mixer=(1)
            fi

            BETA_VAL=$beta_default
            LAMBDA_VAL=$lambda_default
            GAMMA_VAL=0.5

            #if transformation is "none", we can disable orthogonal projection
            if [[ "$transformation" == "none" ]]; then
                disable_orthogonal_projection=1
            else
                disable_orthogonal_projection=0
            fi

            run_deep_models \
                0 \
                $seed \
                $window_size_item \
                "vlinear" \
                "mul" \
                $context \
                "max" \
                "symmetric" \
                $predictor\
                1 \
                $batch_size \
                $disable_orthogonal_projection \
                $epochs \
                $hidden_layer_size \
                $exp_name \
                $transformation \
                $plot_latents

        done
    done
done