source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

lrs=("1e-4")

BETA_VAL=0.01
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

    local main_model="aerca_based"
    local hidden_layer_size=64

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
results_csv="Ablations_SWAT.csv"
seeds=(1)
window_size=(4)


# Ablation options
latent_modes=("mul") #"gate"
contexts=("gate") #"none" 
pools=("split_max") # "split_diff"
#mul,gate,split_max,bipartite,mlp


# fixed for now
coeff_mode_list=("bipartite" "symmetric" "cosine")
predictor_list=("mlp" "linear")
temporal_mixer_list=(0 1)


for seed in "${seeds[@]}"; do
    for window_size_item in "${window_size[@]}"; do

        for latent_mode in "${latent_modes[@]}"; do
            for context in "${contexts[@]}"; do
                for pool in "${pools[@]}"; do


                    for coeff_mode in "${coeff_mode_list[@]}"; do
                        for predictor in "${predictor_list[@]}"; do
                            for temporal_mixer in "${temporal_mixer_list[@]}"; do

                                preprocessing_data=0
                                echo "Running for seed=$seed, window_size=$window_size_item, latent_mode=$latent_mode, context=$context, pool=$pool, coeff_mode=$coeff_mode, predictor=$predictor, temporal_mixer=$temporal_mixer"
                                run_deep_models $preprocessing_data $seed $window_size_item "vlinear" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer

                            done
                        done
                    done


                done
            done
        done

    done
done