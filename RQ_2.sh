source ~/miniforge3/etc/profile.d/conda.sh
conda activate RCAEval

lrs=("1e-4")

BETA_VAL=0.005
LAMBDA_VAL=0.5
GAMMA_VAL=0.5

#-------------------------------------------------------------------
#-------------------------deep models-------------------------------
#-------------------------------------------------------------------

run_deep_models() {
    echo "received parameters: $@"
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
    local epochs=${12}
    local hidden_layer_size=${13}


    echo "Running: dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size_item | latent_mode=$latent_mode | context=$context | pool=$pool | coeff_mode=$coeff_mode | predictor=$predictor | temporal_mixer=$temporal_mixer"

                                cmd="python3 main.py \
                                        --encoder_gamma=$GAMMA_VAL --decoder_gamma=$GAMMA_VAL \
                                        --encoder_lambda=$LAMBDA_VAL --decoder_lambda=$LAMBDA_VAL --beta=$BETA_VAL \

                                    --device=cpu \
                                    
                                    --main_model=$main_model \
                                    --coeff_architecture="$arch" \
                                    --temporal_mixer=$temporal_mixer\
                                    --use_MoM=0 \

                                    --preprocessing_data="$preprocessing_data" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \
                                    --batch_size="$batch_size" \
                                    --epochs="$epochs" \

                                    --latent_mode=$latent_mode \
                                    --context=$context \
                                    --pool=$pool \
                                    --coeff_mode=$coeff_mode \
                                    --predictor=$predictor \
                                    --temporal_mixer=$temporal_mixer \


                                    --training_aerca=0 \
                                    --results_csv="$results_csv" \

                                    --hidden_layer_size="$hidden_layer_size" \
                                "

                                eval $cmd
                        
            
}
#-------------------swat-----------------------


latent_mode=("mul") 
context=("linear_attn") 
pool=("max") 
coeff_mode=("symmetric") 
predictor=("mlp") 
temporal_mixer=(1) 

dataset="swat"
results_csv="RQ_2_SWAT_RaspberryPi_Smaller.csv"
seeds=(1 2 3) 
window_size_item=8
batch_size=512
epochs=200
hidden_layer_size=128
for seed in "${seeds[@]}"; do
    preprocessing_data=0
    #run_deep_models $preprocessing_data $seed $window_size_item "GVAR" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    #run_deep_models $preprocessing_data $seed $window_size_item "cLSTM" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    #run_deep_models $preprocessing_data $seed $window_size_item "CUTS_PLUS" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    #run_deep_models $preprocessing_data $seed $window_size_item "vlinear" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    #run_deep_models $preprocessing_data $seed $window_size_item "cMLP" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    #run_deep_models $preprocessing_data $seed $window_size_item "deep_mlp" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size

    run_deep_models $preprocessing_data $seed $window_size_item "iTransformer" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    run_deep_models $preprocessing_data $seed $window_size_item "TimeMixerpp" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size

done




##-------------------WADI-----------------------


latent_mode=("mul") #"gate"
context=("gate") #"none" "gate" 
pool=("max") # "split_max" "split_diff"
coeff_mode=("symmetric") # "cosine" "bipartite" 
predictor=("linear") #"mlp" "linear"
temporal_mixer=(1) #0 

batch_size=512
dataset="wadi"
results_csv="RQ_2_WADI_RaspberryPi_Smaller.csv"
window_size_item=8
epochs=50
seeds=(2 3)
hidden_layer_size=128
for seed in "${seeds[@]}"; do
    preprocessing_data=0
    #run_deep_models $preprocessing_data $seed $window_size_item "vlinear" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    #run_deep_models $preprocessing_data $seed $window_size_item "GVAR" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    #run_deep_models $preprocessing_data $seed $window_size_item "deep_mlp" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    #run_deep_models $preprocessing_data $seed $window_size_item "cMLP" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    #run_deep_models $preprocessing_data $seed $window_size_item "cLSTM" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    #run_deep_models $preprocessing_data $seed $window_size_item "CUTS_PLUS" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size

    run_deep_models $preprocessing_data $seed $window_size_item "iTransformer" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    run_deep_models $preprocessing_data $seed $window_size_item "TimeMixerpp" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size


done




##-------------------batadal-----------------------
#
#
latent_mode=("mul") #"gate"
context=("linear_attn") #"none" "gate" 
pool=("max") # "split_max" "split_diff"
coeff_mode=("symmetric") # "cosine" "bipartite" 
predictor=("linear") #"mlp" "linear"
temporal_mixer=(1) #0 

dataset="batadal"
results_csv="RQ_2_BATADAL_RaspberryPi_Smaller.csv"
seeds=(1 2 3) # 2 3
window_size_item=16
epochs=200
batch_size=256
hidden_layer_size=256
for seed in "${seeds[@]}"; do
    BETA_VAL=0.0
    LAMBDA_VAL=0.5
    GAMMA_VAL=0.5
    preprocessing_data=0
    #run_deep_models $preprocessing_data $seed $window_size_item "vlinear" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    #run_deep_models $preprocessing_data $seed $window_size_item "GVAR" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    #run_deep_models $preprocessing_data $seed $window_size_item "cLSTM" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    #run_deep_models $preprocessing_data $seed $window_size_item "cMLP" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    #run_deep_models $preprocessing_data $seed $window_size_item "deep_mlp" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    #run_deep_models $preprocessing_data $seed $window_size_item "CUTS_PLUS" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size

    run_deep_models $preprocessing_data $seed $window_size_item "iTransformer" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size
    run_deep_models $preprocessing_data $seed $window_size_item "TimeMixerpp" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer $batch_size $epochs $hidden_layer_size

done

