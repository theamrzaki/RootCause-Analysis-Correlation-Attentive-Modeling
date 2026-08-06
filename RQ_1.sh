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
    local hidden_layer_size=128


    echo "Running: dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size_item | latent_mode=$latent_mode | context=$context | pool=$pool | coeff_mode=$coeff_mode | predictor=$predictor | temporal_mixer=$temporal_mixer"

                                cmd="python3 main.py \
                                        --encoder_gamma=$GAMMA_VAL --decoder_gamma=$GAMMA_VAL \
                                        --encoder_lambda=$LAMBDA_VAL --decoder_lambda=$LAMBDA_VAL --beta=$BETA_VAL \

                                    --main_model=$main_model \
                                    --coeff_architecture="$arch" \
                                    --temporal_mixer=0 \
                                    --use_MoM=0 \

                                    --preprocessing_data="$preprocessing_data" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --latent_mode=$latent_mode \
                                    --context=$context \
                                    --pool=$pool \
                                    --coeff_mode=$coeff_mode \
                                    --predictor=$predictor \
                                    --temporal_mixer=$temporal_mixer \


                                    --training_aerca=1 \
                                    --epochs=200 \
                                    --results_csv="$results_csv" \

                                    --hidden_layer_size="$hidden_layer_size" \
                                "

                                eval $cmd
                        
            
}
#-------------------------------------------------------------------
#-----------------------------rcd & epsilon diagnosis---------------
#-------------------------------------------------------------------



run_experiment_baselines() {
   local preprocessing_data=$1
   local seed=$2
   local window_size_item=$3
   local coeff_architecture=("baro" "torai" "rcd")
   local main_model="aerca_based" # Changed from array to string
   for arch in "${coeff_architecture[@]}"; do
       echo "Running: dataset=$dataset | seed=$seed | arch=$arch | window_size=$window_size_item"
       # only run preprocessing for the first coeff_architecture to avoid redundant preprocessing
       #if [ $arch == "torai" ]; then
       #    preprocessing_flag=$preprocessing_data
       #else
       #    preprocessing_flag=0
       #fi
       preprocessing_flag=$preprocessing_data  # Always use the provided preprocessing_data value
       cmd="python3 main.py \
           --seed=$seed \
           --dataset=$dataset \
           --coeff_architecture=$arch \
           --preprocessing_data=$preprocessing_flag \
           --window_size=$window_size_item \
           --main_model=$main_model \
           --training_aerca=0 \
           --results_csv=$results_csv" # Removed the comment/backslash error here
       eval $cmd
   done
}




#-------------------SMD-----------------------


latent_mode=("mul") #"gate"
context=("linear_attn") #"none" "gate" 
pool=("max") # "split_max" "split_diff"
coeff_mode=("symmetric") # "cosine" "bipartite" 
predictor=("linear") #"mlp" "linear"
temporal_mixer=(1) #0 

dataset="swat"
results_csv="RQ_1_SWAT_ExpandedWindows.csv"
seeds=(1) # 2 3
window_size=(4 8 12) # 8 16 20)  # 6 10 20 NOT 20 — 20 is already done except CUTS_PLUS

for seed in "${seeds[@]}"; do
    for window_size_item in "${window_size[@]}"; do
        preprocessing_data=0
        run_deep_models $preprocessing_data $seed $window_size_item "vlinear" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer
        #preprocessing_data=0
        #run_deep_models $preprocessing_data $seed $window_size_item "GVAR" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer
        #run_experiment_baselines $preprocessing_data $seed $window_size_item
        #run_deep_models $preprocessing_data $seed $window_size_item "cLSTM" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer
        #run_deep_models $preprocessing_data $seed $window_size_item "CUTS_PLUS" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer
    done
done







#-------------------SMD-----------------------


latent_modes=("mul") #"gate"
contexts=("linear_attn") #"none" 
pools=("max") # "split_diff"
coeff_mode_list=("symmetric") # "cosine" "bipartite" 
predictor_list=("linear") #"mlp" "linear"
temporal_mixer_list=(1) #0 

dataset="wadi"
results_csv="RQ_1_WADI_ExpandedWindows.csv"
seeds=(1) # 2 3
window_size=(4 8 12) # 8 16 20)  # 6 10 20 NOT 20 — 20 is already done except CUTS_PLUS

for seed in "${seeds[@]}"; do
    for window_size_item in "${window_size[@]}"; do
        preprocessing_data=0
        run_deep_models $preprocessing_data $seed $window_size_item "vlinear" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer
        #preprocessing_data=0
        #run_deep_models $preprocessing_data $seed $window_size_item "GVAR" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer
        #run_experiment_baselines $preprocessing_data $seed $window_size_item
        #run_deep_models $preprocessing_data $seed $window_size_item "cLSTM" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer
        #run_deep_models $preprocessing_data $seed $window_size_item "CUTS_PLUS" $latent_mode $context $pool $coeff_mode $predictor $temporal_mixer
    done
done



