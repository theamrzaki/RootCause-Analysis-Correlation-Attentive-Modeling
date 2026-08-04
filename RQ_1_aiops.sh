source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

seeds=()
dataset="aiops"
window_size=(8)
lrs=("1e-4")
results_csv="RQ_1_aiops_TSE.csv"

BETA_VAL=0.01
LAMBDA_VAL=0.5
GAMMA_VAL=0.2

#-------------------------------------------------------------------
#-----------------------------deep models-------------------------------
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
                                    --use_MoM=0 \

                                    --preprocessing_data="$preprocessing_data" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --training_aerca=1 \
                                    --epochs=200 \
                                    --results_csv="$results_csv" \

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
for seed in "${seeds[@]}"; do
    for window_size_item in "${window_size[@]}"; do
        preprocessing_data=0 
        run_deep_models $preprocessing_data $seed $window_size_item "vlinear"
        #preprocessing_data=0
        #run_deep_models $preprocessing_data $seed $window_size_item "GVAR"
        #run_experiment_baselines $preprocessing_data $seed $window_size_item
        #run_deep_models $preprocessing_data $seed $window_size_item "cLSTM"
        #run_deep_models $preprocessing_data $seed $window_size_item "CUTS_PLUS"
    done
done


















source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

seeds=()
dataset="gaia"
window_size=(8)
lrs=("1e-4")
results_csv="RQ_1_gaia_TSE.csv"

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
                                    --use_MoM=0 \   

                                    --preprocessing_data="$preprocessing_data" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --training_aerca=1 \
                                    --epochs=200 \
                                    --results_csv="$results_csv" \

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


for seed in "${seeds[@]}"; do
    for window_size_item in "${window_size[@]}"; do
        preprocessing_data=0
        run_deep_models $preprocessing_data $seed $window_size_item "vlinear"
        #preprocessing_data=0
        #run_deep_models $preprocessing_data $seed $window_size_item "GVAR"
        #run_experiment_baselines $preprocessing_data $seed $window_size_item
        #run_deep_models $preprocessing_data $seed $window_size_item "cLSTM"
        #run_deep_models $preprocessing_data $seed $window_size_item "CUTS_PLUS"
    done
done














source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

dataset="smd"
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
    local main_model="aerca_based"
    local hidden_layer_size=64

                                echo "Running:  arch=$arch | dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model | preprocessing_data=$preprocessing_data"


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

#dataset="swat"
#results_csv="RQ_1_swat_TSE.csv"
#seeds=(1)
#window_size=(5 7 10 15 20)  # 6 10 20 NOT 20 — 20 is already done except CUTS_PLUS
#
#for seed in "${seeds[@]}"; do
#    for window_size_item in "${window_size[@]}"; do
#        preprocessing_data=1 
#        run_deep_models $preprocessing_data $seed $window_size_item "vlinear"
#        preprocessing_data=0
#        run_deep_models $preprocessing_data $seed $window_size_item "GVAR"
#        run_experiment_baselines $preprocessing_data $seed $window_size_item
#        run_deep_models $preprocessing_data $seed $window_size_item "cLSTM"
#        run_deep_models $preprocessing_data $seed $window_size_item "CUTS_PLUS"
#    done
#done


dataset="wadi"
results_csv="RQ_1_wadi_TSE.csv"
seeds=(1)
window_size=(6 10 14 16)  # 6 10 20 NOT 20 — 20 is already done except CUTS_PLUS

for seed in "${seeds[@]}"; do
    for window_size_item in "${window_size[@]}"; do
        preprocessing_data=1 
        run_deep_models $preprocessing_data $seed $window_size_item "vlinear"
        preprocessing_data=0
        run_deep_models $preprocessing_data $seed $window_size_item "GVAR"
        run_experiment_baselines $preprocessing_data $seed $window_size_item
        run_deep_models $preprocessing_data $seed $window_size_item "cLSTM"
        run_deep_models $preprocessing_data $seed $window_size_item "CUTS_PLUS"
    done
done



