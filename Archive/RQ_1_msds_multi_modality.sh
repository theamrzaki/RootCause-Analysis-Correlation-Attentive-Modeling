source ~/miniconda3/etc/profile.d/conda.sh
#conda activate RCAEval







# 1. Create environment
conda create -n dgl_env python=3.9.13 -y

# 2. Activate
conda activate dgl_env

# 3. Install PyTorch 1.12.1 (CPU version — add cudatoolkit if you need GPU)
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch -y

# 4. Install scikit-learn
pip install scikit-learn==1.1.2

# 5. Install DGL 0.9.0
pip install dgl==0.9.0 -f https://data.dgl.ai/wheels/repo.html

# 6. Verify
python -c "import torch; print('torch:', torch.__version__)"
python -c "import dgl; print('dgl:', dgl.__version__)"
python -c "import sklearn; print('sklearn:', sklearn.__version__)"




#seeds=(1 2 3)
#dataset="msds_multi_modality"
#window_size=(2)
#lrs=("1e-4")
#results_csv="RQ_1_msds_multi_modality_final.csv"
#
#BETA_VAL=0.01
#LAMBDA_VAL=0.5
#GAMMA_VAL=0.2
#
##-------------------------------------------------------------------
##-----------------------------deep models-------------------------------
##-------------------------------------------------------------------
#
#
#
#run_deep_models() {
#    local preprocessing_data=$1
#    local seed=$2
#    local window_size_item=$3
#    local arch=$4
#    local main_model="aerca_based"
#    local hidden_layer_size=12
#
#        echo "Running:  arch=$arch | dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model | preprocessing_data=$preprocessing_data"
#
#
#        cmd="python3 main.py \
#                --encoder_gamma=$GAMMA_VAL --decoder_gamma=$GAMMA_VAL \
#                --encoder_lambda=$LAMBDA_VAL --decoder_lambda=$LAMBDA_VAL --beta=$BETA_VAL \
#
#            --main_model=$main_model \
#            --coeff_architecture="$arch" \
#            --temporal_mixer=0 \
#
#            --preprocessing_data="$preprocessing_data" \
#
#            --lr="$lrs" \
#            --seed="$seed" \
#            --dataset="$dataset" \
#            --window_size="$window_size_item" \
#
#            --training_aerca=1 \
#            --epochs=200 \
#            --results_csv="$results_csv" \
#
#            --hidden_layer_size="$hidden_layer_size" \
#        "
#
#        eval $cmd
#                        
#            
#}
#
##-------------------------------------------------------------------
##-----------------------------experiment_baselines------------------
##-------------------------------------------------------------------	
#
#
#run_experiment_baselines() {
    local preprocessing_data=$1
    local seed=$2
    local window_size_item=$3

    local coeff_architecture=("torai")
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
#}
#
#
#
#
##---------------------------------------------------------------------------
##-----------------------------run experiments-------------------------------
##---------------------------------------------------------------------------
#for seed in "${seeds[@]}"; do
#    preprocessing_data=0 # already preprocessed data
#    #run_deep_models $preprocessing_data $seed $window_size "vlinear"
#    run_deep_models $preprocessing_data $seed $window_size "Eadro"
#    #run_deep_models $preprocessing_data $seed $window_size "Anofusion"
#    #run_experiment_baselines $preprocessing_data $seed $window_size
#done
#