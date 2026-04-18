seeds=(1 2 3)
dataset=("msds")
lrs=("1e-4")
window_size=(1 2 3 4 5)


#-----------------------------------------------------
#-----------------------------Fedformer---------------
#-----------------------------------------------------
main_model="FEDformer"
attention_dim=16
heads=2
outer_att_dim_val=16
outer_heads_val=2

run_experiment_Fedformer() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model"

                                cmd="python3 main.py \
                                        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                        --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                        --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \

                                    --main_model=$main_model \
                                    --time_freq_representation="mag_phase" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --preprocessing_data=0 \
                                    --results_csv="RQ_1_msds.csv" \

                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            
        done
    done
}
#run_experiment_Fedformer 1


#-----------------------------------------------------
#-----------------------------rcd & epsilon diagnosis---------------
#-----------------------------------------------------
coeff_architecture=("rcd" "epsilon_diagnosis")
main_model="aerca_based"
attention_dim=16
heads=2
outer_att_dim_val=16
outer_heads_val=2

run_experiment_Baselines() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
            for coeff_architecture_item in "${coeff_architecture[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model | coeff_architecture=$coeff_architecture_item"

                                cmd="python3 main.py \
                                        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                        --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                        --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \

                                    --main_model=$main_model \
                                    --time_freq_representation="mag_phase" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --training_aerca=0\
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --preprocessing_data=0 \
                                    --results_csv="RQ_1_msds.csv" \
                                    --coeff_architecture="$coeff_architecture_item" \
                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            done
        done
    done
    }
#run_experiment_Baselines 1


#-----------------------------------------------------
#-----------------------------iTransformer---------------
#-----------------------------------------------------
arch="iTransformer"
main_model="iTransformer"
attention_dim=16
heads=2
outer_att_dim_val=16
outer_heads_val=2

run_experiment_SWAT_iTransformer() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model"

                                cmd="python3 main.py \
                                        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                        --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                        --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \

                                    --main_model=$main_model \
                                    --time_freq_representation="mag_phase" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --preprocessing_data=0 \
                                    --results_csv="RQ_1_msds.csv" \

                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            
        done
    done
}
#run_experiment_SWAT_iTransformer 1



#-----------------------------------------------------
#-----------------------------vLinear---------------
#-----------------------------------------------------
arch="vlinear"
main_model="aerca_based"
attention_dim=16
heads=2
outer_att_dim_val=16
outer_heads_val=2

run_experiment_SWAT_CrGSTA() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model"

                                cmd="python3 main.py \
                                        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                        --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                        --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \

                                    --main_model=$main_model \
                                    --coeff_architecture="$arch" \
                                    --time_freq_representation="mag_phase" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --training_aerca=1 \
                                    --epochs=50 \
                                    --early_stopping=0 \
                                    --preprocessing_data=0 \
                                    --combine_method="attention" \
                                    --results_csv="RQ_1_msds.csv" \

                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            
        done
    done
}
#run_experiment_SWAT_CrGSTA 1



#-----------------------------------------------------
#-----------------------------Deep MLP---------------
#-----------------------------------------------------
coeff_architecture="deep_mlp"
main_model="aerca_based"
lrs="1e-6"

run_experiment_deepmlp() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
        
                    echo "Running: dataset=$dataset | seed=$seed | arch=$coeff_architecture | window_size=$window_size_item | lr=$lrs"

                    cmd="python3 main.py \
                        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                        --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                        --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \
                        --main_model=$main_model \
                        --lr=$lrs \
                        --seed=$seed \
                        --dataset=$dataset \
                        --coeff_architecture=$coeff_architecture \
                        --window_size=$window_size_item \
                        --preprocessing_data=0 \
                        --training_aerca=1 \
                        --epochs=1000 \
                        --early_stopping=1 \
                        --results_csv=RQ_1_msds.csv"

                    eval $cmd
        done
    done
}
#run_experiment_deepmlp




#-----------------------------------------------------
#-----------------------------CausalRCA---------------
#-----------------------------------------------------
arch="causalrca"
main_model="aerca_based"
attention_dim=256
heads=2
outer_att_dim_val=256
outer_heads_val=2

run_experiment_SWAT_CausalRCA() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model | coeff_architecture=$arch"

                                cmd="python3 main.py \
                                        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                        --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                        --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \

                                    --main_model=$main_model \
                                     --coeff_architecture="$arch" \
                                    --time_freq_representation="mag_phase" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --preprocessing_data=0 \
                                    --results_csv="RQ_1_msds.csv" \

                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            
        done
    done
}
#run_experiment_SWAT_CausalRCA 1





#-----------------------------------------------------
#-----------------------------GVAR---------------
#-----------------------------------------------------
arch="GVAR"
main_model="aerca_based"
attention_dim=256
heads=2
outer_att_dim_val=256
outer_heads_val=2

run_experiment_Cuts_MLP() {
    for seed in "${seeds[@]}"; do
        for window_size_item in "${window_size[@]}"; do
                                echo "Running: dataset=$dataset | seed=$seed | window_size=$window_size_item | lr=$lrs | main_model=$main_model | coeff_architecture=$arch"

                                cmd="python3 main.py \
                                        --correlated_KL=0 --mean_std_recon_loss=0 --AMOC_Loss=0 \
                                        --encoder_alpha=0.5 --decoder_alpha=0.5 --encoder_gamma=0.5 --decoder_gamma=0.5 \
                                        --encoder_lambda=0.5 --decoder_lambda=0.5 --beta=0.5 \

                                    --main_model=$main_model \
                                     --coeff_architecture="$arch" \
                                    --time_freq_representation="mag_phase" \

                                    --lr="$lrs" \
                                    --seed="$seed" \
                                    --dataset="$dataset" \
                                    --window_size="$window_size_item" \

                                    --training_aerca=1 \
                                    --epochs=1000 \
                                    --early_stopping=0 \
                                    --preprocessing_data=0 \
                                    --results_csv="RQ_1_msds.csv" \

                                    --attention_dim="$attention_dim" \
                                    --num_attention_heads="$heads" \
                                    --outer_heads_num="$outer_heads_val" \
                                    --outer_hidden_dim="$outer_att_dim_val" \

                                       "

                                eval $cmd
                        
            
        done
    done
}
run_experiment_Cuts_MLP 1



