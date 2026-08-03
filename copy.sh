#!/bin/bash
#IP_ADDRESS="130.63.254.162" #db2003smaller
IP_ADDRESS="130.63.100.216"
#DEVICE_NAME="db2003smaller"
DEVICE_NAME="db2003larger"


#This sets up SSH keys so scp never asks for a password again.
ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
ssh-copy-id $DEVICE_NAME@$IP_ADDRESS

# Give your user permission to read the source folder so you don't need sudo inside the loops
sudo chown -R $(whoami) "/home/db2003/Desktop/Amr/(TSE) RootCause-Analysis-Correlation-Attentive-Modeling/saved_models/"

DATASET_ROOT="/home/db2003/Desktop/Amr/(TSE) RootCause-Analysis-Correlation-Attentive-Modeling/datasets"

dataset="smd"
windows_list=() #4
num_vars=38

for window in "${windows_list[@]}"; do
    model_file=${DATASET_ROOT}/${dataset}/window_${window}_vars_${num_vars}

    echo "---- Copying dataset files for window ${window} and num_vars ${num_vars} to $DEVICE_NAME@$IP_ADDRESS ----"
    data_files=(
        "${DATASET_ROOT}/${dataset}/window_${window}_vars_${num_vars}/label_list.npy"
        "${DATASET_ROOT}/${dataset}/window_${window}_vars_${num_vars}/x_ab_list.npy"
        "${DATASET_ROOT}/${dataset}/window_${window}_vars_${num_vars}/x_n_list.npy"
    )
    for data_file in "${data_files[@]}"; do
        # create the destination directory on the remote server if it doesn't exist
        ssh $DEVICE_NAME@$IP_ADDRESS "mkdir -p /home/$DEVICE_NAME/RootCause-Analysis-Correlation-Attentive-Modeling/datasets/${dataset}/window_${window}#_vars_${num_vars}/"
        scp "$data_file" $DEVICE_NAME@$IP_ADDRESS:/home/$DEVICE_NAME/RootCause-Analysis-Correlation-Attentive-Modeling/datasets/${dataset}/window_${window}#_vars_${num_vars}/
    done

    orth_data_file="${DATASET_ROOT}/${dataset}/window_${window}_vars_${num_vars}/orth_transform_meta/swat_q_matrix_lag${window}.npy"
    ssh $DEVICE_NAME@$IP_ADDRESS "mkdir -p /home/$DEVICE_NAME/RootCause-Analysis-Correlation-Attentive-Modeling/datasets/${dataset}/window_${window}_vars_${num_vars}/orth_transform_meta"
    scp "$orth_data_file" $DEVICE_NAME@$IP_ADDRESS:/home/$DEVICE_NAME/RootCause-Analysis-Correlation-Attentive-Modeling/datasets/${dataset}/window_${window}_vars_${num_vars}/orth_transform_meta/
done






#------------------------------------------------------#
#------------------------------------------------------#
#------------------------------------------------------#
#------------------------------------------------------#
#--------------------Model Files-----------------------#
#------------------------------------------------------#
#------------------------------------------------------#
#------------------------------------------------------#
#------------------------------------------------------#



#!/bin/bash
MODEL_NAMES=("vlinear") # )"cLSTM" "GVAR" "vlinear" "CUTS_PLUS"
DATASET_NAMES=("smd") # "gaia")
SEEDS=(1)
windows_list=(4) #4
for seed in "${SEEDS[@]}"; do
    for model in "${MODEL_NAMES[@]}"; do
        for dataset in "${DATASET_NAMES[@]}"; do

            for window in "${windows_list[@]}"; do
                echo "##### Copying model files for window ${window} and num_vars ${num_vars} to $DEVICE_NAME@$IP_ADDRESS ####"
                #cLSTM_aiops_ws_8_seed_1_numvars_30_lower_decoder.npy
                #cLSTM_aiops_ws_8_seed_1_numvars_30_lower_encoder.npy
                #cLSTM_aiops_ws_8_seed_1_numvars_30_recon_mean.npy
                #cLSTM_aiops_ws_8_seed_1_numvars_30_recon_std.npy
                #cLSTM_aiops_ws_8_seed_1_numvars_30_recon_threshold.npy
                #cLSTM_aiops_ws_8_seed_1_numvars_30_upper_decoder.npy
                #cLSTM_aiops_ws_8_seed_1_numvars_30_upper_encoder.npy
                #cLSTM_aiops_ws_8_seed_1_numvars_30_us_mean_decoder.npy
                #cLSTM_aiops_ws_8_seed_1_numvars_30_us_mean_encoder.npy
                #cLSTM_aiops_ws_8_seed_1_numvars_30_us_std_decoder.npy
                #cLSTM_aiops_ws_8_seed_1_numvars_30_us_std_encoder.npy
                #cLSTM_aiops_ws_8_seed_1_numvars_30.pt
            
                lower_decoder=${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_lower_decoder.npy
                lower_encoder=${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_lower_encoder.npy
                recon_mean=${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_recon_mean.npy
                recon_std=${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_recon_std.npy
                recon_threshold=${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_recon_threshold.npy
                upper_decoder=${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_upper_decoder.npy
                upper_encoder=${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_upper_encoder.npy
                us_mean_decoder=${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_us_mean_decoder.npy
                us_mean_encoder=${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_us_mean_encoder.npy
                us_std_decoder=${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_us_std_decoder.npy
                us_std_encoder=${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_us_std_encoder.npy
                model_file=${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}.pt


                models_to_copy=(
                    "$lower_decoder"
                    "$lower_encoder"
                    "$recon_mean"
                    "$recon_std"
                    "$recon_threshold"
                    "$upper_decoder"
                    "$upper_encoder"
                    "$us_mean_decoder"
                    "$us_mean_encoder"
                    "$us_std_decoder"
                    "$us_std_encoder"
                    "$model_file"
                )

                for model_file in "${models_to_copy[@]}"; do

                    echo "Copying $model_file to $DEVICE_NAME@$IP_ADDRESS:/home/$DEVICE_NAME/RootCause-Analysis-Correlation-Attentive-Modeling/saved_models/"

                    scp "/home/db2003/Desktop/Amr/(TSE) RootCause-Analysis-Correlation-Attentive-Modeling/saved_models/${model_file}" $DEVICE_NAME@$IP_ADDRESS:/home/$DEVICE_NAME/RootCause-Analysis-Correlation-Attentive-Modeling/saved_models/
                done
            done
        done
    done
done
