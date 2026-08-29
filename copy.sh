#!/bin/bash
IP_ADDRESS="130.63.254.143" #db2003smaller
#IP_ADDRESS="130.63.102.34"
DEVICE_NAME="db2003smaller"
#DEVICE_NAME="db2003larger"


# 1. Generate SSH key only if it doesn't already exist
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "Generating new SSH key..."
    ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
fi

# 2. Copy SSH key to remote (using IdentitiesOnly to prevent agent auth errors)
echo "Ensuring SSH key is copied to $DEVICE_NAME@$IP_ADDRESS..."
ssh-copy-id -o IdentitiesOnly=yes -i ~/.ssh/id_rsa.pub $DEVICE_NAME@$IP_ADDRESS || true

# SSH/SCP flags to avoid 'Too many authentication failures'
SSH_OPTS="-o IdentitiesOnly=yes -i ~/.ssh/id_rsa"

# Fix permissions on local directory
sudo chown -R $(whoami) "/home/db2003/Desktop/Amr/(TSE) RootCause-Analysis-Correlation-Attentive-Modeling/saved_models/"

DATASET_ROOT="/home/db2003/Desktop/Amr/(TSE) RootCause-Analysis-Correlation-Attentive-Modeling/datasets"
dataset="batadal"
windows_list=(8)
num_vars=127

# --- COPY DATASET FILES ---
#for window in "${windows_list[@]}"; do
#    echo "---- Copying dataset files for window ${window} and num_vars ${num_vars} to $DEVICE_NAME@$IP_ADDRESS ----"
#    
#    # FIXED: Replaced '#' typo with '_' in the target path
#    REMOTE_DATASET_DIR="/home/$DEVICE_NAME/RootCause-Analysis-Correlation-Attentive-Modeling/datasets/${dataset}/window_${window}_vars_${num_vars}"
#
#    ssh $SSH_OPTS $DEVICE_NAME@$IP_ADDRESS "mkdir -p ${REMOTE_DATASET_DIR}/orth_transform_meta"
#
#    data_files=(
#        "${DATASET_ROOT}/${dataset}/window_${window}_vars_${num_vars}/label_list.npy"
#        "${DATASET_ROOT}/${dataset}/window_${window}_vars_${num_vars}/x_ab_list.npy"
#        "${DATASET_ROOT}/${dataset}/window_${window}_vars_${num_vars}/x_n_list.npy"
#    )
#
#    for data_file in "${data_files[@]}"; do
#        scp $SSH_OPTS "$data_file" $DEVICE_NAME@$IP_ADDRESS:${REMOTE_DATASET_DIR}/
#    done
#
#    orth_data_file="${DATASET_ROOT}/${dataset}/window_${window}_vars_${num_vars}/orth_transform_meta/swat_q_matrix_lag${window}.npy"
#    scp $SSH_OPTS "$orth_data_file" $DEVICE_NAME@$IP_ADDRESS:${REMOTE_DATASET_DIR}/orth_transform_meta/
#done


# --- COPY MODEL FILES ---
MODEL_NAMES=("iTransformer" "TimeMixerpp") # "GVAR" "CUTS_PLUS" "vlinear" "cLSTM"   "deep_mlp" "cMLP"
DATASET_NAMES=("wadi")
SEEDS=(1)

REMOTE_SAVED_MODELS="/home/$DEVICE_NAME/RootCause-Analysis-Correlation-Attentive-Modeling/saved_models/"
ssh $SSH_OPTS $DEVICE_NAME@$IP_ADDRESS "mkdir -p ${REMOTE_SAVED_MODELS}"

for seed in "${SEEDS[@]}"; do
    for model in "${MODEL_NAMES[@]}"; do
        for dataset in "${DATASET_NAMES[@]}"; do
            for window in "${windows_list[@]}"; do
                echo "##### Copying model files for $model ($dataset, window ${window}) to $DEVICE_NAME@$IP_ADDRESS ####"

                models_to_copy=(
                    "${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_lower_decoder.npy"
                    "${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_lower_encoder.npy"
                    "${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_recon_mean.npy"
                    "${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_recon_std.npy"
                    "${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_recon_threshold.npy"
                    "${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_upper_decoder.npy"
                    "${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_upper_encoder.npy"
                    "${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_us_mean_decoder.npy"
                    "${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_us_mean_encoder.npy"
                    "${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_us_std_decoder.npy"
                    "${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}_us_std_encoder.npy"
                    "${model}_${dataset}_ws_${window}_seed_${seed}_numvars_${num_vars}.pt"
                )

                for model_file in "${models_to_copy[@]}"; do
                    LOCAL_PATH="/home/db2003/Desktop/Amr/(TSE) RootCause-Analysis-Correlation-Attentive-Modeling/saved_models/${model_file}"
                    if [ -f "$LOCAL_PATH" ]; then
                        scp $SSH_OPTS "$LOCAL_PATH" $DEVICE_NAME@$IP_ADDRESS:${REMOTE_SAVED_MODELS}
                    else
                        echo "Warning: File $LOCAL_PATH does not exist, skipping."
                    fi
                done
            done
        done
    done
done