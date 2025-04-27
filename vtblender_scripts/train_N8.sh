#!/bin/bash
#SBATCH -A your_account
#SBATCH -p your_partition
#SBATCH -N 8 # number of nodes
#SBATCH -t 4:00:00              # wall time
#SBATCH --time-min 4:00:00 
#SBATCH --ntasks-per-node=8    # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH -J "speechlm_train"            # job name (<< CHANGE ! >>)
#SBATCH --output=slurm_logs/%x=%j --error=slurm_logs/%x=%j

set -x

TOTAL_NUM_GPUS=`expr $SLURM_GPUS_PER_NODE \* $SLURM_JOB_NUM_NODES`
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=`expr $MICRO_BATCH_SIZE \* $TOTAL_NUM_GPUS`


PROJECT_NAME=vtblender_N${SLURM_JOB_NUM_NODES}
SCRIPT_PATH=/code/examples/multimodal/speech_llm/modular_audio_gpt_train.py
CONFIG_PATH=/code/examples/multimodal/speech_llm/conf/full_canary
CONFIG_NAME=singlestage_gemma_enc-ft_adp-ft_llm-lora_lr1e-4_max100k.yaml

EXP_NAME=${CONFIG_NAME}

# NOTE: these depend on your file system structure
# Please customize them according to your file system structure
SLURM_ACCOUNT=yours # <Make sure you dont override SLURM_ACCOUNT!>
USERID=yours
LUSTRE_ACCOUNT_PREFIX=/yours/${SLURM_ACCOUNT}  
WANDB='your_wandb_key'

CONTAINER=/path/to/your/nemo-main-24jun7.sqsh

CODE_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/code/NeMo
DATA_DIR=${LUSTRE_ACCOUNT_PREFIX}/your/data
RESULTS_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/results/$PROJECT_NAME/$EXP_NAME
ALL_RESULTS_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/results

mkdir -p ${RESULTS_DIR}
OUTFILE=${RESULTS_DIR}/slurm-%j-%n.log
ERRFILE=${RESULTS_DIR}/error-%j-%n.log

# Please customize the following according to your file system structure
MOUNTS="--container-mounts=$CODE_DIR:/code,$RESULTS_DIR:/results,$DATA_DIR:/data,$ALL_RESULTS_DIR:/all_results"

export HYDRA_FULL_ERROR=1

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& nvidia-smi \
&& export WANDB_API_KEY=${WANDB} \
&& wandb login ${WANDB} \
&& cd /code \
&& git rev-parse HEAD \
&& pip show torch \
&& export PYTHONPATH="/code/.:${PYTHONPATH}" \
&& export LHOTSE_DILL_ENABLED=1 \
&& export NVTE_MASKED_SOFTMAX_FUSION=0 \
&& export NVTE_FLASH_ATTN=1 \
&& export NVTE_FUSED_ATTN=0 \
&& export HF_HOME="/hfcache/" \
&& export TORCH_HOME="/hfcache/torch" \
&& export NEMO_CACHE_DIR="/hfcache/torch/nemo" \
&& export HF_DATASETS_CACHE="/hfcache/datasets" \
&& export TOKENIZERS_PARALLELISM=false \
&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3 \
&& python -c 'import pytorch_lightning as ptl; print(ptl.__version__)' \
&& echo "Starting training" \
&& HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 python ${SCRIPT_PATH} \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    name=$EXP_NAME \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    model.global_batch_size=$GLOBAL_BATCH_SIZE \
    model.micro_batch_size=$MICRO_BATCH_SIZE \
    exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME}
EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
set +x
