#!/bin/bash
#SBATCH -A your_account_name
#SBATCH -p your_partition_name
#SBATCH -N 1 # number of nodes
#SBATCH -t 4:00:00              # wall time
#SBATCH --time-min 04:00:00  
#SBATCH --ntasks-per-node=1    # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=1
#SBATCH --mem=250G
#SBATCH --cpus-per-task 1
#SBATCH -J "vtblender_test"            # job name (<< CHANGE ! >>)
#SBATCH --output=slurm_logs/%x=%j --error=slurm_logs/%x=%j

set -x

# NOTE: these depend on your file system structure
# Please customize them according to your file system structure
SLURM_ACCOUNT=yours
USERID=yours
LUSTRE_ACCOUNT_PREFIX=/path/to/your/${SLURM_ACCOUNT}
RESULTS_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/results/

BATCH_SIZE=8
USE_GREEDY=True

ALM_CKPT=/path/to/your/model.ckpt

# below is the path to the config used during training
ALM_YAML=version_0/hparams.yaml

TEST_OUTPUT_DIR=$(dirname $ALM_CKPT)/../test_sqa_greedy$USE_GREEDY
mkdir -p ${RESULTS_DIR}/${TEST_OUTPUT_DIR}

PRETRAINED_AUDIO_MODEL="/path/to/pretrained/canary-1b.nemo"
MEGATRON_CKPT=/path/to/your/gemma-chat.llm.nemo

CONTAINER=/path/to/your/nemo-main-24jun7.sqsh

SCRIPT_PATH=/code/examples/multimodal/speech_llm/modular_audio_gpt_eval.py
CONFIG_PATH=/code/examples/multimodal/speech_llm/conf/gemma_chat
CONFIG_NAME=eval

CODE_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/code/NeMo
DATA_DIR=/path/to/your/data

TEST_CFG=/path/to/your/test_config.yaml

ALL_RESULTS_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/results

OUTFILE=${RESULTS_DIR}/${TEST_OUTPUT_DIR}/slurm-%j-%n.log
ERRFILE=${RESULTS_DIR}/${TEST_OUTPUT_DIR}/error-%j-%n.log

MOUNTS="--container-mounts=$CODE_DIR:/code,$RESULTS_DIR:/results,$DATA_DIR:/data,$ALL_RESULTS_DIR:/all_results"

export HYDRA_FULL_ERROR=1

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& nvidia-smi \
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
&& echo "Starting testing" \
&& HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 python ${SCRIPT_PATH} \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.restore_from_path=$MEGATRON_CKPT \
    model.peft.restore_from_path=/results/$ALM_CKPT \
    model.peft.restore_from_hparams_path=/results/$(dirname $ALM_CKPT)/../$ALM_YAML \
    ++model.pretrained_audio_model=$PRETRAINED_AUDIO_MODEL \
    ++model.data.test_ds.input_cfg=$TEST_CFG \
    model.global_batch_size=$BATCH_SIZE \
    model.micro_batch_size=$BATCH_SIZE \
    model.data.test_ds.global_batch_size=$BATCH_SIZE \
    model.data.test_ds.micro_batch_size=$BATCH_SIZE \
    model.data.test_ds.tokens_to_generate=512 \
    ++model.data.test_ds.batch_size=$BATCH_SIZE \
    ++model.data.test_ds.shuffle=False \
    ++model.data.test_ds.shuffle_buffer_size=1 \
    ++model.data.test_ds.metric.name="rouge" \
    ++inference.greedy=$USE_GREEDY \
    ++inference.top_k=50 \
    ++inference.top_p=0.95 \
    ++inference.temperature=0.4 \
    ++inference.repetition_penalty=1.2 \
    ++model.data.test_ds.output_dir=/results/$TEST_OUTPUT_DIR
EOF


srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
