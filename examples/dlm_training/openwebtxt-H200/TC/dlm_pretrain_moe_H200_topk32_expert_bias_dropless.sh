#!/bin/bash
set -e
set -eo pipefail
date
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

########################################################
# Distributed settings - Auto-configured from CUDA_VISIBLE_DEVICES
########################################################

# Count number of GPUs from CUDA_VISIBLE_DEVICES
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Error: CUDA_VISIBLE_DEVICES is not set"
    exit 1
fi

# Count GPUs: split by comma and count
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Detected $GPUS_PER_NODE GPUs from CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Single node setup (can be overridden by environment variables)
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6099}

WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((WORLD_SIZE / MODEL_PARALLEL_SIZE / PIPELINE_MODEL_PARALLEL_SIZE))

CONVERT_CHECKPOINT_ONLY=${1}
CONVERT_CKPT_STEP=${2}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    --node_rank $NODE_RANK
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $MODEL_PARALLEL_SIZE
	--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
)

########################################################
# Training-related settings
########################################################

# Hardcoded paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
CKPT_DIR="$ROOT_DIR"  # Checkpoint directory
LOG_DIR="$ROOT_DIR/logs"  # Log directory

# todo: reset these values
RUN_NAME_BASE=TC_512_loss_free_bias_dropless

# Dataset paths
# Note: Run examples/dlm_training/preprocess_openwebtext.sh first to download and preprocess data
train_data_prefix="/path/to/openwebtext/train_processed_text_document"
valid_data_prefix="/path/to/openwebtext/valid_processed_text_document"

# Use full dataset: Set TRAINING_TOKENS_PER_EPOCH to a large value to ensure using entire dataset
# Training will automatically stop when dataset is exhausted
TRAINING_TOKENS_PER_EPOCH=20000000000  # Large value (20B tokens) to ensure using entire dataset
EPOCHS=10  # Train for 10 epochs to use full dataset 10 times
GLOBAL_BATCH_SIZE=256  # 8-GPU EP: 32 * 8 = 256
SEQ_LENGTH=513
TOKENIZER="$ROOT_DIR/megatron/training/tokenizer/gptneox_tokenizer" # or change this to any huggingface tokenizer; local tokenizer is also supported

# Set RUN_NAME to include topk value for wandb (must be defined before creating directories)
MOE_ROUTER_TOPK=32
RUN_NAME=${RUN_NAME_BASE}_topk${MOE_ROUTER_TOPK}

cd $ROOT_DIR

TRAINING_TOKENS=$((TRAINING_TOKENS_PER_EPOCH * EPOCHS))

LOGS_PATH=${LOG_DIR:-"$CKPT_DIR/cache/difflm/logs"}
data_local_cache_dir="$CKPT_DIR/cache/difflm/data/data_cache/${RUN_NAME}"
mkdir -p $data_local_cache_dir
mkdir -p "${LOGS_PATH}/${RUN_NAME}"
mkdir -p "$CKPT_DIR/cache/difflm/training_checkpoints/${RUN_NAME}"

GPT_MODEL_ARGS=(
    --seq-length $SEQ_LENGTH
    --attention-backend flash
    --attention-softmax-in-fp32
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --no-bias-gelu-fusion
    --no-bias-dropout-fusion
    --no-bias-swiglu-fusion

    --num-layers 16
    --hidden-size 512
    --num-attention-heads 16
    # --group-query-attention
    # --num-query-groups 1
    --ffn-hidden-size 384
    --swiglu
    # --vocab-size: Not needed for HuggingFaceTokenizer, vocab size is auto-detected from tokenizer
    --normalization RMSNorm
    --max-position-embeddings $SEQ_LENGTH
    --norm-epsilon 1e-6
    --rotary-base 10000
    --disable-bias-linear
    # --add-qkv-bias
    --position-embedding-type rope
    --qk-layernorm
    --untie-embeddings-and-output-weights
)

PLMT_ARGS=(
    --model-running-mode difflm-noshift
    --base-model vanilla
    --mask-token 50277
    --attention-mask-type no_mask
    --difflm-varilen-prob 0.0  # Disable variable length sequences
)

########################################################
# MoE-related settings
########################################################

# Auto-configure expert parallelism based on available GPUs
# NUM_EXPERTS must be divisible by GPUS_PER_NODE for even expert distribution
# Changed to 512 experts, each expert has 1/8 parameters (hidden_size=512, ffn_hidden_size=384)
NUM_EXPERTS=512

# Check if num_experts is divisible by GPUS_PER_NODE
if [ $((NUM_EXPERTS % GPUS_PER_NODE)) -ne 0 ]; then
    echo "Error: NUM_EXPERTS ($NUM_EXPERTS) must be divisible by GPUS_PER_NODE ($GPUS_PER_NODE)"
    echo "Please adjust NUM_EXPERTS or CUDA_VISIBLE_DEVICES to satisfy this requirement."
    exit 1
fi

EXPERT_MODEL_PARALLEL_SIZE=$GPUS_PER_NODE

echo "MoE Configuration:"
echo "  NUM_EXPERTS=$NUM_EXPERTS"
echo "  EXPERT_MODEL_PARALLEL_SIZE=$EXPERT_MODEL_PARALLEL_SIZE (auto-configured from $GPUS_PER_NODE GPUs)"
echo "  Local experts per GPU: $((NUM_EXPERTS / EXPERT_MODEL_PARALLEL_SIZE))"

MOE_ARGS=(
    --num-experts $NUM_EXPERTS
    --moe-grouped-gemm
    --expert-model-parallel-size $EXPERT_MODEL_PARALLEL_SIZE
    --moe-router-type topk
    --moe-router-load-balancing-type loss_free
    --moe-router-topk $MOE_ROUTER_TOPK
    # Router softmax: Using post-softmax (default) - softmax is applied after top-k selection
    # To use pre-softmax instead, add: --moe-router-pre-softmax
    # Per-layer logging: Enable per-layer metrics logging for MoE (auxiliary loss, z loss, dynamic topk metrics)
    --moe-per-layer-logging
    # Loss-free load balancing (arXiv:2408.15664): uses dynamic expert biases instead of aux loss
    # Must set --moe-aux-loss-coeff to 0 (incompatible with auxiliary loss)
    --moe-aux-loss-coeff 0.0
    --moe-expert-bias-update-rate 0.001
    # Z-loss: Router stability regularization to encourage router logits to remain small
    # Helps stabilize training and improve router convergence (recommended value: 1e-3)
    --moe-z-loss-coeff 0.001
    # Token dispatcher: alltoall (recommended for EP)
    # See detailed explanation below for differences between dispatcher types
    --moe-token-dispatcher-type alltoall
    # --moe-layer-recompute: Enable activation recomputation for MoE layer to save memory
    # This trades computation time (~20-30% slower) for memory (~30-40% less memory)
    # Recomputes activations during backward pass instead of storing them
    # --moe-layer-recompute
    --moe-shared-expert-intermediate-size 768  # Enable 2 shared experts (2 × ffn_hidden_size)
    --enable-scaling-factor  # Auto-compute scaling factors to balance shared and routing expert norms
    # Capacity buffer enabled: Set capacity_factor to 1.25 to allow 25% buffer for load balancing
    # This limits the maximum number of tokens each expert can process
    # Formula: capacity = ceil((num_tokens / num_experts) * capacity_factor)
    # Tokens exceeding capacity will be dropped (based on routing probabilities)
    # Useful for preventing OOM during early training when router is untrained
    # Can be removed after ~200 steps when router stabilizes
    # dropless: no capacity factor (no token dropping)
)

TARGET_TRAINING_TOKENS=$TRAINING_TOKENS
TOKENS_PER_BATCH=$((GLOBAL_BATCH_SIZE * SEQ_LENGTH))
TRAIN_ITERS=$((TARGET_TRAINING_TOKENS / TOKENS_PER_BATCH))
SAVE_INTERVAL=$((TRAIN_ITERS / EPOCHS))
LOG_INTERVAL=50
NON_PERSISTENT_SAVE_INTERVAL=$((TRAIN_ITERS * 2))
EVAL_INTERVAL=1000  # Evaluate every 1000 steps
TARGET_VAL_TOKENS=10000000 # 10M tokens (use full validation data, ~9.19M tokens)
TOKENS_PER_VAL_BATCH=$((GLOBAL_BATCH_SIZE * SEQ_LENGTH))
VAL_ITERS=$((TARGET_VAL_TOKENS / TOKENS_PER_VAL_BATCH))

# Hardcoded micro batch size
MICRO_BATCH_SIZE=32  # 8-GPU EP: 256 / 8 = 32

TRAINING_ARGS=(
    --seed 1234
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS
    --weight-decay 0.1
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.02
    --clip-grad 1.0 
    --bf16
    --lr 0.0002
    --min-lr 0.00002
    --lr-decay-style WSD
    --lr-warmup-iters 2000
    --lr-decay-iters $TRAIN_ITERS # this includes the warmup phase
    --lr-wsd-decay-style exponential
    --lr-wsd-decay-iters 1000
    # --use-checkpoint-args
    --use-distributed-optimizer
    --num-distributed-optimizer-instances $NUM_NODES
    --rerun-mode validate_results
    --overlap-param-gather
    --overlap-grad-reduce
    --distributed-timeout-minutes 60
    # --recompute-activations  # Activation recomputation to reduce memory usage
    # --tp-comm-overlap #todo turn on this when using TP > 1 and SP > 1
    # --num-layers-per-virtual-pipeline-stage # todo specify this when PP > 1
)

# add these to the training args for stable training (store master weights, weight gradients, and optimizer states in fp32)
# --accumulate-allreduce-grads-in-fp32
# --use-precision-aware-optimizer alongside:
# --main-params-dtype=fp32
# --main-grads-dtype=fp32
# --exp-avg-dtype=fp32
# --exp-avg-sq-dtype=fp32

DATA_ARGS=(
    --train-data-path $train_data_prefix
    --valid-data-path $valid_data_prefix
    --data-cache-path $data_local_cache_dir
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER
)

# Force online mode and export to environment
export WANDB_MODE="online"

CHECKPOINT_PATH=$CKPT_DIR/cache/difflm/training_checkpoints/${RUN_NAME}
EVAL_AND_LOGGING_ARGS=(
    --log-interval $LOG_INTERVAL
    --log-params-norm
    --log-num-zeros-in-grad
    --log-throughput
    --log-progress
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-memory-to-tensorboard
    --save-interval $SAVE_INTERVAL
    --non-persistent-save-interval $NON_PERSISTENT_SAVE_INTERVAL
    --non-persistent-ckpt-type global
    --non-persistent-global-ckpt-dir $CHECKPOINT_PATH/non_persistent
    --ckpt-format torch_dist
    --auto-detect-ckpt-format
    --ckpt-fully-parallel-load
    # --async-save
    --eval-interval $EVAL_INTERVAL 
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --eval-iters $VAL_ITERS
    --tensorboard-dir ${LOGS_PATH}/${RUN_NAME}/tensorboard
    --wandb-project MegaDLM_H200
    --wandb-exp-name $RUN_NAME
    # --wandb-run-id $RUN_NAME  # Commented out to avoid step conflicts when resuming training
    --wandb-save-dir ${LOGS_PATH}/${RUN_NAME}/wandb
    --wandb-mode $WANDB_MODE
    --wandb-tags 'dlm'
    --wandb-notes 'training'
    --wandb-resume allow
)

########################################################
# Run the training script
########################################################

if [ "$CONVERT_CHECKPOINT_ONLY" == "convert_ckpt" ]; then
    echo "Skipping training and converting checkpoint only"
elif [ "$CONVERT_CHECKPOINT_ONLY" == "" ]; then
    NEED_RUN_TRAINING=true
    if [ -f "$CHECKPOINT_PATH/latest_checkpointed_iteration.txt" ]; then
        # if the last checkpoint of the training exists, we will not run the training script
        if [ "$TRAIN_ITERS" -eq "$(cat $CHECKPOINT_PATH/latest_checkpointed_iteration.txt)" ]; then
            echo "The last checkpoint of the training exists. We will not run the training script. Skipping ..."
            NEED_RUN_TRAINING=false
        else
            echo "No last-iteration checkpoint found. Running the training script."
        fi
    else
        echo "No last-iteration checkpoint found. Running the training script."
    fi
    if [ "$NEED_RUN_TRAINING" == "true" ]; then
        torchrun ${DISTRIBUTED_ARGS[@]} pretrain_difflm.py \
            ${GPT_MODEL_ARGS[@]} \
            ${MOE_ARGS[@]} \
            ${TRAINING_ARGS[@]} \
            ${MODEL_PARALLEL_ARGS[@]} \
            ${DATA_ARGS[@]} \
            ${EVAL_AND_LOGGING_ARGS[@]} \
            ${PLMT_ARGS[@]}
    fi
else
    echo "Invalid value for CONVERT_CHECKPOINT_ONLY: $CONVERT_CHECKPOINT_ONLY. Please use 'convert_ckpt' or leave blank."
    exit 1
fi

########################################################
# convert the checkpoints
########################################################

if [ "$CONVERT_CHECKPOINT_ONLY" == "convert_ckpt" ]; then
    # Master node actions
    echo "Master node (Rank $NODE_RANK): Preparing for checkpoint conversion process."

    echo "Converting checkpoints to torch format without optimizer state."

    # GPUS_PER_NODE=$((MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE))
    CUDA_VISIBLE_DEVICES_CONVERT_CKPT=$(seq -s, 0 $((GPUS_PER_NODE_CONVERT_CKPT-1)))

    # make sure NUM_NODES_CONVERT_CKPT equals to NUM_NODES
    if [ "$NUM_NODES_CONVERT_CKPT" != "$NUM_NODES" ]; then
        echo "Error: NUM_NODES_CONVERT_CKPT ($NUM_NODES_CONVERT_CKPT) does not equal to NUM_NODES ($NUM_NODES) that's actually launched."
        exit 1
    fi
    # make sure GPUS_PER_NODE_CONVERT_CKPT is less than 8
    if [ "$GPUS_PER_NODE_CONVERT_CKPT" -gt 8 ]; then
        echo "Error: GPUS_PER_NODE_CONVERT_CKPT ($GPUS_PER_NODE_CONVERT_CKPT) is greater than 8."
        exit 1
    fi

    if [ "$NUM_NODES_CONVERT_CKPT" == "1" ]; then
        MASTER_ADDR=localhost
        MASTER_PORT=6099
        NODE_RANK=0
    fi

    DISTRIBUTED_ARGS=(
        --nproc_per_node $GPUS_PER_NODE_CONVERT_CKPT
        --nnodes $NUM_NODES_CONVERT_CKPT
        --master_addr $MASTER_ADDR 
        --master_port $MASTER_PORT
        --node_rank $NODE_RANK
    )

    # reset the load dir to the training checkpoint dir
    LOAD=$CHECKPOINT_PATH

    if [ "$CONVERT_CKPT_STEP" == "" ]; then
        echo "No checkpoint step provided. Converting all checkpoints."
        
        non_persistent_arg1="--non-persistent-global-ckpt-dir"
        non_persistent_arg2="$CHECKPOINT_PATH/non_persistent"
        EVAL_AND_LOGGING_ARGS=("${EVAL_AND_LOGGING_ARGS[@]/$non_persistent_arg1/}")
        EVAL_AND_LOGGING_ARGS=("${EVAL_AND_LOGGING_ARGS[@]/$non_persistent_arg2/}")

        # Remove --non-persistent-ckpt-type and its value "global" by rebuilding the array
        new_args=()
        skip_next=0
        for arg in "${EVAL_AND_LOGGING_ARGS[@]}"; do
            if [[ "$skip_next" -eq 1 ]]; then
                skip_next=0 # Reset flag and skip this element (the value "global")
                continue
            fi
            if [[ "$arg" == "--non-persistent-ckpt-type" ]]; then
                skip_next=1 # Set flag to skip the next element
                continue    # Skip the current element ("--non-persistent-ckpt-type")
            fi
            # Keep the element if it's not one we want to remove
            new_args+=("$arg")
        done
        EVAL_AND_LOGGING_ARGS=("${new_args[@]}")

        load_arg1="$LOAD"
        EVAL_AND_LOGGING_ARGS=("${EVAL_AND_LOGGING_ARGS[@]/$load_arg1/$CHECKPOINT_PATH}")

        # get all ckpt steps
        ALL_CKPT_STEPS=()
        for d in $(find "$CHECKPOINT_PATH" -maxdepth 1 -type d -name "iter_*" -print); do
            ckpt_name=$(basename "$d")
            step_num_with_zeros=${ckpt_name#iter_}
            # Remove leading zeros
            step_num=$(echo "$step_num_with_zeros" | sed 's/^0*//')
            if [[ -n "$step_num" ]]; then # ensure step_num is not empty after removing zeros (e.g. "0000000")
                ALL_CKPT_STEPS+=("$step_num")
            fi
        done
        # Sort the array numerically
        IFS=$'\n' ALL_CKPT_STEPS=($(printf "%s\n" "${ALL_CKPT_STEPS[@]}" | sort -n))
        unset IFS
        # Optionally subsample checkpoints via SKIP_K_CHECKPOINTS_TO_CONVERT (default 0)
        SKIP_K_CHECKPOINTS_TO_CONVERT="${SKIP_K_CHECKPOINTS_TO_CONVERT:-0}"
        if ! [[ "$SKIP_K_CHECKPOINTS_TO_CONVERT" =~ ^[0-9]+$ ]]; then
            SKIP_K_CHECKPOINTS_TO_CONVERT=0
        fi
        if [ ${#ALL_CKPT_STEPS[@]} -gt 0 ] && [ "$SKIP_K_CHECKPOINTS_TO_CONVERT" -gt 0 ]; then
            stride=$((SKIP_K_CHECKPOINTS_TO_CONVERT + 1))
            filtered_steps=()
            for ((i=0; i<${#ALL_CKPT_STEPS[@]}; i+=stride)); do
                # if the checkpoint step is greater than the latest checkpoint step, skip it
                if [ "${ALL_CKPT_STEPS[i]}" -gt "$(cat $CHECKPOINT_PATH/latest_checkpointed_iteration.txt)" ]; then
                    continue
                fi
                filtered_steps+=("${ALL_CKPT_STEPS[i]}")
            done
            ALL_CKPT_STEPS=("${filtered_steps[@]}")
            unset filtered_steps
        fi
        # If no checkpoints are found, exit or handle as an error
        if [ ${#ALL_CKPT_STEPS[@]} -eq 0 ]; then
            echo "No checkpoint steps found in $CHECKPOINT_PATH matching iter_*"
            # Do NOT create flag here, as this is an error condition (exit 1)
            exit 1 
        fi
    else

        echo "Trying to load checkpoint step $CONVERT_CKPT_STEP from normal checkpoints. Ignoring the non-persistent checkpoints and loading from $CHECKPOINT_PATH."
        # FIRST_STAGE_SAVE_NAME=${FIRST_STAGE_SAVE_NAME}_ckptstep_${CONVERT_CKPT_STEP}
        non_persistent_arg1="--non-persistent-global-ckpt-dir"
        non_persistent_arg2="$CHECKPOINT_PATH/non_persistent"
        EVAL_AND_LOGGING_ARGS=("${EVAL_AND_LOGGING_ARGS[@]/$non_persistent_arg1/}")
        EVAL_AND_LOGGING_ARGS=("${EVAL_AND_LOGGING_ARGS[@]/$non_persistent_arg2/}")

        # Remove --non-persistent-ckpt-type and its value "global" by rebuilding the array
        new_args=()
        skip_next=0
        for arg in "${EVAL_AND_LOGGING_ARGS[@]}"; do
            if [[ "$skip_next" -eq 1 ]]; then
                skip_next=0 # Reset flag and skip this element (the value "global")
                continue
            fi
            if [[ "$arg" == "--non-persistent-ckpt-type" ]]; then
                skip_next=1 # Set flag to skip the next element
                continue    # Skip the current element ("--non-persistent-ckpt-type")
            fi
            # Keep the element if it's not one we want to remove
            new_args+=("$arg")
        done
        EVAL_AND_LOGGING_ARGS=("${new_args[@]}")

        load_arg1="$LOAD"
        EVAL_AND_LOGGING_ARGS=("${EVAL_AND_LOGGING_ARGS[@]/$load_arg1/$CHECKPOINT_PATH}")
        
        ALL_CKPT_STEPS=($CONVERT_CKPT_STEP)
    fi

    for CKPT_STEP_TO_CONVERT in "${ALL_CKPT_STEPS[@]}"; do

        # if the checkpoint step has already been converted, skip it
        if [ -f "$CKPT_DIR/cache/difflm/converted_checkpoints/${RUN_NAME}/ckptstep_${CKPT_STEP_TO_CONVERT}/torch/latest_checkpointed_iteration.txt" ]; then
            echo "Torch checkpoint step $CKPT_STEP_TO_CONVERT has already been converted. Skipping ..."
            continue
        fi

        echo "Converting checkpoint step $CKPT_STEP_TO_CONVERT to torch format without optimizer state."

        CONVERT_CKPT_ARGS=(
            --no-load-optim
            --no-save-optim
            --no-save-rng
            --no-load-rng
            --ckpt-convert-format torch
            --ckpt-convert-save $CKPT_DIR/cache/difflm/converted_checkpoints/${RUN_NAME}/ckptstep_${CKPT_STEP_TO_CONVERT}
            --tensorboard-dir ${LOG_DIR}/${RUN_NAME}/convert_ckpt_temp
            --ckpt-step $CKPT_STEP_TO_CONVERT
        )


        EVAL_AND_LOGGING_ARGS=("${EVAL_AND_LOGGING_ARGS[@]/--log-progress/}")
        EVAL_AND_LOGGING_ARGS=("${EVAL_AND_LOGGING_ARGS[@]/--async-save/}")
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_CONVERT_CKPT torchrun ${DISTRIBUTED_ARGS[@]} pretrain_difflm.py \
            ${GPT_MODEL_ARGS[@]} \
            ${MOE_ARGS[@]} \
            ${TRAINING_ARGS[@]} \
            ${MODEL_PARALLEL_ARGS[@]} \
            ${DATA_ARGS[@]} \
            ${EVAL_AND_LOGGING_ARGS[@]} \
            ${PLMT_ARGS[@]} \
            ${CONVERT_CKPT_ARGS[@]}
    done

    if [ "$NODE_RANK" == "0" ]; then
        for CKPT_STEP_TO_CONVERT in "${ALL_CKPT_STEPS[@]}"; do

            # if the checkpoint step has already been converted, skip it
            if [ -f "$CKPT_DIR/cache/difflm/converted_checkpoints/${RUN_NAME}/ckptstep_${CKPT_STEP_TO_CONVERT}/hf/tokenizer.json" ]; then
                echo "HF checkpoint step $CKPT_STEP_TO_CONVERT has already been converted. Skipping ..."
                continue
            fi

            echo "Converting checkpoint step $CKPT_STEP_TO_CONVERT to hf format."

            # convert the checkpoint to hf format
            MODEL_NAME="gptneox_1.7b_dlm"
        
            python tools/weights_conversion/megatron_to_hf_te.py \
                --model $MODEL_NAME \
                --untieembed \
                --input_dir $CKPT_DIR/cache/difflm/converted_checkpoints/${RUN_NAME}/ckptstep_${CKPT_STEP_TO_CONVERT}/torch \
                --output_dir $CKPT_DIR/cache/difflm/converted_checkpoints/${RUN_NAME}/ckptstep_${CKPT_STEP_TO_CONVERT}/hf \
                --num_output_shards 4 \
                --rope_theta 1e4 # todo: adjust accordingly
        done
    else
        # Worker nodes (NODE_RANK != "0")
        echo "Worker node (Rank $NODE_RANK): Waiting for master node (Rank 0) to complete checkpoint conversion..."
        WAIT_INTERVAL=18000 # seconds
        sleep $WAIT_INTERVAL
    fi

    echo "All checkpoints have been converted to torch format without optimizer state."
    echo "Master node: Main checkpoint conversion process completed successfully."
    echo "Master node: Conversion complete. Exiting."
fi