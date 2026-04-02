#!/bin/bash
set -e
set -eo pipefail
date
export HF_TOKEN=XXX  # todo: set your HuggingFace token
export HUGGING_FACE_HUB_TOKEN=XXX  # todo: set your HuggingFace token
export WANDB_API_KEY=XXX  # todo: set your W&B API key
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

# Multi-node setup (4 nodes x 8 H200 = 32 GPUs)
# Override via environment variables when launching on each node
NUM_NODES=${NUM_NODES:-4}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}  # Set to master node IP when launching
MASTER_PORT=${MASTER_PORT:-6099}

WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((WORLD_SIZE / MODEL_PARALLEL_SIZE / PIPELINE_MODEL_PARALLEL_SIZE))

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
RUN_NAME_BASE=Nemotron_EC_64_h2048_cfg2049_gbs288

# Dataset paths (Megatron indexed dataset prefixes)
NEMOTRON_HQ_ROOT="/path/to/nemontron-data/High-Quality"
NEMOTRON_DQA_ROOT="/path/to/nemontron-data/Diverse-QA"
# Blend weights: 50% High-Quality, 50% Diverse-QA (1:1)
HQ_BLEND_WEIGHT=${HQ_BLEND_WEIGHT:-50}
DQA_BLEND_WEIGHT=${DQA_BLEND_WEIGHT:-50}
# Use first N shards for validation to keep eval cost controlled.
VALID_PREFIX_COUNT=${VALID_PREFIX_COUNT:-8}

# Nemotron-CC-v2: HQ + Diverse-QA combined
TRAINING_TOKENS_PER_EPOCH=400000000000  # 400B tokens
EPOCHS=5  # 400B x 5 = 2T tokens
GLOBAL_BATCH_SIZE=288  # 4 nodes x 8 H200: DP=32, MBS=9, grad_accum=1
SEQ_LENGTH=2049
TOKENIZER="$ROOT_DIR/megatron/training/tokenizer/gptneox_tokenizer" # or change this to any huggingface tokenizer; local tokenizer is also supported
# On-the-fly eval debug switch:
#   OTF_EVAL_DEBUG=1 enables small-sample eval for quick pipeline checks.
#   OTF_EVAL_DEBUG_SAMPLES controls max samples per task.
#   OTF_EVAL_DEBUG_TASKS optionally overrides task list.
OTF_EVAL_DEBUG=${OTF_EVAL_DEBUG:-1}
OTF_EVAL_DEBUG_SAMPLES=${OTF_EVAL_DEBUG_SAMPLES:-512}
OTF_EVAL_DEBUG_TASKS=${OTF_EVAL_DEBUG_TASKS:-mmlu,arc_easy,arc_challenge,piqa,winogrande,copa,boolq,openbookqa,hellaswag}
OTF_EVAL_DATA_DIR=${OTF_EVAL_DATA_DIR:-$ROOT_DIR/tasks/on_the_fly_eval/local_data}
OTF_EVAL_FORCE_LOCAL=${OTF_EVAL_FORCE_LOCAL:-1}

if [ "$OTF_EVAL_FORCE_LOCAL" = "1" ]; then
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
fi

REQUIRED_OTF_EVAL_FILES=(
    mmlu_test.jsonl
    mmlu_dev.jsonl
    arc_easy_test.jsonl
    arc_easy_train.jsonl
    arc_challenge_test.jsonl
    arc_challenge_train.jsonl
    piqa_validation.jsonl
    piqa_train.jsonl
    winogrande_validation.jsonl
    winogrande_train.jsonl
    copa_validation.jsonl
    copa_train.jsonl
    boolq_validation.jsonl
    boolq_train.jsonl
    openbookqa_test.jsonl
    openbookqa_train.jsonl
    hellaswag_validation.jsonl
)
for f in "${REQUIRED_OTF_EVAL_FILES[@]}"; do
    if [ ! -f "$OTF_EVAL_DATA_DIR/$f" ]; then
        echo "Error: missing local on-the-fly eval file: $OTF_EVAL_DATA_DIR/$f"
        echo "Please run: python $ROOT_DIR/examples/dlm_training/scripts/download_on_the_fly_eval_data.py --output-dir $OTF_EVAL_DATA_DIR"
        exit 1
    fi
done

# Set RUN_NAME to include topk value for wandb (must be defined before creating directories)
MOE_ROUTER_TOPK=8
RUN_NAME=${RUN_NAME_BASE}_topk${MOE_ROUTER_TOPK}

cd $ROOT_DIR

TRAINING_TOKENS=$((TRAINING_TOKENS_PER_EPOCH * EPOCHS))

LOGS_PATH=${LOG_DIR:-"$CKPT_DIR/cache/difflm/logs"}
data_local_cache_dir="$CKPT_DIR/cache/difflm/data/data_cache/${RUN_NAME}"
mkdir -p $data_local_cache_dir
mkdir -p "${LOGS_PATH}/${RUN_NAME}"
mkdir -p "$CKPT_DIR/cache/difflm/training_checkpoints/${RUN_NAME}"

# Build per-split data args JSON with weighted HQ + Diverse-QA blend (1:1).
PER_SPLIT_DATA_ARGS_JSON="${data_local_cache_dir}/nemotron_blend_per_split_data_args.json"
mapfile -t HQ_PREFIXES < <(find "$NEMOTRON_HQ_ROOT" -maxdepth 1 -name '*_text_document.idx' | sort | sed 's/\.idx$//')
mapfile -t DQA_PREFIXES < <(find "$NEMOTRON_DQA_ROOT" -maxdepth 1 -name '*_text_document.idx' | sort | sed 's/\.idx$//')
if [ ${#HQ_PREFIXES[@]} -eq 0 ]; then
    echo "Error: No *_text_document.idx found under $NEMOTRON_HQ_ROOT"
    exit 1
fi
if [ ${#DQA_PREFIXES[@]} -eq 0 ]; then
    echo "Error: No *_text_document.idx found under $NEMOTRON_DQA_ROOT"
    exit 1
fi
if [ "$VALID_PREFIX_COUNT" -gt "${#HQ_PREFIXES[@]}" ]; then
    VALID_PREFIX_COUNT=${#HQ_PREFIXES[@]}
fi
python - "$PER_SPLIT_DATA_ARGS_JSON" "$VALID_PREFIX_COUNT" "$HQ_BLEND_WEIGHT" "$DQA_BLEND_WEIGHT" \
    "${#HQ_PREFIXES[@]}" "${HQ_PREFIXES[@]}" "${DQA_PREFIXES[@]}" <<'PY'
import json
import sys

out_path = sys.argv[1]
valid_prefix_count = int(sys.argv[2])
hq_weight = float(sys.argv[3])
dqa_weight = float(sys.argv[4])
num_hq = int(sys.argv[5])
all_prefixes = sys.argv[6:]
hq_prefixes = all_prefixes[:num_hq]
dqa_prefixes = all_prefixes[num_hq:]

# Per-prefix weight so that total HQ weight = hq_weight, total DQA weight = dqa_weight.
w_hq = hq_weight / len(hq_prefixes)
w_dqa = dqa_weight / len(dqa_prefixes)

# Megatron blend format: "weight1 prefix1 weight2 prefix2 ..."
train_parts = []
for p in hq_prefixes:
    train_parts.extend([f"{w_hq:.6f}", p])
for p in dqa_prefixes:
    train_parts.extend([f"{w_dqa:.6f}", p])

# Validation: use HQ shards with equal weights (enables multi-epoch cycling).
valid_shards = hq_prefixes[:valid_prefix_count]
w_valid = 1.0 / len(valid_shards)
valid_parts = []
for p in valid_shards:
    valid_parts.extend([f"{w_valid:.6f}", p])

data_args = {
    "train": " ".join(train_parts),
    "valid": " ".join(valid_parts),
    "test": None,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data_args, f, ensure_ascii=True)
PY
echo "Blend: HQ=${#HQ_PREFIXES[@]} shards (weight=${HQ_BLEND_WEIGHT}), DQA=${#DQA_PREFIXES[@]} shards (weight=${DQA_BLEND_WEIGHT})"
echo "Valid: ${VALID_PREFIX_COUNT} HQ shards (weighted, cycling enabled)"
echo "Per-split data args json: $PER_SPLIT_DATA_ARGS_JSON"

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
    --hidden-size 2048
    --num-attention-heads 16
    # --group-query-attention
    # --num-query-groups 1
    --ffn-hidden-size 1280
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
    --difflm-stable-training  # When enabled: training loss = CE * mask (no 1/t). When disabled: training loss = CE * mask / t (same as val, ELBO weighting)
    # On-the-fly evaluation (runs every 5k steps, independent of val loss eval at 1k steps)
    --use-on-the-fly-eval
    --on-the-fly-eval-shard-mode expert_dp
    --eval-before-train
    --on-the-fly-eval-interval 5000
    --on-the-fly-eval-batch-size 16
    --on-the-fly-eval-tasks mmlu,arc_easy,arc_challenge,piqa,winogrande,copa,hellaswag
    --on-the-fly-eval-mcq-shots 5
    --on-the-fly-eval-data-dir $OTF_EVAL_DATA_DIR
)
if [ "$OTF_EVAL_DEBUG" = "1" ]; then
    echo "On-the-fly eval debug mode enabled: max_samples=$OTF_EVAL_DEBUG_SAMPLES, tasks=$OTF_EVAL_DEBUG_TASKS"
    PLMT_ARGS+=(
        --on-the-fly-eval-max-samples $OTF_EVAL_DEBUG_SAMPLES
        --on-the-fly-eval-tasks $OTF_EVAL_DEBUG_TASKS
    )
fi

########################################################
# MoE-related settings
########################################################

# Auto-configure expert parallelism based on available GPUs
# NUM_EXPERTS must be divisible by GPUS_PER_NODE for even expert distribution
# 64 fine-grained experts (hidden_size=2048, ffn_hidden_size=1280), DeepSeek-MoE style
NUM_EXPERTS=64

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
    --moe-router-type expert_choice
    --moe-router-load-balancing-type softmax_expert_choice
    --moe-router-topk $MOE_ROUTER_TOPK
    # Router softmax: Using post-softmax (default) - softmax is applied after top-k selection
    # To use pre-softmax instead, add: --moe-router-pre-softmax
    # Per-layer logging: Enable per-layer metrics logging for MoE (auxiliary loss, z loss, dynamic topk metrics)
    --moe-per-layer-logging
    # Expert-choice: Set aux_loss_coeff to 0 (or omit) since expert-choice doesn't need load balancing loss
    # Expert-choice inherently balances load by design (experts choose tokens)
    --moe-aux-loss-coeff 0.0
    # Z-loss: Router stability regularization to encourage router logits to remain small
    # Helps stabilize training and improve router convergence (recommended value: 1e-3)
    --moe-z-loss-coeff 0.001
    # Token dispatcher: alltoall (recommended for EP)
    # See detailed explanation below for differences between dispatcher types
    --moe-token-dispatcher-type alltoall
    # --moe-layer-recompute: Enable activation recomputation for MoE layer to save memory
    # This trades computation time (~20-30% slower) for memory (~30-40% less memory)
    # Recomputes activations during backward pass instead of storing them
    # --moe-layer-recompute  # Disabled to improve training speed
    --moe-shared-expert-intermediate-size 2560  # Enable 2 shared experts (2 x ffn_hidden_size)
    --enable-scaling-factor  # Auto-compute scaling factors to balance shared and routing expert norms
    # --moe-expert-capacity-factor 1.0  # Uncomment to enable token dropping (useful for early training)
    # moe-expert-capacity-factor:
    #   Default: None (no token dropping, dropless mode)
    #   Formula: capacity = ceil((num_tokens / num_experts) * capacity_factor)
    #   When set: Tokens exceeding capacity are dropped (based on routing probabilities)
    #   Why related to OOM: Training early (first 100-300 steps), router is untrained,
    #     causing severe load imbalance. Some experts may receive too many tokens,
    #     leading to OOM. Setting capacity limits max tokens per expert.
    #   Typical values: 1.0-1.25 for early training, remove after ~200 steps when router stabilizes
)

TARGET_TRAINING_TOKENS=$TRAINING_TOKENS
TOKENS_PER_BATCH=$((GLOBAL_BATCH_SIZE * SEQ_LENGTH))
TRAIN_ITERS=$((TARGET_TRAINING_TOKENS / TOKENS_PER_BATCH))
SAVE_INTERVAL=40000
LOG_INTERVAL=50
NON_PERSISTENT_SAVE_INTERVAL=$((TRAIN_ITERS * 2))
EVAL_INTERVAL=1000  # Evaluate every 1000 steps
TARGET_VAL_TOKENS=35000000 # 35M tokens for validation
TOKENS_PER_VAL_BATCH=$((GLOBAL_BATCH_SIZE * SEQ_LENGTH))
VAL_ITERS=$((TARGET_VAL_TOKENS / TOKENS_PER_VAL_BATCH))

# Auto-configure micro batch size to keep grad accumulation fixed.
TARGET_GRAD_ACCUM=1
if [ $((GLOBAL_BATCH_SIZE % (DATA_PARALLEL_SIZE * TARGET_GRAD_ACCUM))) -ne 0 ]; then
    echo "Error: GLOBAL_BATCH_SIZE ($GLOBAL_BATCH_SIZE) must be divisible by DATA_PARALLEL_SIZE ($DATA_PARALLEL_SIZE) * TARGET_GRAD_ACCUM ($TARGET_GRAD_ACCUM)"
    exit 1
fi
MICRO_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / DATA_PARALLEL_SIZE / TARGET_GRAD_ACCUM))
ACTUAL_GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE)))
echo "Batch config: GBS=$GLOBAL_BATCH_SIZE, DP=$DATA_PARALLEL_SIZE, MBS=$MICRO_BATCH_SIZE, grad_accum=$ACTUAL_GRAD_ACCUM"

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
    # --recompute-activations  # Activation recomputation to reduce memory usage (disabled to improve training speed)
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
    --per-split-data-args-path $PER_SPLIT_DATA_ARGS_JSON
    --data-cache-path $data_local_cache_dir
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER
    # --reset-dataloader  # Reset consumed_samples to 0 for new data blend (remove after first resume)
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
    --wandb-project MegaDLM_Nemotron_H200
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

NEED_RUN_TRAINING=true
if [ -f "$CHECKPOINT_PATH/latest_checkpointed_iteration.txt" ]; then
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
