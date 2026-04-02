#!/bin/bash
set -e
set -eo pipefail

########################################################
# OpenWebText Dataset Download and Preprocessing Script
########################################################

# Resolve project root. Prefer a valid PROJECT_DIR that contains megatron/tools.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")
if [ -n "$PROJECT_DIR" ] && [ -d "$PROJECT_DIR/megatron" ] && [ -d "$PROJECT_DIR/tools" ]; then
    ROOT_DIR="$PROJECT_DIR"
fi
DATASET_OUTPUT_DIR="${DATASET_OUTPUT_DIR:-/path/to/openwebtext}"
TOKENIZER="$ROOT_DIR/megatron/training/tokenizer/gptneox_tokenizer"

# Configuration
DATASET_NAME=${DATASET_NAME:-"Skylion007/openwebtext"}
MAX_SAMPLES=${MAX_SAMPLES:-}  # Set to limit samples (e.g., 1000000), empty for all
USE_STREAMING=${USE_STREAMING:-true}  # Use streaming for large datasets
NUM_PROC=${NUM_PROC:-1}  # Number of processes for processing (set to 1 to avoid transformers import issues)
TOKENIZER_TYPE=${TOKENIZER_TYPE:-"HuggingFaceTokenizer"}

cd $ROOT_DIR

echo "=========================================="
echo "OpenWebText Dataset Download and Preprocessing"
echo "=========================================="
echo "Dataset: $DATASET_NAME"
echo "Output directory: $DATASET_OUTPUT_DIR"
echo "Tokenizer: $TOKENIZER"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "Streaming: $USE_STREAMING"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$DATASET_OUTPUT_DIR"

########################################################
# Step 1: Download and convert to JSONL (skip if files exist)
########################################################
echo "Step 1: Checking for existing JSONL files..."
echo "----------------------------------------"

if [ -f "$DATASET_OUTPUT_DIR/train.jsonl" ] && [ -f "$DATASET_OUTPUT_DIR/valid.jsonl" ]; then
    echo "✓ Found existing train.jsonl and valid.jsonl"
    echo "  Skipping download step..."
    echo ""
    echo "Step 1 skipped (files already exist)!"
    echo ""
else
    echo "JSONL files not found. Downloading dataset from HuggingFace..."
    echo "----------------------------------------"
    
    python $ROOT_DIR/tools/download_and_preprocess_openwebtext.py \
        --dataset-name "$DATASET_NAME" \
        --output-dir "$DATASET_OUTPUT_DIR" \
        --split train \
        ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES} \
        ${USE_STREAMING:+--streaming} \
        --num-proc $NUM_PROC
    
    if [ $? -ne 0 ]; then
        echo "Error: Dataset download failed!"
        exit 1
    fi
    
    echo ""
    echo "Step 1 completed successfully!"
    echo ""
fi

########################################################
# Step 2: Preprocess training data
########################################################
echo "Step 2: Preprocessing training data..."
echo "----------------------------------------"

python $ROOT_DIR/tools/preprocess_data.py \
    --input "$DATASET_OUTPUT_DIR/train.jsonl" \
    --output-prefix "$DATASET_OUTPUT_DIR/train_processed" \
    --tokenizer-type $TOKENIZER_TYPE \
    --tokenizer-model $TOKENIZER \
    --json-keys text \
    --workers $NUM_PROC \
    --append-eod

if [ $? -ne 0 ]; then
    echo "Error: Training data preprocessing failed!"
    exit 1
fi

echo ""
echo "Step 2 completed successfully!"
echo ""

########################################################
# Step 3: Preprocess validation data
########################################################
echo "Step 3: Preprocessing validation data..."
echo "----------------------------------------"

python $ROOT_DIR/tools/preprocess_data.py \
    --input "$DATASET_OUTPUT_DIR/valid.jsonl" \
    --output-prefix "$DATASET_OUTPUT_DIR/valid_processed" \
    --tokenizer-type $TOKENIZER_TYPE \
    --tokenizer-model $TOKENIZER \
    --json-keys text \
    --workers $NUM_PROC \
    --append-eod

if [ $? -ne 0 ]; then
    echo "Error: Validation data preprocessing failed!"
    exit 1
fi

echo ""
echo "Step 3 completed successfully!"
echo ""

########################################################
# Summary
########################################################
echo "=========================================="
echo "Dataset preprocessing complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  Training data:"
echo "    - $DATASET_OUTPUT_DIR/train_processed_text_document.bin"
echo "    - $DATASET_OUTPUT_DIR/train_processed_text_document.idx"
echo ""
echo "  Validation data:"
echo "    - $DATASET_OUTPUT_DIR/valid_processed_text_document.bin"
echo "    - $DATASET_OUTPUT_DIR/valid_processed_text_document.idx"
echo ""
echo "To use in training script, set:"
echo "  train_data_prefix=\"$DATASET_OUTPUT_DIR/train_processed_text_document\""
echo "  valid_data_prefix=\"$DATASET_OUTPUT_DIR/valid_processed_text_document\""
echo "=========================================="

