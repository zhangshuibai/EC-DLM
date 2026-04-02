#!/usr/bin/env bash
set -euo pipefail

# Convert Nemotron parquet subsets to Megatron .bin/.idx with NeMo AutoModel tool.
# This wrapper avoids feeding huge globs directly to the tool (which can spawn
# too many processes). It processes one parquet file per call with bounded
# file-level parallelism.

CONDA_ROOT="${CONDA_ROOT:-/path/to/miniconda3}"
CONDA_ENV="${CONDA_ENV:-nemo_convert}"
AUTOMODEL_DIR="${AUTOMODEL_DIR:-/path/to/Automodel}"

INPUT_ROOT="${INPUT_ROOT:-/path/to/nemontron-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/nemontron-data-megadlm}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./megatron/training/tokenizer/gptneox_tokenizer}"

# Subsets to convert.
SUBSETS="${SUBSETS:-High-Quality,Translated-Diverse-QA,Diverse-QA}"

# Throughput knobs.
# PARALLEL_FILES: number of parquet files processed simultaneously.
# WORKERS_PER_FILE: --workers for preprocess_megatron_dataset.py per file.
PARALLEL_FILES="${PARALLEL_FILES:-4}"
WORKERS_PER_FILE="${WORKERS_PER_FILE:-2}"
PARQUET_BATCH_SIZE="${PARQUET_BATCH_SIZE:-20000}"
LOG_INTERVAL="${LOG_INTERVAL:-200000}"

# Tokenizer threading.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "[info] CONDA_ROOT=$CONDA_ROOT"
echo "[info] CONDA_ENV=$CONDA_ENV"
echo "[info] AUTOMODEL_DIR=$AUTOMODEL_DIR"
echo "[info] INPUT_ROOT=$INPUT_ROOT"
echo "[info] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[info] TOKENIZER_PATH=$TOKENIZER_PATH"
echo "[info] SUBSETS=$SUBSETS"
echo "[info] PARALLEL_FILES=$PARALLEL_FILES WORKERS_PER_FILE=$WORKERS_PER_FILE PARQUET_BATCH_SIZE=$PARQUET_BATCH_SIZE"
echo "[info] TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM RAYON_NUM_THREADS=$RAYON_NUM_THREADS"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "[error] conda.sh not found: $CONDA_ROOT/etc/profile.d/conda.sh"
    exit 1
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [[ ! -d "$AUTOMODEL_DIR" ]]; then
    echo "[info] Cloning Automodel to $AUTOMODEL_DIR"
    git clone --depth 1 https://github.com/NVIDIA-NeMo/Automodel.git "$AUTOMODEL_DIR"
fi

TOOL_SCRIPT="$AUTOMODEL_DIR/tools/preprocess_megatron_dataset.py"
if [[ ! -f "$TOOL_SCRIPT" ]]; then
    echo "[error] tool script not found: $TOOL_SCRIPT"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

format_hms() {
    local sec="$1"
    if (( sec < 0 )); then
        printf "??:??:??"
        return
    fi
    printf "%02d:%02d:%02d" $((sec / 3600)) $(((sec % 3600) / 60)) $((sec % 60))
}

print_progress() {
    local subset="$1"
    local completed="$2"
    local total="$3"
    local running="$4"
    local start_ts="$5"
    local bar_width=30

    local percent fill empty elapsed eta
    percent=$(( completed * 100 / total ))
    fill=$(( completed * bar_width / total ))
    empty=$(( bar_width - fill ))
    elapsed=$(( $(date +%s) - start_ts ))
    if (( completed > 0 )); then
        eta=$(( (total - completed) * elapsed / completed ))
    else
        eta=-1
    fi

    local bar_filled bar_empty
    bar_filled="$(printf '%*s' "$fill" '' | tr ' ' '#')"
    bar_empty="$(printf '%*s' "$empty" '' | tr ' ' '-')"

    printf "\r[progress] %s [%s%s] %3d%% (%d/%d) running=%d elapsed=%s eta=%s" \
        "$subset" "$bar_filled" "$bar_empty" "$percent" "$completed" "$total" "$running" \
        "$(format_hms "$elapsed")" "$(format_hms "$eta")"
}

run_one_file() {
    local subset="$1"
    local parquet_file="$2"
    local subset_out="$OUTPUT_ROOT/$subset"
    local base
    base="$(basename "${parquet_file%.parquet}")"

    mkdir -p "$subset_out"

    python "$TOOL_SCRIPT" \
        --input "$parquet_file" \
        --input-type parquet \
        --text-column text \
        --parquet-batch-size "$PARQUET_BATCH_SIZE" \
        --output-prefix "${subset}_${base}" \
        --output-path "$subset_out" \
        --workers "$WORKERS_PER_FILE" \
        --log-interval "$LOG_INTERVAL" \
        --pretrained-model-name-or-path "$TOKENIZER_PATH" \
        --append-eod
}

IFS=',' read -r -a subset_array <<< "$SUBSETS"

for subset in "${subset_array[@]}"; do
    subset_dir="$INPUT_ROOT/$subset"
    if [[ ! -d "$subset_dir" ]]; then
        echo "[warn] subset dir not found, skip: $subset_dir"
        continue
    fi

    echo "[start] subset=$subset"
    mapfile -t parquet_files < <(find "$subset_dir" -maxdepth 1 -type f -name "*.parquet" | sort)
    total="${#parquet_files[@]}"
    if [[ "$total" -eq 0 ]]; then
        echo "[warn] no parquet files under $subset_dir"
        continue
    fi
    echo "[info] $subset: $total parquet files"

    running=0
    dispatched=0
    completed=0
    subset_start_ts="$(date +%s)"
    print_progress "$subset" "$completed" "$total" "$running" "$subset_start_ts"

    for parquet_file in "${parquet_files[@]}"; do
        run_one_file "$subset" "$parquet_file" &
        ((running += 1))
        ((dispatched += 1))

        if (( running >= PARALLEL_FILES )); then
            wait -n
            ((running -= 1))
            ((completed += 1))
            print_progress "$subset" "$completed" "$total" "$running" "$subset_start_ts"
        fi
    done

    while (( running > 0 )); do
        wait -n
        ((running -= 1))
        ((completed += 1))
        print_progress "$subset" "$completed" "$total" "$running" "$subset_start_ts"
    done
    echo
    echo "[done] subset=$subset"
done

echo "[all done] output at $OUTPUT_ROOT"
echo "[note] Generated files are sharded prefixes like *_text_document.bin/.idx"
