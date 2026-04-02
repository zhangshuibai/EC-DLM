#!/usr/bin/env bash
#
# Download Nemotron-CC-HQ: the paper's high-quality subset (~1.1T tokens).
# Only the two partitions used in the paper for 1T-token training (MMLU 59.0):
#   - quality=high/kind=actual/kind2=actual
#   - quality=high/kind=synthetic/kind2=diverse_qa_pairs
# See: https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html
# Paper: Nemotron-CC (arXiv 2412.02595), Table 4 "Nemotron-CC-HQ"
#
# Usage:
#   ./download_nemotron_cc_high.sh [OUTPUT_DIR] [PARALLEL_JOBS]
#   OUTPUT_DIR     default: /path/to/nemontron-CC
#   PARALLEL_JOBS  default: 16 (capped at 128)
#
# Examples:
#   ./download_nemotron_cc_high.sh
#   ./download_nemotron_cc_high.sh /other/path 64
#
# Requires: curl, gzip, wget
#

set -euo pipefail

for cmd in curl gzip wget; do
  if ! command -v "${cmd}" &>/dev/null; then
    echo "Error: required command not found: ${cmd}" >&2
    exit 1
  fi
done

BASE_URL="https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC"
PATHS_FILE="data-jsonl.paths.gz"

# Default output: /path/to/nemontron-CC
OUTPUT_DIR="${1:-/path/to/nemontron-CC}"
PARALLEL_JOBS="${2:-16}"

# Validate PARALLEL_JOBS: positive integer, cap at 128
if ! [[ "${PARALLEL_JOBS}" =~ ^[0-9]+$ ]] || [[ "${PARALLEL_JOBS}" -lt 1 ]]; then
  echo "Error: PARALLEL_JOBS must be a positive integer (got: ${PARALLEL_JOBS})" >&2
  exit 1
fi
if [[ "${PARALLEL_JOBS}" -gt 128 ]]; then
  PARALLEL_JOBS=128
  echo "Capping PARALLEL_JOBS at 128"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Paper subset: cache separate from "full quality=high" list
PATHS_LIST="${SCRIPT_DIR}/.nemotron_cc_hq_paper.paths.txt"
FAILED_LOG="${SCRIPT_DIR}/.nemotron_cc_hq_paper.failed"

echo "Output directory: ${OUTPUT_DIR}"
echo "Parallel jobs: ${PARALLEL_JOBS}"
mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

# Step 1: Fetch path list and filter for paper's Nemotron-CC-HQ (actual + diverse_qa_pairs only)
echo "Fetching path list and filtering for paper subset (actual + diverse_qa_pairs)..."
if [[ ! -f "${PATHS_LIST}" ]]; then
  tmp_gz="$(mktemp)"
  curl -sSfL "${BASE_URL}/${PATHS_FILE}" -o "${tmp_gz}"
  gzip -dc "${tmp_gz}" | grep -E "quality=high/kind=actual/kind2=actual|quality=high/kind=synthetic/kind2=diverse_qa_pairs" > "${PATHS_LIST}" || true
  rm -f "${tmp_gz}"
  if [[ ! -s "${PATHS_LIST}" ]]; then
    echo "Error: no paths found. Check URL or grep pattern." >&2
    exit 1
  fi
fi
NUM_FILES=$(wc -l < "${PATHS_LIST}")
echo "Found ${NUM_FILES} files in paper Nemotron-CC-HQ subset (~1.1T tokens)."

rm -f "${FAILED_LOG}"

# Step 2: Download each file (resume-friendly, preserve path under OUTPUT_DIR)
# We are already cd'd to OUTPUT_DIR; path is relative e.g. contrib/Nemotron/Nemotron-CC/data-jsonl/...
download_one() {
  local path="$1"
  local url="https://data.commoncrawl.org/${path}"
  local dir
  dir="$(dirname "${path}")"
  mkdir -p "${dir}"
  if [[ -f "${path}" ]]; then
    return 0
  fi
  if wget -q -c -O "${path}.tmp" "${url}"; then
    mv "${path}.tmp" "${path}"
  else
    rm -f "${path}.tmp"
    echo "FAIL: ${path}" >&2
    echo "${path}" >> "${FAILED_LOG}"
    return 1
  fi
}
export -f download_one
export FAILED_LOG

# Progress reporter: count completed files and show a single updating line
progress_reporter() {
  local total="$1"
  local paths_file="$2"
  while true; do
    count=0
    while IFS= read -r p; do
      [[ -z "$p" ]] && continue
      [[ -f "$p" ]] && ((count++)) || true
    done < "$paths_file"
    pct=0
    [[ "$total" -gt 0 ]] && pct=$((count * 100 / total))
    # Bar: 20 chars wide
    filled=$((pct * 20 / 100))
    bar=""
    for ((i=0; i<20; i++)); do
      [[ $i -lt $filled ]] && bar+="=" || bar+=" "
    done
    printf "\rProgress: %d/%d [%s] %d%%  " "$count" "$total" "$bar" "$pct"
    sleep 2
  done
}

echo "Starting downloads (${PARALLEL_JOBS} parallel)..."
progress_reporter "${NUM_FILES}" "${PATHS_LIST}" &
PROGRESS_PID=$!
trap 'kill "${PROGRESS_PID}" 2>/dev/null || true' EXIT

batch=()
while IFS= read -r path; do
  [[ -z "${path}" ]] && continue
  batch+=("${path}")
  if ((${#batch[@]} >= PARALLEL_JOBS)); then
    for p in "${batch[@]}"; do download_one "${p}" & done
    wait
    batch=()
  fi
done < "${PATHS_LIST}"
for p in "${batch[@]}"; do download_one "${p}" & done
wait

kill "${PROGRESS_PID}" 2>/dev/null || true
trap - EXIT
printf "\n"

if [[ -f "${FAILED_LOG}" ]] && [[ -s "${FAILED_LOG}" ]]; then
  failed_count=$(wc -l < "${FAILED_LOG}")
  echo "Warning: ${failed_count} file(s) failed to download. See: ${FAILED_LOG}" >&2
  exit 1
fi

echo "Done. Data is under: ${OUTPUT_DIR}"
echo "Path list used: ${PATHS_LIST} (delete to re-fetch path list next run)"
