#!/usr/bin/env python3
"""Convert Nemotron parquet subsets to Megatron indexed dataset format.

This script scans subset folders under an input root and converts each subset
into:
  - train_processed_text_document.bin/.idx
  - valid_processed_text_document.bin/.idx

It writes Megatron IndexedDataset files directly (no large JSONL intermediates).
"""

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Tuple

import pyarrow.parquet as pq
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from megatron.core.datasets import indexed_dataset
from megatron.training.tokenizer import build_tokenizer


def _build_hf_tokenizer(tokenizer_model: str):
    tok_args = SimpleNamespace(
        rank=0,
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=tokenizer_model,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
        vocab_extra_ids=0,
        padded_vocab_size=None,
        vocab_size=None,
        tiktoken_pattern=None,
        tiktoken_num_special_tokens=1000,
        tiktoken_special_tokens=None,
    )
    return build_tokenizer(tok_args)


def _discover_subsets(input_root: Path) -> List[Path]:
    subsets = []
    for d in sorted(input_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("."):
            continue
        if list(d.glob("*.parquet")):
            subsets.append(d)
    return subsets


def _iter_parquet_files(subset_dir: Path) -> Iterable[Path]:
    return sorted(subset_dir.glob("*.parquet"))


def _collect_subset_stats(parquet_files: List[Path]) -> Tuple[int, int]:
    total_rows = 0
    total_row_groups = 0
    for parquet_file in parquet_files:
        md = pq.ParquetFile(parquet_file).metadata
        total_row_groups += md.num_row_groups
        for row_group_idx in range(md.num_row_groups):
            total_rows += md.row_group(row_group_idx).num_rows
    return total_rows, total_row_groups


def _convert_subset(
    subset_dir: Path,
    output_root: Path,
    tokenizer,
    valid_ratio: int,
    min_chars: int,
    append_eod: bool,
    log_every_docs: int,
    tokenize_batch_size: int,
    force: bool,
):
    subset_name = subset_dir.name
    out_dir = output_root / subset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_prefix = out_dir / "train_processed_text_document"
    valid_prefix = out_dir / "valid_processed_text_document"
    train_bin = f"{train_prefix}.bin"
    train_idx = f"{train_prefix}.idx"
    valid_bin = f"{valid_prefix}.bin"
    valid_idx = f"{valid_prefix}.idx"

    if (not force) and all(Path(p).exists() for p in [train_bin, train_idx, valid_bin, valid_idx]):
        print(f"[skip] {subset_name}: output already exists")
        return {
            "subset": subset_name,
            "train_prefix": str(train_prefix),
            "valid_prefix": str(valid_prefix),
            "train_docs": None,
            "valid_docs": None,
            "skipped_docs": None,
            "elapsed_sec": 0.0,
        }

    for p in [train_bin, train_idx, valid_bin, valid_idx]:
        if Path(p).exists():
            os.remove(p)

    dtype = indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size)
    train_builder = indexed_dataset.IndexedDatasetBuilder(train_bin, dtype=dtype)
    valid_builder = indexed_dataset.IndexedDatasetBuilder(valid_bin, dtype=dtype)

    train_docs = 0
    valid_docs = 0
    skipped_docs = 0
    accepted_docs = 0
    start = time.time()

    parquet_files = list(_iter_parquet_files(subset_dir))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found under {subset_dir}")

    total_rows, total_row_groups = _collect_subset_stats(parquet_files)
    print(
        f"[start] {subset_name}: {len(parquet_files)} parquet files, "
        f"{total_row_groups:,} row groups, {total_rows:,} rows"
    )

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=total_rows,
            desc=f"{subset_name}",
            unit="doc",
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        )
    else:
        print("  tqdm not installed; falling back to periodic log output.")

    for file_i, parquet_file in enumerate(parquet_files, start=1):
        pf = pq.ParquetFile(parquet_file)
        for row_group_idx in range(pf.num_row_groups):
            table = pf.read_row_group(row_group_idx, columns=["text"])
            text_col = table.column("text")
            row_group_rows = len(text_col)
            tokenization_buffer = []

            def _flush_buffer(buf):
                nonlocal train_docs, valid_docs, skipped_docs, accepted_docs
                if not buf:
                    return
                # Batch tokenization is much faster than per-sample calls.
                batch_ids = tokenizer.tokenize(buf)
                if not batch_ids:
                    skipped_docs += len(buf)
                    return
                # Keep shape consistent when buf has one sample.
                if isinstance(batch_ids[0], int):
                    batch_ids = [batch_ids]

                for token_ids in batch_ids:
                    if not token_ids:
                        skipped_docs += 1
                        continue
                    if append_eod:
                        token_ids.append(tokenizer.eod)

                    doc_len = len(token_ids)
                    if valid_ratio > 0 and accepted_docs % valid_ratio == 0:
                        valid_builder.add_document(token_ids, [doc_len])
                        valid_docs += 1
                    else:
                        train_builder.add_document(token_ids, [doc_len])
                        train_docs += 1

                    accepted_docs += 1
                    if log_every_docs > 0 and accepted_docs % log_every_docs == 0:
                        elapsed = time.time() - start
                        rate = accepted_docs / max(elapsed, 1e-6)
                        if pbar is not None:
                            pbar.set_postfix(
                                accepted=f"{accepted_docs:,}",
                                train=f"{train_docs:,}",
                                valid=f"{valid_docs:,}",
                                skipped=f"{skipped_docs:,}",
                                rate=f"{rate:,.0f}/s",
                                file=f"{file_i}/{len(parquet_files)}",
                            )
                        else:
                            print(
                                f"  [{subset_name}] docs={accepted_docs:,} train={train_docs:,} "
                                f"valid={valid_docs:,} skipped={skipped_docs:,} "
                                f"rate={rate:,.0f} docs/s file={file_i}/{len(parquet_files)}"
                            )

            for j in range(len(text_col)):
                text = text_col[j].as_py()
                if not isinstance(text, str):
                    skipped_docs += 1
                    continue
                text = text.strip()
                if len(text) < min_chars:
                    skipped_docs += 1
                    continue

                tokenization_buffer.append(text)
                if len(tokenization_buffer) >= tokenize_batch_size:
                    _flush_buffer(tokenization_buffer)
                    tokenization_buffer = []

            _flush_buffer(tokenization_buffer)

            if pbar is not None:
                pbar.update(row_group_rows)

    if pbar is not None:
        pbar.close()

    train_builder.finalize(train_idx)
    valid_builder.finalize(valid_idx)
    elapsed = time.time() - start
    print(
        f"[done] {subset_name}: train={train_docs:,} valid={valid_docs:,} "
        f"skipped={skipped_docs:,} elapsed={elapsed/60:.1f}m"
    )
    return {
        "subset": subset_name,
        "train_prefix": str(train_prefix),
        "valid_prefix": str(valid_prefix),
        "train_docs": train_docs,
        "valid_docs": valid_docs,
        "skipped_docs": skipped_docs,
        "elapsed_sec": elapsed,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Convert Nemotron parquet subsets to Megadlm indexed dataset.")
    p.add_argument(
        "--input-root",
        type=str,
        default="/path/to/nemontron-data",
        help="Root directory containing subset folders with parquet files.",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="/path/to/nemontron-data-megadlm",
        help="Output root directory for generated .bin/.idx files.",
    )
    p.add_argument(
        "--tokenizer-model",
        type=str,
        default="./megatron/training/tokenizer/gptneox_tokenizer",
        help="Tokenizer path passed to HuggingFaceTokenizer.",
    )
    p.add_argument(
        "--subsets",
        type=str,
        default="High-Quality,Translated-Diverse-QA,Diverse-QA",
        help="Comma-separated subset names. Empty means auto-detect all subset dirs with parquet files.",
    )
    p.add_argument(
        "--valid-ratio",
        type=int,
        default=10000,
        help="One out of N accepted docs goes to valid set. Set <=0 to disable valid split.",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=50,
        help="Skip docs shorter than this many characters.",
    )
    p.add_argument(
        "--no-append-eod",
        action="store_true",
        help="Disable appending EOD token per document.",
    )
    p.add_argument(
        "--log-every-docs",
        type=int,
        default=200000,
        help="Progress print interval in accepted documents.",
    )
    p.add_argument(
        "--tokenize-batch-size",
        type=int,
        default=512,
        help="Batch size for tokenizer calls. Larger is usually faster but uses more RAM.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files for subsets.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    if args.subsets.strip():
        selected = [s.strip() for s in args.subsets.split(",") if s.strip()]
        subset_dirs = [input_root / s for s in selected]
        missing = [str(s) for s in subset_dirs if not s.exists()]
        if missing:
            raise FileNotFoundError(f"Subset dirs not found: {missing}")
    else:
        subset_dirs = _discover_subsets(input_root)

    if not subset_dirs:
        raise RuntimeError(f"No subset directories with parquet files found under {input_root}")

    print("Subsets to convert:")
    for d in subset_dirs:
        print(f"  - {d.name}")

    tokenizer = _build_hf_tokenizer(args.tokenizer_model)

    summary_rows = []
    for subset_dir in subset_dirs:
        row = _convert_subset(
            subset_dir=subset_dir,
            output_root=output_root,
            tokenizer=tokenizer,
            valid_ratio=args.valid_ratio,
            min_chars=args.min_chars,
            append_eod=not args.no_append_eod,
            log_every_docs=args.log_every_docs,
            tokenize_batch_size=args.tokenize_batch_size,
            force=args.force,
        )
        summary_rows.append(row)

    manifest = output_root / "manifest.tsv"
    with manifest.open("w", encoding="utf-8") as f:
        f.write("subset\ttrain_prefix\tvalid_prefix\ttrain_docs\tvalid_docs\tskipped_docs\telapsed_sec\n")
        for r in summary_rows:
            f.write(
                f"{r['subset']}\t{r['train_prefix']}\t{r['valid_prefix']}\t"
                f"{r['train_docs']}\t{r['valid_docs']}\t{r['skipped_docs']}\t{r['elapsed_sec']}\n"
            )

    print("\nAll done. Manifest:")
    print(f"  {manifest}")
    print("\nExample training prefixes (use one subset or blend multiple):")
    for r in summary_rows:
        print(f"  train: {r['train_prefix']}")
        print(f"  valid: {r['valid_prefix']}")


if __name__ == "__main__":
    main()
