#!/usr/bin/env python3
"""
Download and preprocess OpenWebText dataset from HuggingFace.

This script:
1. Downloads the Skylion007/openwebtext dataset from HuggingFace
2. Preprocesses the data (cleaning, filtering, etc.)
3. Saves as JSONL format for Megatron preprocessing
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterator, Dict, Any
import time

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not found. Install it with: pip install datasets")
    sys.exit(1)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Remove very short texts (less than 50 characters)
    if len(text) < 50:
        return ""
    
    return text.strip()


def process_dataset(
    dataset_name: str = "Skylion007/openwebtext",
    output_dir: str = "./data/openwebtext",
    split: str = "train",
    max_samples: int = None,
    streaming: bool = False,
    num_proc: int = 1,
) -> Dict[str, str]:
    """
    Download and preprocess the dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Output directory for processed files
        split: Dataset split to download (train, validation, test)
        max_samples: Maximum number of samples to process (None for all)
        streaming: Whether to use streaming mode for large datasets
        num_proc: Number of processes for processing
    
    Returns:
        Dictionary with paths to output files
    """
    print(f"Loading dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Streaming mode: {streaming}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    try:
        if streaming:
            dataset = load_dataset(dataset_name, split=split, streaming=True)
        else:
            dataset = load_dataset(dataset_name, split=split, num_proc=num_proc)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying to load with trust_remote_code=True...")
        try:
            if streaming:
                dataset = load_dataset(
                    dataset_name, split=split, streaming=True, trust_remote_code=True
                )
            else:
                dataset = load_dataset(
                    dataset_name, split=split, num_proc=num_proc, trust_remote_code=True
                )
        except Exception as e2:
            print(f"Failed to load dataset: {e2}")
            sys.exit(1)
    
    # Determine output file paths
    train_file = output_path / "train.jsonl"
    valid_file = output_path / "valid.jsonl"
    
    # Process and save data
    print(f"\nProcessing dataset...")
    print(f"Output directory: {output_path}")
    
    train_count = 0
    valid_count = 0
    skipped_count = 0
    
    # Open output files
    train_fp = open(train_file, "w", encoding="utf-8")
    valid_fp = open(valid_file, "w", encoding="utf-8")
    
    try:
        # Process samples
        total = max_samples if max_samples else None
        iterator = enumerate(dataset)
        if tqdm is not None:
            pbar = tqdm(iterator, total=total, desc="Downloading", unit="doc", dynamic_ncols=True)
        else:
            pbar = iterator

        for idx, example in pbar:
            if max_samples and idx >= max_samples:
                break

            # Get text field (OpenWebText uses 'text' field)
            text = example.get("text", "")

            # Clean text
            cleaned_text = clean_text(text)

            # Skip empty texts
            if not cleaned_text:
                skipped_count += 1
                continue

            # Create JSON object
            json_obj = {"text": cleaned_text}

            # Split into train/validation (99.9/0.1 split)
            if idx % 1000 == 0:  # 0.1% (1/1000) for validation
                json.dump(json_obj, valid_fp, ensure_ascii=False)
                valid_fp.write("\n")
                valid_count += 1
            else:  # 99.9% for training
                json.dump(json_obj, train_fp, ensure_ascii=False)
                train_fp.write("\n")
                train_count += 1

            # Progress update
            if tqdm is not None and hasattr(pbar, 'set_postfix'):
                if (idx + 1) % 5000 == 0:
                    pbar.set_postfix(train=train_count, valid=valid_count, skip=skipped_count)
            elif (idx + 1) % 10000 == 0:
                print(f"Processed {idx + 1} samples (train: {train_count}, valid: {valid_count}, skipped: {skipped_count})")
    
    finally:
        train_fp.close()
        valid_fp.close()
    
    print(f"\nProcessing complete!")
    print(f"Total samples processed: {train_count + valid_count}")
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {valid_count}")
    print(f"Skipped samples: {skipped_count}")
    print(f"\nOutput files:")
    print(f"  Training: {train_file}")
    print(f"  Validation: {valid_file}")
    
    return {
        "train": str(train_file),
        "valid": str(valid_file),
        "train_count": train_count,
        "valid_count": valid_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess OpenWebText dataset"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Skylion007/openwebtext",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/openwebtext",
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None for all)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of processes for processing",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OpenWebText Dataset Download and Preprocessing")
    print("=" * 60)
    
    start_time = time.time()
    
    result = process_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
        streaming=args.streaming,
        num_proc=args.num_proc,
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    print("=" * 60)
    
    # Print summary for easy copy-paste
    print("\nNext steps:")
    print(f"1. Run Megatron preprocessing:")
    print(f"   python tools/preprocess_data.py \\")
    print(f"       --input {result['train']} \\")
    print(f"       --output-prefix {args.output_dir}/train_processed \\")
    print(f"       --tokenizer-type HuggingFaceTokenizer \\")
    print(f"       --tokenizer-model <path_to_tokenizer> \\")
    print(f"       --workers 8 \\")
    print(f"       --append-eod")
    print(f"\n   python tools/preprocess_data.py \\")
    print(f"       --input {result['valid']} \\")
    print(f"       --output-prefix {args.output_dir}/valid_processed \\")
    print(f"       --tokenizer-type HuggingFaceTokenizer \\")
    print(f"       --tokenizer-model <path_to_tokenizer> \\")
    print(f"       --workers 8 \\")
    print(f"       --append-eod")


if __name__ == "__main__":
    main()

