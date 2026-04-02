#!/usr/bin/env python3
"""Download and materialize on-the-fly eval datasets as local JSONL files.

This script writes files that exactly match the names expected by:
  megadlm/tasks/on_the_fly_eval/data_utils.py
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Tuple


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _to_rows_mmlu(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "question": x["question"],
            "choices": list(x["choices"]),
            "answer": int(x["answer"]),
            "subject": x["subject"],
        }
        for x in items
    ]


def _to_rows_arc(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
    rows = []
    for x in items:
        choices = list(x["choices"]["text"])
        answer_idx = label_map.get(x["answerKey"], 0)
        if answer_idx < len(choices):
            rows.append(
                {
                    "question": x["question"],
                    "choices": choices,
                    "answer": answer_idx,
                    "subject": "science",
                }
            )
    return rows


def _to_rows_piqa(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for x in items:
        label = int(x["label"])
        if label >= 0:
            rows.append(
                {
                    "question": f"Goal: {x['goal']}",
                    "choices": [x["sol1"], x["sol2"]],
                    "answer": label,
                    "subject": "physical_intuition",
                }
            )
    return rows


def _to_rows_winogrande(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for x in items:
        if x["answer"] in ("1", "2"):
            rows.append(
                {
                    "question": x["sentence"],
                    "choices": [x["option1"], x["option2"]],
                    "answer": int(x["answer"]) - 1,
                    "subject": "commonsense",
                }
            )
    return rows


def _to_rows_copa(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for x in items:
        if x["question"] == "cause":
            q = f"{x['premise']} What was the cause of this?"
        else:
            q = f"{x['premise']} What happened as a result?"
        rows.append(
            {
                "question": q,
                "choices": [x["choice1"], x["choice2"]],
                "answer": int(x["label"]),
                "subject": "causal_reasoning",
            }
        )
    return rows


def _to_rows_boolq(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "question": f"{x['passage']}\nQuestion: {x['question']}",
            "choices": ["No", "Yes"],
            "answer": int(x["answer"]),
            "subject": "yes_no_question",
        }
        for x in items
    ]


def _to_rows_openbookqa(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    rows = []
    for x in items:
        choices = list(x["choices"]["text"])
        answer_idx = label_map.get(x["answerKey"], 0)
        if answer_idx < len(choices):
            rows.append(
                {
                    "question": x["question_stem"],
                    "choices": choices,
                    "answer": answer_idx,
                    "subject": "science",
                }
            )
    return rows


def _to_rows_hellaswag(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Keep a minimal normalized schema that data_utils can load directly.
    rows = []
    for x in items:
        label = int(x["label"])
        rows.append(
            {
                "query": x.get("query", ""),
                "endings": list(x["endings"]),
                "label": label,
            }
        )
    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _run_with_retry(
    desc: str,
    fn: Callable[[], List[Dict[str, Any]]],
    retries: int,
    retry_sleep: float,
) -> List[Dict[str, Any]]:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == retries:
                break
            wait_s = retry_sleep * attempt
            print(f"[warn] {desc} failed (attempt {attempt}/{retries}): {exc}")
            print(f"[warn] retrying in {wait_s:.1f}s ...")
            time.sleep(wait_s)
    assert last_exc is not None
    raise last_exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Download on-the-fly eval datasets to local JSONL files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tasks/on_the_fly_eval/local_data",
        help="Directory to write local JSONL files.",
    )
    parser.add_argument("--retries", type=int, default=8, help="Max retries per dataset split.")
    parser.add_argument("--retry-sleep", type=float, default=20.0, help="Base retry sleep seconds.")
    offline_group = parser.add_mutually_exclusive_group()
    offline_group.add_argument(
        "--offline",
        dest="offline",
        action="store_true",
        help="Use only local HuggingFace cache (default).",
    )
    offline_group.add_argument(
        "--online",
        dest="offline",
        action="store_false",
        help="Allow online access to HuggingFace Hub.",
    )
    parser.set_defaults(offline=True)
    args = parser.parse_args()

    repo_root = _repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from datasets import load_dataset  # type: ignore

    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("[info] offline mode enabled: using local HuggingFace cache only.")

    os.makedirs(args.output_dir, exist_ok=True)

    jobs: List[Tuple[str, Callable[[], List[Dict[str, Any]]]]] = [
        (
            "mmlu_test.jsonl",
            lambda: _to_rows_mmlu(
                load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
            ),
        ),
        (
            "mmlu_dev.jsonl",
            lambda: _to_rows_mmlu(
                load_dataset("cais/mmlu", "all", split="dev", trust_remote_code=True)
            ),
        ),
        (
            "arc_easy_test.jsonl",
            lambda: _to_rows_arc(
                load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", trust_remote_code=True)
            ),
        ),
        (
            "arc_easy_train.jsonl",
            lambda: _to_rows_arc(
                load_dataset("allenai/ai2_arc", "ARC-Easy", split="train", trust_remote_code=True)
            ),
        ),
        (
            "arc_challenge_test.jsonl",
            lambda: _to_rows_arc(
                load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
            ),
        ),
        (
            "arc_challenge_train.jsonl",
            lambda: _to_rows_arc(
                load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", trust_remote_code=True)
            ),
        ),
        (
            "piqa_validation.jsonl",
            lambda: _to_rows_piqa(load_dataset("piqa", split="validation", trust_remote_code=True)),
        ),
        (
            "piqa_train.jsonl",
            lambda: _to_rows_piqa(load_dataset("piqa", split="train", trust_remote_code=True)),
        ),
        (
            "winogrande_validation.jsonl",
            lambda: _to_rows_winogrande(
                load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
            ),
        ),
        (
            "winogrande_train.jsonl",
            lambda: _to_rows_winogrande(
                load_dataset("winogrande", "winogrande_xl", split="train", trust_remote_code=True)
            ),
        ),
        (
            "copa_validation.jsonl",
            lambda: _to_rows_copa(
                load_dataset("super_glue", "copa", split="validation", trust_remote_code=True)
            ),
        ),
        (
            "copa_train.jsonl",
            lambda: _to_rows_copa(
                load_dataset("super_glue", "copa", split="train", trust_remote_code=True)
            ),
        ),
        (
            "boolq_validation.jsonl",
            lambda: _to_rows_boolq(
                load_dataset("google/boolq", split="validation", trust_remote_code=True)
            ),
        ),
        (
            "boolq_train.jsonl",
            lambda: _to_rows_boolq(
                load_dataset("google/boolq", split="train", trust_remote_code=True)
            ),
        ),
        (
            "openbookqa_test.jsonl",
            lambda: _to_rows_openbookqa(
                load_dataset("allenai/openbookqa", "main", split="test", trust_remote_code=True)
            ),
        ),
        (
            "openbookqa_train.jsonl",
            lambda: _to_rows_openbookqa(
                load_dataset("allenai/openbookqa", "main", split="train", trust_remote_code=True)
            ),
        ),
        (
            "hellaswag_validation.jsonl",
            lambda: _to_rows_hellaswag(
                load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
            ),
        ),
    ]

    for filename, loader in jobs:
        out_path = os.path.join(args.output_dir, filename)
        print(f"[start] {filename}")
        rows = _run_with_retry(
            desc=filename,
            fn=loader,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        _write_jsonl(out_path, rows)
        print(f"[done] {filename}: {len(rows)} rows -> {out_path}")

    print(f"[ok] all files written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
