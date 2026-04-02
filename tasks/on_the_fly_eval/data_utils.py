"""Data loading and prompt formatting for on-the-fly evaluation.

All MCQ tasks return items in a unified format:
    {"question": str, "choices": [str], "answer": int (0-indexed), "subject": str}

HellaSwag uses a different format:
    {"query": str, "endings": [str], "label": int (0-3)}
"""

import json
import os
import random
import re


# ---------------------------------------------------------------------------
# Generic MCQ prompt formatting
# ---------------------------------------------------------------------------

_LETTERS = "ABCDEFGH"

TASK_HEADERS = {
    "mmlu": "The following is a multiple choice question about {subject}.",
    "arc_easy": "The following is a multiple choice question about science.",
    "arc_challenge": "The following is a multiple choice question about science.",
    "piqa": "The following is a multiple choice question about physical intuition.",
    "winogrande": "The following is a sentence completion question.",
    "copa": "The following is a multiple choice question about causal reasoning.",
    "boolq": "The following is a yes/no question.",
    "openbookqa": "The following is a multiple choice question about science.",
}


def get_task_header(task_name, item=None):
    """Get the prompt header for a task, filling {subject} if present."""
    template = TASK_HEADERS.get(task_name, "The following is a multiple choice question.")
    if "{subject}" in template and item:
        return template.format(subject=_format_subject(item.get("subject", "")))
    return template


def format_mcq_prompt(item, header, few_shot_examples=None):
    """Format a generic multiple-choice question prompt.

    Args:
        item: dict with 'question' and 'choices'.
        header: Task header string (already formatted).
        few_shot_examples: Optional list of items with 'question', 'choices', 'answer'.

    Returns:
        Formatted prompt string ending with "Answer:"
    """
    parts = [header, "\n\n"]

    if few_shot_examples:
        for ex in few_shot_examples:
            parts.append(_format_question_block(ex["question"], ex["choices"]))
            parts.append(f"Answer: {_LETTERS[ex['answer']]}\n\n")

    parts.append(_format_question_block(item["question"], item["choices"]))
    parts.append("Answer:")

    return "".join(parts)


def _format_question_block(question, choices):
    """Format question text + lettered choices."""
    lines = [question + "\n"]
    for i, c in enumerate(choices):
        lines.append(f"{_LETTERS[i]}. {c}\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# HellaSwag prompt formatting (log-likelihood based, no MCQ format)
# ---------------------------------------------------------------------------

def format_hellaswag_prompt(ctx, ending):
    """Format a HellaSwag context+ending pair.

    Returns:
        (context_str, ending_str) tuple.
    """
    return ctx, " " + ending.strip()


def _preprocess_hellaswag_text(text):
    """Apply the same text cleanup used by common HellaSwag eval setups."""
    text = (text or "").strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def _normalize_hellaswag_doc(item):
    """Convert raw HellaSwag record to normalized {query, endings, label}."""
    if "query" in item and "endings" in item:
        query = _preprocess_hellaswag_text(item["query"])
        endings = [_preprocess_hellaswag_text(x) for x in item["endings"]]
        label = int(item["gold"]) if "gold" in item else int(item["label"])
        return {"query": query, "endings": endings, "label": label}

    if "query" in item and "choices" in item:
        query = _preprocess_hellaswag_text(item["query"])
        endings = [_preprocess_hellaswag_text(x) for x in item["choices"]]
        label = int(item["gold"]) if "gold" in item else int(item["label"])
        return {"query": query, "endings": endings, "label": label}

    if all(k in item for k in ("activity_label", "ctx_a", "ctx_b", "endings")):
        ctx = f"{item['ctx_a']} {str(item['ctx_b']).capitalize()}"
        query = _preprocess_hellaswag_text(f"{item['activity_label']}: {ctx}")
    else:
        query = _preprocess_hellaswag_text(item.get("ctx", ""))

    endings = [_preprocess_hellaswag_text(x) for x in item["endings"]]
    return {"query": query, "endings": endings, "label": int(item["label"])}


def _format_subject(subject):
    """Convert subject slug to human-readable form."""
    return subject.replace("_", " ")


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def _require_datasets():
    """Import and return HuggingFace load_dataset, or raise with instructions."""
    try:
        from datasets import load_dataset
        return load_dataset
    except ImportError:
        raise RuntimeError(
            "HuggingFace `datasets` library is required for on-the-fly evaluation. "
            "Install with `pip install datasets`, or provide "
            "--on-the-fly-eval-data-dir with local JSONL files."
        )


def _subsample(data, num_samples, seed=42):
    """Randomly subsample data if num_samples is set and smaller than data."""
    if num_samples is not None and num_samples < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, num_samples)
    return data


def _load_local_jsonl(path):
    """Load a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

def load_mmlu(split="validation", num_samples=None, data_dir=None, seed=42):
    """Load MMLU dataset.

    Returns:
        List of dicts: {question, choices, answer (0-3), subject}.
    """
    if data_dir is not None:
        data = _load_local_jsonl(os.path.join(data_dir, f"mmlu_{split}.jsonl"))
    else:
        load_dataset = _require_datasets()
        ds = load_dataset("cais/mmlu", "all", split=split, trust_remote_code=True)
        data = [
            {"question": item["question"], "choices": list(item["choices"]),
             "answer": int(item["answer"]), "subject": item["subject"]}
            for item in ds
        ]
    return _subsample(data, num_samples, seed)


def load_mmlu_few_shot(data_dir=None, seed=42):
    """Load MMLU dev split grouped by subject for few-shot.

    Returns:
        Dict: {subject: [items]}.  Caller slices [:num_shots].
    """
    if data_dir is not None:
        data = _load_local_jsonl(os.path.join(data_dir, "mmlu_dev.jsonl"))
    else:
        load_dataset = _require_datasets()
        ds = load_dataset("cais/mmlu", "all", split="dev", trust_remote_code=True)
        data = [
            {"question": item["question"], "choices": list(item["choices"]),
             "answer": int(item["answer"]), "subject": item["subject"]}
            for item in ds
        ]

    by_subject = {}
    for item in data:
        by_subject.setdefault(item["subject"], []).append(item)

    return by_subject


# ---------------------------------------------------------------------------
# ARC (Easy / Challenge)
# ---------------------------------------------------------------------------

def load_arc(split="validation", config="ARC-Easy", num_samples=None,
             data_dir=None, seed=42):
    """Load ARC-Easy or ARC-Challenge.

    Returns:
        List of dicts: {question, choices, answer (0-indexed), subject}.
    """
    local_name = "arc_easy" if config == "ARC-Easy" else "arc_challenge"
    if data_dir is not None:
        data = _load_local_jsonl(os.path.join(data_dir, f"{local_name}_{split}.jsonl"))
    else:
        load_dataset = _require_datasets()
        ds = load_dataset("allenai/ai2_arc", config, split=split, trust_remote_code=True)
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
                     "1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
        data = []
        for item in ds:
            choices = list(item["choices"]["text"])
            answer_idx = label_map.get(item["answerKey"], 0)
            if answer_idx < len(choices):
                data.append({
                    "question": item["question"],
                    "choices": choices,
                    "answer": answer_idx,
                    "subject": "science",
                })
    return _subsample(data, num_samples, seed)


def load_arc_few_shot(config="ARC-Easy", data_dir=None, seed=42):
    """Load ARC train split for few-shot pool.

    Returns:
        List of items (flat, deterministic source order). Caller slices [:num_shots].
    """
    local_name = "arc_easy" if config == "ARC-Easy" else "arc_challenge"
    if data_dir is not None:
        data = _load_local_jsonl(os.path.join(data_dir, f"{local_name}_train.jsonl"))
    else:
        load_dataset = _require_datasets()
        ds = load_dataset("allenai/ai2_arc", config, split="train", trust_remote_code=True)
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
                     "1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
        data = []
        for item in ds:
            choices = list(item["choices"]["text"])
            answer_idx = label_map.get(item["answerKey"], 0)
            if answer_idx < len(choices):
                data.append({
                    "question": item["question"],
                    "choices": choices,
                    "answer": answer_idx,
                    "subject": "science",
                })
    return data


# ---------------------------------------------------------------------------
# PIQA
# ---------------------------------------------------------------------------

def load_piqa(split="validation", num_samples=None, data_dir=None, seed=42):
    """Load PIQA dataset.

    Returns:
        List of dicts: {question, choices (2), answer (0-1), subject}.
    """
    if data_dir is not None:
        data = _load_local_jsonl(os.path.join(data_dir, f"piqa_{split}.jsonl"))
    else:
        load_dataset = _require_datasets()
        ds = load_dataset("piqa", split=split, trust_remote_code=True)
        data = [
            {"question": f"Goal: {item['goal']}",
             "choices": [item["sol1"], item["sol2"]],
             "answer": int(item["label"]),
             "subject": "physical_intuition"}
            for item in ds
            if int(item["label"]) >= 0  # test labels are -1
        ]
    return _subsample(data, num_samples, seed)


def load_piqa_few_shot(data_dir=None, seed=42):
    """Load PIQA train split for few-shot pool."""
    if data_dir is not None:
        data = _load_local_jsonl(os.path.join(data_dir, "piqa_train.jsonl"))
    else:
        load_dataset = _require_datasets()
        ds = load_dataset("piqa", split="train", trust_remote_code=True)
        data = [
            {"question": f"Goal: {item['goal']}",
             "choices": [item["sol1"], item["sol2"]],
             "answer": int(item["label"]),
             "subject": "physical_intuition"}
            for item in ds
            if int(item["label"]) >= 0
        ]
    return data


# ---------------------------------------------------------------------------
# WinoGrande
# ---------------------------------------------------------------------------

def load_winogrande(split="validation", num_samples=None, data_dir=None, seed=42):
    """Load WinoGrande dataset.

    Returns:
        List of dicts: {question, choices (2), answer (0-1), subject}.
    """
    if data_dir is not None:
        data = _load_local_jsonl(os.path.join(data_dir, f"winogrande_{split}.jsonl"))
    else:
        load_dataset = _require_datasets()
        ds = load_dataset("winogrande", "winogrande_xl", split=split,
                          trust_remote_code=True)
        data = [
            {"question": item["sentence"],
             "choices": [item["option1"], item["option2"]],
             "answer": int(item["answer"]) - 1,  # "1"/"2" -> 0/1
             "subject": "commonsense"}
            for item in ds
            if item["answer"] in ("1", "2")
        ]
    return _subsample(data, num_samples, seed)


def load_winogrande_few_shot(data_dir=None, seed=42):
    """Load WinoGrande train split for few-shot pool."""
    if data_dir is not None:
        data = _load_local_jsonl(os.path.join(data_dir, "winogrande_train.jsonl"))
    else:
        load_dataset = _require_datasets()
        ds = load_dataset("winogrande", "winogrande_xl", split="train",
                          trust_remote_code=True)
        data = [
            {"question": item["sentence"],
             "choices": [item["option1"], item["option2"]],
             "answer": int(item["answer"]) - 1,
             "subject": "commonsense"}
            for item in ds
            if item["answer"] in ("1", "2")
        ]
    return data


# ---------------------------------------------------------------------------
# COPA
# ---------------------------------------------------------------------------

def load_copa(split="validation", num_samples=None, data_dir=None, seed=42):
    """Load COPA dataset.

    Returns:
        List of dicts: {question, choices (2), answer (0-1), subject}.
    """
    if data_dir is not None:
        data = _load_local_jsonl(os.path.join(data_dir, f"copa_{split}.jsonl"))
    else:
        load_dataset = _require_datasets()
        ds = load_dataset("super_glue", "copa", split=split, trust_remote_code=True)
        data = []
        for item in ds:
            if item["question"] == "cause":
                q = f"{item['premise']} What was the cause of this?"
            else:
                q = f"{item['premise']} What happened as a result?"
            data.append({
                "question": q,
                "choices": [item["choice1"], item["choice2"]],
                "answer": int(item["label"]),
                "subject": "causal_reasoning",
            })
    return _subsample(data, num_samples, seed)


def load_copa_few_shot(data_dir=None, seed=42):
    """Load COPA train split for few-shot pool."""
    if data_dir is not None:
        data = _load_local_jsonl(os.path.join(data_dir, "copa_train.jsonl"))
    else:
        load_dataset = _require_datasets()
        ds = load_dataset("super_glue", "copa", split="train", trust_remote_code=True)
        data = []
        for item in ds:
            if item["question"] == "cause":
                q = f"{item['premise']} What was the cause of this?"
            else:
                q = f"{item['premise']} What happened as a result?"
            data.append({
                "question": q,
                "choices": [item["choice1"], item["choice2"]],
                "answer": int(item["label"]),
                "subject": "causal_reasoning",
            })
    return data


# ---------------------------------------------------------------------------
# BoolQ
# ---------------------------------------------------------------------------

def load_boolq(split="validation", num_samples=None, data_dir=None, seed=42):
    """Load BoolQ dataset.

    Returns:
        List of dicts: {question, choices (2), answer (0-1), subject}.
    """
    if data_dir is not None:
        data = _load_local_jsonl(os.path.join(data_dir, f"boolq_{split}.jsonl"))
    else:
        load_dataset = _require_datasets()
        ds = load_dataset("google/boolq", split=split, trust_remote_code=True)
        data = [
            {"question": f"{item['passage']}\nQuestion: {item['question']}",
             "choices": ["No", "Yes"],
             "answer": int(item["answer"]),
             "subject": "yes_no_question"}
            for item in ds
        ]
    return _subsample(data, num_samples, seed)


def load_boolq_few_shot(data_dir=None, seed=42):
    """Load BoolQ train split for few-shot pool."""
    if data_dir is not None:
        data = _load_local_jsonl(os.path.join(data_dir, "boolq_train.jsonl"))
    else:
        load_dataset = _require_datasets()
        ds = load_dataset("google/boolq", split="train", trust_remote_code=True)
        data = [
            {"question": f"{item['passage']}\nQuestion: {item['question']}",
             "choices": ["No", "Yes"],
             "answer": int(item["answer"]),
             "subject": "yes_no_question"}
            for item in ds
        ]
    return data


# ---------------------------------------------------------------------------
# OpenBookQA
# ---------------------------------------------------------------------------

def load_openbookqa(split="test", num_samples=None, data_dir=None, seed=42):
    """Load OpenBookQA dataset.

    Returns:
        List of dicts: {question, choices (4), answer (0-3), subject}.
    """
    if data_dir is not None:
        data = _load_local_jsonl(os.path.join(data_dir, f"openbookqa_{split}.jsonl"))
    else:
        load_dataset = _require_datasets()
        ds = load_dataset("allenai/openbookqa", "main", split=split, trust_remote_code=True)
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        data = []
        for item in ds:
            choices = list(item["choices"]["text"])
            answer_idx = label_map.get(item["answerKey"], 0)
            if answer_idx < len(choices):
                data.append({
                    "question": item["question_stem"],
                    "choices": choices,
                    "answer": answer_idx,
                    "subject": "science",
                })
    return _subsample(data, num_samples, seed)


def load_openbookqa_few_shot(data_dir=None, seed=42):
    """Load OpenBookQA train split for few-shot pool."""
    if data_dir is not None:
        data = _load_local_jsonl(os.path.join(data_dir, "openbookqa_train.jsonl"))
    else:
        load_dataset = _require_datasets()
        ds = load_dataset("allenai/openbookqa", "main", split="train", trust_remote_code=True)
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        data = []
        for item in ds:
            choices = list(item["choices"]["text"])
            answer_idx = label_map.get(item["answerKey"], 0)
            if answer_idx < len(choices):
                data.append({
                    "question": item["question_stem"],
                    "choices": choices,
                    "answer": answer_idx,
                    "subject": "science",
                })
    return data


# ---------------------------------------------------------------------------
# HellaSwag
# ---------------------------------------------------------------------------

def load_hellaswag(split="validation", num_samples=None, data_dir=None, seed=42):
    """Load HellaSwag dataset.

    Returns:
        List of dicts: {query, endings (list of 4 str), label (int 0-3)}.
    """
    if data_dir is not None:
        raw_data = _load_local_jsonl(os.path.join(data_dir, f"hellaswag_{split}.jsonl"))
    else:
        load_dataset = _require_datasets()
        ds = load_dataset("Rowan/hellaswag", split=split, trust_remote_code=True)
        raw_data = list(ds)

    data = []
    for item in raw_data:
        normalized = _normalize_hellaswag_doc(item)
        if normalized["label"] >= 0:
            data.append(normalized)
    return _subsample(data, num_samples, seed)
