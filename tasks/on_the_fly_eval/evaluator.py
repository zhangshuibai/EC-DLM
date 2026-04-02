"""On-the-fly benchmark evaluation (MMLU / ARC / PIQA / WinoGrande / COPA /
HellaSwag) during training.

Callbacks for the Megatron training loop:
  - ``on_the_fly_eval_func``  →  ``non_loss_data_func`` (ALL ranks)
  - ``process_on_the_fly_eval_results``  →  ``process_non_loss_data_func``
    (last rank only, logs to TensorBoard / WandB)
"""

import math

import torch
import torch.nn.functional as F

from megatron.training import get_args, get_tokenizer, print_rank_0, is_last_rank
from megatron.core import mpu

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from .data_utils import (
    load_mmlu, load_mmlu_few_shot,
    load_arc, load_arc_few_shot,
    load_piqa, load_piqa_few_shot,
    load_winogrande, load_winogrande_few_shot,
    load_copa, load_copa_few_shot,
    load_boolq, load_boolq_few_shot,
    load_openbookqa, load_openbookqa_few_shot,
    load_hellaswag,
    format_mcq_prompt, get_task_header,
    format_hellaswag_prompt,
)

# Task categories
MCQ_TASKS = {"mmlu", "arc_easy", "arc_challenge", "piqa", "winogrande", "copa", "boolq", "openbookqa"}
LOGLIKELIHOOD_TASKS = {"hellaswag"}

# Module-level cache  {key: data}
_CACHE = {}

# Call counter: tracks how many times on_the_fly_eval_func has been invoked
# (incremented each eval_interval steps)
_EVAL_CALL_COUNT = 0


def _is_rank_0():
    """Check if current process is rank 0."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _progress(iterable, total=None, desc=""):
    """Wrap iterable with tqdm on rank 0, plain iterator on other ranks."""
    if _HAS_TQDM and _is_rank_0():
        return tqdm(iterable, total=total, desc=desc, leave=False)
    return iterable


def _get_shard_info(args):
    """Return on-the-fly eval sharding metadata.

    Returns:
        (mode, shard_rank, shard_world_size, shard_group)
    """
    mode = getattr(args, "on_the_fly_eval_shard_mode", "none")
    if mode == "none":
        return mode, 0, 1, None

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return mode, 0, 1, None

    if mode == "expert_dp":
        try:
            shard_group = mpu.get_expert_data_parallel_group()
            shard_rank = mpu.get_expert_data_parallel_rank()
            shard_world_size = torch.distributed.get_world_size(group=shard_group)
        except Exception as exc:
            raise ValueError(
                "--on-the-fly-eval-shard-mode expert_dp requires initialized expert "
                "data-parallel groups."
            ) from exc
        return mode, shard_rank, shard_world_size, shard_group

    raise ValueError(f"Unsupported on-the-fly eval shard mode: {mode}")


def _shard_data(data, shard_rank, shard_world_size):
    """Shard a list deterministically by rank using strided slicing."""
    if shard_world_size <= 1:
        return data
    return data[shard_rank::shard_world_size]


def _all_reduce_counts(values, group):
    """All-reduce sum over integer counters in *values*.

    Args:
        values: Sequence of integer-like counters.
        group:  Process group or None.

    Returns:
        List[int]: Summed counters.
    """
    if group is None or not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return [int(v) for v in values]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor([float(v) for v in values], device=device, dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group)
    return [int(v) for v in tensor.tolist()]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _get_base_model(model):
    """Unwrap DDP / FP16 wrappers to reach the raw GPTModel."""
    if isinstance(model, (list, tuple)):
        model = model[0]
    while hasattr(model, "module"):
        model = model.module
    return model


def _difflm_forward_for_eval(model, input_ids, position_ids):
    """Clean forward pass bypassing DiffLM diffusion noise.

    Calls: embedding → rotary_pos_emb → decoder(attention_mask=None) → output_layer

    When dynamic topk routing is enabled, computes mask_ratio from the MASK
    tokens present in ``input_ids`` so that the MoE router uses the same
    dynamic topk schedule as during training / val-loss evaluation.

    Args:
        model: The wrapped model (list / DDP).
        input_ids:    [batch, seq_len]  long tensor on CUDA.
        position_ids: [batch, seq_len]  long tensor on CUDA.

    Returns:
        logits: [batch, seq_len, vocab_size]  float tensor.
    """
    base = _get_base_model(model)

    # 1. Embedding  →  [s, b, h]  (Megatron seq-first)
    decoder_input = base.embedding(input_ids=input_ids, position_ids=position_ids)

    # 2. Rotary position embeddings
    rotary_pos_emb = None
    if base.position_embedding_type == 'rope' and not base.config.multi_latent_attention:
        seq_len = input_ids.shape[1]
        rotary_pos_emb = base.rotary_pos_emb(seq_len)

    # 3. Compute mask_ratio for dynamic topk routing (if enabled)
    mask_ratio = None
    masked_indices = None
    if base.config.moe_router_dynamic_topk:
        args = get_args()
        mask_token_id = args.mask_token
        masked_indices = (input_ids == mask_token_id)  # [b, seq_len]
        mask_ratio = masked_indices.float().sum(dim=1) / input_ids.shape[1]  # [b]

    # 4. Decoder forward – bidirectional (attention_mask=None)
    hidden_states = base.decoder(
        hidden_states=decoder_input,
        attention_mask=None,
        inference_params=None,
        rotary_pos_emb=rotary_pos_emb,
        mask_ratio=mask_ratio,
        masked_indices=masked_indices,
    )

    # 5. Output layer  →  logits [s, b, vocab]
    output_weight = None
    if base.share_embeddings_and_output_weights:
        output_weight = base.shared_embedding_or_output_weight()

    logits, _ = base.output_layer(
        hidden_states, weight=output_weight, runtime_gather_output=True,
    )

    # [s, b, vocab] → [b, s, vocab]
    return logits.transpose(0, 1).contiguous()


# ---------------------------------------------------------------------------
# Batch size helper
# ---------------------------------------------------------------------------

def _get_eval_batch_size(args):
    """Return evaluation batch size: explicit arg or 2x micro_batch_size."""
    if getattr(args, "on_the_fly_eval_batch_size", None) is not None:
        return args.on_the_fly_eval_batch_size
    return getattr(args, "micro_batch_size", 4) * 2


def _parse_non_negative_int_list(raw, *, arg_name, sep=","):
    """Parse a separated integer list (non-negative values only)."""
    values = []
    for token in str(raw).split(sep):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid integer in {arg_name}: '{token}'") from exc
        if value < 0:
            raise ValueError(f"{arg_name} must contain non-negative integers, got {value}.")
        values.append(value)
    return values


def _get_mcq_shots_by_task(args):
    """Return per-task shot overrides from args.on_the_fly_eval_mcq_shots_by_task.

    Format: "task:shots,task:shots", where shots can be:
      - single integer:    mmlu:5
      - pipe-separated:    mmlu:0|5
    """
    raw = getattr(args, "on_the_fly_eval_mcq_shots_by_task", "")
    if raw is None:
        return {}
    raw = raw.strip()
    if not raw:
        return {}

    overrides = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "--on-the-fly-eval-mcq-shots-by-task format must be task:shots, got "
                f"'{item}'."
            )
        task_name, shot_spec = item.split(":", 1)
        task_name = task_name.strip()
        if task_name not in MCQ_TASKS:
            raise ValueError(
                "--on-the-fly-eval-mcq-shots-by-task contains unsupported task "
                f"'{task_name}'. Supported: {sorted(MCQ_TASKS)}"
            )
        shots = _parse_non_negative_int_list(
            shot_spec, arg_name=f"--on-the-fly-eval-mcq-shots-by-task[{task_name}]", sep="|"
        )
        if not shots:
            raise ValueError(
                "--on-the-fly-eval-mcq-shots-by-task has empty shots for "
                f"task '{task_name}'."
            )
        # Deduplicate while preserving order.
        overrides[task_name] = list(dict.fromkeys(shots))

    return overrides


# ---------------------------------------------------------------------------
# Answer token resolution
# ---------------------------------------------------------------------------

def _resolve_answer_token_ids(tokenizer, max_choices=5):
    """Get token IDs for answer letters A … E (with leading space).

    Uses ``add_special_tokens=False`` to avoid BOS / EOS.
    """
    ids = []
    for letter in list("ABCDEFGH")[:max_choices]:
        toks = tokenizer.tokenize(" " + letter, add_special_tokens=False)
        if len(toks) >= 1:
            ids.append(toks[0])
        else:
            toks = tokenizer.tokenize(letter, add_special_tokens=False)
            ids.append(toks[0])
    return ids


# ---------------------------------------------------------------------------
# Dataset / few-shot loading with caching
# ---------------------------------------------------------------------------

def _get_dataset(task_name, args, shard_rank=0, shard_world_size=1):
    """Load and cache the evaluation dataset for *task_name*."""
    if task_name in _CACHE:
        return _shard_data(_CACHE[task_name], shard_rank, shard_world_size)

    data_dir = getattr(args, "on_the_fly_eval_data_dir", None)
    max_samples = getattr(args, "on_the_fly_eval_max_samples", None)
    mmlu_samples = (
        args.on_the_fly_eval_mmlu_samples
        if args.on_the_fly_eval_mmlu_samples is not None
        else max_samples
    )
    hellaswag_samples = (
        args.on_the_fly_eval_hellaswag_samples
        if args.on_the_fly_eval_hellaswag_samples is not None
        else max_samples
    )

    if task_name == "mmlu":
        data = load_mmlu("test", mmlu_samples, data_dir)
    elif task_name == "arc_easy":
        data = load_arc("test", "ARC-Easy", max_samples, data_dir)
    elif task_name == "arc_challenge":
        data = load_arc("test", "ARC-Challenge", max_samples, data_dir)
    elif task_name == "piqa":
        data = load_piqa("validation", max_samples, data_dir)
    elif task_name == "winogrande":
        data = load_winogrande("validation", max_samples, data_dir)
    elif task_name == "copa":
        data = load_copa("validation", max_samples, data_dir)
    elif task_name == "boolq":
        data = load_boolq("validation", max_samples, data_dir)
    elif task_name == "openbookqa":
        data = load_openbookqa("test", max_samples, data_dir)
    elif task_name == "hellaswag":
        data = load_hellaswag("validation", hellaswag_samples, data_dir)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    _CACHE[task_name] = data
    return _shard_data(data, shard_rank, shard_world_size)


def _get_few_shot_pool(task_name, args):
    """Load and cache the few-shot candidate pool for *task_name*.

    Returns:
        - dict {subject: [items]}  for MMLU  (per-subject few-shot)
        - list [items]             for other MCQ tasks  (flat pool)
    """
    key = f"{task_name}_fs"
    if key in _CACHE:
        return _CACHE[key]

    data_dir = getattr(args, "on_the_fly_eval_data_dir", None)

    if task_name == "mmlu":
        pool = load_mmlu_few_shot(data_dir)
    elif task_name == "arc_easy":
        pool = load_arc_few_shot("ARC-Easy", data_dir)
    elif task_name == "arc_challenge":
        pool = load_arc_few_shot("ARC-Challenge", data_dir)
    elif task_name == "piqa":
        pool = load_piqa_few_shot(data_dir)
    elif task_name == "winogrande":
        pool = load_winogrande_few_shot(data_dir)
    elif task_name == "copa":
        pool = load_copa_few_shot(data_dir)
    elif task_name == "boolq":
        pool = load_boolq_few_shot(data_dir)
    elif task_name == "openbookqa":
        pool = load_openbookqa_few_shot(data_dir)
    else:
        pool = []

    _CACHE[key] = pool
    return pool


def _select_few_shot(few_shot_pool, item, num_shots):
    """Select few-shot examples for one item.

    Handles both dict (MMLU per-subject) and list (flat pool) formats.
    """
    if few_shot_pool is None or num_shots <= 0:
        return None
    if isinstance(few_shot_pool, dict):
        candidates = few_shot_pool.get(item.get("subject", ""), [])
    else:
        candidates = few_shot_pool
    examples = candidates[:num_shots]
    return examples if examples else None


# ---------------------------------------------------------------------------
# Generic MCQ evaluation
# ---------------------------------------------------------------------------

def _evaluate_mcq(model, tokenizer, args, task_name, dataset, few_shot_pool,
                  num_shots, shard_group=None, shard_world_size=1):
    """Evaluate a multiple-choice task using mask-based scoring.

    Strategy (1 forward pass per batch):
      - Prompt ends with "Answer: [MASK]"
      - Extract logits at MASK position for answer letter tokens
      - argmax → predicted answer

    Returns:
        dict with accuracy, correct, total, per_subject.
    """
    mask_token_id = args.mask_token
    max_seq_len = args.seq_length - 1  # model processes (seq_length - 1) tokens
    batch_size = _get_eval_batch_size(args)
    answer_token_ids = _resolve_answer_token_ids(tokenizer, max_choices=5)

    total_correct = 0
    total_count = 0
    subject_results = {}

    total_batches = math.ceil(len(dataset) / batch_size)
    batch_iter = _progress(
        range(0, len(dataset), batch_size),
        total=total_batches,
        desc=f"    {task_name} {num_shots}-shot",
    )

    for batch_start in batch_iter:
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_items = dataset[batch_start:batch_end]
        cur_bs = len(batch_items)

        input_ids_list = []
        mask_positions = []
        answers = []
        num_choices_list = []

        for item in batch_items:
            # Few-shot selection
            few_shot = _select_few_shot(few_shot_pool, item, num_shots)

            # Format prompt
            header = get_task_header(task_name, item)
            prompt = format_mcq_prompt(item, header, few_shot)
            tokens = tokenizer.tokenize(prompt, add_special_tokens=False)

            # Truncate from LEFT if too long (preserve the question + choices)
            if len(tokens) > max_seq_len - 1:
                tokens = tokens[-(max_seq_len - 1):]

            # Append MASK token for the answer position
            tokens.append(mask_token_id)
            mask_pos = len(tokens) - 1

            # Pad to max_seq_len
            pad_len = max_seq_len - len(tokens)
            tokens = tokens + [tokenizer.eod] * pad_len

            input_ids_list.append(tokens)
            mask_positions.append(mask_pos)
            answers.append(item["answer"])
            num_choices_list.append(len(item["choices"]))

        # Build tensors
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device="cuda")
        position_ids = (torch.arange(max_seq_len, device="cuda")
                        .unsqueeze(0).expand(cur_bs, -1))

        # Forward pass (ALL EP ranks must execute for MoE all-to-all)
        logits = _difflm_forward_for_eval(model, input_ids, position_ids)

        # Score each item
        for i in range(cur_bs):
            n_choices = num_choices_list[i]
            mask_logits = logits[i, mask_positions[i], :]
            choice_logits = mask_logits[answer_token_ids[:n_choices]]
            predicted = choice_logits.argmax().item()

            if predicted == answers[i]:
                total_correct += 1
            total_count += 1

            subj = batch_items[i].get("subject", "default")
            if subj not in subject_results:
                subject_results[subj] = {"correct": 0, "total": 0}
            subject_results[subj]["total"] += 1
            if predicted == answers[i]:
                subject_results[subj]["correct"] += 1

        if hasattr(batch_iter, "set_postfix"):
            batch_iter.set_postfix(acc=f"{total_correct/max(total_count,1):.3f}")

    global_correct, global_total = _all_reduce_counts([total_correct, total_count], shard_group)
    accuracy = global_correct / max(global_total, 1)
    per_subject_acc = {
        s: r["correct"] / max(r["total"], 1)
        for s, r in subject_results.items()
    } if shard_world_size <= 1 else {}

    return {
        "accuracy": accuracy,
        "correct": global_correct,
        "total": global_total,
        "per_subject": per_subject_acc,
    }


# ---------------------------------------------------------------------------
# HellaSwag evaluation (log-likelihood based, always 0-shot)
# ---------------------------------------------------------------------------

def _evaluate_hellaswag(model, tokenizer, args, dataset, shard_group=None):
    """Evaluate HellaSwag accuracy.

    Strategy (log-likelihood, multi-question batching):
      - Each question has 4 continuations, each becomes one row in the batch.
      - Multiple questions are packed into one forward pass to fill eval_batch_size.
      - Input:  [context_tokens] + [MASK × len(continuation)]
      - Labels: [context_tokens] + [continuation_tokens]
      - Score = Σ log P(true_token) at continuation positions
      - Pick continuation with highest score per question.

    Returns:
        dict with acc, acc_norm, accuracy(acc_norm), correct(acc_norm), total.
    """
    mask_token_id = args.mask_token
    max_seq_len = args.seq_length - 1
    eval_bs = _get_eval_batch_size(args)
    # Each question produces 4 rows; pack multiple questions per forward pass
    questions_per_batch = max(eval_bs // 4, 1)

    total_correct_acc = 0
    total_correct_acc_norm = 0
    total_count = 0

    total_batches = math.ceil(len(dataset) / questions_per_batch)
    batch_iter = _progress(
        range(0, len(dataset), questions_per_batch),
        total=total_batches,
        desc="    hellaswag 0-shot",
    )

    for batch_q_start in batch_iter:
        batch_q_end = min(batch_q_start + questions_per_batch, len(dataset))
        batch_questions = dataset[batch_q_start:batch_q_end]

        all_input_ids = []
        all_label_ids = []
        all_cont_masks = []

        for item in batch_questions:
            ctx_str = item.get("query", item.get("ctx", ""))
            for ending in item["endings"]:
                context_str, ending_str = format_hellaswag_prompt(ctx_str, ending)
                ctx_tokens = tokenizer.tokenize(context_str, add_special_tokens=False)
                cont_tokens = tokenizer.tokenize(ending_str, add_special_tokens=False)

                # Truncate context from left if total exceeds max_seq_len
                total_len = len(ctx_tokens) + len(cont_tokens)
                if total_len > max_seq_len:
                    excess = total_len - max_seq_len
                    ctx_tokens = ctx_tokens[excess:]

                # Input: context (real) + continuation (MASK)
                masked_input = ctx_tokens + [mask_token_id] * len(cont_tokens)
                label_tokens = ctx_tokens + cont_tokens
                cont_mask = [0] * len(ctx_tokens) + [1] * len(cont_tokens)

                # Pad to max_seq_len
                pad_len = max_seq_len - len(masked_input)
                masked_input = masked_input + [tokenizer.eod] * pad_len
                label_tokens = label_tokens + [tokenizer.eod] * pad_len
                cont_mask = cont_mask + [0] * pad_len

                all_input_ids.append(masked_input)
                all_label_ids.append(label_tokens)
                all_cont_masks.append(cont_mask)

        cur_bs = len(all_input_ids)  # questions_in_batch * 4
        input_ids = torch.tensor(all_input_ids, dtype=torch.long, device="cuda")
        label_ids = torch.tensor(all_label_ids, dtype=torch.long, device="cuda")
        cont_masks = torch.tensor(all_cont_masks, dtype=torch.float, device="cuda")
        position_ids = (torch.arange(max_seq_len, device="cuda")
                        .unsqueeze(0).expand(cur_bs, -1))

        # Forward pass (ALL EP ranks)
        logits = _difflm_forward_for_eval(model, input_ids, position_ids)

        # Compute log-probs at continuation positions.
        log_probs = F.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(2, label_ids.unsqueeze(-1)).squeeze(-1)
        sum_scores = (gathered * cont_masks).sum(dim=-1)
        cont_len = cont_masks.sum(dim=-1).clamp(min=1)
        norm_scores = sum_scores / cont_len  # [cur_bs] mean log P per token

        # Score per question: reshape to [num_questions, 4] and argmax
        num_q = len(batch_questions)
        sum_scores = sum_scores.view(num_q, 4)
        norm_scores = norm_scores.view(num_q, 4)
        predictions_acc = sum_scores.argmax(dim=-1)
        predictions_acc_norm = norm_scores.argmax(dim=-1)

        for i, item in enumerate(batch_questions):
            if predictions_acc[i].item() == item["label"]:
                total_correct_acc += 1
            if predictions_acc_norm[i].item() == item["label"]:
                total_correct_acc_norm += 1
            total_count += 1

        if hasattr(batch_iter, "set_postfix"):
            batch_iter.set_postfix(
                acc=f"{total_correct_acc/max(total_count,1):.3f}",
                acc_norm=f"{total_correct_acc_norm/max(total_count,1):.3f}",
            )

    global_correct_acc, global_correct_acc_norm, global_total = _all_reduce_counts(
        [total_correct_acc, total_correct_acc_norm, total_count], shard_group
    )
    acc = global_correct_acc / max(global_total, 1)
    acc_norm = global_correct_acc_norm / max(global_total, 1)
    return {
        "accuracy": acc_norm,
        "acc": acc,
        "acc_norm": acc_norm,
        "correct": global_correct_acc_norm,
        "total": global_total,
    }


# ---------------------------------------------------------------------------
# Training loop callbacks
# ---------------------------------------------------------------------------

def on_the_fly_eval_func(model):
    """``non_loss_data_func`` callback.

    Called on **all** ranks inside ``torch.no_grad()`` during evaluation.
    All EP ranks must execute the same forward passes for MoE all-to-all.

    For MCQ tasks: uses per-task shots from --on-the-fly-eval-mcq-shots-by-task
    (fallback: --on-the-fly-eval-mcq-shots).
    For HellaSwag: runs 0-shot only.

    Returns:
        dict  {"{task}_{n}shot": {"accuracy": ..., "correct": ..., "total": ..., ...}}
        or None if disabled.
    """
    args = get_args()

    if not getattr(args, "use_on_the_fly_eval", False):
        return None

    force_run = getattr(args, "_on_the_fly_eval_force_run", False)

    # Interval gating: this function is called every eval_interval steps.
    # If on_the_fly_eval_interval is set, only run when the step aligns.
    if not force_run:
        global _EVAL_CALL_COUNT
        _EVAL_CALL_COUNT += 1
        otf_interval = getattr(args, "on_the_fly_eval_interval", None)
        if otf_interval is not None:
            eval_interval = args.eval_interval
            if eval_interval is None:
                raise ValueError(
                    "--eval-interval must be set when --on-the-fly-eval-interval is used."
                )
            if otf_interval <= 0:
                raise ValueError(
                    f"--on-the-fly-eval-interval must be > 0, got {otf_interval}."
                )
            if eval_interval <= 0:
                raise ValueError(
                    f"--eval-interval must be > 0 when on-the-fly eval is enabled, got {eval_interval}."
                )
            if otf_interval % eval_interval != 0:
                raise ValueError(
                    "--on-the-fly-eval-interval must be an integer multiple of --eval-interval. "
                    f"Got on_the_fly_eval_interval={otf_interval}, eval_interval={eval_interval}."
                )
            # Run every (otf_interval / eval_interval) calls
            run_every_n = otf_interval // eval_interval
            if run_every_n > 1 and _EVAL_CALL_COUNT % run_every_n != 0:
                current_step = _EVAL_CALL_COUNT * eval_interval
                print_rank_0(f"> Skipping on-the-fly eval at ~step {current_step} "
                            f"(next at ~step {(((_EVAL_CALL_COUNT // run_every_n) + 1) * otf_interval)})")
                return None
    else:
        print_rank_0("> Running forced on-the-fly eval before training start.")

    tasks = [t.strip() for t in args.on_the_fly_eval_tasks.split(",") if t.strip()]
    if not tasks:
        return None

    default_mcq_shots = _parse_non_negative_int_list(
        args.on_the_fly_eval_mcq_shots,
        arg_name="--on-the-fly-eval-mcq-shots",
        sep=",",
    )
    mcq_shots_by_task = _get_mcq_shots_by_task(args)

    tokenizer = get_tokenizer()
    shard_mode, shard_rank, shard_world_size, shard_group = _get_shard_info(args)
    if shard_world_size > 1:
        print_rank_0(
            f"> On-the-fly eval sharding enabled (mode={shard_mode}, shards={shard_world_size})."
        )
    results = {}

    print_rank_0("> Starting on-the-fly evaluation ...")

    for task in tasks:
        if task in MCQ_TASKS:
            dataset = _get_dataset(task, args, shard_rank, shard_world_size)
            task_shots = mcq_shots_by_task.get(task, default_mcq_shots)
            if not task_shots:
                print_rank_0(
                    f"  No shot config found for {task}, skipping. "
                    "Set --on-the-fly-eval-mcq-shots or --on-the-fly-eval-mcq-shots-by-task."
                )
                continue

            for n_shots in task_shots:
                few_shot_pool = _get_few_shot_pool(task, args) if n_shots > 0 else None

                print_rank_0(f"  Evaluating {task} ({n_shots}-shot) ...")
                task_results = _evaluate_mcq(
                    model,
                    tokenizer,
                    args,
                    task,
                    dataset,
                    few_shot_pool,
                    n_shots,
                    shard_group=shard_group,
                    shard_world_size=shard_world_size,
                )
                result_key = f"{task}_{n_shots}shot"
                results[result_key] = task_results
                print_rank_0(
                    f"    {task} {n_shots}-shot accuracy: "
                    f"{task_results['accuracy']:.4f} "
                    f"({task_results['correct']}/{task_results['total']})"
                )

            torch.cuda.empty_cache()

        elif task in LOGLIKELIHOOD_TASKS:
            dataset = _get_dataset(task, args, shard_rank, shard_world_size)

            print_rank_0(f"  Evaluating {task} (0-shot) ...")
            task_results = _evaluate_hellaswag(
                model, tokenizer, args, dataset, shard_group=shard_group
            )
            results[f"{task}_0shot"] = task_results
            print_rank_0(
                f"    {task} 0-shot acc_norm: {task_results['acc_norm']:.4f}, "
                f"acc: {task_results['acc']:.4f} "
                f"(acc_norm correct {task_results['correct']}/{task_results['total']})"
            )
            torch.cuda.empty_cache()

        else:
            print_rank_0(f"  Unknown task: {task}, skipping.")

    print_rank_0("> On-the-fly evaluation complete.")
    return results


def process_on_the_fly_eval_results(collected_data, iteration, writer):
    """``process_non_loss_data_func`` callback.

    Called only on ``is_last_rank()`` with the TensorBoard writer.
    Logs metrics to TensorBoard and WandB.
    """
    if collected_data is None:
        return

    from megatron.training.global_vars import get_wandb_writer
    wandb_writer = get_wandb_writer()

    # Collect MCQ accuracies grouped by shot count for averaging
    # key format: "{task}_{n}shot" → e.g. "mmlu_0shot", "arc_easy_3shot"
    mcq_by_shot = {}  # {n_shots: [accuracy, ...]}

    for key, task_results in collected_data.items():
        acc = task_results.get("accuracy")
        if acc is not None:
            tag = f"eval/{key}_accuracy"
            if writer:
                writer.add_scalar(tag, acc, iteration)
            if wandb_writer:
                wandb_writer.log({tag: acc}, iteration)

            # Collect MCQ results for averaging (exclude hellaswag)
            # key = "{task}_{n}shot", extract task name and shot count
            parts = key.rsplit("_", 1)  # e.g. ["arc_easy", "0shot"]
            if len(parts) == 2 and parts[0] in MCQ_TASKS:
                shot_key = parts[1]  # e.g. "0shot"
                mcq_by_shot.setdefault(shot_key, []).append(acc)

        for metric_name in ("acc", "acc_norm"):
            metric_value = task_results.get(metric_name)
            if metric_value is not None:
                tag = f"eval/{key}_{metric_name}"
                if writer:
                    writer.add_scalar(tag, metric_value, iteration)
                if wandb_writer:
                    wandb_writer.log({tag: metric_value}, iteration)

        # Per-subject accuracy (mainly useful for MMLU)
        per_subject = task_results.get("per_subject", {})
        for subject, subj_acc in per_subject.items():
            tag = f"eval/{key}_subject/{subject}"
            if writer:
                writer.add_scalar(tag, subj_acc, iteration)
            if wandb_writer:
                wandb_writer.log({tag: subj_acc}, iteration)

    # Log average accuracy across MCQ tasks for each shot count
    for shot_key, accs in sorted(mcq_by_shot.items()):
        avg_acc = sum(accs) / len(accs)
        tag = f"eval/mcq_avg_{shot_key}_accuracy"
        if writer:
            writer.add_scalar(tag, avg_acc, iteration)
        if wandb_writer:
            wandb_writer.log({tag: avg_acc}, iteration)
