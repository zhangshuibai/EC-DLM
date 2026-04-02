# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math
import json
import os
from typing import Optional

import numpy as np
import torch

from megatron.core import parallel_state


# Module-level cache for auto-computed EC scaling factors
_EC_SCALING_FACTORS_CACHE = {}

# Module-level cache for auto-computed TC scaling factors
_TC_SCALING_FACTOR_CACHE = {}

_SCALING_FACTOR_CACHE_VERSION = 1


def _normalize_cache_path(cache_path: Optional[str]) -> Optional[str]:
    if cache_path is None:
        return None
    cache_path = str(cache_path).strip()
    if not cache_path:
        return None
    return os.path.expanduser(cache_path)


def _tc_cache_key(
    num_routing_experts: int, topk: int, num_shared_experts: int, act: str, renorm: bool
) -> str:
    return (
        f"experts={num_routing_experts}|topk={topk}|shared={num_shared_experts}|"
        f"act={act}|renorm={int(renorm)}"
    )


def _ec_cache_key(num_routing_experts: int, num_shared_experts: int) -> str:
    return f"experts={num_routing_experts}|shared={num_shared_experts}"


def _load_scaling_cache_file(cache_path: str) -> dict:
    if not os.path.exists(cache_path):
        return {
            "version": _SCALING_FACTOR_CACHE_VERSION,
            "tc_scaling_factors": {},
            "ec_scaling_factors": {},
        }

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # Corrupted cache should not break training.
        return {
            "version": _SCALING_FACTOR_CACHE_VERSION,
            "tc_scaling_factors": {},
            "ec_scaling_factors": {},
        }

    if not isinstance(data, dict):
        data = {}
    data.setdefault("version", _SCALING_FACTOR_CACHE_VERSION)
    data.setdefault("tc_scaling_factors", {})
    data.setdefault("ec_scaling_factors", {})
    return data


def _save_scaling_cache_file(cache_path: str, payload: dict) -> None:
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, cache_path)


def _read_scaling_factor_from_disk(cache_path: Optional[str], section: str, cache_key: str):
    cache_path = _normalize_cache_path(cache_path)
    if cache_path is None:
        return None

    payload = _load_scaling_cache_file(cache_path)
    section_payload = payload.get(section, {})
    if not isinstance(section_payload, dict):
        return None
    return section_payload.get(cache_key, None)


def _write_scaling_factor_to_disk(
    cache_path: Optional[str], section: str, cache_key: str, value: dict
) -> None:
    cache_path = _normalize_cache_path(cache_path)
    if cache_path is None:
        return

    lock_path = f"{cache_path}.lock"
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    # Best-effort exclusive update to avoid clobbering updates from other ranks.
    with open(lock_path, "a+", encoding="utf-8") as lock_f:
        try:
            import fcntl

            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        except Exception:
            # If locking fails, continue best-effort.
            pass

        payload = _load_scaling_cache_file(cache_path)
        section_payload = payload.get(section, {})
        if not isinstance(section_payload, dict):
            section_payload = {}
        section_payload[cache_key] = value
        payload[section] = section_payload
        payload["version"] = _SCALING_FACTOR_CACHE_VERSION
        _save_scaling_cache_file(cache_path, payload)

        try:
            import fcntl

            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass


def compute_ec_scaling_factors(
    num_routing_experts, num_shared_experts, num_samples=10000, cache_path: Optional[str] = None
):
    """Compute scaling factors for Expert-Choice MoE routing via Monte Carlo simulation.

    In Expert-Choice routing, the number of experts assigned to a token is variable.
    This function estimates the optimal scaling factor for each possible number of
    activated routing experts (k_routed), to balance shared and routing expert norms.

    Args:
        num_routing_experts (int): Number of routing experts.
        num_shared_experts (int): Number of shared experts (always activated).
        num_samples (int): Number of Monte Carlo simulations for estimation.

    Returns:
        torch.Tensor: A tensor of length (num_routing_experts + 1), where index k
            is the scaling factor for tokens assigned to k routing experts.
    """
    cache_key = (num_routing_experts, num_shared_experts)
    if cache_key in _EC_SCALING_FACTORS_CACHE:
        return _EC_SCALING_FACTORS_CACHE[cache_key]
    disk_cache_key = _ec_cache_key(num_routing_experts, num_shared_experts)

    disk_entry = _read_scaling_factor_from_disk(
        cache_path, "ec_scaling_factors", disk_cache_key
    )
    if isinstance(disk_entry, dict) and "scaling_factors" in disk_entry:
        try:
            result = torch.tensor(disk_entry["scaling_factors"], dtype=torch.float32)
            _EC_SCALING_FACTORS_CACHE[cache_key] = result
            print(
                f"[compute_ec_scaling_factors] loaded from cache: {cache_path} "
                f"(num_routing_experts={num_routing_experts}, num_shared_experts={num_shared_experts})"
            )
            return result
        except Exception:
            pass

    # Use fixed seed for determinism across ranks
    rng_state = np.random.get_state()
    np.random.seed(42)

    scaling_factors = [1.0]  # Index 0: no routing expert (doesn't matter, gates are 0)

    shared_norm = np.sqrt(num_shared_experts)

    for k_routed in range(1, num_routing_experts + 1):
        observed_factors = []
        for _ in range(num_samples):
            logits = np.random.randn(num_routing_experts)
            e_x = np.exp(logits - np.max(logits))
            all_probs = e_x / e_x.sum()

            selected_probs = all_probs[:k_routed]
            prob_sum = selected_probs.sum()
            if prob_sum > 0:
                selected_probs = selected_probs / prob_sum

            routing_norm = np.sqrt(np.sum(selected_probs ** 2))
            if routing_norm > 1e-9:
                observed_factors.append(shared_norm / routing_norm)

        avg_factor = np.mean(observed_factors) if observed_factors else 1.0
        scaling_factors.append(avg_factor)

    # Restore numpy RNG state
    np.random.set_state(rng_state)

    result = torch.tensor(scaling_factors, dtype=torch.float32)
    _EC_SCALING_FACTORS_CACHE[cache_key] = result
    _write_scaling_factor_to_disk(
        cache_path,
        "ec_scaling_factors",
        disk_cache_key,
        {
            "num_routing_experts": int(num_routing_experts),
            "num_shared_experts": int(num_shared_experts),
            "num_samples": int(num_samples),
            "scaling_factors": [float(x) for x in scaling_factors],
        },
    )

    print(f"[compute_ec_scaling_factors] num_routing_experts={num_routing_experts}, "
          f"num_shared_experts={num_shared_experts}, num_samples={num_samples}")
    for k in range(len(scaling_factors)):
        print(f"  k_routed={k}: scaling_factor={scaling_factors[k]:.4f}")

    return result


def compute_tc_scaling_factor(
    num_routing_experts,
    topk,
    num_shared_experts,
    act='softmax',
    renorm=True,
    num_samples=10000,
    cache_path: Optional[str] = None,
):
    """Compute scaling factor for Token-Choice MoE routing via Monte Carlo simulation.

    In TC routing, each token selects exactly topk routing experts. This function
    estimates the scaling factor to balance shared and routing expert output norms.

    Args:
        num_routing_experts (int): Number of routing experts.
        topk (int): Number of experts selected per token.
        num_shared_experts (int): Number of shared experts (always activated).
        act (str): Activation function ('softmax' or 'sigmoid').
        renorm (bool): Whether to renormalize the selected probabilities. Default True.
        num_samples (int): Number of Monte Carlo simulations for estimation.

    Returns:
        float: The scaling factor.
    """
    cache_key = (num_routing_experts, topk, num_shared_experts, act, renorm)
    if cache_key in _TC_SCALING_FACTOR_CACHE:
        return _TC_SCALING_FACTOR_CACHE[cache_key]
    disk_cache_key = _tc_cache_key(num_routing_experts, topk, num_shared_experts, act, renorm)

    disk_entry = _read_scaling_factor_from_disk(
        cache_path, "tc_scaling_factors", disk_cache_key
    )
    if isinstance(disk_entry, dict) and "scaling_factor" in disk_entry:
        try:
            result = float(disk_entry["scaling_factor"])
            _TC_SCALING_FACTOR_CACHE[cache_key] = result
            print(
                f"[compute_tc_scaling_factor] loaded from cache: {cache_path} "
                f"(num_routing_experts={num_routing_experts}, topk={topk}, "
                f"num_shared_experts={num_shared_experts}, act={act}, renorm={renorm}, "
                f"scaling_factor={result:.4f})"
            )
            return result
        except Exception:
            pass

    rng_state = np.random.get_state()
    np.random.seed(42)

    factors = []
    for _ in range(num_samples):
        logits = np.random.randn(num_routing_experts)
        if act == 'softmax':
            e_x = np.exp(logits - np.max(logits))
            p = e_x / e_x.sum()
        elif act == 'sigmoid':
            p = 1 / (1 + np.exp(-logits))
        else:
            raise ValueError(f"Unknown activation function: {act}")
        p = np.sort(p)[::-1][:topk]
        if renorm:
            p_sum = p.sum()
            if p_sum > 0:
                p = p / p_sum
        norm = np.sqrt(np.sum(p ** 2))
        if norm > 1e-9:
            factors.append(np.sqrt(num_shared_experts) / norm)

    np.random.set_state(rng_state)

    result = np.mean(factors) if factors else 1.0
    _TC_SCALING_FACTOR_CACHE[cache_key] = result
    _write_scaling_factor_to_disk(
        cache_path,
        "tc_scaling_factors",
        disk_cache_key,
        {
            "num_routing_experts": int(num_routing_experts),
            "topk": int(topk),
            "num_shared_experts": int(num_shared_experts),
            "act": str(act),
            "renorm": bool(renorm),
            "num_samples": int(num_samples),
            "scaling_factor": float(result),
        },
    )

    print(f"[compute_tc_scaling_factor] num_routing_experts={num_routing_experts}, "
          f"topk={topk}, num_shared_experts={num_shared_experts}, "
          f"act={act}, renorm={renorm}, scaling_factor={result:.4f}")

    return result


def switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    moe_aux_loss_coeff: float,
    sequence_partition_group=None,
):
    """Calculate the auxiliary loss for load balancing.
    Refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

    Args:
        probs (torch.Tensor): Softmax probabilities output by the router for each token.
                              Shape in [num_tokens, num_experts].
        tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.
                                          Shape in [num_experts]
        topk (int): The number of experts selected for each token.
        moe_aux_loss_coeff (float): The coefficient for the auxiliary loss.
        sequence_partition_group (optional): The parallel group over which the sequence is
                                             partitioned. If None, no partitioning is applied.
                                             Defaults to None.

    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    num_sub_sequence = 1

    # If the sequence is partitioned by certain parallelism strategies like Sequence Parallelism
    # or Context Parallelism, compute the gradient of the auxiliary loss with respect to the full
    # sequence.
    if sequence_partition_group is not None:
        # We can keep `aggregated_probs_per_expert` local since we don't need the gradient for
        # `tokens_per_expert`, saving one allreduce operation for `aggregated_probs_per_expert`.
        num_sub_sequence = torch.distributed.get_world_size(sequence_partition_group)
        torch.distributed.all_reduce(tokens_per_expert, group=sequence_partition_group)

    num_tokens = probs.shape[0] * num_sub_sequence
    num_experts = probs.shape[1]

    # The formula of aux_loss: aux_loss = sum((probs_per_expert/num_tokens) *
    # (tokens_per_expert/(num_tokens*topk))) * num_experts * moe_aux_loss_coeff.
    # This can be simplified to fuse the division and multiplication operations.
    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (num_tokens * num_tokens * topk)
    )
    return aux_loss

def ec_load_balancing_loss_func(
    probs: torch.Tensor,
    experts_per_token: torch.Tensor,
    topk: int,
    moe_aux_loss_coeff: float,
    sequence_partition_group=None,
):
    """Calculate the auxiliary loss for expert choice load balancing.

    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    # num_sub_sequence = 1

    # If the sequence is partitioned by certain parallelism strategies like Sequence Parallelism
    # or Context Parallelism, compute the gradient of the auxiliary loss with respect to the full
    # sequence.
    if sequence_partition_group is not None:
        # We can keep `aggregated_probs_per_expert` local since we don't need the gradient for
        # `experts_per_token`, saving one allreduce operation for `aggregated_probs_per_expert`.
        # num_sub_sequence = torch.distributed.get_world_size(sequence_partition_group)
        torch.distributed.all_reduce(experts_per_token, group=sequence_partition_group)

    # num_tokens = probs.shape[0] * num_sub_sequence
    num_experts = probs.shape[1]

    # The formula of aux_loss: aux_loss = sum((probs_per_token/num_experts) *
    # (experts_per_token/(num_tokens*topk))) * num_tokens * moe_aux_loss_coeff.
    # This can be simplified to fuse the division and multiplication operations.
    # In uniform routing, the aux_loss_original is 1.0, which has the same principle as the switch transformer design.
    aggregated_probs_per_token = probs.sum(dim=1)
    aux_loss_original = torch.sum(aggregated_probs_per_token * experts_per_token) / (
        (num_experts * topk)
    )
    aux_loss = aux_loss_original * moe_aux_loss_coeff
    return aux_loss, aux_loss_original


def sequence_load_balancing_loss_func(
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    batch_size: int,
    seq_length: int,
    topk: int,
    moe_aux_loss_coeff: float,
    sequence_partition_group=None,
):
    """
    Calculate the auxiliary loss in sequence-level by computing the loss for each individual sample.
    Refer to the DeepSeek-V2 huggingface repo
    (https://huggingface.co/deepseek-ai/DeepSeek-V2) for details.
    """
    num_sub_sequence = 1

    # If the sequence is partitioned by certain parallelism strategies like Sequence Parallelism
    # or Context Parallelism, compute the gradient of the auxiliary loss with respect to the full
    # sequence.
    if sequence_partition_group is not None:
        # We can keep `aggregated_probs_per_expert` local since we don't need the gradient for
        # `tokens_per_expert`, saving one allreduce operation for `aggregated_probs_per_expert`.
        num_sub_sequence = torch.distributed.get_world_size(sequence_partition_group)
        torch.distributed.all_reduce(tokens_per_expert, group=sequence_partition_group)

    assert num_sub_sequence == 1, "Do not support sequence aux loss in sequence partition case"

    num_experts = probs.shape[1]

    probs_for_aux_loss = probs.view(seq_length, batch_size, -1)
    cost_coeff = routing_map.view(seq_length, batch_size, -1).sum(dim=0).float()
    cost_coeff.div_(seq_length * topk / num_experts)
    seq_aux_loss = (cost_coeff * probs_for_aux_loss.mean(dim=0)).sum(dim=1).mean()
    seq_aux_loss *= moe_aux_loss_coeff

    return seq_aux_loss


def z_loss_func(logits, z_loss_coeff):
    """Encourages the router's logits to remain small to enhance stability.
    Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

    Args:
        logits (torch.Tensor): The logits of the router.

    Returns:
        torch.Tensor: The logits after applying the z-loss.
    """

    z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * z_loss_coeff
    return z_loss


def sinkhorn(cost: torch.Tensor, tol: float = 0.0001):
    """Sinkhorn based MoE routing function"""
    cost = torch.exp(cost)
    d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
    d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)

    eps = 0.00000001
    error = 1e9
    d1_old = d1
    while error > tol:
        d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
        d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
        error = torch.mean(torch.abs(d1_old - d1))
        d1_old = d1
    return d1 * cost * d0.unsqueeze(1)


def get_capacity(num_tokens: int, num_experts: int, capacity_factor: float, min_capacity=None):
    """
    Calculate the capacity of each expert.

    Args:
        num_tokens (int): num of the input tokens.
        num_experts (int): num of the experts.
        capacity_factor (float): Capacity factor.
        min_capacity (int, optional): Minimum capacity. Defaults to None.

    Returns:
        Tensor: Capacity of each expert.
    """
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    return capacity


class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for auxiliary loss."""

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss
                                               gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the aux loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale


def permute(tokens, routing_map, num_out_tokens: int = None):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
    """
    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]

    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.bool().T.contiguous()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    )
    sorted_indices = token_indices.masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The indices used to sort the tokens.
        restore_shape (torch.Size): The shape of the unpermuted tensor.
        probs (torch.Tensor, optional): The unpermuted probs tensor,
        routing_map (torch.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].

    Returns:
        torch.Tensor: The tokens restored to their original order.
    """
    _, hidden = restore_shape

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = torch.zeros(
        restore_shape, device=permuted_tokens.device, dtype=permuted_tokens.dtype
    )
    # Scatter add the permuted_input back to the original positions
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output_tokens


def sort_chunks_by_idxs(input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor):
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs], dim=0)
    return output


def device_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    moe_router_topk_limited_devices: int,
):
    """Perform top-k routing on a subset of expert parallel ranks.

    Selects N ranks for each token, then conducts top-k selection among experts on these devices.
    See DeepSeek-V2 technical report (https://arxiv.org/pdf/2405.04434) for details.

    Args:
        scores (torch.Tensor): Softmax scores from the router.
        topk (int): The number of experts to select for each token.
        num_tokens (int): The number of tokens.
        num_experts (int): The number of experts.
        moe_router_topk_limited_devices (int): Number of expert parallel ranks to consider for
            each token during routing. None means no device limitation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Probs and indices tensor.
    """

    # Organize the experts into groups
    num_group = (
        parallel_state.get_expert_model_parallel_world_size()
    )  # num_group equals to expert parallel size
    group_scores = scores.view(num_tokens, num_group, -1).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=moe_router_topk_limited_devices, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    # Mask the experts based on selection groups
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_group, num_experts // num_group)
        .reshape(num_tokens, -1)
    )

    masked_scores = scores.masked_fill(~score_mask.bool(), 0.0)
    probs, top_indices = torch.topk(masked_scores, k=topk, dim=-1)

    return probs, top_indices


def topk_softmax_with_capacity(
    config,
    logits: torch.Tensor,
    topk: int,
    capacity_factor: Optional[float] = None,
    pad_to_capacity: bool = False,
    drop_policy: str = "probs",
    use_pre_softmax: bool = False,
    moe_router_topk_limited_devices: int = None,
    moe_router_topk_scaling_factor: float = None,
    deterministic_mode: bool = False,
    log_tracker = None,
):
    """Apply capacity and padding to the top-k selection.
    Args:
        logits (torch.Tensor): Logits tensor.
        topk (int): The number of experts to select for each token.
        capacity_factor (int): The capacity factor of each expert. Will drop tokens if the number
                               of tokens exceeds the capacity.
        pad_to_capacity (bool): Whether to need padding in token drop mode.
        drop_policy (str): The policy to drop tokens. Can be either "prob" or "position".
                           If "prob", the tokens with the lowest probabilities will be dropped.
                           If "position", tokens at the end of each batch will be dropped.
        use_pre_softmax (bool): Whether to apply softmax before top-k selection.
        moe_router_topk_limited_devices (int): Number of expert parallel ranks to consider for
            each token during routing. None means no device limitation.
        moe_router_topk_scaling_factor (float): Scaling factor for routing score in top-k
            selection, only works when use_pre_softmax enabled.
        deterministic_mode (bool): Deprecated.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - routing_probs (torch.Tensor): A tensor of shape [num_tokens, num_experts] containing
              the routing probabilities for each token to each expert.
            - routing_map (torch.Tensor): A mask tensor of shape [num_tokens, num_experts]
              indicating which experts were selected for each token. True values represent
              the selected experts.
            - tokens_per_expert (torch.Tensor): A tensor of shape [num_experts] containing
              the number of local tokens assigned to each expert before dropping and padding.
    """
    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."

    num_tokens = logits.shape[0]
    num_experts = logits.shape[1]
    if use_pre_softmax:
        # Pre softmax
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)

        if moe_router_topk_limited_devices:
            probs, top_indices = device_limited_topk(
                scores, topk, num_tokens, num_experts, moe_router_topk_limited_devices
            )
        else:
            probs, top_indices = torch.topk(scores, k=topk, dim=1)

        # Normalize the probs.
        if moe_router_topk_scaling_factor:
            probs = probs * moe_router_topk_scaling_factor
    else:
        # Post softmax
        if topk == 1:
            # Requires applying softmax before selecting the top-k when k is 1,
            # since softmax on a [num_tokens, 1] would yield a zero gradient.
            raise ValueError("Please use --moe-router-pre-softmax when topk is 1.")
        # assert (
        #     moe_router_topk_scaling_factor is None
        # ), "moe_router_topk_scaling_factor is not supported with post-softmax"
        if moe_router_topk_limited_devices:
            scores, top_indices = device_limited_topk(
                logits, topk, num_tokens, num_experts, moe_router_topk_limited_devices
            )
        else:
            scores, top_indices = torch.topk(logits, k=topk, dim=1)
            
        if config.moe_router_activation_function == "softmax":
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
        elif config.moe_router_activation_function == "sigmoid":
            probs = torch.sigmoid(scores, dtype=torch.float32).type_as(logits)
            sigmoid_denominator = torch.sum(probs, dim=-1, keepdim=True) + 1e-20
            probs = probs / sigmoid_denominator
        else:
            raise ValueError(f"Invalid activation function: {config.moe_router_activation_function}")
        
        if moe_router_topk_scaling_factor:
            probs = probs * moe_router_topk_scaling_factor

    # TODO Try using element-wise operations instead of scatter?
    topk_masked_gates = torch.zeros_like(logits).scatter(1, top_indices, probs)
    topk_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    tokens_per_expert = topk_map.sum(dim=0)

    if config.enable_scaling_factor:
        if config.moe_shared_expert_intermediate_size is None:
            raise ValueError(
                "enable_scaling_factor requires shared experts (set --moe-shared-expert-intermediate-size)."
            )
        num_shared = config.moe_shared_expert_intermediate_size // config.moe_ffn_hidden_size
        act = config.moe_router_activation_function
        sf = compute_tc_scaling_factor(
            num_experts,
            topk,
            num_shared,
            act=act,
            renorm=True,
            cache_path=getattr(config, "moe_scaling_factor_cache_path", None),
        )
        topk_masked_gates = topk_masked_gates * sf

    if log_tracker is not None:

        e_capacity = get_capacity(
            num_tokens=num_tokens * topk, num_experts=num_experts, capacity_factor=1.0
        )
        
        log_tracker(
            "topk_expert_capacity_utilization_ratio_mean", (tokens_per_expert / e_capacity).mean()
        )
        log_tracker(
            "topk_expert_capacity_utilization_ratio_min", (tokens_per_expert / e_capacity).min()
        )
        log_tracker(
            "topk_expert_capacity_utilization_ratio_max", (tokens_per_expert / e_capacity).max()
        )
        log_tracker(
            "topk_expert_capacity_utilization_ratio_std", (tokens_per_expert / e_capacity).std()
        )

    if capacity_factor is None:
        # TopK without capacity
        return topk_masked_gates, topk_map, tokens_per_expert
    else:
        # TopK with capacity
        expert_capacity = get_capacity(
            num_tokens=num_tokens * topk, num_experts=num_experts, capacity_factor=capacity_factor
        )

        # Maskout exceeded tokens
        if drop_policy == "probs":
            _, capacity_indices = torch.topk(
                topk_masked_gates, k=expert_capacity, dim=0, sorted=False
            )
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1).bool()
        elif drop_policy == "position":
            _, capacity_indices = torch.topk(topk_map.int(), k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1).bool()
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

        if pad_to_capacity:
            final_map = capacity_mask
            final_probs = topk_masked_gates * final_map
        else:
            # Get exceed mask and maskout exceeded probs and indices
            final_map = torch.logical_and(topk_map, capacity_mask)
            final_probs = topk_masked_gates * final_map
            
        if log_tracker is not None:
            num_dropped_tokens = topk_map.shape[0] * topk * capacity_factor - torch.logical_and(topk_map, capacity_mask).sum()
            log_tracker("topk_token_drop_ratio", num_dropped_tokens / (topk_map.shape[0] * topk * capacity_factor))
            # Actual drop ratio: fraction of topk assignments dropped due to capacity limit
            actual_topk_total = topk_map.sum()
            actual_retained = torch.logical_and(topk_map, capacity_mask).sum()
            log_tracker("topk_capacity_drop_ratio", (actual_topk_total - actual_retained) / actual_topk_total)
            
        return final_probs, final_map, tokens_per_expert


def expert_choice_softmax_with_capacity(
    config,
    logits: torch.Tensor,
    bsz: int,
    seq_length: int,
    topk: int,
    use_batch_level_expert_choice: bool = False,
    use_pre_softmax: bool = False,
    moe_router_topk_limited_devices: int = None,
    moe_router_topk_scaling_factor: float = None,
    deterministic_mode: bool = False,
    dynamic_topk: torch.Tensor = None,
):
    """Apply capacity and padding to the top-k selection.
    Args:
        logits (torch.Tensor): Logits tensor.
        topk (int): The average number of token to choose for each expert.
        use_pre_softmax (bool): Whether to apply softmax before top-k selection.
        moe_router_topk_limited_devices (int): Number of expert parallel ranks to consider for
            each token during routing. None means no device limitation.
        moe_router_topk_scaling_factor (float): Scaling factor for routing score in top-k
            selection, only works when use_pre_softmax enabled.
        deterministic_mode (bool): Deprecated.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - routing_probs (torch.Tensor): A tensor of shape [num_tokens, num_experts] containing
              the routing probabilities for each token to each expert.
            - routing_map (torch.Tensor): A mask tensor of shape [num_tokens, num_experts]
              indicating which experts were selected for each token. True values represent
              the selected experts.
            - tokens_per_expert (torch.Tensor): A tensor of shape [num_experts] containing
              the number of local tokens assigned to each expert before dropping and padding.
    """
    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
    if config.moe_shared_expert_intermediate_size is not None and config.shared_experts_with_logits:
        num_tokens = logits.shape[0]
        num_experts = logits.shape[1]
        num_shared_experts = config.moe_shared_expert_intermediate_size // config.moe_ffn_hidden_size
        num_experts = num_experts - num_shared_experts
        logits = logits[:, :-num_shared_experts] # shared expert logits are not used in the routing
        shared_expert_logits = logits[:, -num_shared_experts:]
    else:
        num_tokens = logits.shape[0]
        num_experts = logits.shape[1]
        
    assert num_tokens // bsz == seq_length, f"Not match: num_tokens: {num_tokens}, bsz: {bsz}, seq_length: {seq_length}"
    
    # Handle per-sample dynamic topk
    use_per_sample_topk = (dynamic_topk is not None)
    if use_per_sample_topk:
        # dynamic_topk shape: [batch_size], per-sample topk values
        # Convert to int tensor
        dynamic_topk = dynamic_topk.long()
        # Calculate per-sample topk_ec for sequence-level expert choice
        if not use_batch_level_expert_choice:
            # topk_ec_per_sample: [batch_size]
            topk_ec_per_sample = torch.clamp(
                (seq_length * dynamic_topk) // num_experts,
                min=1
            )
            max_topk_ec = int(topk_ec_per_sample.max().item())
        else:
            # For batch-level, use max topk
            topk = int(dynamic_topk.max().item())
    else:
        max_topk_ec = None
    
    if use_pre_softmax:
        # Pre softmax
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)

        if moe_router_topk_limited_devices:
            raise NotImplementedError("device_limited_topk is not supported with expert_choice_softmax")
        else:
            if use_batch_level_expert_choice:
                if use_per_sample_topk:
                    topk_ec = max(num_tokens * topk // num_experts, 1)
                else:
                    topk_ec = max(num_tokens * topk // num_experts, 1)
                probs, top_indices = torch.topk(scores, k=topk_ec, dim=0)
            else:
                # Sequence-level expert choice with per-sample topk support
                if use_per_sample_topk:
                    raise NotImplementedError(
                        "Dynamic per-sample topk is not yet optimized for pre-softmax EC routing."
                    )
                else:
                    topk_ec = max(seq_length * topk // num_experts, 1)
                    # probs_intermediate: [topk_ec, bsz, num_experts]
                    # local_seq_indices: [topk_ec, bsz, num_experts] (values are seq_idx from 0 to seq_length-1)
                    probs_intermediate, local_seq_indices = torch.topk(scores.view(seq_length, bsz, num_experts), k=topk_ec, dim=0)
                    
                    # Reshape probs_intermediate to [topk_ec * bsz, num_experts]
                    probs = probs_intermediate.reshape(topk_ec * bsz, num_experts)

                    # Calculate global token indices: g_idx = s_idx * bsz + b_idx
                    batch_idx_tensor = torch.arange(
                        bsz, device=scores.device, dtype=local_seq_indices.dtype
                    ).view(1, bsz, 1)
                    global_token_indices = local_seq_indices * bsz + batch_idx_tensor
                    
                    # Reshape global_token_indices to [topk_ec * bsz, num_experts]
                    top_indices = global_token_indices.reshape(topk_ec * bsz, num_experts)
        
        if config.enable_scaling_factor:
            raise NotImplementedError("enable_scaling_factor is not supported with pre-softmax EC routing yet.")

        if config.expert_choice_routing_expert_scaling_factors is not None:
            raise NotImplementedError("expert_choice_routing_expert_scaling_factors is not supported with pre-softmax at present. But it can be implemented.")

        # scale the norm of the routing experts to balance the shared and routing experts
        if moe_router_topk_scaling_factor:
            probs = probs * moe_router_topk_scaling_factor
            
        topk_masked_gates = torch.zeros_like(logits).scatter(0, top_indices, probs)
        topk_map = torch.zeros_like(logits).int().scatter(0, top_indices, 1).bool()
        experts_per_token = topk_map.sum(dim=1)
    else:
        # Post softmax
        _use_transposed = False  # Flag for transposed (coalesced) topk/scatter path
        _dynamic_valid_mask_T = None  # Validity mask for dynamic per-sample topk (transposed layout)
        if moe_router_topk_scaling_factor:
            raise NotImplementedError("moe_router_topk_scaling_factor is not supported with expert_choice_softmax")

        if moe_router_topk_limited_devices:
            raise NotImplementedError("device_limited_topk is not supported with expert_choice_softmax")
        else:
            if use_batch_level_expert_choice:
                if use_per_sample_topk:
                    topk_ec = max(num_tokens * topk // num_experts, 1)
                else:
                    topk_ec = max(num_tokens * topk // num_experts, 1)
                scores, top_indices = torch.topk(logits, k=topk_ec, dim=0)
            else:
                # Sequence-level expert choice with per-sample topk support
                if use_per_sample_topk:
                    # Batched topk with transposed coalesced layout (replaces Python for-loop)
                    # [num_tokens, E] → [seq, bsz, E] → [E, bsz, seq] for coalesced topk on dim=2
                    logits_ef = logits.view(seq_length, bsz, num_experts).permute(2, 1, 0).contiguous()
                    scores_ef, local_seq_indices_ef = torch.topk(logits_ef, k=max_topk_ec, dim=2)
                    # scores_ef, local_seq_indices_ef: [E, bsz, max_topk_ec]

                    # Global token indices: g_idx = s_idx * bsz + b_idx
                    batch_idx_tensor = torch.arange(
                        bsz, device=logits.device, dtype=local_seq_indices_ef.dtype
                    ).view(1, -1, 1)
                    global_token_indices_ef = local_seq_indices_ef * bsz + batch_idx_tensor

                    # Validity mask: position k is valid if k < topk_ec_per_sample[b]
                    # Shape [1, bsz, max_topk_ec], broadcast over E dimension
                    _dynamic_valid_mask_ef = (
                        torch.arange(max_topk_ec, device=logits.device).view(1, 1, -1)
                        < topk_ec_per_sample.view(1, -1, 1)
                    )

                    # Reshape to transposed 2D: [E, max_topk_ec * bsz]
                    top_indices_T = global_token_indices_ef.reshape(num_experts, max_topk_ec * bsz)
                    scores_T = scores_ef.reshape(num_experts, max_topk_ec * bsz)
                    _dynamic_valid_mask_T = _dynamic_valid_mask_ef.expand(
                        num_experts, -1, -1
                    ).contiguous().reshape(num_experts, max_topk_ec * bsz)

                    # Invalid entries get -inf scores → prob=0 after softmax, harmless in scatter
                    scores_T = scores_T.masked_fill(~_dynamic_valid_mask_T, float('-inf'))
                    _use_transposed = True
                else:
                    topk_ec = max(seq_length * topk // num_experts, 1)
                    # Transpose to expert-first layout for coalesced topk
                    # [seq_length, bsz, num_experts] -> [num_experts, bsz, seq_length]
                    # seq_length is last dim -> stride=1 -> fully coalesced topk
                    logits_ef = logits.view(seq_length, bsz, num_experts).permute(2, 1, 0).contiguous()
                    # Coalesced topk along seq_length (dim=2, stride=1)
                    scores_ef, local_seq_indices_ef = torch.topk(logits_ef, k=topk_ec, dim=2)
                    # scores_ef: [num_experts, bsz, topk_ec]

                    # Calculate global token indices: g_idx = s_idx * bsz + b_idx
                    batch_idx_tensor = torch.arange(
                        bsz, device=logits.device, dtype=local_seq_indices_ef.dtype
                    ).view(1, bsz, 1)
                    global_token_indices_ef = local_seq_indices_ef * bsz + batch_idx_tensor

                    # Transposed layout: [num_experts, topk_ec * bsz]
                    top_indices_T = global_token_indices_ef.reshape(num_experts, topk_ec * bsz)
                    scores_T = scores_ef.reshape(num_experts, topk_ec * bsz)
                    _use_transposed = True
        
        if config.moe_shared_expert_intermediate_size is not None and config.shared_experts_with_logits and use_per_sample_topk:
            raise NotImplementedError(
                "Dynamic per-sample topk is not supported with shared_experts_with_logits."
            )

        if config.moe_shared_expert_intermediate_size is not None and config.shared_experts_with_logits:
            # Fallback: shared_experts_with_logits requires full-matrix softmax
            # because shared expert logits are concatenated before activation
            topk_map = torch.zeros_like(logits).int().scatter(0, top_indices, 1).bool()
            scores = torch.zeros_like(logits).scatter(0, top_indices, scores)
            scores = torch.cat([scores, shared_expert_logits], dim=-1)
            topk_map_mask = torch.cat([~topk_map, torch.zeros_like(shared_expert_logits).bool()], dim=-1)

            if config.moe_router_activation_function == "softmax":
                scores[topk_map_mask] = -float('inf')
                topk_masked_gates = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
                topk_masked_gates[torch.isnan(topk_masked_gates)] = 0.0
            elif config.moe_router_activation_function == "sigmoid":
                scores[topk_map_mask] = -float('inf')
                topk_masked_gates = torch.sigmoid(scores, dtype=torch.float32).type_as(logits)
                sigmoid_denominator = torch.sum(topk_masked_gates, dim=-1, keepdim=True) + 1e-20
                topk_masked_gates = topk_masked_gates / sigmoid_denominator
            else:
                raise ValueError(f"Invalid activation function: {config.moe_router_activation_function}")
        else:
            # Sparse softmax: only process selected elements instead of full [num_tokens, num_experts] matrix
            # Flatten from transposed or original layout (scatter ops are order-independent)
            if _use_transposed:
                flat_indices = top_indices_T.flatten().long()
                flat_scores = scores_T.flatten()
            else:
                flat_indices = top_indices.flatten().long()
                flat_scores = scores.flatten()

            if config.moe_router_activation_function == "softmax":
                # Sparse softmax grouped by token index
                flat_scores_fp32 = flat_scores.float()
                max_per_token = torch.full((num_tokens,), -float('inf'), device=logits.device, dtype=torch.float32)
                max_per_token.scatter_reduce_(0, flat_indices, flat_scores_fp32, reduce='amax')
                # Clamp to prevent NaN from -inf - (-inf) when all entries for a token are padded
                max_per_token = max_per_token.clamp(min=torch.finfo(torch.float32).min)
                flat_exp = torch.exp(flat_scores_fp32 - max_per_token[flat_indices])
                # Non-in-place scatter_add to preserve gradient flow through denominator
                sum_per_token = torch.zeros(num_tokens, device=logits.device, dtype=torch.float32).scatter_add(0, flat_indices, flat_exp)
                flat_probs = (flat_exp / (sum_per_token[flat_indices] + 1e-20)).to(logits.dtype)
            elif config.moe_router_activation_function == "sigmoid":
                # Sparse sigmoid with per-token normalization
                flat_sigmoid = torch.sigmoid(flat_scores.float())
                # Non-in-place scatter_add to preserve gradient flow through denominator
                sum_per_token = torch.zeros(num_tokens, device=logits.device, dtype=torch.float32).scatter_add(0, flat_indices, flat_sigmoid)
                flat_probs = (flat_sigmoid / (sum_per_token[flat_indices] + 1e-20)).to(logits.dtype)
            else:
                raise ValueError(f"Invalid activation function: {config.moe_router_activation_function}")

            if _use_transposed:
                # Coalesced scatter in transposed space (dim=1), then transpose back
                probs_T = flat_probs.view_as(top_indices_T)
                topk_masked_gates = torch.zeros(
                    num_experts, num_tokens, device=logits.device, dtype=logits.dtype
                ).scatter(1, top_indices_T, probs_T).T.contiguous()
                if _dynamic_valid_mask_T is not None:
                    # Only mark valid (non-padded) entries in topk_map
                    topk_map = torch.zeros(
                        num_experts, num_tokens, device=logits.device, dtype=torch.int32
                    ).scatter(1, top_indices_T, _dynamic_valid_mask_T.int()).T.contiguous().bool()
                else:
                    topk_map = torch.zeros(
                        num_experts, num_tokens, device=logits.device, dtype=torch.int32
                    ).scatter(1, top_indices_T, 1).T.contiguous().bool()
            else:
                probs = flat_probs.view_as(top_indices)
                topk_masked_gates = torch.zeros_like(logits).scatter(0, top_indices, probs)
                topk_map = torch.zeros_like(logits).int().scatter(0, top_indices, 1).bool()
        
        experts_per_token = topk_map.sum(dim=1)

        if config.enable_scaling_factor:
            if config.expert_choice_routing_expert_scaling_factors is not None:
                raise ValueError(
                    "Cannot use enable_scaling_factor together with expert_choice_routing_expert_scaling_factors."
                )
            if config.moe_router_topk_scaling_factor is not None:
                raise ValueError(
                    "Cannot use enable_scaling_factor together with moe_router_topk_scaling_factor."
                )
            if config.moe_shared_expert_intermediate_size is None:
                raise ValueError(
                    "enable_scaling_factor requires shared experts (set --moe-shared-expert-intermediate-size)."
                )

            num_shared = config.moe_shared_expert_intermediate_size // config.moe_ffn_hidden_size
            factors_tensor = compute_ec_scaling_factors(
                num_experts,
                num_shared,
                cache_path=getattr(config, "moe_scaling_factor_cache_path", None),
            )
            factors_tensor = factors_tensor.to(device=logits.device, dtype=logits.dtype)

            scaling_factors = factors_tensor[experts_per_token]  # [num_tokens, ]
            topk_masked_gates = topk_masked_gates * scaling_factors.unsqueeze(1)  # [num_tokens, num_experts]

        elif config.expert_choice_routing_expert_scaling_factors is not None:
            if config.moe_shared_expert_intermediate_size is not None and config.shared_experts_with_logits:
                raise NotImplementedError("expert_choice_routing_expert_scaling_factors is not supported with shared_experts_with_logits.")

            defined_scaling_factors = config.expert_choice_routing_expert_scaling_factors.split(",")
            defined_scaling_factors = torch.tensor([float(factor) for factor in defined_scaling_factors], device=logits.device, dtype=logits.dtype)
            assert len(defined_scaling_factors) == num_experts + 1, f"The number of defined scaling factors ({len(defined_scaling_factors)}) does not match the number of experts + 1: ({num_experts + 1})."

            scaling_factors = defined_scaling_factors[experts_per_token]  # [num_tokens, ]
            topk_masked_gates = topk_masked_gates * scaling_factors.unsqueeze(1)  # [num_tokens, num_experts]

    return topk_masked_gates, topk_map, experts_per_token


def save_to_aux_losses_tracker(
    name: str,
    loss: torch.Tensor,
    layer_number: int,
    num_layers: int,
    reduce_group: torch.distributed.ProcessGroup = None,
    avg_group: torch.distributed.ProcessGroup = None,
):
    """Save the auxiliary loss for logging.
    Args:
        name (str): The name of the loss.
        loss (torch.Tensor): The loss tensor.
        layer_number (int): Layer index of the loss.
        num_layers (int): The number of total layers.
        reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
        mean_group (torch.distributed.ProcessGroup): The group for averaging the loss.
    """
    # Skip aux loss logging if layer_number is None.
    if layer_number is None:
        return

    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    if name not in tracker:
        tracker[name] = {}
        tracker[name]["values"] = torch.zeros(num_layers, device=loss.device)
    tracker[name]["values"][layer_number - 1] += loss.detach()  # Aggregate the loss for the layer.
    tracker[name]["reduce_group"] = reduce_group
    tracker[name]["avg_group"] = avg_group


def clear_aux_losses_tracker():
    """Clear the auxiliary losses."""
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    for name in tracker:
        tracker[name]["values"].zero_()
        tracker[name]["reduce_group"] = None
        tracker[name]["avg_group"] = None


def reduce_aux_losses_tracker_across_ranks():
    """Collect and reduce the auxiliary losses across ranks."""
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    for name in tracker:
        values = tracker[name]["values"]
        # Collect aux losses across PP.
        torch.distributed.all_reduce(
            values, group=parallel_state.get_pipeline_model_parallel_group()
        )
        # Reduce aux losses across ranks.
        if tracker[name].get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker[name].get('reduce_group'))
        if tracker[name].get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker[name]['avg_group'], op=torch.distributed.ReduceOp.AVG
            )


def track_moe_metrics(
    loss_scale,
    iteration,
    writer,
    wandb_writer=None,
    total_loss_dict=None,
    per_layer_logging=False,
    name_prefix: str = "",
):
    """Track the MoE metrics for logging."""
    # Aux loss logging
    reduce_aux_losses_tracker_across_ranks()
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    if writer is not None:
        aux_losses = {k: v['values'].float() * loss_scale for k, v in tracker.items()}
        for name, loss_list in aux_losses.items():
            prefixed_name = f"{name_prefix}{name}" if name_prefix else name
            if total_loss_dict is not None:
                if name not in total_loss_dict:
                    total_loss_dict[name] = loss_list.mean()
                else:
                    total_loss_dict[name] += loss_list.mean()

            # currently when using add_scalars,
            # torch.utils.add_scalars makes each timer its own run, which
            # polutes the runs list, so we just add each as a scalar
            writer.add_scalar(prefixed_name, loss_list.mean(), iteration)
            if per_layer_logging:
                for i, loss in enumerate(loss_list.tolist()):
                    layer_name = f"{name_prefix}moe/{name}_layer_{i}" if name_prefix else f"moe/{name}_layer_{i}"
                    writer.add_scalar(layer_name, loss, iteration)

            # W&B logging lacks support for logging multiple scalars simultaneously.
            # As a workaround, we log each scalar individually first, then we can create
            # a custom panel to manually group them to a single plot.
            if wandb_writer:
                wandb_writer.log({prefixed_name: loss_list.mean()}, iteration)
                if per_layer_logging:
                    wandb_writer.log(
                        {
                            (
                                f"{name_prefix}moe/{name}_layer_{i}"
                                if name_prefix
                                else f"moe/{name}_layer_{i}"
                            ): loss
                            for i, loss in enumerate(loss_list.tolist())
                        },
                        iteration,
                    )

    clear_aux_losses_tracker()
