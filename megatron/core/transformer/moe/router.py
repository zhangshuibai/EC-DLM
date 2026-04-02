# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable
import math

import torch

from megatron.core import parallel_state
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
    sequence_load_balancing_loss_func,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    expert_choice_softmax_with_capacity,
    z_loss_func,
    ec_load_balancing_loss_func,
    get_capacity,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None
        self.layer_number = None

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
        )
        if config.perform_initialization:
            config.init_method(self.weight)
        self.weight.data = self.weight.data.to(dtype=config.params_dtype)
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == 'cpu':
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        logits = torch.nn.functional.linear(input, self.weight)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mapping.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor, dynamic_topk: torch.Tensor = None):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
            dynamic_topk (torch.Tensor, optional): Dynamic topk values per sample [batch_size].
                Ignored for TopKRouter.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.input_jitter = None

        # Loss-free load balancing: register expert bias buffer (arXiv:2408.15664)
        if self.routing_type == "loss_free":
            self.register_buffer(
                'expert_bias', torch.zeros(self.config.num_moe_experts, dtype=torch.float32)
            )
            # Accumulator for tokens_per_expert across micro-batches within a training step
            self.register_buffer(
                '_accumulated_tokens_per_expert',
                torch.zeros(self.config.num_moe_experts, dtype=torch.float32),
                persistent=False,  # Not saved in checkpoint
            )
            self._num_microbatch_accumulations = 0
            self.bias_update_rate = self.config.moe_expert_bias_update_rate

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
        else:
            logits = _sinkhorn_activation(logits)
            _, indices = torch.topk(logits, k=self.topk, dim=1)
        map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        scores = logits * map
        return scores, map

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mask of token to experts assignment.
        """
        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            self.config,
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            moe_router_topk_limited_devices=self.config.moe_router_topk_limited_devices,
            moe_router_topk_scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            log_tracker=partial(save_to_aux_losses_tracker, layer_number=self.layer_number, num_layers=self.config.num_layers),
        )

        if self.training:
            # Apply load balancing loss
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            aux_loss_func = partial(
                switch_load_balancing_loss_func,
                probs=scores,
                tokens_per_expert=tokens_per_expert,
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )
        return probs, routing_map

    def seq_aux_loss_load_balancing(self, logits: torch.Tensor, bsz: int, seq_length: int):
        """Apply loss-based load balancing to the logits tensor."""

        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            moe_router_topk_limited_devices=self.config.moe_router_topk_limited_devices,
            moe_router_topk_scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
        )

        if self.training:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            aux_loss_func = partial(
                sequence_load_balancing_loss_func,
                probs=scores,
                routing_map=routing_map,
                tokens_per_expert=tokens_per_expert,
                batch_size=bsz,
                seq_length=seq_length,
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )

        return probs, routing_map

    def loss_free_load_balancing(self, logits: torch.Tensor):
        """Apply auxiliary-loss-free load balancing (arXiv:2408.15664).

        Uses dynamic expert-wise biases to influence top-K selection without
        adding any auxiliary loss. Biases are updated after each forward pass
        based on observed expert load.

        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mask of token to experts assignment.
        """
        # Use biased logits for top-K selection only
        biased_logits = logits + self.expert_bias.unsqueeze(0)

        # Run top-K selection with biased logits (probs will be computed from biased logits
        # internally, but we only use routing_map from this call)
        _, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            self.config,
            biased_logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            moe_router_topk_limited_devices=self.config.moe_router_topk_limited_devices,
            moe_router_topk_scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
        )

        # Recompute probs from original (unbiased) logits using the routing_map
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        probs = scores * routing_map

        # Accumulate tokens_per_expert for global-batch-level bias update
        if self.training:
            with torch.no_grad():
                self._accumulated_tokens_per_expert += tokens_per_expert.float()
                self._num_microbatch_accumulations += 1

            # Log bias statistics for monitoring
            save_to_aux_losses_tracker(
                "expert_bias_mean", self.expert_bias.mean(),
                self.layer_number, self.config.num_layers,
            )
            save_to_aux_losses_tracker(
                "expert_bias_std", self.expert_bias.std(),
                self.layer_number, self.config.num_layers,
            )
            save_to_aux_losses_tracker(
                "expert_bias_max", self.expert_bias.max(),
                self.layer_number, self.config.num_layers,
            )
            save_to_aux_losses_tracker(
                "expert_bias_min", self.expert_bias.min(),
                self.layer_number, self.config.num_layers,
            )

        return probs, routing_map

    def update_expert_bias(self):
        """Update expert biases from accumulated stats and reset accumulators.

        Called once per training step (after all micro-batches) to match the
        paper's Algorithm 1: update biases based on global batch load statistics.
        """
        if self._num_microbatch_accumulations == 0:
            return
        with torch.no_grad():
            total_tokens_per_expert = self._accumulated_tokens_per_expert
            mean_load = total_tokens_per_expert.mean()
            error = mean_load - total_tokens_per_expert
            self.expert_bias += self.bias_update_rate * torch.sign(error)
            # Reset accumulators
            self._accumulated_tokens_per_expert.zero_()
            self._num_microbatch_accumulations = 0

    def apply_load_balancing_loss(
        self, activation: torch.Tensor, load_balancing_loss_func: Callable
    ):
        """Calculate auxiliary loss, attach gradient function to activation and add to logging."""
        moe_aux_loss_coeff = self.config.moe_aux_loss_coeff
        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            moe_aux_loss_coeff /= parallel_state.get_tensor_model_parallel_world_size()
        else:
            sequence_partition_group = parallel_state.get_tensor_and_context_parallel_group()

        aux_loss = load_balancing_loss_func(
            moe_aux_loss_coeff=moe_aux_loss_coeff, sequence_partition_group=sequence_partition_group
        )
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss / moe_aux_loss_coeff,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training:
            moe_z_loss_coeff = (
                self.config.moe_z_loss_coeff
                / parallel_state.get_tensor_and_context_parallel_world_size()
            )
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            save_to_aux_losses_tracker(
                "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, self.config.num_layers
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    
    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        if self.routing_type == "sinkhorn":
            scores, routing_map = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, routing_map = self.aux_loss_load_balancing(logits)
        elif self.routing_type == "seq_aux_loss":
            scores, routing_map = self.seq_aux_loss_load_balancing(logits, bsz, seq_length)
        elif self.routing_type == "loss_free":
            scores, routing_map = self.loss_free_load_balancing(logits)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, routing_map, _ = topk_softmax_with_capacity(
                self.config,
                logits,
                self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_token_drop_policy,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                moe_router_topk_scaling_factor=self.config.moe_router_topk_scaling_factor,
                deterministic_mode=self.config.deterministic_mode,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

        return scores, routing_map

    def forward(self, input: torch.Tensor, dynamic_topk: torch.Tensor = None):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
            dynamic_topk (torch.Tensor, optional): Dynamic topk values per sample [batch_size].
                Ignored for TopKRouter.
        """

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        scores, routing_map = self.routing(logits)
        # print(routing_map.shape)
        # print(routing_map.sum(0))
        # print(routing_map.sum(1))
        return scores, routing_map


class ExpertChoiceRouter(Router):
    """Route each token with expert choice."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        self.routing_type = self.config.moe_router_load_balancing_type
        self.input_jitter = None
        self.topk = self.config.moe_router_topk
        
        if self.config.shared_experts_with_logits:
            # Add the shared experts weights to the router to be applied later
            self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts + self.config.moe_shared_expert_intermediate_size // self.config.moe_ffn_hidden_size, self.config.hidden_size), dtype=torch.float32)
            )
            if config.perform_initialization:
                config.init_method(self.weight)
            self.weight.data = self.weight.data.to(dtype=config.params_dtype)
            setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

    def apply_load_balancing_loss(
        self, activation: torch.Tensor, load_balancing_loss_func: Callable
    ):
        """Calculate auxiliary loss, attach gradient function to activation and add to logging."""
        moe_aux_loss_coeff = self.config.moe_aux_loss_coeff
        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            moe_aux_loss_coeff /= parallel_state.get_tensor_model_parallel_world_size()
        else:
            sequence_partition_group = parallel_state.get_tensor_and_context_parallel_group()

        aux_loss, aux_loss_original = load_balancing_loss_func(
            moe_aux_loss_coeff=moe_aux_loss_coeff, sequence_partition_group=sequence_partition_group
        )
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss_original,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training:
            moe_z_loss_coeff = (
                self.config.moe_z_loss_coeff
                / parallel_state.get_tensor_and_context_parallel_world_size()
            )
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            save_to_aux_losses_tracker(
                "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, self.config.num_layers
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def apply_softmax_expert_choice_load_balancing(self, logits: torch.Tensor, bsz: int, seq_length: int, dynamic_topk: torch.Tensor = None):
        """Apply softmax-based expert choice to the logits tensor.
        """
        # Use dynamic_topk if provided, otherwise use default topk
        if dynamic_topk is not None:
            # dynamic_topk shape: [batch_size], per-sample topk values
            # Pass per-sample topk to the function
            probs, routing_map, experts_per_token = expert_choice_softmax_with_capacity(
                self.config,
                logits,
                bsz,
                seq_length,
                topk=self.topk,  # Keep default topk for backward compatibility
                dynamic_topk=dynamic_topk,  # Pass per-sample topk
                use_batch_level_expert_choice=self.config.moe_batch_level_expert_choice,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                moe_router_topk_limited_devices=self.config.moe_router_topk_limited_devices,
                moe_router_topk_scaling_factor=self.config.moe_router_topk_scaling_factor,
                deterministic_mode=self.config.deterministic_mode,
            )
            # Use mean topk for aux loss calculation (aux loss needs a scalar)
            topk_for_loss = int(dynamic_topk.float().mean().item())
        else:
            topk_for_loss = self.topk
            probs, routing_map, experts_per_token = expert_choice_softmax_with_capacity(
                self.config,
                logits,
                bsz,
                seq_length,
                topk=self.topk,
                dynamic_topk=None,
                use_batch_level_expert_choice=self.config.moe_batch_level_expert_choice,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                moe_router_topk_limited_devices=self.config.moe_router_topk_limited_devices,
                moe_router_topk_scaling_factor=self.config.moe_router_topk_scaling_factor,
                deterministic_mode=self.config.deterministic_mode,
            )
        
        if self.training and self.config.moe_aux_loss_coeff > 0:
            scores = torch.softmax(logits, dim=0, dtype=torch.float32)
            aux_loss_func = partial(
                ec_load_balancing_loss_func,
                probs=scores,
                experts_per_token=experts_per_token,
                topk=topk_for_loss,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )

        return probs, routing_map
    
    def apply_sinkhorn_expert_choice_load_balancing(self, logits: torch.Tensor, bsz: int, seq_length: int):
        """Apply sinkhorn-based expert choice to the logits tensor.
        """
        raise NotImplementedError("Sinkhorn-based expert choice is not supported yet.")
    
    def routing(self, logits: torch.Tensor, dynamic_topk: torch.Tensor = None):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.
            dynamic_topk (torch.Tensor, optional): Dynamic topk values per sample [batch_size]. 
                If None, uses the default topk from config.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        if self.config.moe_shared_expert_intermediate_size is None or not self.config.shared_experts_with_logits:
            logits = logits.view(-1, self.config.num_moe_experts)
        else:
            num_shared_experts = self.config.moe_shared_expert_intermediate_size // self.config.moe_ffn_hidden_size
            logits = logits.view(-1, self.config.num_moe_experts + num_shared_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        if self.routing_type == "softmax_expert_choice":
            scores, routing_map = self.apply_softmax_expert_choice_load_balancing(logits, bsz, seq_length, dynamic_topk=dynamic_topk)
        elif self.routing_type == "sinkhorn_expert_choice":
            scores, routing_map = self.apply_sinkhorn_expert_choice_load_balancing(logits, bsz, seq_length)
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

        return scores, routing_map

    def forward(self, input: torch.Tensor, dynamic_topk: torch.Tensor = None):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
            dynamic_topk (torch.Tensor, optional): Dynamic topk values per sample [batch_size]. 
                If None, uses the default topk from config.
        """

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)
        scores, routing_map = self.routing(logits, dynamic_topk=dynamic_topk)
        
        if self.training:
            experts_per_token = routing_map.sum(1)
            save_to_aux_losses_tracker(
                "ec_token_drop_ratio", (experts_per_token == 0).int().sum() / routing_map.shape[0], self.layer_number, self.config.num_layers
            )
            experts_per_token_f = experts_per_token.float()
            save_to_aux_losses_tracker(
                "ec_experts_per_token_mean", experts_per_token_f.mean(), self.layer_number, self.config.num_layers
            )
            save_to_aux_losses_tracker(
                "ec_experts_per_token_min", experts_per_token_f.min(), self.layer_number, self.config.num_layers
            )
            save_to_aux_losses_tracker(
                "ec_experts_per_token_max", experts_per_token_f.max(), self.layer_number, self.config.num_layers
            )
            save_to_aux_losses_tracker(
                "ec_experts_per_token_std", experts_per_token_f.std(), self.layer_number, self.config.num_layers
            )
        
        # Track dynamic topk compute cost metrics if dynamic_topk is enabled
        if dynamic_topk is not None:
            # input shape: [seq_len, batch_size, hidden_size]
            seq_len, batch_size = input.shape[:2]
            
            # Calculate compute cost per sample: num_tokens * per_sample_topk
            # dynamic_topk shape: [batch_size]
            compute_cost_per_sample = seq_len * dynamic_topk.float()  # [batch_size]
            
            # Track per-sample mean compute cost (sum of all samples / num_samples)
            # This will be aggregated across all micro batches in a training step
            total_compute_cost = compute_cost_per_sample.sum()  # Scalar: total compute cost for this batch
            num_samples = batch_size  # Number of samples in this batch
            
            # Record mean compute cost per sample (will be averaged across micro batches)
            save_to_aux_losses_tracker(
                "dynamic_topk_compute_cost_per_sample_mean", 
                compute_cost_per_sample.mean(), 
                self.layer_number, 
                self.config.num_layers
            )
            
            # Record compute cost distribution
            save_to_aux_losses_tracker(
                "dynamic_topk_compute_cost_per_sample_min",
                compute_cost_per_sample.min(),
                self.layer_number,
                self.config.num_layers
            )
            save_to_aux_losses_tracker(
                "dynamic_topk_compute_cost_per_sample_max",
                compute_cost_per_sample.max(),
                self.layer_number,
                self.config.num_layers
            )
            save_to_aux_losses_tracker(
                "dynamic_topk_compute_cost_per_sample_std",
                compute_cost_per_sample.std(),
                self.layer_number,
                self.config.num_layers
            )
            
            # Record mean topk per sample (compute_cost_mean / seq_len)
            save_to_aux_losses_tracker(
                "dynamic_topk_mean",
                dynamic_topk.float().mean(),
                self.layer_number,
                self.config.num_layers
            )
            
            # Record topk distribution
            save_to_aux_losses_tracker(
                "dynamic_topk_min",
                dynamic_topk.float().min(),
                self.layer_number,
                self.config.num_layers
            )
            save_to_aux_losses_tracker(
                "dynamic_topk_max",
                dynamic_topk.float().max(),
                self.layer_number,
                self.config.num_layers
            )
            save_to_aux_losses_tracker(
                "dynamic_topk_std",
                dynamic_topk.float().std(),
                self.layer_number,
                self.config.num_layers
            )
            
            # Record compute cost ratio compared to fixed topk
            # This shows how much compute is saved/increased compared to fixed topk
            fixed_topk = self.topk
            fixed_compute_cost_per_sample = seq_len * fixed_topk
            compute_cost_ratio = compute_cost_per_sample.mean() / fixed_compute_cost_per_sample
            save_to_aux_losses_tracker(
                "dynamic_topk_compute_cost_ratio",
                compute_cost_ratio,
                self.layer_number,
                self.config.num_layers
            )

        # print(routing_map.shape)
        # print(routing_map.sum(0))
        # print(routing_map.sum(1))
        return scores, routing_map


def update_loss_free_expert_biases(model):
    """Update expert biases for all loss-free routers in the model.

    Should be called once per training step, after all micro-batches have been
    processed (forward + backward), to match the paper's Algorithm 1.

    Args:
        model: The model (or list of model chunks) to update.
    """
    if isinstance(model, list):
        for model_chunk in model:
            update_loss_free_expert_biases(model_chunk)
        return
    for module in model.modules():
        if isinstance(module, TopKRouter) and hasattr(module, 'update_expert_bias'):
            module.update_expert_bias()