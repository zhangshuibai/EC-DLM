# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.router import TopKRouter, ExpertChoiceRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.moe_layer_recompute = config.moe_layer_recompute

        # Initialize router
        self.router = TopKRouter(config=self.config)

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(self.submodules.experts, self.num_local_experts, self.config)

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def forward(self, hidden_states: torch.Tensor, mask_ratio: torch.Tensor = None, masked_indices: torch.Tensor = None):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # Calculate dynamic topk if enabled
        dynamic_topk = None
        if self.config.moe_router_dynamic_topk and mask_ratio is not None:
            # hidden_states shape: [seq_len, batch_size, hidden_size]
            seq_len, batch_size = hidden_states.shape[:2]
            num_experts = self.config.num_moe_experts
            max_topk = self.config.moe_router_dynamic_topk_max
            min_topk = self.config.moe_router_dynamic_topk_min

            # mask_ratio shape: [batch_size]
            # All schedulers compute a factor in [0, 1], then map to topk in [min_topk, max_topk]
            # via: topk = round(min_topk + (max_topk - min_topk) * factor)
            scheduler = self.config.moe_router_dynamic_topk_scheduler
            if scheduler in ("cosine", "cosine_reverse"):
                import math
                ratio = mask_ratio if scheduler == "cosine" else (1.0 - mask_ratio)
                cosine_factor = 0.5 * (1 + torch.cos(math.pi * (1 - ratio)))
                dynamic_topk = torch.clamp(
                    torch.round(min_topk + cosine_factor * (max_topk - min_topk)).long(),
                    min=min_topk,
                    max=max_topk
                )
            elif scheduler in ("linear", "linear_reverse"):
                ratio = mask_ratio if scheduler == "linear" else (1.0 - mask_ratio)
                dynamic_topk = torch.clamp(
                    torch.round(min_topk + ratio * (max_topk - min_topk)).long(),
                    min=min_topk,
                    max=max_topk
                )
            elif scheduler == "bell":
                # Bell-shaped scheduler: topk = min_topk at mask_ratio=0 and 1, topk = max_topk at mask_ratio=0.5
                # factor = 1 - 4 * (mask_ratio - 0.5)^2: 0 at edges, 1 at center
                bell_factor = 1.0 - 4.0 * torch.pow(mask_ratio - 0.5, 2)
                bell_factor = torch.clamp(bell_factor, min=0.0, max=1.0)
                dynamic_topk = torch.clamp(
                    torch.round(min_topk + (max_topk - min_topk) * bell_factor).long(),
                    min=min_topk,
                    max=max_topk
                )
            elif scheduler in ("bell_gaussian", "bell_gaussian_reverse"):
                # Gaussian bell-shaped scheduler: topk = min_topk at edges, max_topk at center
                # Reverse: topk = max_topk at edges, min_topk at center
                import math
                sigma = self.config.moe_router_dynamic_topk_bell_sigma
                gaussian_factor = torch.exp(-torch.pow(mask_ratio - 0.5, 2) / (2 * sigma ** 2))
                # Normalize: factor = 0 at edges, factor = 1 at center
                edge_factor = math.exp(-0.25 / (2 * sigma ** 2))
                bell_factor = (gaussian_factor - edge_factor) / (1.0 - edge_factor)
                bell_factor = torch.clamp(bell_factor, min=0.0, max=1.0)
                if scheduler == "bell_gaussian_reverse":
                    bell_factor = 1.0 - bell_factor
                dynamic_topk = torch.clamp(
                    torch.round(min_topk + (max_topk - min_topk) * bell_factor).long(),
                    min=min_topk,
                    max=max_topk
                )
            else:
                raise ValueError(f"Unsupported dynamic topk scheduler: {scheduler}")

        # process MoE
        def custom_forward(hidden_states):
            probs, routing_map = self.router(hidden_states, dynamic_topk=dynamic_topk)
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, routing_map
            )
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            if self.use_shared_expert and not self.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
                output += self.shared_experts(hidden_states)
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias


class MoELayerExpertChoice(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayerExpertChoice, self).__init__(config=config, layer_number=layer_number)
        self.moe_layer_recompute = config.moe_layer_recompute

        # Initialize router
        self.router = ExpertChoiceRouter(config=self.config)

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(self.submodules.experts, self.num_local_experts, self.config)

        # Initialize shared experts
        if self.use_shared_expert:
            if not self.config.shared_experts_with_logits:
                self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)
            else:
                single_shared_expert_dim = self.config.moe_ffn_hidden_size
                n_shared_experts = self.config.moe_shared_expert_intermediate_size // single_shared_expert_dim
                total_shared_expert_dim = self.config.moe_shared_expert_intermediate_size
                self.config.moe_shared_expert_intermediate_size = single_shared_expert_dim
                self.shared_experts = torch.nn.ModuleList([build_module(self.submodules.shared_experts, config=self.config) for _ in range(n_shared_experts)])
                self.config.moe_shared_expert_intermediate_size = total_shared_expert_dim
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def forward(self, hidden_states: torch.Tensor, mask_ratio: torch.Tensor = None, masked_indices: torch.Tensor = None):
        # for i in range(self.num_local_experts):
        #     weight1 = getattr(self.experts.linear_fc1, f"weight{i}")
        #     weight2 = getattr(self.experts.linear_fc2, f"weight{i}")
        #     print(torch.isnan(weight1).any(), torch.isnan(weight2).any())
        #     print(torch.isinf(weight1).any(), torch.isinf(weight2).any())
        #     print(weight1.shape, weight2.shape)
        # import sys; sys.exit()
        
        # print(getattr(self.experts.linear_fc2, f"weight0"))
        # print(torch.isnan(getattr(self.experts.linear_fc2, f"weight0")).sum())
        # print(getattr(self.experts.linear_fc2, f"weight0").shape)
        # for num in getattr(self.experts.linear_fc2, f"weight0"):
        #     for num1 in num:
        #         if torch.isnan(num1).any():
        #             print(num1)
        # import sys; sys.exit()
        
        # print(self.shared_experts.linear_fc1.weight)
        # print(self.shared_experts.linear_fc1.weight.shape)
        # print(torch.isnan(self.shared_experts.linear_fc1.weight).sum())
        # print(torch.isinf(self.shared_experts.linear_fc1.weight).sum())
        # import sys; sys.exit()
        
        # print(self.router.weight)
        # print(self.router.weight.shape)
        # print(torch.isnan(self.router.weight).sum())
        # print(torch.isinf(self.router.weight).sum())
        # print(self.router.weight.max(), self.router.weight.min())
        # import sys; sys.exit()
        
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # Calculate dynamic topk if enabled
        dynamic_topk = None
        if self.config.moe_router_dynamic_topk and mask_ratio is not None:
            # hidden_states shape: [seq_len, batch_size, hidden_size]
            seq_len, batch_size = hidden_states.shape[:2]
            num_experts = self.config.num_moe_experts
            max_topk = self.config.moe_router_dynamic_topk_max
            min_topk = self.config.moe_router_dynamic_topk_min

            # mask_ratio shape: [batch_size]
            # All schedulers compute a factor in [0, 1], then map to topk in [min_topk, max_topk]
            # via: topk = round(min_topk + (max_topk - min_topk) * factor)
            scheduler = self.config.moe_router_dynamic_topk_scheduler
            if scheduler in ("cosine", "cosine_reverse"):
                import math
                ratio = mask_ratio if scheduler == "cosine" else (1.0 - mask_ratio)
                cosine_factor = 0.5 * (1 + torch.cos(math.pi * (1 - ratio)))
                dynamic_topk = torch.clamp(
                    torch.round(min_topk + cosine_factor * (max_topk - min_topk)).long(),
                    min=min_topk,
                    max=max_topk
                )
            elif scheduler in ("linear", "linear_reverse"):
                ratio = mask_ratio if scheduler == "linear" else (1.0 - mask_ratio)
                dynamic_topk = torch.clamp(
                    torch.round(min_topk + ratio * (max_topk - min_topk)).long(),
                    min=min_topk,
                    max=max_topk
                )
            elif scheduler == "bell":
                # Bell-shaped scheduler: topk = min_topk at mask_ratio=0 and 1, topk = max_topk at mask_ratio=0.5
                # factor = 1 - 4 * (mask_ratio - 0.5)^2: 0 at edges, 1 at center
                bell_factor = 1.0 - 4.0 * torch.pow(mask_ratio - 0.5, 2)
                bell_factor = torch.clamp(bell_factor, min=0.0, max=1.0)
                dynamic_topk = torch.clamp(
                    torch.round(min_topk + (max_topk - min_topk) * bell_factor).long(),
                    min=min_topk,
                    max=max_topk
                )
            elif scheduler in ("bell_gaussian", "bell_gaussian_reverse"):
                # Gaussian bell-shaped scheduler: topk = min_topk at edges, max_topk at center
                # Reverse: topk = max_topk at edges, min_topk at center
                import math
                sigma = self.config.moe_router_dynamic_topk_bell_sigma
                gaussian_factor = torch.exp(-torch.pow(mask_ratio - 0.5, 2) / (2 * sigma ** 2))
                # Normalize: factor = 0 at edges, factor = 1 at center
                edge_factor = math.exp(-0.25 / (2 * sigma ** 2))
                bell_factor = (gaussian_factor - edge_factor) / (1.0 - edge_factor)
                bell_factor = torch.clamp(bell_factor, min=0.0, max=1.0)
                if scheduler == "bell_gaussian_reverse":
                    bell_factor = 1.0 - bell_factor
                dynamic_topk = torch.clamp(
                    torch.round(min_topk + (max_topk - min_topk) * bell_factor).long(),
                    min=min_topk,
                    max=max_topk
                )
            else:
                raise ValueError(f"Unsupported dynamic topk scheduler: {scheduler}")

        # process MoE
        def custom_forward(hidden_states):
            probs, routing_map = self.router(hidden_states, dynamic_topk=dynamic_topk)
            if self.use_shared_expert and self.config.shared_experts_with_logits:
                n_shared_experts = self.config.moe_shared_expert_intermediate_size // self.config.moe_ffn_hidden_size
                shared_expert_logits = probs[:, -n_shared_experts:]
                probs = probs[:, :-n_shared_experts]
                
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, routing_map
            )
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            if self.use_shared_expert:
                if not self.shared_expert_overlap:
                    # if shared_expert_overlap is True, the expert calculation happens in
                    # the token_dispatcher to overlap communications and computations
                    if self.config.shared_experts_with_logits:
                        seq_len, bsz, hidden_dim = hidden_states.shape
                        share_expert_outputs = [
                            shared_expert(hidden_states) * shared_expert_logits[:, s_i].view(seq_len, bsz, 1)
                            for s_i, shared_expert in enumerate(self.shared_experts)
                            ]
                        for _out_state in share_expert_outputs:
                            output += _out_state
                        # output += torch.sum(torch.stack(share_expert_outputs, dim=2), dim=2)
                        
                    else:
                        output += self.shared_experts(hidden_states)
                elif self.shared_expert_overlap and self.config.shared_experts_with_logits:
                    raise NotImplementedError("Shared experts with logits not implemented for self.shared_expert_overlap.")
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias