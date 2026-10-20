# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/functions/ms_deform_attn_func.py
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/multi_scale_deform_attn.py
# ------------------------------------------------------------------------------------------------
# Modified for VisDINO by Shahriar Shakir Sumit, Murat Tahtali and Seyedali Mirjalili, 2026.
# The modifications are released under the MIT License. See LICENSE and NOTICE.
# ------------------------------------------------------------------------------------------------

"""
Occlusion-Gated Sampling (OGS) Multi-Scale Deformable Attention

Core innovation of VisDINO: predicts per-sampling-point visibility weights
and uses them to gate attention aggregation. Supervised by CrowdHuman
visible/full box geometry.

Key design choices:
1. Gate at sampling point level (not per-query heatmap) → memory efficient
2. Visibility gate is a lightweight MLP → minimal overhead
3. Gates initialized to "all visible" (bias=+2.0) → stable warm-start
4. Fully vectorized → no loops over queries
"""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from detrex.layers.multi_scale_deform_attn import (
    MultiScaleDeformableAttention,
    MultiScaleDeformableAttnFunction,
    multi_scale_deformable_attn_pytorch,
    _is_power_of_2,
)

try:
    from detrex import _C
    HAS_CUDA_OPS = True
except ImportError:
    HAS_CUDA_OPS = False


class OGSMultiScaleDeformableAttention(nn.Module):
    """
    Occlusion-Gated Sampling (OGS) Multi-Scale Deformable Attention.
    
    Extends standard deformable attention with per-sampling-point visibility
    gates that suppress features from occluded regions.
    
    The visibility gates are:
    - Predicted from query features + sampling point offsets
    - Supervised by visible/full box geometry (BCE loss)
    - Applied as multiplicative weights on attention scores
    
    Args:
        embed_dim (int): Embedding dimension. Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        num_levels (int): Number of feature map levels. Default: 4.
        num_points (int): Number of sampling points per query per head per level. Default: 4.
        img2col_step (int): im2col step. Default: 64.
        dropout (float): Dropout rate. Default: 0.1.
        batch_first (bool): If True, input/output as (bs, n, embed_dim). Default: True.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        img2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}"
            )
        head_dim = embed_dim // num_heads

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.im2col_step = img2col_step
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        if not _is_power_of_2(head_dim):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make sure that "
                "each dim of the attention head a power of 2."
            )

        # Standard deformable attention projections
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # ============================================================
        # OGS: Visibility gate MLP
        # Input: query features (embed_dim)
        # Output: one scalar per sampling point (num_heads * num_levels * num_points)
        # ============================================================
        self.visibility_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, num_heads * num_levels * num_points),
        )

        self.init_weights()

    def init_weights(self):
        """Initialize weights. OGS gates initialized to 'visible' (bias=+2.0)."""
        # Standard deformable attention init
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

        # OGS gate initialization: bias=+2.0 → sigmoid(2.0) ≈ 0.88
        # This means "mostly visible" at initialization → stable warm-start
        constant_(self.visibility_gate[-1].weight.data, 0.0)
        constant_(self.visibility_gate[-1].bias.data, 2.0)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with OGS visibility gating.
        
        Returns:
            output (Tensor): Attention output, shape (bs, num_query, embed_dim)
            
        Side effect:
            Stores self._sampling_locations and self._visibility_weights
            for loss computation and visualization.
        """
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        # Value projection
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)

        # Sampling offsets
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        # Attention weights (standard)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points,
        )

        # ============================================================
        # OGS: Compute visibility gates
        # ============================================================
        vis_gates = self.visibility_gate(query)  # (bs, num_query, H*L*P)
        vis_gates = torch.sigmoid(vis_gates)
        vis_gates = vis_gates.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        # Apply visibility gates to attention weights
        # Gated weights = attention_weights * vis_gates
        gated_attention_weights = attention_weights * vis_gates

        # Renormalize to maintain attention sum ≈ 1
        gated_attention_weights = gated_attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        weight_sum = gated_attention_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        gated_attention_weights = gated_attention_weights / weight_sum
        gated_attention_weights = gated_attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points,
        )

        # Compute sampling locations (same as standard deformable attention)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}"
            )

        # Store for loss computation
        self._sampling_locations = sampling_locations.detach()
        self._visibility_weights = vis_gates  # Keep gradient for L_point loss

        # Forward attention with gated weights
        if torch.cuda.is_available() and value.is_cuda and HAS_CUDA_OPS:
            output = MultiScaleDeformableAttnFunction.apply(
                value.to(torch.float32) if value.dtype == torch.float16 else value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                gated_attention_weights,
                self.im2col_step,
            )
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, gated_attention_weights
            )

        if value.dtype == torch.float16:
            output = output.to(torch.float16)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

    def get_visibility_weights(self):
        """Return stored visibility weights for loss computation."""
        return getattr(self, '_visibility_weights', None)

    def get_sampling_locations(self):
        """Return stored sampling locations for loss computation."""
        return getattr(self, '_sampling_locations', None)
