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
# Modified for VisDINO by Shahriar Shakir Sumit, Murat Tahtali and Seyedali Mirjalili, 2026.
# The modifications are released under the MIT License. See LICENSE and NOTICE.
# ------------------------------------------------------------------------------------------------

"""
VisDINO Transformer: Encoder-Decoder with OGS cross-attention and dual-box refinement.

Changes from standard DINO Transformer:
1. Decoder cross-attention uses OGS (visibility-gated deformable attention)
2. Decoder produces dual-box predictions (fbox + vbox) per layer
3. Visibility ratio prediction per query
"""

import torch
import torch.nn as nn

from detrex.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    MultiheadAttention,
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
    get_sine_pos_embed,
)
from detrex.utils import inverse_sigmoid

from .ogs_attention import OGSMultiScaleDeformableAttention

try:
    from projects.visdino.tomm_extension.modeling import MSOGSMultiScaleDeformableAttention
except ImportError:
    MSOGSMultiScaleDeformableAttention = None

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None


class VisDINOTransformerDecoder(TransformerLayerSequence):
    """
    VisDINO Decoder with:
    - OGS-enhanced cross-attention (visibility-gated sampling)
    - Dual-box iterative refinement (fbox + vbox)
    - Visibility ratio prediction
    
    Args:
        use_ogs (bool): If True, use OGS visibility-gated cross-attention.
            If False, use standard deformable attention (for ablation studies).
        cross_attn_override (nn.Module, optional): If provided, use this module
            as cross-attention instead of automatically selecting based on use_ogs.
            This allows plugging in MS-OGS or other custom attention modules.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        return_intermediate: bool = True,
        num_feature_levels: int = 4,
        look_forward_twice: bool = True,
        use_checkpoint: bool = True,
        use_ogs: bool = True,
        cross_attn_override: nn.Module = None,
    ):
        # Select cross-attention type
        if cross_attn_override is not None:
            cross_attn_module = cross_attn_override
        elif use_ogs:
            cross_attn_module = OGSMultiScaleDeformableAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
                num_levels=num_feature_levels,
            )
        else:
            cross_attn_module = MultiScaleDeformableAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
                num_levels=num_feature_levels,
            )

        super(VisDINOTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    # Self-attention (standard)
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=True,
                    ),
                    # Cross-attention: OGS or standard
                    cross_attn_module,
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate
        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)

        # Full-body box refinement (standard DINO)
        self.bbox_embed = None
        self.class_embed = None

        # Visible-body box refinement (NEW)
        self.vbox_embed = None

        # Visibility ratio prediction (NEW)
        self.vis_ratio_head = None

        self.look_forward_twice = look_forward_twice
        self.norm = nn.LayerNorm(embed_dim)

        if use_checkpoint and checkpoint_wrapper is not None:
            for layer in self.layers:
                layer = checkpoint_wrapper(layer)

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        reference_points=None,
        valid_ratios=None,
        **kwargs,
    ):
        output = query
        bs, num_queries, _ = output.size()
        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)

        intermediate = []
        intermediate_reference_points = []
        intermediate_vbox_reference_points = []
        intermediate_vis_ratios = []

        # Initialize vbox reference points = fbox reference points
        vbox_reference_points = reference_points.clone()

        # Collect OGS visibility weights per layer for loss computation
        all_vis_weights = []
        all_sampling_locs = []

        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            # Forward through decoder layer (with OGS cross-attention)
            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                **kwargs,
            )

            # Collect OGS info from cross-attention layer
            cross_attn = layer.attentions[1]  # Second attention is cross-attn (OGS)
            if hasattr(cross_attn, 'get_visibility_weights'):
                vis_w = cross_attn.get_visibility_weights()
                samp_loc = cross_attn.get_sampling_locations()
                if vis_w is not None:
                    all_vis_weights.append(vis_w)
                if samp_loc is not None:
                    all_sampling_locs.append(samp_loc)

            # === Full-body box refinement (standard DINO) ===
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            # === Visible-body box refinement (NEW) ===
            if self.vbox_embed is not None:
                vbox_tmp = self.vbox_embed[layer_idx](output)
                if vbox_reference_points.shape[-1] == 4:
                    new_vbox_ref = vbox_tmp + inverse_sigmoid(vbox_reference_points)
                    new_vbox_ref = new_vbox_ref.sigmoid()
                else:
                    new_vbox_ref = vbox_tmp
                    new_vbox_ref[..., :2] = vbox_tmp[..., :2] + inverse_sigmoid(vbox_reference_points)
                    new_vbox_ref = new_vbox_ref.sigmoid()
                vbox_reference_points = new_vbox_ref.detach()

            # === Visibility ratio prediction (NEW) ===
            vis_ratio = None
            if self.vis_ratio_head is not None:
                vis_ratio = self.vis_ratio_head[layer_idx](output)  # (bs, nq, 1)

            # Store intermediates
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.look_forward_twice:
                    intermediate_reference_points.append(new_reference_points)
                    if self.vbox_embed is not None:
                        intermediate_vbox_reference_points.append(new_vbox_ref)
                else:
                    intermediate_reference_points.append(reference_points)
                    if self.vbox_embed is not None:
                        intermediate_vbox_reference_points.append(vbox_reference_points)
                if vis_ratio is not None:
                    intermediate_vis_ratios.append(vis_ratio)

        if self.return_intermediate:
            result = {
                'inter_states': torch.stack(intermediate),
                'inter_references': torch.stack(intermediate_reference_points),
                'vis_weights': all_vis_weights,
                'sampling_locs': all_sampling_locs,
            }
            if intermediate_vbox_reference_points:
                result['inter_vbox_references'] = torch.stack(intermediate_vbox_reference_points)
            if intermediate_vis_ratios:
                result['inter_vis_ratios'] = torch.stack(intermediate_vis_ratios)
            return result

        return output, reference_points


class VisDINOTransformer(nn.Module):
    """
    VisDINO Transformer: DINO transformer with OGS decoder.
    
    Same encoder as DINO, but decoder uses OGS cross-attention
    and produces dual-box + visibility ratio predictions.
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        num_feature_levels=4,
        two_stage_num_proposals=900,
        learnt_init_query=True,
    ):
        super(VisDINOTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals

        self.embed_dim = self.encoder.embed_dim

        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dim)
        )
        self.learnt_init_query = learnt_init_query
        if self.learnt_init_query:
            self.tgt_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dim)
        self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
        self.enc_output_norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        _attn_types = (MultiScaleDeformableAttention, OGSMultiScaleDeformableAttention)
        if MSOGSMultiScaleDeformableAttention is not None:
            _attn_types = _attn_types + (MSOGSMultiScaleDeformableAttention,)
        for m in self.modules():
            if isinstance(m, _attn_types):
                m.init_weights()
        nn.init.normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """Generate encoder output proposals (same as DINO)."""
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur: (_cur + H * W)].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += H * W

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get reference points for decoder (same as DINO)."""
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_position_embeddings,
        query_embed,
        attn_masks,
        **kwargs,
    ):
        """Forward pass (same interface as DINO transformer)."""
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_position_embeddings)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)

        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in multi_level_masks], 1
        )

        # ============================================================
        # Encoder
        # ============================================================
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat_flatten.device
        )
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            **kwargs,
        )

        # ============================================================
        # Two-stage proposal generation (same as DINO)
        # ============================================================
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )

        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = (
            self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        )  # unsigmoided

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )  # unsigmoided
        reference_points = topk_coords_unact.detach().sigmoid()

        # Prepare init queries
        init_query_label, init_query_bbox = query_embed
        if init_query_label is not None:
            # Denoising queries from CDN: prepend DN reference points
            reference_points = torch.cat([init_query_bbox.sigmoid(), reference_points], 1)

        init_reference_out = reference_points

        # Extract region features
        target_unact = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )
        if self.learnt_init_query:
            target = self.tgt_embed.weight[None].repeat(bs, 1, 1)
        else:
            target = target_unact.detach()
        if init_query_label is not None:
            target = torch.cat([init_query_label, target], dim=1)

        # ============================================================
        # Decoder (with OGS cross-attention)
        # ============================================================
        decoder_out = self.decoder(
            query=target,
            key=memory,
            value=memory,
            attn_masks=attn_masks,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        if isinstance(decoder_out, dict):
            inter_states = decoder_out['inter_states']
            inter_references = decoder_out['inter_references']
        else:
            inter_states, inter_references = decoder_out

        # Return format compatible with DINO
        return (
            inter_states,
            init_reference_out,
            inter_references,
            target_unact,
            topk_coords_unact.sigmoid(),
            decoder_out if isinstance(decoder_out, dict) else {},
        )

    @staticmethod
    def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
