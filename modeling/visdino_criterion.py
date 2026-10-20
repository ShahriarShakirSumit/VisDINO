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
VisDINO Criterion: Multi-task loss for visibility-aware detection.

Losses:
1. L_cls:       Focal loss for classification (standard DINO)
2. L_fbox:      L1 + GIoU for full-body box (standard DINO)
3. L_vbox:      L1 + GIoU for visible-body box
4. L_ratio:     SmoothL1 for visibility ratio (scalar regression)
5. L_point:     BCE for OGS point visibility (core innovation, with warmup)
6. DN losses:   Denoising losses for all above

Architecture:
- Inherits from DINOCriterion (TwoStageCriterion → SetCriterion)
- Adds vbox, ratio, and point losses
- Uses standard Hungarian matching on fbox (proven to work)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou
from detrex.utils import get_world_size, is_dist_avail_and_initialized

from projects.dino.modeling.dn_criterion import DINOCriterion


class VisDINOCriterion(DINOCriterion):
    """
    VisDINO loss = DINOCriterion + L_vbox + L_ratio + L_point.
    
    The matching is still on fbox only (same as DINO). This is intentional:
    - fbox matching is stable and proven
    - vbox, ratio, and point losses are supervised on matched pairs
    
    V1: SmoothL1 for vis_ratio regression, optional point warmup.
    """

    # Occlusion level bins: ratio thresholds
    OCC_THRESHOLDS = [0.85, 0.50]  # none: >0.85, partial: 0.50-0.85, heavy: <=0.50

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses=["class", "boxes"],
        eos_coef=None,
        loss_class_type="focal_loss",
        alpha: float = 0.25,
        gamma: float = 2.0,
        two_stage_binary_cls=False,
        # VisDINO-specific
        vbox_loss_coef: float = 2.0,
        vbox_giou_coef: float = 1.0,
        ratio_loss_coef: float = 1.0,
        point_loss_coef: float = 0.5,
        point_warmup_iters: int = 0,
        point_region: str = "vbox",
    ):
        super().__init__(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            eos_coef=eos_coef,
            loss_class_type=loss_class_type,
            alpha=alpha,
            gamma=gamma,
            two_stage_binary_cls=two_stage_binary_cls,
        )
        self.point_region = point_region  # "vbox" (default) or "boxes" (full-box ablation)
        self.vbox_loss_coef = vbox_loss_coef
        self.vbox_giou_coef = vbox_giou_coef
        self.ratio_loss_coef = ratio_loss_coef
        self.point_loss_coef = point_loss_coef
        self.point_warmup_iters = point_warmup_iters
        self._current_iter = 0

    def forward(self, outputs, targets, dn_metas=None):
        """
        Compute all losses: standard DINO losses + VisDINO losses.
        """
        self._current_iter += 1

        # Standard DINO losses (cls + fbox + giou + enc + dn)
        losses = super().forward(outputs, targets, dn_metas)

        # Compute number of boxes for normalization
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Get matching indices for the final layer
        outputs_without_aux = {k: v for k, v in outputs.items() if k not in ["aux_outputs", "enc_outputs"]}
        indices = self.matcher(outputs_without_aux, targets)

        # === VisDINO losses on final layer ===
        vis_losses = self._compute_visdino_losses(outputs, targets, indices, num_boxes)
        losses.update(vis_losses)

        # === VisDINO losses on auxiliary layers ===
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                aux_vis_losses = self._compute_visdino_losses(
                    aux_outputs, targets, aux_indices, num_boxes
                )
                aux_vis_losses = {k + f"_{i}": v for k, v in aux_vis_losses.items()}
                losses.update(aux_vis_losses)

        # === VisDINO DN losses ===
        if dn_metas and "output_known_lbs_bboxes" in dn_metas:
            dn_outputs = dn_metas["output_known_lbs_bboxes"]
            dn_num = dn_metas["dn_num"]
            single_padding = dn_metas["single_padding"]

            # Build DN indices
            dn_idx = []
            for i_img in range(len(targets)):
                if len(targets[i_img]["labels"]) > 0:
                    t = torch.arange(0, len(targets[i_img]["labels"])).long().cuda()
                    t = t.unsqueeze(0).repeat(dn_num, 1)
                    tgt_idx = t.flatten()
                    output_idx = (
                        torch.tensor(range(dn_num)) * single_padding
                    ).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()
                dn_idx.append((output_idx, tgt_idx))

            dn_vis_losses = self._compute_visdino_losses(
                dn_outputs, targets, dn_idx, num_boxes * dn_num
            )
            dn_vis_losses = {k + "_dn": v for k, v in dn_vis_losses.items()}
            losses.update(dn_vis_losses)

            # DN aux losses
            if "aux_outputs" in dn_outputs:
                for i, aux_dn_out in enumerate(dn_outputs["aux_outputs"]):
                    aux_dn_vis = self._compute_visdino_losses(
                        aux_dn_out, targets, dn_idx, num_boxes * dn_num
                    )
                    aux_dn_vis = {k + f"_dn_{i}": v for k, v in aux_dn_vis.items()}
                    losses.update(aux_dn_vis)

        # === L_point loss (OGS visibility supervision) ===
        point_loss = self._compute_point_loss(outputs, targets, indices, num_boxes)
        losses.update(point_loss)

        return losses

    def _compute_visdino_losses(self, outputs, targets, indices, num_boxes):
        """Compute vbox + ratio losses for given outputs and indices."""
        losses = {}

        # === L_vbox: visible box L1 + GIoU ===
        if "pred_vboxes" in outputs:
            idx = self._get_src_permutation_idx(indices)

            src_vboxes = outputs["pred_vboxes"][idx]
            target_vboxes = torch.cat(
                [t["vbox"][j] for t, (_, j) in zip(targets, indices)], dim=0
            )

            if src_vboxes.numel() > 0:
                # Clamp for numerical stability (randomly-initialized heads may produce NaN)
                src_vboxes_safe = torch.nan_to_num(src_vboxes, nan=0.0)
                target_vboxes_safe = torch.nan_to_num(target_vboxes, nan=0.0)
                
                loss_vbox = F.l1_loss(src_vboxes_safe, target_vboxes_safe, reduction="none")
                losses["loss_vbox"] = loss_vbox.sum() / num_boxes

                src_vboxes_xyxy = box_cxcywh_to_xyxy(src_vboxes_safe)
                tgt_vboxes_xyxy = box_cxcywh_to_xyxy(target_vboxes_safe)
                # Ensure valid xyxy (w,h > 0) — no inplace ops for autograd safety
                src_xy_min = src_vboxes_xyxy[:, :2]
                src_xy_max = torch.max(src_vboxes_xyxy[:, 2:], src_xy_min + 1e-4)
                src_vboxes_xyxy = torch.cat([src_xy_min, src_xy_max], dim=1)
                tgt_xy_min = tgt_vboxes_xyxy[:, :2]
                tgt_xy_max = torch.max(tgt_vboxes_xyxy[:, 2:], tgt_xy_min + 1e-4)
                tgt_vboxes_xyxy = torch.cat([tgt_xy_min, tgt_xy_max], dim=1)
                
                giou_vals = generalized_box_iou(src_vboxes_xyxy, tgt_vboxes_xyxy)
                loss_vbox_giou = 1 - torch.diag(giou_vals)
                loss_vbox_giou = torch.nan_to_num(loss_vbox_giou, nan=0.0)
                losses["loss_vbox_giou"] = loss_vbox_giou.sum() / num_boxes
            else:
                losses["loss_vbox"] = outputs["pred_vboxes"].sum() * 0.0
                losses["loss_vbox_giou"] = outputs["pred_vboxes"].sum() * 0.0

        # === L_ratio: visibility ratio regression (SmoothL1) ===
        # Predicts scalar vis_ratio for matched queries
        if "pred_vis_ratio" in outputs:
            idx = self._get_src_permutation_idx(indices)
            src_ratio = outputs["pred_vis_ratio"][idx].squeeze(-1)  # (N,)
            target_ratio = torch.cat(
                [t["vis_ratio"][j] for t, (_, j) in zip(targets, indices)], dim=0
            )

            if src_ratio.numel() > 0:
                losses["loss_ratio"] = F.smooth_l1_loss(
                    src_ratio.sigmoid(), target_ratio, reduction="mean"
                )
            else:
                losses["loss_ratio"] = outputs["pred_vis_ratio"].sum() * 0.0

        return losses

    def _compute_point_loss(self, outputs, targets, indices, num_boxes):
        """
        L_point: BCE loss on OGS visibility gates (matched queries only).
        
        For each sampling point of a MATCHED query, determine if it falls inside
        the visible box. Points inside vbox → label 1, outside → label 0.
        
        Key design: Only compute on matched queries (not all 900+180 queries).
        Unmatched queries have no GT visibility info, so including them would
        just add noise and cause massive loss values.
        """
        losses = {}

        vis_weights_list = outputs.get("ogs_vis_weights", [])
        sampling_locs_list = outputs.get("ogs_sampling_locs", [])

        if not vis_weights_list or not sampling_locs_list:
            losses["loss_point"] = torch.tensor(0.0, device=next(iter(outputs.values())).device)
            return losses

        total_point_loss = 0.0
        total_points = 0
        n_layers = min(len(vis_weights_list), len(sampling_locs_list))

        for layer_idx in range(n_layers):
            vis_weights = vis_weights_list[layer_idx]  # (bs, nq, H, L, P)
            sampling_locs = sampling_locs_list[layer_idx]  # (bs, nq, H, L*P, 2)

            bs = vis_weights.shape[0]
            H = vis_weights.shape[2]
            L = vis_weights.shape[3]
            P = vis_weights.shape[4]

            # Reshape sampling_locs to (bs, nq, H, L, P, 2)
            sampling_locs = sampling_locs.view(bs, -1, H, L, P, 2)

            matched_preds = []
            matched_targets = []

            for b_idx in range(bs):
                if b_idx >= len(indices):
                    continue
                src_idx, tgt_idx = indices[b_idx]
                if len(src_idx) == 0:
                    continue

                # Get matched GT vboxes in normalized [cx, cy, w, h] → xyxy
                gt_vbox_cxcywh = targets[b_idx][self.point_region][tgt_idx]  # (M, 4)
                gt_vbox_xyxy = box_cxcywh_to_xyxy(gt_vbox_cxcywh)  # (M, 4)

                M = len(src_idx)

                # Extract vis_weights and sampling_locs for matched queries only
                matched_vis = vis_weights[b_idx, src_idx]  # (M, H, L, P)
                matched_locs = sampling_locs[b_idx, src_idx]  # (M, H, L, P, 2)

                # Test if each point is inside the vbox
                vbox_x1 = gt_vbox_xyxy[:, 0].view(M, 1, 1, 1)
                vbox_y1 = gt_vbox_xyxy[:, 1].view(M, 1, 1, 1)
                vbox_x2 = gt_vbox_xyxy[:, 2].view(M, 1, 1, 1)
                vbox_y2 = gt_vbox_xyxy[:, 3].view(M, 1, 1, 1)

                px = matched_locs[..., 0]  # (M, H, L, P)
                py = matched_locs[..., 1]  # (M, H, L, P)

                inside = (
                    (px >= vbox_x1) & (px <= vbox_x2) &
                    (py >= vbox_y1) & (py <= vbox_y2)
                ).float()  # (M, H, L, P)

                matched_preds.append(matched_vis.reshape(-1))
                matched_targets.append(inside.reshape(-1))

            if matched_preds:
                all_preds = torch.cat(matched_preds, dim=0)
                all_targets = torch.cat(matched_targets, dim=0)

                # Disable autocast for BCE (unsafe under AMP autocast)
                with torch.cuda.amp.autocast(enabled=False):
                    # Ensure valid range and no NaN
                    pred_safe = all_preds.float()
                    pred_safe = torch.nan_to_num(pred_safe, nan=0.5, posinf=1.0, neginf=0.0)
                    pred_safe = pred_safe.clamp(1e-6, 1 - 1e-6)
                    loss_pt = F.binary_cross_entropy(
                        pred_safe,
                        all_targets.float(),
                        reduction="mean",
                    )
                total_point_loss += loss_pt
                total_points += 1  # Count layers that contributed

        # Average over layers
        if total_points > 0:
            losses["loss_point"] = total_point_loss / total_points
        else:
            losses["loss_point"] = torch.tensor(0.0, device=next(iter(outputs.values())).device)

        # Apply warmup: linearly ramp from 0 to full weight
        if self.point_warmup_iters > 0:
            warmup_scale = min(1.0, self._current_iter / self.point_warmup_iters)
            losses["loss_point"] = losses["loss_point"] * warmup_scale

        return losses
