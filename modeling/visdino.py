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
VisDINO: Visibility-aware DINO for Occluded Person Detection

Extends DINO with:
1. OGS-enhanced cross-attention (visibility-gated deformable sampling)
2. Dual-box prediction: full-body (fbox) + visible-body (vbox) per query
3. Visibility ratio prediction: scalar per query
4. Visibility-aware matching in criterion
"""

import copy
import math
import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb

try:
    from projects.visdino.tomm_extension.modeling import prepare_for_ogs_cdn
except ImportError:
    prepare_for_ogs_cdn = None


class VisDINO(nn.Module):
    """
    VisDINO: Visibility-aware DINO for Occluded Person Detection.
    
    Same interface as DINO, but adds:
    - vbox_embed: visible box prediction head (per decoder layer)
    - vis_ratio_head: visibility ratio prediction (per decoder layer)
    - OGS cross-attention in decoder (integrated in transformer)
    """

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: nn.Module,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        aux_loss: bool = True,
        select_box_nums_for_evaluation: int = 300,
        device="cuda",
        dn_number: int = 100,
        label_noise_ratio: float = 0.2,
        box_noise_scale: float = 1.0,
        input_format: Optional[str] = "RGB",
        vis_period: int = 0,
        use_ogs_cdn: bool = False,
        ogs_cdn_alpha: float = 0.5,
        ogs_cdn_boundary_fraction: float = 0.3,
    ):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.neck = neck
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.transformer = transformer

        # Standard DINO heads
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes

        # === NEW: Visible box head ===
        self.vbox_embed = MLP(embed_dim, embed_dim, 4, 3)

        # === NEW: Visibility ratio head ===
        # Predicts scalar visibility ratio (0-1) per query via SmoothL1
        self.vis_ratio_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, 1),  # single scalar ratio
        )

        self.aux_loss = aux_loss
        self.criterion = criterion

        # Denoising
        self.label_enc = nn.Embedding(num_classes, embed_dim)
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # OGS-CDN settings
        self.use_ogs_cdn = use_ogs_cdn
        self.ogs_cdn_alpha = ogs_cdn_alpha
        self.ogs_cdn_boundary_fraction = ogs_cdn_boundary_fraction

        # Normalizer
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # Eval selection
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        # Visualization
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        # Initialize weights
        self._init_weights()

        # Create per-layer copies (num_decoder_layers + 1 for two-stage)
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.ModuleList(
            [copy.deepcopy(self.class_embed) for _ in range(num_pred)]
        )
        self.bbox_embed = nn.ModuleList(
            [copy.deepcopy(self.bbox_embed) for _ in range(num_pred)]
        )
        self.vbox_embed = nn.ModuleList(
            [copy.deepcopy(self.vbox_embed) for _ in range(num_pred)]
        )
        self.vis_ratio_head = nn.ModuleList(
            [copy.deepcopy(self.vis_ratio_head) for _ in range(num_pred)]
        )

        # Initialize bbox head biases
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        nn.init.constant_(self.vbox_embed[0].layers[-1].bias.data[2:], -2.0)

        # Wire heads into decoder for iterative refinement
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.vbox_embed = self.vbox_embed
        self.transformer.decoder.vis_ratio_head = self.vis_ratio_head

        # Two-stage handling, following DINO
        for bbox_layer in self.bbox_embed:
            nn.init.constant_(bbox_layer.layers[-1].bias.data[2:], 0.0)
        for vbox_layer in self.vbox_embed:
            nn.init.constant_(vbox_layer.layers[-1].bias.data[2:], 0.0)

    def _init_weights(self):
        """Initialize weights for heads."""
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.vbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.vbox_embed.layers[-1].bias.data, 0)

        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

    def forward(self, batched_inputs):
        """Forward pass."""
        images = self.preprocess_image(batched_inputs)

        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)

        # Backbone + neck
        features = self.backbone(images.tensor)
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(
                self.position_embedding(multi_level_masks[-1])
            )

        # Denoising
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            if self.use_ogs_cdn and prepare_for_ogs_cdn is not None:
                input_query_label, input_query_bbox, attn_mask, dn_meta = prepare_for_ogs_cdn(
                    targets,
                    dn_number=self.dn_number,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                    num_queries=self.num_queries,
                    num_classes=self.num_classes,
                    hidden_dim=self.embed_dim,
                    label_enc=self.label_enc,
                    occlusion_noise_alpha=self.ogs_cdn_alpha,
                    use_occ_boundary_negatives=True,
                    occ_boundary_fraction=self.ogs_cdn_boundary_fraction,
                )
            else:
                input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                    targets,
                    dn_number=self.dn_number,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                    num_queries=self.num_queries,
                    num_classes=self.num_classes,
                    hidden_dim=self.embed_dim,
                    label_enc=self.label_enc,
                )
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
            targets = None
        query_embeds = (input_query_label, input_query_bbox)

        # Transformer (encoder + OGS decoder)
        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,
            decoder_extra,
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
        )

        # Distributed-training handling
        inter_states[0] += self.label_enc.weight[0, 0] * 0.0

        # === Compute predictions per decoder layer ===
        outputs_classes = []
        outputs_coords = []  # fbox
        outputs_vcoords = []  # vbox
        outputs_vis_ratios = []

        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            # Classification
            outputs_class = self.class_embed[lvl](inter_states[lvl])

            # Full-body box
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            # Visible box (uses same reference as fbox for residual prediction)
            vtmp = self.vbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                vtmp += reference
            else:
                vtmp[..., :2] += reference
            outputs_vcoord = vtmp.sigmoid()

            # Visibility ratio (scalar)
            vis_ratio = self.vis_ratio_head[lvl](inter_states[lvl])  # (bs, nq, 1)

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_vcoords.append(outputs_vcoord)
            outputs_vis_ratios.append(vis_ratio)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_vcoord = torch.stack(outputs_vcoords)
        outputs_vis_ratio = torch.stack(outputs_vis_ratios)

        # Denoising post-process (for both fbox and vbox)
        if dn_meta is not None:
            outputs_class, outputs_coord, outputs_vcoord, outputs_vis_ratio = \
                self.dn_post_process(
                    outputs_class, outputs_coord, dn_meta,
                    outputs_vcoord=outputs_vcoord,
                    outputs_vis_ratio=outputs_vis_ratio,
                )

        # Build output dict
        output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_vboxes": outputs_vcoord[-1],
            "pred_vis_ratio": outputs_vis_ratio[-1],
        }

        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_vcoord, outputs_vis_ratio
            )

        # Two-stage encoder output
        interm_coord = enc_reference
        interm_class = self.transformer.decoder.class_embed[-1](enc_state)
        output["enc_outputs"] = {
            "pred_logits": interm_class,
            "pred_boxes": interm_coord,
        }

        # OGS visibility info for L_point loss
        if isinstance(decoder_extra, dict):
            output["ogs_vis_weights"] = decoder_extra.get("vis_weights", [])
            output["ogs_sampling_locs"] = decoder_extra.get("sampling_locs", [])

        if self.training:
            # Visualize
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    box_cls = output["pred_logits"]
                    box_pred = output["pred_boxes"]
                    results = self.inference(box_cls, box_pred, images.image_sizes)
                    self.visualize_training(batched_inputs, results)

            # Compute loss
            loss_dict = self.criterion(output, targets, dn_meta)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            box_vpred = output.get("pred_vboxes", None)
            results = self.inference(box_cls, box_pred, images.image_sizes, box_vpred=box_vpred)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_vcoord, outputs_vis_ratio):
        """Set auxiliary losses for all decoder layers except the last."""
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_vboxes": c,
                "pred_vis_ratio": d,
            }
            for a, b, c, d in zip(
                outputs_class[:-1],
                outputs_coord[:-1],
                outputs_vcoord[:-1],
                outputs_vis_ratio[:-1],
            )
        ]

    def dn_post_process(
        self, outputs_class, outputs_coord, dn_metas,
        outputs_vcoord=None, outputs_vis_ratio=None,
    ):
        """Post-process denoising queries (split DN from matching queries)."""
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]

            # DN outputs
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            # Matching outputs
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]

            out = {
                "pred_logits": output_known_class[-1],
                "pred_boxes": output_known_coord[-1],
            }

            # Handle vbox DN
            if outputs_vcoord is not None:
                output_known_vcoord = outputs_vcoord[:, :, :padding_size, :]
                outputs_vcoord = outputs_vcoord[:, :, padding_size:, :]
                out["pred_vboxes"] = output_known_vcoord[-1]

            # Handle vis_ratio DN
            if outputs_vis_ratio is not None:
                output_known_vis_ratio = outputs_vis_ratio[:, :, :padding_size, :]
                outputs_vis_ratio = outputs_vis_ratio[:, :, padding_size:, :]
                out["pred_vis_ratio"] = output_known_vis_ratio[-1]

            if self.aux_loss:
                if outputs_vcoord is not None and outputs_vis_ratio is not None:
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                            "pred_vboxes": c,
                            "pred_vis_ratio": d,
                        }
                        for a, b, c, d in zip(
                            output_known_class[:-1],
                            output_known_coord[:-1],
                            output_known_vcoord[:-1],
                            output_known_vis_ratio[:-1],
                        )
                    ]
                else:
                    out["aux_outputs"] = self._set_aux_loss(
                        output_known_class, output_known_coord,
                        output_known_class,  # dummy
                        output_known_class[:, :, :, :1],  # dummy
                    )

            dn_metas["output_known_lbs_bboxes"] = out

        return outputs_class, outputs_coord, outputs_vcoord, outputs_vis_ratio

    def prepare_for_cdn(
        self, targets, dn_number, label_noise_ratio, box_noise_scale,
        num_queries, num_classes, hidden_dim, label_enc,
    ):
        """CDN denoising preparation (same as DINO)."""
        if dn_number <= 0:
            return None, None, None, None
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["labels"])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None

        dn_number = dn_number // (int(max(known_num) * 2))
        if dn_number == 0:
            dn_number = 1

        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, num_classes)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)

        single_padding = int(max(known_num))
        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = (
            torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        )
        positive_idx += (
            torch.tensor(range(dn_number)) * len(boxes) * 2
        ).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)

        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            )
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to("cuda")
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )
            map_known_indice = torch.cat(
                [map_known_indice + single_padding * i for i in range(2 * dn_number)]
            ).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0
        attn_mask[pad_size:, :pad_size] = True
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_padding * 2 * i: single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1): pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[
                    single_padding * 2 * i: single_padding * 2 * (i + 1),
                    : single_padding * i * 2,
                ] = True
            else:
                attn_mask[
                    single_padding * 2 * i: single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1): pad_size,
                ] = True
                attn_mask[
                    single_padding * 2 * i: single_padding * 2 * (i + 1),
                    : single_padding * 2 * i,
                ] = True

        dn_meta = {
            "single_padding": single_padding * 2,
            "dn_num": dn_number,
        }
        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def inference(self, box_cls, box_pred, image_sizes, box_vpred=None):
        """Inference: same as DINO, but also outputs vbox if available."""
        assert len(box_cls) == len(image_sizes)
        results = []
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # Gather vbox predictions using the same top-k indices
        vboxes = None
        if box_vpred is not None:
            vboxes = torch.gather(box_vpred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            # Attach vbox predictions
            if vboxes is not None:
                vbox_per_image = vboxes[i]
                result.pred_vboxes = Boxes(box_cxcywh_to_xyxy(vbox_per_image))
                result.pred_vboxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            results.append(result)
        return results

    def prepare_targets(self, targets):
        """Prepare targets: normalize boxes and include vbox + ratio."""
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=self.device
            )
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)

            target = {"labels": gt_classes, "boxes": gt_boxes}

            # Visible box (if available)
            if hasattr(targets_per_image, "gt_vbox"):
                gt_vbox = targets_per_image.gt_vbox / image_size_xyxy
                gt_vbox = box_xyxy_to_cxcywh(gt_vbox)
                target["vbox"] = gt_vbox
            else:
                # If no vbox, use fbox (fully visible)
                target["vbox"] = gt_boxes.clone()

            # Visibility ratio (if available)
            if hasattr(targets_per_image, "gt_ratio"):
                target["vis_ratio"] = targets_per_image.gt_ratio
            else:
                target["vis_ratio"] = torch.ones(
                    len(gt_classes), device=self.device
                )

            new_targets.append(target)
        return new_targets

    def visualize_training(self, batched_inputs, results):
        from detectron2.utils.visualizer import Visualizer
        storage = get_event_storage()
        max_vis_box = 20
        for input, results_per_image in zip(batched_inputs, results):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=results_per_image.pred_boxes[:max_vis_box].tensor.detach().cpu().numpy()
            )
            pred_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, pred_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted boxes"
            storage.put_image(vis_name, vis_img)
            break
