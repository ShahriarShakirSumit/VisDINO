"""
VisDINO R50 4-scale CrowdHuman Config - 300k Extended Training

Exact same setup as V1 (180k) but max_iter=300000.
Based on DINO R50 baseline config, adding:
- OGS-enhanced deformable cross-attention
- Dual-box heads (fbox + vbox)
- Visibility ratio prediction
- VisDINO criterion with L_vbox, L_ratio, L_point

LAMBDA_P SENSITIVITY (2026-08-15): identical to the canonical config in every respect
except lambda_p (weight on L_point) 0.5 -> 0.25. AMP stays ON, exactly as canonical, so
the comparison is clean: lambda_p is the ONLY difference.
Gives the 3-point curve lambda_p = 0 (mAP 48.49) / 0.25 (?) / 0.5 (mAP 50.20).
NOTE lambda_p=1.0 diverged under AMP at iter 132 but ran 400 iters clean in fp32,
so that failure is a mixed-precision interaction, not an optimization failure.
"""

import copy
import torch.nn as nn
from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.config import get_config
from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine

# Import DINO encoder (shared with VisDINO)
from projects.dino.modeling import DINOTransformerEncoder

# Import VisDINO modules
from projects.visdino.modeling import (
    VisDINO,
    VisDINOTransformerDecoder,
    VisDINOTransformer,
    VisDINOCriterion,
)

# ============================================================================
# Base configs
# ============================================================================
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# ============================================================================
# Model
# ============================================================================
model = L(VisDINO)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res3", "res4", "res5"],
        freeze_at=1,
    ),
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    neck=L(ChannelMapper)(
        input_shapes={
            "res3": ShapeSpec(channels=512),
            "res4": ShapeSpec(channels=1024),
            "res5": ShapeSpec(channels=2048),
        },
        in_features=["res3", "res4", "res5"],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
    ),
    transformer=L(VisDINOTransformer)(
        encoder=L(DINOTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            post_norm=False,
            num_feature_levels=4,
            use_checkpoint=False,
        ),
        decoder=L(VisDINOTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels=4,
            use_checkpoint=False,
        ),
        num_feature_levels=4,
        two_stage_num_proposals="${..num_queries}",
    ),
    embed_dim=256,
    num_classes=1,  # person only
    num_queries=900,
    aux_loss=True,
    criterion=L(VisDINOCriterion)(
        num_classes="${..num_classes}",
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            # Standard DINO losses
            "loss_class": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn": 1.0,
            "loss_bbox_dn": 5.0,
            "loss_giou_dn": 2.0,
            # VisDINO losses
            "loss_vbox": 2.0,
            "loss_vbox_giou": 1.0,
            "loss_ratio": 1.0,
            "loss_point": 0.25,  # <<<--- SENSITIVITY: lambda_p 0.5 -> 0.25
            # DN variants of VisDINO losses
            "loss_vbox_dn": 2.0,
            "loss_vbox_giou_dn": 1.0,
            "loss_ratio_dn": 1.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        two_stage_binary_cls=False,
    ),
    dn_number=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    vis_period=0,
    input_format="RGB",
    device="cuda",
)

# ============================================================================
# Set aux loss weight dict (for all decoder layers + enc)
# ============================================================================
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    # Encoder losses
    aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
    # Auxiliary decoder layer losses
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict

# ============================================================================
# Training
# ============================================================================
train.init_checkpoint = "output/dino_r50_crowdhuman_baseline/model_final.pth"
train.output_dir = "./output/ablation_lambdap_0.25_300k"
train.device = "cuda"

# 300k iters (extended from V1's 180k, same LR schedule scales automatically)
train.max_iter = 300000
train.eval_period = 10000
train.log_period = 50
train.checkpointer.period = 10000

# Gradient clipping
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# AMP
train.amp.enabled = True

# ============================================================================
# Dataset: CrowdHuman
# ============================================================================
from pathlib import Path
import importlib

_detrex = importlib.import_module('detrex')
_p = Path(_detrex.__file__).resolve()
candidates = [_p.parents[2], _p.parents[1], Path.cwd()]
CH = None
for cand in candidates:
    dd = cand / 'datasets' / 'crowdhuman'
    if dd.exists():
        CH = dd
        break
if CH is None:
    CH = _p.parents[2] / 'datasets' / 'crowdhuman'
print('[VisDINO CrowdHuman] data root =', CH)

from detectron2.data.datasets.coco import load_coco_json
from detectron2.data import DatasetCatalog, MetadataCatalog

train_json = str(CH / 'annotation_train.json')
val_json = str(CH / 'annotation_val.json')
train_imgs = str(CH / 'images_train' / 'Images')
val_imgs = str(CH / 'images_val' / 'Images')

# CRITICAL: Use extra_annotation_keys to preserve visible_bbox / full_bbox
# register_coco_instances does NOT pass this, so visible_bbox was silently dropped!
# BUG FIX (2026-03-11): Force-deregister first to guarantee vbox is loaded.
_EXTRA_ANN_KEYS = ["visible_bbox", "full_bbox"]

def _force_register_with_vbox(name, json_file, image_root, extra_keys):
    """Register COCO dataset preserving custom annotation fields.
    
    Force-deregisters first to prevent silent fallback to a prior
    registration that lacks extra_annotation_keys.
    """
    # Remove any existing registration (e.g. from register_coco_instances)
    if name in DatasetCatalog:
        DatasetCatalog.remove(name)
        print(f'[VisDINO] Removed stale registration for {name}')
    
    DatasetCatalog.register(
        name,
        lambda jf=json_file, ir=image_root, n=name, ek=extra_keys:
            load_coco_json(jf, ir, n, extra_annotation_keys=ek)
    )
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco"
    )

_force_register_with_vbox("crowdhuman_train", train_json, train_imgs, _EXTRA_ANN_KEYS)
_force_register_with_vbox("crowdhuman_val", val_json, val_imgs, _EXTRA_ANN_KEYS)
print('[VisDINO CrowdHuman] Registered datasets (with visible_bbox!)')

# Runtime assertion: verify visible_bbox is actually loaded
_sample = DatasetCatalog.get("crowdhuman_val")[:3]
for _s in _sample:
    for _a in _s.get("annotations", []):
        assert "visible_bbox" in _a, (
            f"BUG: visible_bbox missing from loaded annotations! "
            f"Keys: {list(_a.keys())}. Check extra_annotation_keys."
        )
        break
    break
print('[VisDINO CrowdHuman] VERIFIED: visible_bbox present in loaded annotations ✓')

# ============================================================================
# Dataloader with VisDINO mapper (loads vbox + ratio)
# ============================================================================
from detectron2.config import LazyCall as L
import detectron2.data.transforms as T
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from projects.visdino.data import VisDINOCrowdHumanMapper

# Train dataloader with VisDINO mapper
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="crowdhuman_train"),
    mapper=L(VisDINOCrowdHumanMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=2,
    num_workers=8,
)

# Test dataloader (standard, no vbox needed for eval)
from detrex.data import DetrDatasetMapper
dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="crowdhuman_val", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

# Evaluator
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="crowdhuman_val",
    output_dir=train.output_dir,
)

# ============================================================================
# Optimizer
# ============================================================================
optimizer.lr = 2e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
