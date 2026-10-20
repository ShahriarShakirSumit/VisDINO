"""
DINO R50 4-scale CrowdHuman Config

Based on official detrex DINO implementation with CrowdHuman dataset.
Uses the "hacked trainer" which achieves 49.4-50.0% COCO AP.

Key settings:
- ResNet-50 backbone (ImageNet pretrained)
- 4 feature scales
- 900 queries + 100 denoising queries
- Single class (person)
- Gradient clipping enabled
- AMP (mixed precision) for memory efficiency

Reference:
- DINO paper: https://arxiv.org/abs/2203.03605
- detrex DINO: projects/dino/README.md
"""

import copy
from detrex.config import get_config
from projects.dino.configs.models.dino_r50 import model

# Base configs
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# ============================================================================
# Model Configuration
# ============================================================================

# Single class for CrowdHuman (person only)
model.num_classes = 1
model.criterion.num_classes = 1

# DINO defaults (keep as reference)
model.num_queries = 900  # DINO uses 900 queries
model.dn_number = 100    # denoising queries

# Device
model.device = "cuda"

# ============================================================================
# Training Configuration
# ============================================================================

# Use ImageNet pretrained ResNet-50 (standard for DINO)
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/dino_r50_crowdhuman_baseline"
train.device = "cuda"

# Training schedule
# CrowdHuman: ~15k train images, batch_size 2 = ~7500 iters/epoch
# 180k iters ≈ 24 epochs (similar to DINO 24ep on COCO)
train.max_iter = 180000
train.eval_period = 10000
train.log_period = 50
train.checkpointer.period = 10000

# Gradient clipping (essential for DINO training stability)
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# AMP for memory efficiency on single GPU
train.amp.enabled = True

# ============================================================================
# Dataset Configuration (CrowdHuman)
# ============================================================================

from pathlib import Path
import importlib

# Find CrowdHuman dataset
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
print('[DINO CrowdHuman] data root =', CH)

# Register CrowdHuman datasets
from detectron2.data.datasets import register_coco_instances
train_json = str(CH / 'annotation_train.json')
val_json = str(CH / 'annotation_val.json')
train_imgs = str(CH / 'images_train' / 'Images')
val_imgs = str(CH / 'images_val' / 'Images')

try:
    register_coco_instances("crowdhuman_train", {}, train_json, train_imgs)
    register_coco_instances("crowdhuman_val", {}, val_json, val_imgs)
    print('[DINO CrowdHuman] Registered datasets')
except Exception as e:
    print('[warn] CrowdHuman registration failed:', e)

# ============================================================================
# Dataloader Configuration
# ============================================================================

from detectron2.config import LazyCall as L
import detectron2.data.transforms as T
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detrex.data import DetrDatasetMapper
from detectron2.evaluation import COCOEvaluator

# Single GPU settings - reduced batch size to prevent OOM on crowded images
dataloader.train.total_batch_size = 2  # RTX 4090 24GB (reduced from 4 due to OOM)
dataloader.train.num_workers = 8

# Train dataloader with DINO-style augmentation
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="crowdhuman_train"),
    mapper=L(DetrDatasetMapper)(
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

# Test dataloader
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
# Optimizer Configuration (DINO hacked trainer style)
# ============================================================================

# Note: The actual optimizer is constructed in train_net.py with special param groups:
# - backbone: lr = 2e-5
# - reference_points, sampling_offsets: lr = 2e-5  
# - rest: lr = 2e-4
# This config is for reference; train_net.py overrides it.
optimizer.lr = 2e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4

# ============================================================================
# Loss weights (DINO defaults)
# ============================================================================

# Already set in dino_r50.py, but ensure single class
model.criterion.weight_dict = {
    "loss_class": 1,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,
    "loss_class_dn": 1,
    "loss_bbox_dn": 5.0,
    "loss_giou_dn": 2.0,
}

# Set aux loss weight dict
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict
