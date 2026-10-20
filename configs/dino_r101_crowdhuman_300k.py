"""
DINO R101 4-scale CrowdHuman Baseline - 300K Iterations.

Backbone generalization study (Section 7 of TOMM paper).
Standard DINO with ResNet-101 backbone, ImageNet-pretrained init.
Same training recipe as R50 baseline (300K iters, lr=1e-4, AdamW).
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

from projects.dino.modeling import (
    DINO,
    DINOTransformerEncoder,
    DINOTransformerDecoder,
    DINOTransformer,
    DINOCriterion,
)

# ============================================================================
# Base configs
# ============================================================================
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# ============================================================================
# Model: Standard DINO with ResNet-101
# ============================================================================
model = L(DINO)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=101,
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
    transformer=L(DINOTransformer)(
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
        decoder=L(DINOTransformerDecoder)(
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
    num_classes=1,
    num_queries=900,
    aux_loss=True,
    criterion=L(DINOCriterion)(
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
            "loss_class": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn": 1.0,
            "loss_bbox_dn": 5.0,
            "loss_giou_dn": 2.0,
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
# Aux loss weight dict
# ============================================================================
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict

# ============================================================================
# Training - ImageNet R101 pretrained
# ============================================================================
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-101.pkl"
train.output_dir = "./output/dino_r101_crowdhuman_300k"
train.device = "cuda"
train.max_iter = 300000
train.eval_period = 10000
train.log_period = 50
train.checkpointer.period = 10000

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2
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
print('[DINO R101 CrowdHuman] data root =', CH)

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

train_json = str(CH / 'annotation_train.json')
val_json   = str(CH / 'annotation_val.json')
train_imgs = str(CH / 'images_train' / 'Images')
val_imgs   = str(CH / 'images_val'   / 'Images')

if "crowdhuman_train" not in DatasetCatalog:
    register_coco_instances("crowdhuman_train", {}, train_json, train_imgs)
    register_coco_instances("crowdhuman_val", {}, val_json, val_imgs)
    print('[DINO R101 CrowdHuman] Registered datasets')

# ============================================================================
# Dataloader
# ============================================================================
import detectron2.data.transforms as T
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detrex.data import DetrDatasetMapper
from detectron2.evaluation import COCOEvaluator

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

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="crowdhuman_val",
    output_dir="./output/dino_r101_crowdhuman_300k",
)

# ============================================================================
# Optimizer - standard DINO R101 settings
# ============================================================================
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

print(f"[DINO R101 CrowdHuman] Output dir: {train.output_dir}")
print(f"[DINO R101 CrowdHuman] Max iter: {train.max_iter}")
print(f"[DINO R101 CrowdHuman] Init checkpoint: {train.init_checkpoint}")
