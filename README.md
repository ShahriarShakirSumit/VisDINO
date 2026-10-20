# Visible-Region Attention Supervision for Occluded Pedestrian Detection

Shahriar Shakir Sumit<sup>1</sup>, Murat Tahtali<sup>1</sup>, Seyedali Mirjalili<sup>2</sup>

<sup>1</sup>UNSW Canberra, Australia   <sup>2</sup>Torrens University Australia

IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2027

## Overview

A DINO-style detector does not read the whole image for each candidate person. It samples a small
set of points inside the region it is attending to and combines what it finds there. Under
occlusion much of the full-body box belongs to the occluder rather than to the target, and
nothing in the usual training signal tells the detector which of its sampled points lie on the
person it is trying to detect.

VisDINO adds a gate inside the decoder cross-attention. For every sampling point the gate
produces a value between 0 and 1, multiplies it into the attention weight and renormalises, so
the total attention for a candidate is unchanged and only its distribution moves. During training
the gate is supervised at the level of individual sampling points using the CrowdHuman visible
boxes: a point inside the visible box is a positive and a point inside the full box but outside
the visible box is a negative. Visible boxes are needed only for training. At inference the model
takes an ordinary image and still predicts full-body boxes, and the gate adds 148,608 parameters,
or 0.31% of DINO.

This repository contains the model, the configurations used for the reported runs, and the
evaluation and analysis scripts.

## Results

CrowdHuman validation. AP@0.5, MR<sup>-2</sup> and JI are from the official CrowdDet evaluation;
mAP is COCO AP@[.50:.95]. Lower is better for MR<sup>-2</sup>.

| Backbone | Method | AP@0.5 | mAP | MR<sup>-2</sup> | JI |
|---|---|---|---|---|---|
| R-50 | DINO | 93.07 | 48.13 | 40.97 | 82.99 |
| R-50 | VisDINO | 94.15 | 50.20 | 37.98 | 84.41 |
| R-101 | DINO | 90.36 | - | 46.65 | 79.38 |
| R-101 | VisDINO | 92.16 | - | 43.13 | 81.93 |

The gain concentrates on occluded pedestrians. At a score threshold of 0.3 recall improves by
2.07 on unoccluded, 2.53 on partially occluded and 2.92 on heavily occluded pedestrians.
Removing only the point loss, while keeping the gate and every other visible-box loss, recovers
0.36 of the 2.07 mAP gain, so the improvement comes from the supervision rather than from the
added capacity.

## Repository structure

```
modeling/ogs_attention.py        the visibility gate inside deformable cross-attention
modeling/visdino.py              the detector
modeling/visdino_transformer.py  encoder and decoder
modeling/visdino_criterion.py    losses, including the point loss on the gate
configs/                         configurations for every run reported in the paper
tools/                           evaluation, analysis and table-generation scripts
```

The configurations cover the main runs and the ablations:

```
dino_r50_4scale_crowdhuman.py           DINO R-50 baseline, the 180k starting checkpoint
visdino_r50_4scale_crowdhuman_300k.py   VisDINO R-50, the reported model
dino_r101_crowdhuman_300k.py            DINO R-101 baseline
visdino_r101_crowdhuman_300k.py         VisDINO R-101
ablation_no_lpoint_300k.py              gate kept, point loss removed
ablation_no_ogs_300k.py                 gate removed
ablation_fullbox_point_300k.py          gate supervised by the full box instead of the visible box
ablation_lambdap_0.25_300k.py           point-loss weight 0.25
ablation_lambdap_1.0_300k.py            point-loss weight 1.0
```

## Installation

This is a project directory for [detrex](https://github.com/IDEA-Research/detrex) rather than a
standalone package. Install detectron2 and detrex first, then place this directory under
`projects/visdino` of the detrex tree.

```
git clone https://github.com/ShahriarShakirSumit/VisDINO.git detrex/projects/visdino
```

## Data

CrowdHuman is expected in COCO format with both the full-body box and the visible box retained,
since the visible box supplies the training signal. NightOwls is used only to test transfer and
needs no visible boxes.

## Training

The reported R-50 model starts from a DINO R-50 checkpoint trained on CrowdHuman for 180k
iterations and then trains for 300k iterations.

```
python tools/train_net.py \
    --config-file projects/visdino/configs/visdino_r50_4scale_crowdhuman_300k.py \
    --num-gpus 1
```

AdamW, learning rate 2e-4 dropped by 0.1 at 275k iterations, weight decay 1e-4, batch size 2,
AMP, one RTX 4090, about 60 hours. No hyperparameter search was performed for the reported run.
The gate bias is initialised to +2.0, so the gate starts near 1 and training begins close to the
ordinary DINO attention path.

## Evaluation

```
python tools/per_occlusion_eval.py     # recall and AP by occlusion level
python tools/eval_nightowls.py         # zero-shot transfer to NightOwls
python tools/temporal_consistency.py   # frame-to-frame stability on NightOwls sequences
python tools/visualize_gates.py        # gate values at the sampling points
python tools/statistical_analysis.py   # paired image-level bootstrap
python tools/benchmark_speed.py        # parameters, throughput and peak memory
python tools/generate_paper_tables.py  # the tables of the paper from the evaluation outputs
```

Two details that affect how the numbers should be read. In `per_occlusion_eval.py` each occlusion
bin is scored against only the ground truth in that bin, so a detection matching a pedestrian
from another bin counts as a false positive rather than being ignored. This lowers absolute
per-bin AP, applies identically to every model, and leaves recall unaffected, since recall is
computed over in-bin ground truth. In `statistical_analysis.py` the bootstrap resamples images
rather than detections, because MR<sup>-2</sup> is a dataset-level quantity that does not
decompose per image.

## Reproducing the reported numbers

The reported R-50 result is the final checkpoint at 300k iterations with seed 34908048, giving
50.20 mAP and 37.98 MR<sup>-2</sup>. An independently trained second seed reached 38.16
MR<sup>-2</sup>, so run-to-run variation of about 0.2 points should be expected. Every number in
the paper comes from the final checkpoint of a run rather than from the best checkpoint.

## Licence

The code is released under the MIT licence (LICENSE). Parts of modeling/visdino.py,
modeling/visdino_transformer.py, modeling/visdino_criterion.py and modeling/ogs_attention.py are
derived from detrex and Deformable DETR and remain under the Apache License 2.0. Their original
copyright notices are kept in those files, and NOTICE and LICENSE-APACHE give the details.

## Citation

```bibtex
@inproceedings{sumit2027visdino,
  title     = {Visible-Region Attention Supervision for Occluded Pedestrian Detection},
  author    = {Sumit, Shahriar Shakir and Tahtali, Murat and Mirjalili, Seyedali},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## Contact

For questions or problems, please open a GitHub issue or contact Shahriar Shakir Sumit
(m.sumit@unsw.edu.au or shahriar9121@gmail.com).

## Acknowledgements

Built on [detrex](https://github.com/IDEA-Research/detrex) and
[detectron2](https://github.com/facebookresearch/detectron2). CrowdHuman and NightOwls are the
work of their respective authors and remain subject to their own terms of use.
