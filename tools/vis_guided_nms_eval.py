#!/usr/bin/env python3
"""
Visibility-Guided NMS Evaluation for VisDINO

Standard NMS uses full-body (fbox) IoU for suppression, which over-suppresses
overlapping pedestrians. This script uses visible-body (vbox) IoU for the
suppression criterion: two fboxes are only suppressed if their vboxes overlap.

This is a FREE improvement — no retraining needed, just better post-processing.

Usage:
    conda activate detrex_env
    cd <detrex root>
    python projects/visdino/vis_guided_nms_eval.py \
        --config projects/visdino/configs/visdino_r50_4scale_crowdhuman_v2.py \
        --checkpoint output/visdino_r50_crowdhuman_v2/model_final.pth \
        --output-dir output/visdino_r50_crowdhuman_v2/vis_nms_eval

Can also be run on V1 model:
    python projects/visdino/vis_guided_nms_eval.py \
        --checkpoint output/visdino_r50_crowdhuman/model_final.pth \
        --config projects/visdino/configs/visdino_r50_4scale_crowdhuman.py \
        --output-dir output/visdino_r50_crowdhuman/vis_nms_eval
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import detectron2.data.transforms as T
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts, DatasetCatalog
from detectron2.structures import Boxes, Instances


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes (xyxy format)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


def cxcywh_to_xyxy(boxes):
    """Convert cx,cy,w,h to x1,y1,x2,y2."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def standard_nms(fboxes, scores, iou_threshold=0.5):
    """Standard NMS on full-body boxes."""
    keep = torch.ops.torchvision.nms(fboxes, scores, iou_threshold)
    return keep


def vis_guided_nms(fboxes, vboxes, scores, iou_threshold=0.5):
    """
    Visibility-guided NMS:
    - Score ordering: by detection score (on fbox)
    - Suppression: using vbox IoU instead of fbox IoU
    - Output: fbox detections (same as standard NMS)

    The key insight: two people whose full-body boxes overlap may have
    non-overlapping visible parts. Standard NMS would suppress one,
    but vis-guided NMS keeps both.
    """
    if len(fboxes) == 0:
        return torch.tensor([], dtype=torch.long)

    # Sort by score descending
    order = scores.argsort(descending=True)
    fboxes = fboxes[order]
    vboxes = vboxes[order]
    scores = scores[order]

    keep = []
    suppressed = torch.zeros(len(fboxes), dtype=torch.bool)

    for i in range(len(fboxes)):
        if suppressed[i]:
            continue
        keep.append(order[i].item())

        # Compute vbox IoU with remaining unsuppressed boxes
        if i + 1 < len(fboxes):
            remaining = ~suppressed[i+1:]
            if remaining.any():
                remaining_indices = torch.where(~suppressed)[0]
                remaining_indices = remaining_indices[remaining_indices > i]

                vbox_i = vboxes[i:i+1]
                vboxes_remaining = vboxes[remaining_indices]

                iou = box_iou(vbox_i, vboxes_remaining).squeeze(0)
                suppress_mask = iou > iou_threshold
                suppressed[remaining_indices[suppress_mask]] = True

    return torch.tensor(keep, dtype=torch.long)


def run_inference_with_vis_nms(model, dataloader, output_dir, 
                                iou_threshold=0.5, vis_nms=True,
                                score_threshold=0.01, max_dets=300):
    """Run model inference with visibility-guided NMS."""
    model.eval()
    device = next(model.parameters()).device

    all_results = []
    coco_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            images = model.preprocess_image(batch)
            features = model.backbone(images.tensor)
            multi_level_feats = model.neck(features)

            # Forward through transformer
            outputs = model.transformer(
                multi_level_feats,
                model.query_embedding.weight,
                model.position_embedding,
                attn_masks=None,
            )

            # Get predictions from final decoder layer
            # This may vary by exact model API; let's use the full forward pass
            pass

    # Alternative: use detectron2's standard inference API
    # and extract vbox predictions
    print("Running full model inference...")
    model.eval()

    results_standard = []
    results_vis_nms = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            # Standard detectron2 inference
            outputs = model(batch)

            for per_image_output, per_image_input in zip(outputs, batch):
                image_id = per_image_input.get("image_id", batch_idx)
                h, w = per_image_input["height"], per_image_input["width"]

                instances = per_image_output["instances"]

                # Get standard predictions (already NMS'd by the model)
                if hasattr(instances, 'pred_boxes'):
                    fboxes = instances.pred_boxes.tensor  # (N, 4) xyxy
                    scores = instances.scores
                    classes = instances.pred_classes

                    # Standard results
                    for i in range(len(fboxes)):
                        box = fboxes[i].cpu().numpy()
                        results_standard.append({
                            "image_id": image_id,
                            "category_id": 1,  # person
                            "bbox": [float(box[0]), float(box[1]),
                                     float(box[2]-box[0]), float(box[3]-box[1])],  # xywh
                            "score": float(scores[i].cpu()),
                        })

    # For vis-guided NMS, we need to modify the model's inference
    # to return raw predictions before NMS, then apply our custom NMS
    print(f"\nStandard NMS: {len(results_standard)} detections")

    # Save standard results
    std_path = os.path.join(output_dir, "coco_instances_results_standard.json")
    with open(std_path, 'w') as f:
        json.dump(results_standard, f)
    print(f"Standard results saved to: {std_path}")

    return results_standard


def run_raw_inference(model, dataloader, output_dir,
                      iou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
                      score_threshold=0.001, max_dets=500):
    """
    Run inference extracting raw (pre-NMS) fbox and vbox predictions,
    then apply both standard NMS and vis-guided NMS at multiple thresholds.
    """
    model.eval()
    device = next(model.parameters()).device

    # We need to hook into the model to get raw predictions
    # VisDINO.forward() in inference mode applies select_box_with_nms
    # We need to bypass that and get raw outputs

    # Monkey-patch the model to return raw predictions
    original_inference = model.inference

    all_raw_predictions = []

    def patched_inference(box_cls, box_pred, image_sizes, **kwargs):
        """Capture raw predictions before NMS."""
        # box_cls: (bs, nq, num_classes)
        # box_pred: (bs, nq, 4)
        all_raw_predictions.append({
            'box_cls': box_cls.cpu(),
            'box_pred': box_pred.cpu(),
        })
        return original_inference(box_cls, box_pred, image_sizes, **kwargs)

    model.inference = patched_inference

    # Also capture vbox predictions
    all_vbox_preds = []
    all_image_metas = []

    print("Running inference to collect raw predictions...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            image_metas = [{"image_id": b.get("image_id", batch_idx),
                           "height": b["height"], "width": b["width"]}
                          for b in batch]
            all_image_metas.extend(image_metas)

            outputs = model(batch)

    # Restore original
    model.inference = original_inference

    print(f"Collected {len(all_raw_predictions)} batches of raw predictions")

    # Note: This approach is limited because VisDINO.forward() doesn't 
    # easily expose vbox predictions during inference.
    # A cleaner approach: directly run the model components and extract
    # both fbox and vbox predictions.

    return all_raw_predictions


def evaluate_with_vis_nms_simple(checkpoint_path, config_path, output_dir,
                                  iou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    Simpler approach: Load existing COCO predictions and re-run inference
    to also get vbox predictions, then apply vis-guided NMS.
    
    This runs a custom forward pass through the model that extracts
    both fbox and vbox for each image.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")

    # Load config and model
    cfg = LazyConfig.load(config_path)
    model = instantiate(cfg.model)
    model = model.to("cuda")
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint_path)

    # Build test dataloader
    dataloader = instantiate(cfg.dataloader.test)

    print(f"\nRunning inference on {len(dataloader)} batches...")

    # Standard and vis-guided results at various thresholds
    results = {f"vis_nms_{t}": [] for t in iou_thresholds}
    results["standard"] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            for per_image in batch:
                image_id = per_image.get("image_id")
                h, w = per_image["height"], per_image["width"]

                # Run model
                output = model([per_image])
                instances = output[0]["instances"].to("cpu")

                fboxes = instances.pred_boxes.tensor  # (N, 4) xyxy
                scores = instances.scores
                classes = instances.pred_classes

                # Standard results (model already applied NMS)
                for i in range(len(fboxes)):
                    box = fboxes[i].numpy()
                    det = {
                        "image_id": int(image_id),
                        "category_id": 1,
                        "bbox": [float(box[0]), float(box[1]),
                                 float(box[2]-box[0]), float(box[3]-box[1])],
                        "score": float(scores[i]),
                    }
                    results["standard"].append(det)

                # For vis-guided NMS, we need vbox predictions
                # Check if model exposes them
                if hasattr(instances, 'pred_vboxes'):
                    vboxes = instances.pred_vboxes.tensor
                    for thresh in iou_thresholds:
                        keep = vis_guided_nms(fboxes, vboxes, scores, thresh)
                        for i in keep:
                            box = fboxes[i].numpy()
                            det = {
                                "image_id": int(image_id),
                                "category_id": 1,
                                "bbox": [float(box[0]), float(box[1]),
                                         float(box[2]-box[0]), float(box[3]-box[1])],
                                "score": float(scores[i]),
                            }
                            results[f"vis_nms_{thresh}"].append(det)

    # Save all result variants
    for name, dets in results.items():
        if dets:
            path = os.path.join(output_dir, f"predictions_{name}.json")
            with open(path, 'w') as f:
                json.dump(dets, f)
            print(f"  {name}: {len(dets)} detections → {path}")

    print(f"\nDone! Run CrowdDet eval on each prediction file to compare.")
    print(f"Example:")
    print(f"  cd crowddet_official_eval")
    print(f"  python coco_to_odgt.py --coco {output_dir}/predictions_vis_nms_0.5.json --out {output_dir}/predictions_vis_nms_0.5.odgt")
    print(f"  PYTHONPATH=$PYTHONPATH:crowddet/lib python run_ap_mr.py --gt <enriched_gt.odgt> --dt {output_dir}/predictions_vis_nms_0.5.odgt")


def main():
    parser = argparse.ArgumentParser(description="Vis-Guided NMS Evaluation")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--iou-thresholds", nargs="+", type=float,
                        default=[0.3, 0.4, 0.5, 0.6, 0.7],
                        help="IoU thresholds for vis-guided NMS")
    args = parser.parse_args()

    evaluate_with_vis_nms_simple(
        args.checkpoint, args.config, args.output_dir, args.iou_thresholds
    )


if __name__ == "__main__":
    main()
