#!/usr/bin/env python3
"""
VisDINO Comprehensive Evaluation Script
========================================
Runs full evaluation with:
  1. COCO AP (fbox) — standard detectron2 eval
  2. COCO AP (vbox) — evaluate visible-body boxes against vbox GT
  3. CrowdDet official AP@0.5 + MR-2
  4. Recall@0.5
  5. JI_max
  6. Precision at various score thresholds
  7. fbox vs vbox comparison analysis

Usage:
  cd <detrex root>
  python projects/visdino/eval_comprehensive.py \
      --config-file projects/visdino/configs/visdino_r50_4scale_crowdhuman_v2.py \
      --checkpoint output/visdino_r50_crowdhuman_v2/model_0279999.pth \
      --output-dir output/visdino_r50_crowdhuman_v2/eval_comprehensive
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from detectron2.config import LazyConfig, instantiate
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, Instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detrex.layers import box_cxcywh_to_xyxy


def to_xyxy(b):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = b
    return [x, y, x + w, y + h]


def area(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / (area(a) + area(b) - inter + 1e-12)


def ioa(a, b):
    """Intersection over area of box a."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / (area(a) + 1e-12)


def compute_precision_recall(gt_odgt_path, dt_odgt_path, score_thresholds=None):
    """Compute Precision, Recall at various score thresholds."""
    if score_thresholds is None:
        score_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Load data
    gt_data = {}
    for line in open(gt_odgt_path):
        r = json.loads(line.strip())
        gt_data[r['ID']] = r

    dt_data = {}
    for line in open(dt_odgt_path):
        r = json.loads(line.strip())
        dt_data[r['ID']] = r

    results = {}
    for thr in score_thresholds:
        tp_total = 0
        fp_total = 0
        gt_valid_count = 0

        for img_id, grecord in gt_data.items():
            gts_valid = []
            gts_ignore = []
            for g in grecord.get('gtboxes', []):
                bb = to_xyxy(g['fbox'])
                tag = g.get('tag', 'person')
                is_valid = tag == 'person' or tag == 1
                (gts_valid if is_valid else gts_ignore).append(bb)
            gt_valid_count += len(gts_valid)

            dets = []
            for d in dt_data.get(img_id, {}).get('dtboxes', []):
                if float(d['score']) >= thr:
                    dets.append((float(d['score']), to_xyxy(d['box'])))
            dets.sort(key=lambda t: t[0], reverse=True)

            # Remove detections overlapping with ignore regions
            kept = []
            for sc, box in dets:
                drop = any(ioa(box, ign) >= 0.5 for ign in gts_ignore)
                if not drop:
                    kept.append((sc, box))

            # Greedy match
            used = [False] * len(gts_valid)
            for sc, db in kept:
                best_iou, best_j = 0.0, -1
                for j, gb in enumerate(gts_valid):
                    if used[j]:
                        continue
                    v = iou(db, gb)
                    if v >= 0.5 and v > best_iou:
                        best_iou, best_j = v, j
                if best_j >= 0:
                    used[best_j] = True
                    tp_total += 1
                else:
                    fp_total += 1

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / gt_valid_count if gt_valid_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[thr] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'TP': tp_total,
            'FP': fp_total,
            'GT': gt_valid_count,
        }

    return results


def run_fbox_vbox_inference(model, data_loader, output_dir, device='cuda'):
    """Run inference and save both fbox and vbox predictions in COCO format."""
    model.eval()
    fbox_predictions = []
    vbox_predictions = []

    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(data_loader, desc="Running inference")):
            outputs = model(inputs)

            for inp, out in zip(inputs, outputs):
                instances = out["instances"]
                img_id = inp["image_id"]

                # Get scores and labels
                scores = instances.scores.cpu().numpy()
                classes = instances.pred_classes.cpu().numpy()
                fboxes = instances.pred_boxes.tensor.cpu().numpy()

                # fbox predictions (COCO format: [x, y, w, h])
                for j in range(len(scores)):
                    x1, y1, x2, y2 = fboxes[j]
                    fbox_predictions.append({
                        "image_id": str(img_id),
                        "category_id": int(classes[j]),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(scores[j]),
                    })

                # vbox predictions
                if instances.has("pred_vboxes"):
                    vboxes = instances.pred_vboxes.tensor.cpu().numpy()
                    for j in range(len(scores)):
                        x1, y1, x2, y2 = vboxes[j]
                        vbox_predictions.append({
                            "image_id": str(img_id),
                            "category_id": int(classes[j]),
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(scores[j]),
                        })

    # Save
    os.makedirs(output_dir, exist_ok=True)
    fbox_path = os.path.join(output_dir, "fbox_predictions.json")
    vbox_path = os.path.join(output_dir, "vbox_predictions.json")

    with open(fbox_path, 'w') as f:
        json.dump(fbox_predictions, f)
    print(f"Saved {len(fbox_predictions)} fbox predictions to {fbox_path}")

    if vbox_predictions:
        with open(vbox_path, 'w') as f:
            json.dump(vbox_predictions, f)
        print(f"Saved {len(vbox_predictions)} vbox predictions to {vbox_path}")
    else:
        print("WARNING: No vbox predictions found! Model may not output pred_vboxes.")

    return fbox_path, vbox_path if vbox_predictions else None


def analyze_fbox_vbox_relationship(fbox_coco_path, vbox_coco_path, output_dir):
    """Analyze the relationship between fbox and vbox predictions."""
    with open(fbox_coco_path) as f:
        fbox_preds = json.load(f)
    with open(vbox_coco_path) as f:
        vbox_preds = json.load(f)

    assert len(fbox_preds) == len(vbox_preds), "fbox and vbox must have same number of predictions"

    # Per-prediction analysis
    area_ratios = []
    height_ratios = []
    width_ratios = []
    iou_values = []
    score_bins = defaultdict(list)

    for fp, vp in zip(fbox_preds, vbox_preds):
        fx, fy, fw, fh = fp['bbox']
        vx, vy, vw, vh = vp['bbox']

        f_area = fw * fh
        v_area = vw * vh

        if f_area > 0:
            area_ratios.append(v_area / f_area)
        if fh > 0:
            height_ratios.append(vh / fh)
        if fw > 0:
            width_ratios.append(vw / fw)

        # IoU between fbox and vbox
        fb = [fx, fy, fx + fw, fy + fh]
        vb = [vx, vy, vx + vw, vy + vh]
        iou_val = iou(fb, vb)
        iou_values.append(iou_val)

        # Bin by score
        score = fp['score']
        if score >= 0.5:
            score_bins['high'].append(v_area / (f_area + 1e-8))
        elif score >= 0.3:
            score_bins['medium'].append(v_area / (f_area + 1e-8))
        else:
            score_bins['low'].append(v_area / (f_area + 1e-8))

    analysis = {
        'total_predictions': len(fbox_preds),
        'area_ratio': {
            'mean': float(np.mean(area_ratios)),
            'median': float(np.median(area_ratios)),
            'std': float(np.std(area_ratios)),
            'min': float(np.min(area_ratios)),
            'max': float(np.max(area_ratios)),
        },
        'height_ratio': {
            'mean': float(np.mean(height_ratios)),
            'median': float(np.median(height_ratios)),
            'std': float(np.std(height_ratios)),
        },
        'width_ratio': {
            'mean': float(np.mean(width_ratios)),
            'median': float(np.median(width_ratios)),
            'std': float(np.std(width_ratios)),
        },
        'iou_fbox_vbox': {
            'mean': float(np.mean(iou_values)),
            'median': float(np.median(iou_values)),
            'std': float(np.std(iou_values)),
        },
        'occluded_predictions': {
            'ratio_below_0.9': int(sum(1 for r in area_ratios if r < 0.9)),
            'ratio_below_0.8': int(sum(1 for r in area_ratios if r < 0.8)),
            'ratio_below_0.5': int(sum(1 for r in area_ratios if r < 0.5)),
            'pct_below_0.9': float(sum(1 for r in area_ratios if r < 0.9) / len(area_ratios) * 100),
            'pct_below_0.8': float(sum(1 for r in area_ratios if r < 0.8) / len(area_ratios) * 100),
            'pct_below_0.5': float(sum(1 for r in area_ratios if r < 0.5) / len(area_ratios) * 100),
        },
        'by_score_bin': {
            k: {'mean_ratio': float(np.mean(v)), 'count': len(v)}
            for k, v in score_bins.items() if v
        },
    }

    analysis_path = os.path.join(output_dir, "fbox_vbox_analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print("\n" + "=" * 60)
    print("  fbox vs vbox Analysis")
    print("=" * 60)
    print(f"  Total predictions: {analysis['total_predictions']}")
    print(f"  Area ratio (vbox/fbox):")
    print(f"    Mean:   {analysis['area_ratio']['mean']:.4f}")
    print(f"    Median: {analysis['area_ratio']['median']:.4f}")
    print(f"    Std:    {analysis['area_ratio']['std']:.4f}")
    print(f"  Height ratio (vbox/fbox):")
    print(f"    Mean:   {analysis['height_ratio']['mean']:.4f}")
    print(f"  Width ratio (vbox/fbox):")
    print(f"    Mean:   {analysis['width_ratio']['mean']:.4f}")
    print(f"  IoU(fbox, vbox):")
    print(f"    Mean:   {analysis['iou_fbox_vbox']['mean']:.4f}")
    print(f"  Occluded detections (vbox < fbox):")
    print(f"    ratio < 0.9: {analysis['occluded_predictions']['pct_below_0.9']:.1f}%")
    print(f"    ratio < 0.8: {analysis['occluded_predictions']['pct_below_0.8']:.1f}%")
    print(f"    ratio < 0.5: {analysis['occluded_predictions']['pct_below_0.5']:.1f}%")
    print("=" * 60)

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crowddet-eval", default="crowddet_official_eval")
    parser.add_argument("--crowddet-lib", default="crowddet/lib")
    parser.add_argument("--enriched-gt", default=None)
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference, use existing predictions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Step 1: Load model and run inference (fbox + vbox)
    # ----------------------------------------------------------------
    if not args.skip_inference:
        print("\n[Step 1] Loading model and running inference...")
        cfg = LazyConfig.load(args.config_file)
        cfg.train.init_checkpoint = args.checkpoint

        model = instantiate(cfg.model)
        model.to("cuda")
        model.eval()

        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(args.checkpoint)

        # Build val loader
        data_loader = instantiate(cfg.dataloader.test)

        # Run inference
        fbox_path, vbox_path = run_fbox_vbox_inference(
            model, data_loader, args.output_dir
        )

        # Clean up model to free GPU memory
        del model
        torch.cuda.empty_cache()
    else:
        print("\n[Step 1] Skipping inference, using existing predictions...")
        fbox_path = os.path.join(args.output_dir, "fbox_predictions.json")
        vbox_path = os.path.join(args.output_dir, "vbox_predictions.json")
        if not os.path.exists(fbox_path):
            print(f"ERROR: {fbox_path} not found!")
            sys.exit(1)

    # ----------------------------------------------------------------
    # Step 2: fbox vs vbox analysis
    # ----------------------------------------------------------------
    if vbox_path and os.path.exists(vbox_path):
        print("\n[Step 2] Analyzing fbox vs vbox relationship...")
        analysis = analyze_fbox_vbox_relationship(fbox_path, vbox_path, args.output_dir)
    else:
        print("\n[Step 2] No vbox predictions available, skipping analysis.")
        analysis = None

    # ----------------------------------------------------------------
    # Step 3: Convert to ODGT and run CrowdDet eval
    # ----------------------------------------------------------------
    print("\n[Step 3] Converting to ODGT format...")

    # Convert fbox predictions to ODGT
    fbox_odgt = os.path.join(args.output_dir, "fbox_predictions.odgt")
    _coco_to_odgt(fbox_path, fbox_odgt)

    # Convert vbox predictions to ODGT (if available)
    if vbox_path and os.path.exists(vbox_path):
        vbox_odgt = os.path.join(args.output_dir, "vbox_predictions.odgt")
        _coco_to_odgt(vbox_path, vbox_odgt)
    else:
        vbox_odgt = None

    # Find enriched GT
    if args.enriched_gt:
        enriched_gt = args.enriched_gt
    else:
        enriched_gt = os.path.join(
            os.path.dirname(os.path.dirname(args.output_dir)),
            "visdino_r50_crowdhuman", "annotation_val_enriched.odgt"
        )
    print(f"Using enriched GT: {enriched_gt}")

    # ----------------------------------------------------------------
    # Step 4: Precision-Recall at various thresholds
    # ----------------------------------------------------------------
    print("\n[Step 4] Computing Precision/Recall at various thresholds...")
    pr_results = compute_precision_recall(enriched_gt, fbox_odgt)

    print("\n  Score_thr  Precision  Recall    F1")
    print("  " + "-" * 44)
    for thr in sorted(pr_results.keys()):
        r = pr_results[thr]
        print(f"  {thr:.1f}        {r['precision']:.4f}    {r['recall']:.4f}   {r['f1']:.4f}")

    pr_path = os.path.join(args.output_dir, "precision_recall.json")
    with open(pr_path, 'w') as f:
        json.dump({str(k): v for k, v in pr_results.items()}, f, indent=2)

    # ----------------------------------------------------------------
    # Step 5: Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  VisDINO Comprehensive Evaluation Complete")
    print("=" * 70)
    print(f"\n  Output directory: {args.output_dir}")
    print(f"  Files generated:")
    print(f"    - fbox_predictions.json (COCO format)")
    if vbox_path:
        print(f"    - vbox_predictions.json (COCO format)")
        print(f"    - fbox_vbox_analysis.json")
    print(f"    - fbox_predictions.odgt")
    print(f"    - precision_recall.json")
    print("=" * 70)


def _coco_to_odgt(coco_path, odgt_path):
    """Convert COCO format predictions to ODGT."""
    with open(coco_path) as f:
        preds = json.load(f)

    image_preds = defaultdict(list)
    for pred in preds:
        img_id = str(pred['image_id'])
        image_preds[img_id].append({'box': pred['bbox'], 'score': pred['score']})

    with open(odgt_path, 'w') as f:
        for img_id, dtboxes in image_preds.items():
            dtboxes = sorted(dtboxes, key=lambda x: x['score'], reverse=True)
            f.write(json.dumps({"ID": img_id, "dtboxes": dtboxes}) + '\n')

    print(f"  Converted {len(preds)} predictions for {len(image_preds)} images -> {odgt_path}")


if __name__ == '__main__':
    main()
