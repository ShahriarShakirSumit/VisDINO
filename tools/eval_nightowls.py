#!/usr/bin/env python3
"""
NightOwls Cross-Dataset Evaluation Script
==========================================
Evaluates CrowdHuman-trained DINO/VisDINO models on NightOwls validation set
using the official NightOwls MR-2 evaluation protocol.

Evaluation subsets (from NightOwls official eval):
  - Reasonable:        height >= 50, vis_ratio >= 0.65
  - Reasonable_small:  height in [50, 75], vis_ratio >= 0.65
  - Reasonable_occ=heavy: height >= 50, vis_ratio in [0.2, 0.65]
  - All:               height >= 20, vis_ratio >= 0.2
  - Avg:               average of all 4 subsets

Usage:
  cd <detrex root>
  python projects/visdino/eval_nightowls.py \
      --model-type visdino \
      --config-file projects/visdino/configs/visdino_r50_4scale_crowdhuman_300k.py \
      --checkpoint output/visdino_r50_crowdhuman_300k/model_final.pth \
      --output-dir output/nightowls_eval/visdino_300k

  # Or for baseline DINO:
  python projects/visdino/eval_nightowls.py \
      --model-type dino \
      --config-file projects/dino_crowdhuman/configs/dino_r50_4scale_crowdhuman.py \
      --checkpoint output/dino_r50_crowdhuman_baseline/model_final.pth \
      --output-dir output/nightowls_eval/dino_baseline
"""

import argparse
import json
import os
import sys
import time
import copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from detectron2.config import LazyConfig, instantiate

# ──────────────────── paths ────────────────────
DETREX_ROOT = Path(__file__).resolve().parents[2]  # detrex/detrex/
DETREX_WORKSPACE = DETREX_ROOT.parent               # detrex/
NIGHTOWLS_DIR = DETREX_WORKSPACE / "datasets" / "nightowls"
NIGHTOWLS_EVAL_DIR = NIGHTOWLS_DIR / "eval_tools" / "python"
NIGHTOWLS_VAL_JSON = NIGHTOWLS_DIR / "annotations" / "nightowls_validation.json"
NIGHTOWLS_VAL_JSON_D2 = NIGHTOWLS_DIR / "annotations" / "nightowls_validation_d2.json"
NIGHTOWLS_VAL_IMGS = NIGHTOWLS_DIR / "images_val"

sys.path.insert(0, str(DETREX_ROOT))
sys.path.insert(0, str(NIGHTOWLS_EVAL_DIR))


# ─────────────── Dataset Registration ───────────────
def register_nightowls():
    """Register NightOwls validation set as a detectron2 dataset.
    
    NightOwls has 4 categories in JSON: pedestrian(1), bicycledriver(2), 
    motorbikedriver(3), ignore(4). Our models output a single 'person' class.
    We register all 4 NightOwls categories but map only pedestrian(1) → class 0
    for the model. The predictions are all emitted as category_id=1 (pedestrian).
    """
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog, DatasetCatalog

    dataset_name = "nightowls_val"
    if dataset_name in DatasetCatalog:
        return dataset_name

    # Use cleaned JSON (no 'ignore' field, pedestrian-only) for detectron2.
    # The original JSON with all categories is used for official MR-2 eval.
    register_coco_instances(
        dataset_name,
        {},
        str(NIGHTOWLS_VAL_JSON_D2),
        str(NIGHTOWLS_VAL_IMGS),
    )
    # Map NightOwls pedestrian (category_id=1) to model class 0
    meta = MetadataCatalog.get(dataset_name)
    meta.set(thing_dataset_id_to_contiguous_id={1: 0})
    return dataset_name


def register_nightowls_train():
    """Register NightOwls training set."""
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog, DatasetCatalog

    dataset_name = "nightowls_train"
    if dataset_name in DatasetCatalog:
        return dataset_name

    train_json = NIGHTOWLS_DIR / "annotations" / "nightowls_training.json"
    train_imgs = NIGHTOWLS_DIR / "images_train"

    register_coco_instances(
        dataset_name,
        {"thing_classes": ["pedestrian"]},
        str(train_json),
        str(train_imgs),
    )
    meta = MetadataCatalog.get(dataset_name)
    meta.set(thing_dataset_id_to_contiguous_id={1: 0})
    return dataset_name


# ─────────────── Model Loading ───────────────
def load_model(config_file, checkpoint, model_type="visdino"):
    """Load a trained DINO/VisDINO model from config + checkpoint."""
    from detectron2.checkpoint import DetectionCheckpointer

    cfg = LazyConfig.load(config_file)

    # Override test dataset to NightOwls
    dataset_name = register_nightowls()
    cfg.dataloader.test.dataset.names = dataset_name

    # Build model
    model = instantiate(cfg.model)
    model.to("cuda")
    model.eval()

    # Load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint)

    return model, cfg


# ─────────────── Inference ───────────────
def run_inference(model, cfg, output_dir, max_images=None):
    """Run inference on NightOwls val set, save COCO-format predictions."""
    from detectron2.data import build_detection_test_loader

    os.makedirs(output_dir, exist_ok=True)

    # Build test loader for NightOwls using lazy config
    test_dl_cfg = cfg.dataloader.test
    dataset = instantiate(test_dl_cfg.dataset)
    mapper = instantiate(test_dl_cfg.mapper)
    num_workers = test_dl_cfg.get("num_workers", 4)
    data_loader = build_detection_test_loader(
        dataset=dataset,
        mapper=mapper,
        num_workers=num_workers,
    )

    predictions = []  # COCO format for NightOwls eval
    coco_predictions = []  # detectron2 COCO format

    model.eval()
    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(data_loader, desc="NightOwls inference")):
            if max_images is not None and idx >= max_images:
                break

            outputs = model(inputs)

            for inp, out in zip(inputs, outputs):
                instances = out["instances"].to("cpu")
                img_id = inp["image_id"]

                scores = instances.scores.numpy()
                classes = instances.pred_classes.numpy()
                boxes = instances.pred_boxes.tensor.numpy()  # xyxy

                for j in range(len(scores)):
                    x1, y1, x2, y2 = boxes[j]
                    w, h = x2 - x1, y2 - y1

                    # NightOwls format (category_id=1 for pedestrian)
                    predictions.append({
                        "image_id": int(img_id),
                        "category_id": 1,  # pedestrian
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(scores[j]),
                    })

                    # detectron2 COCO format (internal class ids)
                    coco_predictions.append({
                        "image_id": int(img_id),
                        "category_id": int(classes[j]),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(scores[j]),
                    })

    # Save predictions
    pred_path = os.path.join(output_dir, "nightowls_predictions.json")
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)
    print(f"Saved {len(predictions)} predictions to {pred_path}")

    coco_pred_path = os.path.join(output_dir, "coco_predictions.json")
    with open(coco_pred_path, 'w') as f:
        json.dump(coco_predictions, f)

    return pred_path


# ─────────────── NightOwls Official Evaluation ───────────────
def run_nightowls_official_eval(pred_json, output_dir):
    """
    Run official NightOwls MR-2 evaluation using their eval tools.

    4 setups:
      0 - Reasonable:           height [50, inf], vis [0.65, inf]
      1 - Reasonable_small:     height [50, 75],  vis [0.65, inf]
      2 - Reasonable_occ=heavy: height [50, inf], vis [0.2, 0.65]
      3 - All:                  height [20, inf], vis [0.2, inf]

    Reports MR-2 (log-average miss rate on FPPI [0.01, 1.0]) for each setup.
    """
    # Import NightOwls eval tools
    from coco import COCO as NightOwlsCOCO
    from eval_MR_multisetup import COCOeval as NightOwlsCOCOeval

    ann_file = str(NIGHTOWLS_VAL_JSON)
    setup_labels = ['Reasonable', 'Reasonable_small', 'Reasonable_occ=heavy', 'All']

    results = {}
    results_file = os.path.join(output_dir, "nightowls_mr_results.txt")
    res_file = open(results_file, "w")

    for id_setup in range(4):
        cocoGt = NightOwlsCOCO(ann_file)
        cocoDt = cocoGt.loadRes(pred_json)
        imgIds = sorted(cocoGt.getImgIds())

        cocoEval = NightOwlsCOCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        mr = cocoEval.summarize(id_setup, res_file)

        if mr is not None and mr >= 0:
            results[setup_labels[id_setup]] = float(mr * 100)
        else:
            results[setup_labels[id_setup]] = -1.0

    # Compute average
    valid_mrs = [v for v in results.values() if v >= 0]
    results['Avg'] = float(np.mean(valid_mrs)) if valid_mrs else -1.0

    res_file.write(f"\n{'='*60}\n")
    res_file.write(f"Average MR-2: {results['Avg']:.2f}%\n")
    res_file.close()

    # Save JSON results
    results_json_path = os.path.join(output_dir, "nightowls_mr_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("NightOwls MR-2 Results:")
    print(f"{'='*60}")
    for label, mr in results.items():
        print(f"  {label:25s}: {mr:.2f}%")
    print(f"{'='*60}")
    print(f"Results saved to {results_file}")

    return results


# ─────────────── COCO-style AP Evaluation ───────────────
def run_coco_ap_eval(cfg, model, output_dir):
    """Run standard COCO AP evaluation on NightOwls (for AP, AP50, AP75, etc.)."""
    from detectron2.data import build_detection_test_loader
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset

    dataset_name = "nightowls_val"
    os.makedirs(output_dir, exist_ok=True)

    evaluator = COCOEvaluator(
        dataset_name,
        output_dir=output_dir,
        tasks=("bbox",),
    )

    data_loader = build_detection_test_loader(
        dataset=instantiate(cfg.dataloader.test.dataset),
        mapper=instantiate(cfg.dataloader.test.mapper),
        num_workers=cfg.dataloader.test.get("num_workers", 4),
    )

    results = inference_on_dataset(model, data_loader, evaluator)
    return results


# ─────────────── Custom Per-Occlusion Analysis ───────────────
def per_occlusion_analysis(pred_json, output_dir, score_thr=0.3, iou_thr=0.5):
    """
    Analyze detection performance by occlusion level on NightOwls.

    NightOwls uses binary 'occluded' field:
      - occluded=False → vis_ratio=1.0 (None occlusion)
      - occluded=True  → vis_ratio=0.5 (Partial/Heavy occlusion)

    We also break down by height for a finer analysis.
    """
    # Load GT
    with open(str(NIGHTOWLS_VAL_JSON)) as f:
        gt_data = json.load(f)

    # Build GT index (pedestrians only)
    gt_by_image = defaultdict(list)
    occ_stats = {"not_occluded": 0, "occluded": 0, "ignore": 0}
    height_bins = {"small": 0, "medium": 0, "large": 0}

    for ann in gt_data['annotations']:
        if ann['category_id'] == 4:  # ignore region
            occ_stats["ignore"] += 1
            continue
        if ann['category_id'] != 1:  # not pedestrian
            continue

        x, y, w, h = ann['bbox']
        occluded = ann.get('occluded', False)
        vis_ratio = 0.5 if occluded else 1.0

        gt_entry = {
            'bbox': [x, y, x + w, y + h],  # xyxy
            'occluded': occluded,
            'vis_ratio': vis_ratio,
            'height': h,
            'ignore': ann.get('ignore', 0),
        }
        gt_by_image[ann['image_id']].append(gt_entry)

        if not ann.get('ignore', 0):
            if occluded:
                occ_stats["occluded"] += 1
            else:
                occ_stats["not_occluded"] += 1

            if h < 50:
                height_bins["small"] += 1
            elif h < 75:
                height_bins["medium"] += 1
            else:
                height_bins["large"] += 1

    # Load predictions
    with open(pred_json) as f:
        preds = json.load(f)

    pred_by_image = defaultdict(list)
    for p in preds:
        if p['score'] >= score_thr:
            x, y, w, h = p['bbox']
            pred_by_image[p['image_id']].append({
                'bbox': [x, y, x + w, y + h],
                'score': p['score'],
            })

    # Compute recall by occlusion level
    def compute_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        aa = (ax2 - ax1) * (ay2 - ay1)
        ab = (bx2 - bx1) * (by2 - by1)
        return inter / (aa + ab - inter + 1e-12)

    # Per-occlusion recall
    recall_stats = {
        "Not Occluded": {"matched": 0, "total": 0},
        "Occluded": {"matched": 0, "total": 0},
    }

    # Per-height recall
    height_recall = {
        "Small (<50px)": {"matched": 0, "total": 0},
        "Medium (50-75px)": {"matched": 0, "total": 0},
        "Large (>75px)": {"matched": 0, "total": 0},
    }

    for img_id, gts in gt_by_image.items():
        dets = sorted(pred_by_image.get(img_id, []), key=lambda d: -d['score'])
        used = [False] * len(gts)

        for det in dets:
            best_iou, best_j = 0, -1
            for j, gt in enumerate(gts):
                if used[j] or gt['ignore']:
                    continue
                v = compute_iou(det['bbox'], gt['bbox'])
                if v >= iou_thr and v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                used[best_j] = True

        # Count per-category recall
        for j, gt in enumerate(gts):
            if gt['ignore']:
                continue

            occ_key = "Occluded" if gt['occluded'] else "Not Occluded"
            recall_stats[occ_key]["total"] += 1
            if used[j]:
                recall_stats[occ_key]["matched"] += 1

            h = gt['height']
            if h < 50:
                hk = "Small (<50px)"
            elif h < 75:
                hk = "Medium (50-75px)"
            else:
                hk = "Large (>75px)"
            height_recall[hk]["total"] += 1
            if used[j]:
                height_recall[hk]["matched"] += 1

    # Print results
    print(f"\n{'='*60}")
    print(f"NightOwls Per-Occlusion Recall @ IoU={iou_thr}, score_thr={score_thr}")
    print(f"{'='*60}")
    print(f"GT distribution: {occ_stats}")
    print(f"Height distribution: {height_bins}")
    print()

    total_matched = 0
    total_gt = 0
    for occ_label, stats in recall_stats.items():
        r = stats['matched'] / stats['total'] * 100 if stats['total'] > 0 else 0
        total_matched += stats['matched']
        total_gt += stats['total']
        print(f"  {occ_label:20s}: {stats['matched']:5d}/{stats['total']:5d} = {r:.2f}%")

    overall_r = total_matched / total_gt * 100 if total_gt > 0 else 0
    print(f"  {'Overall':20s}: {total_matched:5d}/{total_gt:5d} = {overall_r:.2f}%")

    print(f"\nPer-Height Recall:")
    for hk, stats in height_recall.items():
        r = stats['matched'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {hk:20s}: {stats['matched']:5d}/{stats['total']:5d} = {r:.2f}%")

    # Save
    analysis = {
        "score_threshold": score_thr,
        "iou_threshold": iou_thr,
        "gt_distribution": occ_stats,
        "height_distribution": height_bins,
        "per_occlusion_recall": {
            k: {
                "matched": v["matched"],
                "total": v["total"],
                "recall": v["matched"] / v["total"] * 100 if v["total"] > 0 else 0,
            }
            for k, v in recall_stats.items()
        },
        "per_height_recall": {
            k: {
                "matched": v["matched"],
                "total": v["total"],
                "recall": v["matched"] / v["total"] * 100 if v["total"] > 0 else 0,
            }
            for k, v in height_recall.items()
        },
        "overall_recall": overall_r,
    }

    analysis_path = os.path.join(output_dir, "per_occlusion_analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved to {analysis_path}")

    return analysis


# ─────────────── Summary Writer ───────────────
def write_summary(output_dir, model_name, mr_results, occ_analysis=None, coco_results=None):
    """Write a comprehensive evaluation summary."""
    summary_path = os.path.join(output_dir, "eval_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"NightOwls Evaluation Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"NightOwls MR-2 Results (↓ lower is better):\n")
        f.write(f"{'-'*40}\n")
        for label, mr in mr_results.items():
            f.write(f"  {label:25s}: {mr:.2f}%\n")

        if coco_results and 'bbox' in coco_results:
            f.write(f"\nCOCO-style AP Results:\n")
            f.write(f"{'-'*40}\n")
            for k, v in coco_results['bbox'].items():
                f.write(f"  {k:10s}: {v:.4f}\n")

        if occ_analysis:
            f.write(f"\nPer-Occlusion Recall @ IoU={occ_analysis['iou_threshold']}, "
                    f"score_thr={occ_analysis['score_threshold']}:\n")
            f.write(f"{'-'*40}\n")
            for label, stats in occ_analysis['per_occlusion_recall'].items():
                f.write(f"  {label:20s}: {stats['recall']:.2f}% "
                        f"({stats['matched']}/{stats['total']})\n")
            f.write(f"  {'Overall':20s}: {occ_analysis['overall_recall']:.2f}%\n")

            f.write(f"\nPer-Height Recall:\n")
            for label, stats in occ_analysis['per_height_recall'].items():
                f.write(f"  {label:20s}: {stats['recall']:.2f}% "
                        f"({stats['matched']}/{stats['total']})\n")

    print(f"\nSummary saved to {summary_path}")
    return summary_path


# ─────────────── Main ───────────────
def main():
    parser = argparse.ArgumentParser(description="NightOwls Cross-Dataset Evaluation")
    parser.add_argument("--model-type", choices=["dino", "visdino"], required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit inference to N images (for testing)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference, use existing predictions")
    parser.add_argument("--skip-coco-ap", action="store_true",
                        help="Skip COCO AP eval (use for cross-dataset where class mapping may differ)")
    args = parser.parse_args()

    print(f"NightOwls Evaluation Pipeline")
    print(f"{'='*60}")
    print(f"Model type:  {args.model_type}")
    print(f"Config:      {args.config_file}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Output:      {args.output_dir}")
    print()

    # Verify NightOwls data exists
    if not NIGHTOWLS_VAL_JSON.exists():
        raise FileNotFoundError(f"NightOwls val annotations not found: {NIGHTOWLS_VAL_JSON}")
    if not NIGHTOWLS_VAL_IMGS.exists():
        raise FileNotFoundError(f"NightOwls val images not found: {NIGHTOWLS_VAL_IMGS}")

    # Check image count
    img_count = len(list(NIGHTOWLS_VAL_IMGS.glob("*.png")))
    print(f"NightOwls val images found: {img_count}")

    os.makedirs(args.output_dir, exist_ok=True)

    pred_json = os.path.join(args.output_dir, "nightowls_predictions.json")

    if not args.skip_inference:
        # Step 1: Load model
        print("\n[Step 1/4] Loading model...")
        model, cfg = load_model(args.config_file, args.checkpoint, args.model_type)

        # Step 2: Run inference
        print("\n[Step 2/4] Running inference on NightOwls val...")
        pred_json = run_inference(model, cfg, args.output_dir, args.max_images)
    else:
        print("\n[Step 1-2] Skipping inference, using existing predictions")
        if not os.path.exists(pred_json):
            raise FileNotFoundError(f"Predictions not found: {pred_json}")

    # Step 3: Official NightOwls MR-2 evaluation
    print("\n[Step 3/4] Running NightOwls official MR-2 evaluation...")
    mr_results = run_nightowls_official_eval(pred_json, args.output_dir)

    # Step 4: Per-occlusion analysis
    print("\n[Step 4/4] Running per-occlusion analysis...")
    occ_analysis = per_occlusion_analysis(pred_json, args.output_dir)

    # Write summary
    model_name = f"{args.model_type.upper()} — {os.path.basename(args.checkpoint)}"
    write_summary(args.output_dir, model_name, mr_results, occ_analysis)

    print(f"\n{'='*60}")
    print(f"ALL DONE! Results in {args.output_dir}")
    print(f"{'='*60}")

    return mr_results, occ_analysis


if __name__ == "__main__":
    main()
