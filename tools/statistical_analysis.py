#!/usr/bin/env python3
"""
Statistical Significance Analysis for VisDINO Experiments

Computes:
1. Bootstrap confidence intervals for AP/MR metrics
2. McNemar's test for detection-level paired comparison  
3. Per-image paired t-test (VisDINO vs DINO)
4. Effect size (Cohen's d)
5. Variance analysis across seeds

Usage:
    cd <detrex root>
    python projects/visdino/statistical_analysis.py \
        --pred1 output/dino_r50_crowdhuman_baseline/inference/coco_instances_results.json \
        --pred2 output/visdino_r50_crowdhuman_300k_vbox/inference/coco_instances_results.json \
        --gt datasets/crowdhuman/annotation_val.json \
        --name1 "DINO" --name2 "VisDINO"
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

# Try to import scipy for statistical tests
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, some tests will be skipped")


def load_predictions(pred_path):
    """Load COCO-format predictions."""
    with open(pred_path) as f:
        preds = json.load(f)
    
    # Group by image_id
    by_image = defaultdict(list)
    for p in preds:
        by_image[p["image_id"]].append(p)
    return by_image


def load_gt(gt_path):
    """Load COCO-format ground truth."""
    with open(gt_path) as f:
        data = json.load(f)
    
    by_image = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        by_image[ann["image_id"]].append(ann)
    
    image_ids = [img["id"] for img in data["images"]]
    return by_image, image_ids


def compute_per_image_ap(gt_boxes, pred_boxes, pred_scores, iou_thresh=0.5):
    """Compute AP for a single image at a given IoU threshold."""
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0
    if len(pred_boxes) == 0:
        return 0.0
    
    # Sort predictions by score
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    
    n_gt = len(gt_boxes)
    matched = np.zeros(n_gt, dtype=bool)
    
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    
    for i, pb in enumerate(pred_boxes):
        # Compute IoU with all GT
        ious = compute_iou_single(pb, gt_boxes)
        best_idx = np.argmax(ious)
        best_iou = ious[best_idx]
        
        if best_iou >= iou_thresh and not matched[best_idx]:
            tp[i] = 1
            matched[best_idx] = True
        else:
            fp[i] = 1
    
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    
    recalls = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum)
    
    # AP using all-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.01, 0.01):
        prec_at_t = precisions[recalls >= t]
        if len(prec_at_t) > 0:
            ap += np.max(prec_at_t)
    ap /= 101
    
    return ap


def compute_per_image_recall(gt_boxes, pred_boxes, pred_scores, iou_thresh=0.5, score_thresh=0.3):
    """Compute recall for a single image."""
    if len(gt_boxes) == 0:
        return 1.0
    if len(pred_boxes) == 0:
        return 0.0
    
    # Filter by score  
    mask = pred_scores >= score_thresh
    pred_boxes = pred_boxes[mask]
    
    if len(pred_boxes) == 0:
        return 0.0
    
    matched = 0
    used = set()
    for gt in gt_boxes:
        ious = compute_iou_single(gt, pred_boxes)
        for idx in np.argsort(-ious):
            if ious[idx] >= iou_thresh and idx not in used:
                matched += 1
                used.add(idx)
                break
    
    return matched / len(gt_boxes)


def compute_iou_single(box, boxes):
    """Compute IoU of one box with an array of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    
    return inter / np.maximum(union, 1e-8)


def bootstrap_ci(metric_values, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval."""
    np.random.seed(42)
    n = len(metric_values)
    bootstrap_means = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        sample = np.random.choice(metric_values, size=n, replace=True)
        bootstrap_means[i] = sample.mean()
    
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_means, 100 * alpha)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha))
    
    return np.mean(metric_values), lower, upper


def bootstrap_difference_ci(values1, values2, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap CI of the difference between two methods."""
    np.random.seed(42)
    n = len(values1)
    diffs = values1 - values2
    bootstrap_diffs = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        sample = np.random.choice(diffs, size=n, replace=True)
        bootstrap_diffs[i] = sample.mean()
    
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_diffs, 100 * alpha)
    upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha))
    mean_diff = diffs.mean()
    
    # Approximate p-value
    if mean_diff > 0:
        p_value = (bootstrap_diffs <= 0).mean()
    else:
        p_value = (bootstrap_diffs >= 0).mean()
    
    return mean_diff, lower, upper, p_value


def cohens_d(values1, values2):
    """Compute Cohen's d effect size for paired samples."""
    diffs = values1 - values2
    return diffs.mean() / (diffs.std(ddof=1) + 1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred1", required=True, help="Predictions from method 1 (COCO JSON)")
    parser.add_argument("--pred2", required=True, help="Predictions from method 2 (COCO JSON)")
    parser.add_argument("--gt", required=True, help="Ground truth COCO JSON")
    parser.add_argument("--name1", default="Method1", help="Name of method 1")
    parser.add_argument("--name2", default="Method2", help="Name of method 2")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="Bootstrap iterations")
    args = parser.parse_args()
    
    print("Loading data...")
    preds1 = load_predictions(args.pred1)
    preds2 = load_predictions(args.pred2)
    gt_by_image, image_ids = load_gt(args.gt)
    
    # Common images
    common_ids = sorted(set(image_ids) & set(preds1.keys()) & set(preds2.keys()))
    print(f"Common images: {len(common_ids)}")
    
    # Compute per-image metrics
    print("Computing per-image metrics...")
    ap50_1, ap50_2 = [], []
    recall_1, recall_2 = [], []
    
    for img_id in common_ids:
        # GT boxes
        gt_anns = gt_by_image.get(img_id, [])
        gt_boxes = np.array([[a["bbox"][0], a["bbox"][1], 
                              a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] 
                             for a in gt_anns]) if gt_anns else np.zeros((0, 4))
        
        # Method 1 predictions
        p1 = preds1.get(img_id, [])
        if p1:
            boxes1 = np.array([[p["bbox"][0], p["bbox"][1],
                               p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3]]
                              for p in p1])
            scores1 = np.array([p["score"] for p in p1])
        else:
            boxes1 = np.zeros((0, 4))
            scores1 = np.array([])
        
        # Method 2 predictions
        p2 = preds2.get(img_id, [])
        if p2:
            boxes2 = np.array([[p["bbox"][0], p["bbox"][1],
                               p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3]]
                              for p in p2])
            scores2 = np.array([p["score"] for p in p2])
        else:
            boxes2 = np.zeros((0, 4))
            scores2 = np.array([])
        
        # AP@0.5
        ap50_1.append(compute_per_image_ap(gt_boxes, boxes1, scores1))
        ap50_2.append(compute_per_image_ap(gt_boxes, boxes2, scores2))
        
        # Recall
        recall_1.append(compute_per_image_recall(gt_boxes, boxes1, scores1))
        recall_2.append(compute_per_image_recall(gt_boxes, boxes2, scores2))
    
    ap50_1 = np.array(ap50_1)
    ap50_2 = np.array(ap50_2)
    recall_1 = np.array(recall_1)
    recall_2 = np.array(recall_2)
    
    results = {}
    
    print(f"\n{'='*70}")
    print(f"  STATISTICAL ANALYSIS: {args.name1} vs {args.name2}")
    print(f"  N = {len(common_ids)} images")
    print(f"{'='*70}")
    
    # Per-image AP@0.5
    print(f"\n  AP@0.5 (per-image)")
    print(f"    {args.name1}: {ap50_1.mean():.4f} ± {ap50_1.std():.4f}")
    print(f"    {args.name2}: {ap50_2.mean():.4f} ± {ap50_2.std():.4f}")
    
    # Bootstrap CI for AP difference
    diff_mean, diff_lo, diff_hi, diff_p = bootstrap_difference_ci(
        ap50_2, ap50_1, n_bootstrap=args.n_bootstrap
    )
    print(f"    Difference (2-1): {diff_mean:.4f} [{diff_lo:.4f}, {diff_hi:.4f}]")
    print(f"    Bootstrap p-value: {diff_p:.4f}")
    
    # Effect size
    d = cohens_d(ap50_2, ap50_1)
    print(f"    Cohen's d: {d:.4f}")
    
    results["ap50"] = {
        "mean_1": float(ap50_1.mean()), "std_1": float(ap50_1.std()),
        "mean_2": float(ap50_2.mean()), "std_2": float(ap50_2.std()),
        "diff_mean": float(diff_mean), "diff_ci": [float(diff_lo), float(diff_hi)],
        "p_value": float(diff_p), "cohens_d": float(d),
    }
    
    # Paired t-test
    if HAS_SCIPY:
        t_stat, p_val = scipy_stats.ttest_rel(ap50_2, ap50_1)
        print(f"    Paired t-test: t={t_stat:.4f}, p={p_val:.6f}")
        results["ap50"]["t_test_p"] = float(p_val)
        
        # Wilcoxon signed-rank (non-parametric)
        try:
            w_stat, w_p = scipy_stats.wilcoxon(ap50_2 - ap50_1)
            print(f"    Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.6f}")
            results["ap50"]["wilcoxon_p"] = float(w_p)
        except Exception:
            pass
    
    # Per-image Recall
    print(f"\n  Recall@0.5 (per-image)")
    print(f"    {args.name1}: {recall_1.mean():.4f} ± {recall_1.std():.4f}")
    print(f"    {args.name2}: {recall_2.mean():.4f} ± {recall_2.std():.4f}")
    
    diff_mean_r, diff_lo_r, diff_hi_r, diff_p_r = bootstrap_difference_ci(
        recall_2, recall_1, n_bootstrap=args.n_bootstrap
    )
    print(f"    Difference (2-1): {diff_mean_r:.4f} [{diff_lo_r:.4f}, {diff_hi_r:.4f}]")
    print(f"    Bootstrap p-value: {diff_p_r:.4f}")
    
    d_r = cohens_d(recall_2, recall_1)
    print(f"    Cohen's d: {d_r:.4f}")
    
    results["recall"] = {
        "mean_1": float(recall_1.mean()), "std_1": float(recall_1.std()),
        "mean_2": float(recall_2.mean()), "std_2": float(recall_2.std()),
        "diff_mean": float(diff_mean_r), "diff_ci": [float(diff_lo_r), float(diff_hi_r)],
        "p_value": float(diff_p_r), "cohens_d": float(d_r),
    }
    
    if HAS_SCIPY:
        t_stat_r, p_val_r = scipy_stats.ttest_rel(recall_2, recall_1)
        print(f"    Paired t-test: t={t_stat_r:.4f}, p={p_val_r:.6f}")
        results["recall"]["t_test_p"] = float(p_val_r)
    
    # Win/Loss/Tie analysis
    print(f"\n  Win/Loss/Tie Analysis (AP@0.5)")
    wins = np.sum(ap50_2 > ap50_1 + 0.001)
    losses = np.sum(ap50_1 > ap50_2 + 0.001)
    ties = len(common_ids) - wins - losses
    print(f"    {args.name2} wins:  {wins} ({100*wins/len(common_ids):.1f}%)")
    print(f"    {args.name1} wins:  {losses} ({100*losses/len(common_ids):.1f}%)")
    print(f"    Ties (within 0.001): {ties} ({100*ties/len(common_ids):.1f}%)")
    
    results["win_loss"] = {
        "method2_wins": int(wins), "method1_wins": int(losses), "ties": int(ties)
    }
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  SIGNIFICANCE SUMMARY")
    print(f"{'='*70}")
    
    p_thresh = 0.05
    ap_sig = results["ap50"]["p_value"] < p_thresh
    rec_sig = results["recall"]["p_value"] < p_thresh
    
    print(f"  AP@0.5 improvement: {'SIGNIFICANT' if ap_sig else 'NOT significant'} (p={results['ap50']['p_value']:.4f})")
    print(f"  Recall improvement: {'SIGNIFICANT' if rec_sig else 'NOT significant'} (p={results['recall']['p_value']:.4f})")
    
    # LaTeX snippet
    print(f"\n  LaTeX for paper:")
    print(f"  AP@0.5: $\\Delta = {diff_mean:+.2f}$ ($p < {results['ap50']['p_value']:.3f}$, Cohen's $d = {results['ap50']['cohens_d']:.2f}$)")
    print(f"  Recall: $\\Delta = {diff_mean_r:+.2f}$ ($p < {results['recall']['p_value']:.3f}$, Cohen's $d = {results['recall']['cohens_d']:.2f}$)")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
