#!/usr/bin/env python3
"""
Per-Occlusion Breakdown Evaluation for CrowdHuman

Splits GT annotations by occlusion level based on vbox/fbox visibility ratio:
  - Heavy:   ratio < 0.50  (more than 50% occluded)
  - Partial: 0.50 <= ratio < 0.85
  - None:    ratio >= 0.85

For each occlusion level, computes:
  - Recall@0.5 (IoU threshold = 0.5 with fbox)
  - AP@0.5 (per-category AP)
  - MR⁻² (log-average miss rate)
  - Count of GT instances

This is the key analysis showing VisDINO's advantage on occluded pedestrians.
"""

import argparse
import json
import numpy as np
from collections import defaultdict


def to_xyxy(b):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = b
    return [x, y, x + w, y + h]


def area_xyxy(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / (area_xyxy(a) + area_xyxy(b) - inter + 1e-12)


def visibility_ratio(fbox, vbox):
    """Compute visibility ratio = area(vbox) / area(fbox)."""
    fa = fbox[2] * fbox[3]  # w * h
    va = vbox[2] * vbox[3]
    if fa <= 0:
        return 1.0
    return va / fa


def classify_occlusion(ratio):
    """Classify into occlusion level."""
    if ratio < 0.50:
        return "heavy"
    elif ratio < 0.85:
        return "partial"
    else:
        return "none"


def compute_per_occlusion_metrics(gt_odgt_path, coco_results_path):
    """
    Compute per-occlusion metrics.
    
    Args:
        gt_odgt_path: Path to original CrowdHuman annotation_val.odgt
        coco_results_path: Path to COCO format results (coco_instances_results.json)
    
    Returns:
        dict with per-occlusion metrics
    """
    # Load GT
    gt_data = {}
    for line in open(gt_odgt_path, 'r'):
        if not line.strip():
            continue
        rec = json.loads(line)
        gt_data[rec['ID']] = rec
    
    # Load COCO predictions and group by image_id
    with open(coco_results_path, 'r') as f:
        coco_preds = json.load(f)
    
    pred_by_image = defaultdict(list)
    for pred in coco_preds:
        img_id = str(pred['image_id'])
        pred_by_image[img_id].append({
            'bbox': pred['bbox'],  # [x, y, w, h]
            'score': pred['score']
        })
    
    # Sort predictions by score descending for each image
    for img_id in pred_by_image:
        pred_by_image[img_id].sort(key=lambda x: x['score'], reverse=True)
    
    # Categorize GT by occlusion level
    gt_by_occlusion = defaultdict(list)  # level -> [(img_id, fbox_xywh, idx)]
    gt_per_image = defaultdict(lambda: defaultdict(list))  # img_id -> level -> [fbox_xyxy]
    
    for img_id, rec in gt_data.items():
        for gi, g in enumerate(rec.get('gtboxes', [])):
            tag = g.get('tag', 'person')
            if tag not in ('person', 1):
                continue
            
            fbox = g.get('fbox')
            vbox = g.get('vbox')
            if fbox is None:
                continue
            
            if vbox is None:
                vbox = fbox  # treat as no occlusion
            
            ratio = visibility_ratio(fbox, vbox)
            level = classify_occlusion(ratio)
            
            gt_by_occlusion[level].append({
                'img_id': img_id,
                'fbox': fbox,
                'ratio': ratio
            })
            gt_per_image[img_id][level].append(to_xyxy(fbox))
    
    # Compute metrics per occlusion level
    results = {}
    all_levels = ["heavy", "partial", "none"]
    
    for level in all_levels:
        gt_entries = gt_by_occlusion.get(level, [])
        n_gt = len(gt_entries)
        
        if n_gt == 0:
            results[level] = {
                'count': 0,
                'Recall': 0.0,
                'AP': 0.0,
                'MR': 1.0
            }
            continue
        
        # Collect all scored detections matched against this level's GT
        all_scores = []
        all_tp = []
        tp_count = 0
        
        # Process per image
        images_with_gt = set(e['img_id'] for e in gt_entries)
        
        for img_id in images_with_gt:
            gt_boxes = gt_per_image[img_id].get(level, [])
            if not gt_boxes:
                continue
            
            preds = pred_by_image.get(img_id, [])
            matched = [False] * len(gt_boxes)
            
            for pred in preds:
                pred_box = to_xyxy(pred['bbox'])
                score = pred['score']
                
                best_iou = 0.0
                best_gi = -1
                for gi, gt_box in enumerate(gt_boxes):
                    if matched[gi]:
                        continue
                    iou_val = iou_xyxy(pred_box, gt_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gi = gi
                
                if best_iou >= 0.5 and best_gi >= 0:
                    matched[best_gi] = True
                    all_scores.append(score)
                    all_tp.append(1)
                else:
                    all_scores.append(score)
                    all_tp.append(0)
            
            tp_count += sum(matched)
        
        recall = tp_count / n_gt if n_gt > 0 else 0.0
        
        # Compute AP
        if len(all_scores) > 0:
            sorted_indices = np.argsort(-np.array(all_scores))
            sorted_tp = np.array(all_tp)[sorted_indices]
            
            cum_tp = np.cumsum(sorted_tp)
            cum_fp = np.cumsum(1 - sorted_tp)
            
            precisions = cum_tp / (cum_tp + cum_fp + 1e-12)
            recalls = cum_tp / n_gt
            
            # VOC-style AP (11-point interpolation)
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                p_at_r = precisions[recalls >= t]
                ap += np.max(p_at_r) if len(p_at_r) > 0 else 0.0
            ap /= 11.0
            
            # MR⁻² (log-average miss rate at 9 FPPI points)
            fppi_per_image = cum_fp / len(images_with_gt)
            miss_rates = 1.0 - recalls
            
            ref_fppi = np.logspace(-2, 0, 9)  # 0.01 to 1.0
            log_miss_rates = []
            for ref in ref_fppi:
                idx = np.searchsorted(fppi_per_image, ref)
                if idx >= len(miss_rates):
                    log_miss_rates.append(miss_rates[-1])
                elif idx == 0:
                    log_miss_rates.append(miss_rates[0])
                else:
                    log_miss_rates.append(miss_rates[idx])
            
            mr = np.exp(np.mean(np.log(np.maximum(1e-10, log_miss_rates))))
        else:
            ap = 0.0
            mr = 1.0
        
        results[level] = {
            'count': n_gt,
            'Recall': round(recall, 4),
            'AP': round(ap, 4),
            'MR': round(mr, 4),
            'mean_ratio': round(np.mean([e['ratio'] for e in gt_entries]), 4)
        }
    
    # Summary
    total_gt = sum(r['count'] for r in results.values())
    results['summary'] = {
        'total_gt': total_gt,
        'heavy_fraction': results.get('heavy', {}).get('count', 0) / max(1, total_gt),
        'partial_fraction': results.get('partial', {}).get('count', 0) / max(1, total_gt),
        'none_fraction': results.get('none', {}).get('count', 0) / max(1, total_gt),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Per-occlusion evaluation for CrowdHuman')
    parser.add_argument('--gt', required=True, help='Original CrowdHuman annotation_val.odgt')
    parser.add_argument('--coco-results', required=True, help='COCO format predictions (coco_instances_results.json)')
    parser.add_argument('--out', required=True, help='Output JSON with per-occlusion metrics')
    args = parser.parse_args()
    
    results = compute_per_occlusion_metrics(args.gt, args.coco_results)
    
    # Print
    print("\n" + "=" * 70)
    print("Per-Occlusion Breakdown")
    print("=" * 70)
    print(f"{'Level':<10} {'Count':>7} {'Recall':>8} {'AP@0.5':>8} {'MR⁻²':>8} {'Mean VR':>8}")
    print("-" * 70)
    for level in ["heavy", "partial", "none"]:
        m = results.get(level, {})
        print(f"{level:<10} {m.get('count', 0):>7d} {m.get('Recall', 0):>8.4f} "
              f"{m.get('AP', 0):>8.4f} {m.get('MR', 0):>8.4f} {m.get('mean_ratio', 0):>8.4f}")
    print("-" * 70)
    s = results.get('summary', {})
    print(f"Total GT: {s.get('total_gt', 0)}")
    print(f"Heavy: {s.get('heavy_fraction', 0):.1%}, Partial: {s.get('partial_fraction', 0):.1%}, None: {s.get('none_fraction', 0):.1%}")
    print("=" * 70)
    
    json.dump(results, open(args.out, 'w'), indent=2)
    print(f"\nSaved to: {args.out}")


if __name__ == '__main__':
    main()
