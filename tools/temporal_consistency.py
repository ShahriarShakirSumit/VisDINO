#!/usr/bin/env python3
"""
Temporal Consistency Analysis on NightOwls Video Sequences
==========================================================
Evaluates detection STABILITY across consecutive frames in NightOwls
video sequences. Uses CrowdHuman-trained models (cross-dataset) to
measure how consistent detections are in time.

Metrics:
  1. Detection Flicker Rate (DFR): fraction of detections that appear/disappear
     between consecutive frames without a matching detection
  2. BBox Jitter: std deviation of bbox center/size across matched detections
     in consecutive frames (lower = more stable)
  3. Inter-Frame IoU (IFIoU): mean IoU of matched detections between
     consecutive frames (higher = more stable)
  4. Track Fragmentation: how often a tracked pedestrian's detection breaks
     (appears → disappears → reappears)
  5. Score Stability: std of confidence scores for matched detections (lower = better)

Usage:
  cd <detrex root>
  python projects/visdino/temporal_consistency.py \
      --model-type visdino \
      --config-file projects/visdino/configs/visdino_r50_4scale_crowdhuman_300k.py \
      --checkpoint output/visdino_r50_crowdhuman_300k/model_final.pth \
      --output-dir output/temporal_eval/visdino_300k \
      --max-sequences 50

  # Baseline DINO:
  python projects/visdino/temporal_consistency.py \
      --model-type dino \
      --config-file projects/dino_crowdhuman/configs/dino_r50_4scale_crowdhuman.py \
      --checkpoint ../output/dino_r50_crowdhuman_baseline/model_final.pth \
      --output-dir output/temporal_eval/dino_baseline \
      --max-sequences 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# ──────────────────── paths ────────────────────
DETREX_ROOT = Path(__file__).resolve().parents[2]  # detrex/detrex/
DETREX_WORKSPACE = DETREX_ROOT.parent               # detrex/
NIGHTOWLS_DIR = DETREX_WORKSPACE / "datasets" / "nightowls"
NIGHTOWLS_VAL_JSON = NIGHTOWLS_DIR / "annotations" / "nightowls_validation.json"
NIGHTOWLS_VAL_IMGS = NIGHTOWLS_DIR / "images_val"

sys.path.insert(0, str(DETREX_ROOT))


# ──────────────────── Sequence Extraction ────────────────────
def extract_sequences(json_path, min_seq_len=5, max_gap_ms=100000):
    """
    Extract consecutive frame sequences from NightOwls annotations.
    Groups images by recording_id, sorts by timestamp, and splits
    sequences where time gap exceeds max_gap_ms.
    
    Returns: list of sequences, each is a list of image dicts sorted by time
    """
    with open(json_path) as f:
        data = json.load(f)
    
    # Group by recording_id
    recordings = defaultdict(list)
    for img in data['images']:
        rec_id = img.get('recordings_id')
        recordings[rec_id].append(img)
    
    all_sequences = []
    for rec_id, frames in recordings.items():
        frames = sorted(frames, key=lambda x: x['timestamp'])
        
        # Split into sub-sequences based on time gaps
        current_seq = [frames[0]]
        for i in range(1, len(frames)):
            gap = frames[i]['timestamp'] - frames[i-1]['timestamp']
            if gap < max_gap_ms:
                current_seq.append(frames[i])
            else:
                if len(current_seq) >= min_seq_len:
                    all_sequences.append(current_seq)
                current_seq = [frames[i]]
        
        if len(current_seq) >= min_seq_len:
            all_sequences.append(current_seq)
    
    # Sort by sequence length (longest first)
    all_sequences.sort(key=lambda s: len(s), reverse=True)
    
    return all_sequences


# ──────────────────── Model Loading ────────────────────
def load_model(config_file, checkpoint, model_type="visdino"):
    """Load a trained DINO/VisDINO model."""
    from detectron2.config import LazyConfig, instantiate
    from detectron2.checkpoint import DetectionCheckpointer
    
    cfg = LazyConfig.load(config_file)
    
    # Build model
    model = instantiate(cfg.model)
    model.to("cuda")
    model.eval()
    
    # Load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint)
    
    return model, cfg


# ──────────────────── Single Image Inference ────────────────────
def infer_single_image(model, img_path, score_thr=0.3):
    """
    Run inference on a single image. Returns list of detections:
    [{'bbox': [x1,y1,x2,y2], 'score': float, 'center': [cx,cy], 'size': [w,h]}]
    """
    import torchvision.transforms as T
    from detectron2.structures import ImageList
    
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    
    # Prepare input in detectron2 format
    h, w = img_np.shape[:2]
    
    # Standard DINO/VisDINO preprocessing
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(img).to("cuda")
    
    # Create detectron2-style input
    inp = {
        "image": (img_tensor * 255).to(torch.uint8),  # detectron2 expects uint8
        "height": h,
        "width": w,
        "image_id": 0,
    }
    
    # But actually detectron2 models expect BGR uint8 images
    # Let's use the proper approach
    import cv2
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return []
    
    h, w = img_bgr.shape[:2]
    img_tensor = torch.as_tensor(img_bgr.transpose(2, 0, 1).astype("float32"))
    
    inp = {
        "image": img_tensor,
        "height": h,
        "width": w,
        "image_id": 0,
    }
    
    with torch.no_grad():
        outputs = model([inp])
    
    instances = outputs[0]["instances"].to("cpu")
    scores = instances.scores.numpy()
    boxes = instances.pred_boxes.tensor.numpy()  # xyxy
    
    detections = []
    for j in range(len(scores)):
        if scores[j] < score_thr:
            continue
        
        x1, y1, x2, y2 = boxes[j]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1
        
        detections.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'score': float(scores[j]),
            'center': [float(cx), float(cy)],
            'size': [float(bw), float(bh)],
        })
    
    return detections


# ──────────────────── Matching ────────────────────
def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / max(union, 1e-8)


def match_detections(dets1, dets2, iou_thr=0.3):
    """
    Match detections between two consecutive frames using IoU.
    Returns: matched_pairs (list of (i, j, iou)), unmatched_1, unmatched_2
    """
    if not dets1 or not dets2:
        return [], list(range(len(dets1))), list(range(len(dets2)))
    
    n1, n2 = len(dets1), len(dets2)
    iou_matrix = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            iou_matrix[i, j] = compute_iou(dets1[i]['bbox'], dets2[j]['bbox'])
    
    # Greedy matching (highest IoU first)
    matched_pairs = []
    used1 = set()
    used2 = set()
    
    while True:
        if len(used1) == n1 or len(used2) == n2:
            break
        
        # Find best remaining match
        best_iou = -1
        best_i, best_j = -1, -1
        for i in range(n1):
            if i in used1:
                continue
            for j in range(n2):
                if j in used2:
                    continue
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_i, best_j = i, j
        
        if best_iou < iou_thr:
            break
        
        matched_pairs.append((best_i, best_j, best_iou))
        used1.add(best_i)
        used2.add(best_j)
    
    unmatched_1 = [i for i in range(n1) if i not in used1]
    unmatched_2 = [j for j in range(n2) if j not in used2]
    
    return matched_pairs, unmatched_1, unmatched_2


# ──────────────────── Temporal Metrics ────────────────────
def compute_temporal_metrics(sequence_detections, iou_thr=0.3):
    """
    Compute temporal consistency metrics for a sequence of frame detections.
    
    Args:
        sequence_detections: list of (frame_info, detections) for each frame in sequence
        iou_thr: IoU threshold for matching detections between frames
    
    Returns:
        dict with temporal metrics
    """
    if len(sequence_detections) < 2:
        return None
    
    # Pair-wise metrics between consecutive frames
    all_ious = []           # IoU between matched detections
    all_center_diffs = []   # Euclidean distance of bbox centers (matched)
    all_size_diffs = []     # Relative size change (matched)
    all_score_diffs = []    # Score difference (matched)
    
    total_dets_t = 0        # Total detections at time t
    total_dets_t1 = 0       # Total detections at time t+1
    total_matched = 0       # Total matched between consecutive frames
    total_appeared = 0      # Detections that appeared (in t+1 but not t)
    total_disappeared = 0   # Detections that disappeared (in t but not t+1)
    
    # Per-frame detection counts for variance analysis
    det_counts = []
    
    for t in range(len(sequence_detections) - 1):
        _, dets_t = sequence_detections[t]
        _, dets_t1 = sequence_detections[t + 1]
        
        det_counts.append(len(dets_t))
        
        matched, unmatched_t, unmatched_t1 = match_detections(dets_t, dets_t1, iou_thr=iou_thr)
        
        total_dets_t += len(dets_t)
        total_dets_t1 += len(dets_t1)
        total_matched += len(matched)
        total_disappeared += len(unmatched_t)
        total_appeared += len(unmatched_t1)
        
        for i, j, iou in matched:
            all_ious.append(iou)
            
            # Center displacement
            cx_diff = dets_t[i]['center'][0] - dets_t1[j]['center'][0]
            cy_diff = dets_t[i]['center'][1] - dets_t1[j]['center'][1]
            center_dist = np.sqrt(cx_diff**2 + cy_diff**2)
            all_center_diffs.append(center_dist)
            
            # Relative size change
            w_ratio = dets_t1[j]['size'][0] / max(dets_t[i]['size'][0], 1e-8)
            h_ratio = dets_t1[j]['size'][1] / max(dets_t[i]['size'][1], 1e-8)
            size_change = abs(w_ratio - 1.0) + abs(h_ratio - 1.0)
            all_size_diffs.append(size_change)
            
            # Score difference
            score_diff = abs(dets_t[i]['score'] - dets_t1[j]['score'])
            all_score_diffs.append(score_diff)
    
    # Add last frame count
    det_counts.append(len(sequence_detections[-1][1]))
    
    # Compute aggregate metrics
    n_pairs = len(sequence_detections) - 1
    
    # Detection Flicker Rate: fraction of detections appearing/disappearing
    total_transitions = total_appeared + total_disappeared
    total_possible = total_dets_t + total_dets_t1
    flicker_rate = total_transitions / max(total_possible, 1)
    
    # Stability ratio: matched / total
    stability_ratio = (2 * total_matched) / max(total_dets_t + total_dets_t1, 1)
    
    metrics = {
        'num_frames': len(sequence_detections),
        'num_frame_pairs': n_pairs,
        'total_detections': sum(det_counts),
        'mean_dets_per_frame': np.mean(det_counts),
        'std_dets_per_frame': np.std(det_counts),
        
        # Matching metrics
        'total_matched': total_matched,
        'total_appeared': total_appeared,
        'total_disappeared': total_disappeared,
        'stability_ratio': float(stability_ratio),
        'flicker_rate': float(flicker_rate),
        
        # Inter-frame IoU
        'mean_inter_frame_iou': float(np.mean(all_ious)) if all_ious else 0.0,
        'std_inter_frame_iou': float(np.std(all_ious)) if all_ious else 0.0,
        'median_inter_frame_iou': float(np.median(all_ious)) if all_ious else 0.0,
        
        # BBox jitter (center displacement in pixels)
        'mean_center_jitter': float(np.mean(all_center_diffs)) if all_center_diffs else 0.0,
        'std_center_jitter': float(np.std(all_center_diffs)) if all_center_diffs else 0.0,
        'median_center_jitter': float(np.median(all_center_diffs)) if all_center_diffs else 0.0,
        
        # Size stability
        'mean_size_change': float(np.mean(all_size_diffs)) if all_size_diffs else 0.0,
        'std_size_change': float(np.std(all_size_diffs)) if all_size_diffs else 0.0,
        
        # Score stability  
        'mean_score_diff': float(np.mean(all_score_diffs)) if all_score_diffs else 0.0,
        'std_score_diff': float(np.std(all_score_diffs)) if all_score_diffs else 0.0,
    }
    
    return metrics


# ──────────────────── Main Pipeline ────────────────────
def run_temporal_analysis(model, sequences, output_dir, score_thr=0.3, 
                          max_sequences=None, max_frames_per_seq=100, iou_thr=0.3):
    """
    Run temporal consistency analysis on NightOwls sequences.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    
    all_metrics = []
    
    print(f"\n{'='*60}")
    print(f"Temporal Consistency Analysis")
    print(f"{'='*60}")
    print(f"Sequences to analyze: {len(sequences)}")
    print(f"Score threshold: {score_thr}")
    print(f"IoU matching threshold: {iou_thr}")
    print(f"Max frames per sequence: {max_frames_per_seq}")
    print()
    
    for seq_idx, sequence in enumerate(sequences):
        # Truncate very long sequences
        if len(sequence) > max_frames_per_seq:
            sequence = sequence[:max_frames_per_seq]
        
        seq_len = len(sequence)
        rec_id = sequence[0].get('recordings_id', 'unknown')
        
        print(f"[Sequence {seq_idx+1}/{len(sequences)}] Recording {rec_id}, "
              f"{seq_len} frames, ts {sequence[0]['timestamp']}->{sequence[-1]['timestamp']}")
        
        # Run inference on each frame
        sequence_detections = []
        for frame in tqdm(sequence, desc=f"  Seq {seq_idx+1}", leave=False):
            img_path = NIGHTOWLS_VAL_IMGS / frame['file_name']
            dets = infer_single_image(model, img_path, score_thr=score_thr)
            sequence_detections.append((frame, dets))
        
        # Compute temporal metrics
        metrics = compute_temporal_metrics(sequence_detections, iou_thr=iou_thr)
        if metrics is not None:
            metrics['sequence_idx'] = seq_idx
            metrics['recording_id'] = rec_id
            all_metrics.append(metrics)
            
            print(f"  → Stability: {metrics['stability_ratio']:.3f}, "
                  f"Flicker: {metrics['flicker_rate']:.3f}, "
                  f"IoU: {metrics['mean_inter_frame_iou']:.3f}, "
                  f"Jitter: {metrics['mean_center_jitter']:.1f}px, "
                  f"Score Δ: {metrics['mean_score_diff']:.3f}")
    
    # Aggregate across all sequences
    if all_metrics:
        aggregate = compute_aggregate_metrics(all_metrics)
        
        # Save results
        results = {
            'config': {
                'score_threshold': score_thr,
                'iou_threshold': iou_thr,
                'max_sequences': max_sequences,
                'max_frames_per_seq': max_frames_per_seq,
                'num_sequences_analyzed': len(all_metrics),
                'total_frames_analyzed': sum(m['num_frames'] for m in all_metrics),
            },
            'aggregate': aggregate,
            'per_sequence': all_metrics,
        }
        
        results_path = os.path.join(output_dir, "temporal_consistency_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print_summary(aggregate, output_dir)
        
        return results
    
    return None


def compute_aggregate_metrics(all_metrics):
    """Compute aggregate statistics across all sequences."""    
    keys = [
        'stability_ratio', 'flicker_rate',
        'mean_inter_frame_iou', 'mean_center_jitter',
        'mean_size_change', 'mean_score_diff',
        'mean_dets_per_frame', 'std_dets_per_frame',
    ]
    
    aggregate = {}
    for key in keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            aggregate[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
    
    # Total counts
    aggregate['total_frames'] = sum(m['num_frames'] for m in all_metrics)
    aggregate['total_sequences'] = len(all_metrics)
    aggregate['total_detections'] = sum(m['total_detections'] for m in all_metrics)
    aggregate['total_matched'] = sum(m['total_matched'] for m in all_metrics)
    aggregate['total_appeared'] = sum(m['total_appeared'] for m in all_metrics)
    aggregate['total_disappeared'] = sum(m['total_disappeared'] for m in all_metrics)
    
    # Global stability
    total_m = aggregate['total_matched']
    total_a = aggregate['total_appeared']
    total_d = aggregate['total_disappeared']
    aggregate['global_stability_ratio'] = (2 * total_m) / max(2 * total_m + total_a + total_d, 1)
    aggregate['global_flicker_rate'] = (total_a + total_d) / max(2 * total_m + total_a + total_d, 1)
    
    return aggregate


def print_summary(aggregate, output_dir):
    """Print a human-readable summary of temporal consistency metrics."""
    summary_lines = []
    
    def p(line=""):
        print(line)
        summary_lines.append(line)
    
    p()
    p("=" * 70)
    p("TEMPORAL CONSISTENCY ANALYSIS — SUMMARY")
    p("=" * 70)
    p(f"Total sequences: {aggregate['total_sequences']}")
    p(f"Total frames analyzed: {aggregate['total_frames']}")
    p(f"Total detections: {aggregate['total_detections']}")
    p()
    p("─── Key Metrics ───")
    
    for key, label in [
        ('stability_ratio', 'Stability Ratio ↑ (matched/total)'),
        ('flicker_rate', 'Flicker Rate ↓ (appear+disappear/total)'),
        ('mean_inter_frame_iou', 'Inter-Frame IoU ↑'),
        ('mean_center_jitter', 'Center Jitter ↓ (pixels)'),
        ('mean_size_change', 'Size Change ↓ (relative)'),
        ('mean_score_diff', 'Score Instability ↓'),
    ]:
        if key in aggregate:
            stats = aggregate[key]
            p(f"  {label}:")
            p(f"    Mean: {stats['mean']:.4f}  Std: {stats['std']:.4f}  "
              f"Median: {stats['median']:.4f}  [Min: {stats['min']:.4f}, Max: {stats['max']:.4f}]")
    
    p()
    p(f"Global Stability Ratio: {aggregate['global_stability_ratio']:.4f}")
    p(f"Global Flicker Rate:    {aggregate['global_flicker_rate']:.4f}")
    p("=" * 70)
    
    # Save text summary
    summary_path = os.path.join(output_dir, "temporal_summary.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\nSummary saved to {summary_path}")


# ──────────────────── Entry Point ────────────────────
def main():
    parser = argparse.ArgumentParser(description="Temporal Consistency Analysis on NightOwls")
    parser.add_argument("--model-type", choices=["dino", "visdino"], required=True)
    parser.add_argument("--config-file", required=True, help="Path to config .py file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold")
    parser.add_argument("--max-sequences", type=int, default=50, 
                        help="Max number of sequences to analyze")
    parser.add_argument("--max-frames-per-seq", type=int, default=100,
                        help="Max frames per sequence")
    parser.add_argument("--min-seq-len", type=int, default=10,
                        help="Minimum sequence length to include")
    parser.add_argument("--iou-thr", type=float, default=0.3,
                        help="IoU threshold for matching detections between frames")
    args = parser.parse_args()
    
    print("=" * 60)
    print("VisDINO Temporal Consistency Analysis")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Config: {args.config_file}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print()
    
    # 1. Extract sequences from NightOwls
    print("Extracting video sequences from NightOwls val...")
    sequences = extract_sequences(
        NIGHTOWLS_VAL_JSON, 
        min_seq_len=args.min_seq_len
    )
    total_frames = sum(len(s) for s in sequences)
    print(f"Found {len(sequences)} sequences with ≥{args.min_seq_len} frames "
          f"({total_frames} total frames)")
    
    # Show sequence length distribution
    lens = [len(s) for s in sequences]
    print(f"Sequence lengths: min={min(lens)}, max={max(lens)}, "
          f"mean={np.mean(lens):.1f}, median={np.median(lens):.1f}")
    print()
    
    # 2. Load model
    print("Loading model...")
    model, cfg = load_model(args.config_file, args.checkpoint, args.model_type)
    print("Model loaded successfully.")
    print()
    
    # 3. Run temporal analysis
    results = run_temporal_analysis(
        model, sequences, args.output_dir,
        score_thr=args.score_thr,
        max_sequences=args.max_sequences,
        max_frames_per_seq=args.max_frames_per_seq,
        iou_thr=args.iou_thr,
    )
    
    if results is None:
        print("ERROR: No results generated!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
