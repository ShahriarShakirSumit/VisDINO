#!/usr/bin/env python3
"""
Visualize VisDINO OGS Gate Activations

Generates qualitative visualizations showing:
1. Input image with GT fbox (green) and vbox (yellow) overlay
2. Per-query visibility gate heatmaps overlaid on image
3. Sampling points colored by visibility gate value
4. Side-by-side comparison: occluded vs non-occluded detections

Usage:
    cd <detrex root>
    python projects/visdino/visualize_gates.py \
        --config projects/visdino/configs/visdino_r50_4scale_crowdhuman_300k.py \
        --checkpoint output/visdino_r50_crowdhuman_300k_vbox/model_final.pth \
        --images datasets/crowdhuman/images_val/Images \
        --output projects/visdino/paper/figures/gates \
        --num-images 10

Outputs: PNG figures for paper and supplementary material.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path

DETREX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(DETREX_DIR, ".."))

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image


def get_ogs_layers(model):
    """Extract OGS/MS-OGS attention layers from the decoder."""
    ogs_layers = []
    # Method 1: Original OGS (has visibility_gate attribute)
    decoder = model.transformer.decoder
    for i, layer in enumerate(decoder.layers):
        for sublayer in layer.attn:
            if hasattr(sublayer, 'visibility_gate'):
                ogs_layers.append((i, sublayer))
    # Method 2: MS-OGS (class name contains MSOGSMultiScaleDeformable)
    if not ogs_layers:
        for name, mod in model.named_modules():
            if 'MSOGSMultiScaleDeformable' in type(mod).__name__:
                # Extract layer index from name
                layer_idx = -1
                for part in name.split('.'):
                    if part.isdigit():
                        layer_idx = int(part)
                ogs_layers.append((layer_idx, mod))
    return ogs_layers


def run_inference_with_gates(model, image, device):
    """Run inference and collect gate activations from all OGS layers."""
    model.eval()
    
    # Prepare input
    h, w = image.shape[:2]
    img_tensor = torch.as_tensor(image.transpose(2, 0, 1).astype("float32")).to(device)
    inputs = [{"image": img_tensor, "height": h, "width": w}]
    
    with torch.no_grad():
        outputs = model(inputs)
    
    # Collect gate activations
    ogs_layers = get_ogs_layers(model)
    gate_data = []
    for layer_idx, ogs_layer in ogs_layers:
        vis_weights = ogs_layer.get_visibility_weights()
        samp_locs = ogs_layer.get_sampling_locations()
        if vis_weights is not None:
            gate_data.append({
                "layer": layer_idx,
                "vis_weights": vis_weights.cpu().numpy(),  # (bs, nq, H, L, P)
                "sampling_locs": samp_locs.cpu().numpy() if samp_locs is not None else None,
            })
    
    return outputs, gate_data


def draw_boxes(image, boxes, color, thickness=2, label=None):
    """Draw bounding boxes on image."""
    img = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(c) for c in box[:4]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label and i == 0:
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def create_gate_heatmap(image, gate_values, sampling_locs, query_idx, h, w):
    """
    Create a heatmap visualization of gate values for a specific query.
    
    Args:
        image: Original image (H, W, 3)
        gate_values: (num_query, num_heads, num_levels, num_points) 
        sampling_locs: (num_query, num_heads, num_levels, num_points, 2) normalized coords
        query_idx: Which query to visualize
        h, w: Image dimensions
    """
    # Get gate values for this query, average over heads
    gates = gate_values[query_idx]  # (H, L, P)
    locs = sampling_locs[query_idx]   # (H, L, P, 2) 
    
    # Average over heads
    gates_mean = gates.mean(axis=0)  # (L, P)
    locs_mean = locs.mean(axis=0)    # (L, P, 2)
    
    # Create overlay
    overlay = image.copy().astype(np.float32)
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # Place gaussian at each sampling point, weighted by gate
    sigma = max(h, w) / 50
    for l in range(gates_mean.shape[0]):
        for p in range(gates_mean.shape[1]):
            gate_val = gates_mean[l, p]
            loc = locs_mean[l, p]  # Normalized coordinates
            px = int(loc[0] * w)
            py = int(loc[1] * h)
            if 0 <= px < w and 0 <= py < h:
                # Draw gaussian blob
                y_grid, x_grid = np.ogrid[max(0, py-30):min(h, py+30), max(0, px-30):min(w, px+30)]
                gauss = np.exp(-((x_grid - px)**2 + (y_grid - py)**2) / (2 * sigma**2))
                heatmap[max(0, py-30):min(h, py+30), max(0, px-30):min(w, px+30)] += gauss * gate_val
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Apply colormap
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    result = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
    
    return result


def create_sampling_point_viz(image, gate_values, sampling_locs, query_idx, h, w, pred_box=None):
    """
    Draw sampling points as circles colored by their visibility gate value.
    Red = low visibility (occluded), Green = high visibility (visible)
    """
    img = image.copy()
    
    gates = gate_values[query_idx]  # (H, L, P)
    locs = sampling_locs[query_idx]   # (H, L, P, 2)
    
    # Draw for each head (using different marker intensity)
    num_heads = gates.shape[0]
    for h_idx in range(num_heads):
        for l in range(gates.shape[1]):
            for p in range(gates.shape[2]):
                gate_val = gates[h_idx, l, p]
                loc = locs[h_idx, l, p]
                px = int(loc[0] * w)
                py = int(loc[1] * h)
                if 0 <= px < w and 0 <= py < h:
                    # Color: green for visible (gate≈1), red for occluded (gate≈0)
                    green = int(gate_val * 255)
                    red = int((1 - gate_val) * 255)
                    color = (0, green, red)  # BGR
                    radius = 3 if num_heads <= 8 else 2
                    cv2.circle(img, (px, py), radius, color, -1)
    
    # Draw prediction box if available
    if pred_box is not None:
        x1, y1, x2, y2 = [int(c) for c in pred_box[:4]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Cyan
    
    return img


def create_gate_distribution_plot(gate_data, save_path):
    """Plot gate value distributions across layers."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping distribution plot")
        return
    
    fig, axes = plt.subplots(1, len(gate_data), figsize=(4 * len(gate_data), 4))
    if len(gate_data) == 1:
        axes = [axes]
    
    for i, gd in enumerate(gate_data):
        vis_w = gd["vis_weights"][0]  # First image in batch
        gates_flat = vis_w.flatten()
        axes[i].hist(gates_flat, bins=50, density=True, alpha=0.7, color='steelblue')
        axes[i].set_xlabel("Visibility Gate Value")
        axes[i].set_ylabel("Density")
        axes[i].set_title(f"Decoder Layer {gd['layer']}")
        axes[i].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        axes[i].set_xlim(0, 1)
        
        mean_gate = gates_flat.mean()
        axes[i].axvline(x=mean_gate, color='green', linestyle='-', alpha=0.7, label=f"mean={mean_gate:.3f}")
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Gate distribution saved to {save_path}")


def create_per_layer_comparison(image, gate_data, query_idx, h, w, save_path):
    """Show gate heatmaps across decoder layers for one query."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    n_layers = len(gate_data)
    fig, axes = plt.subplots(1, n_layers + 1, figsize=(4 * (n_layers + 1), 4))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input")
    axes[0].axis('off')
    
    for i, gd in enumerate(gate_data):
        vis_w = gd["vis_weights"][0]  # batch dim
        samp_l = gd["sampling_locs"][0] if gd["sampling_locs"] is not None else None
        
        if samp_l is not None and query_idx < vis_w.shape[0]:
            hm = create_gate_heatmap(image, vis_w, samp_l, query_idx, h, w)
            axes[i + 1].imshow(cv2.cvtColor(hm, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(f"Layer {gd['layer']}")
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_gt_annotations(annotation_path, image_id):
    """Load ground truth annotations for a specific image."""
    with open(annotation_path) as f:
        data = json.load(f)
    
    # Find image
    img_info = None
    for img in data["images"]:
        if img["id"] == image_id or img.get("file_name", "").replace(".jpg", "") == str(image_id):
            img_info = img
            break
    
    if img_info is None:
        return None, None
    
    # Find annotations
    anns = [a for a in data["annotations"] if a["image_id"] == img_info["id"]]
    
    fboxes = []
    vboxes = []
    for a in anns:
        if "bbox" in a:
            x, y, w, h = a["bbox"]
            fboxes.append([x, y, x + w, y + h])
        if "visible_bbox" in a:
            vx, vy, vw, vh = a["visible_bbox"]
            vboxes.append([vx, vy, vx + vw, vy + vh])
    
    return np.array(fboxes) if fboxes else None, np.array(vboxes) if vboxes else None


def main():
    parser = argparse.ArgumentParser(description="Visualize VisDINO OGS gates")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--images", required=True, help="Image directory or single image")
    parser.add_argument("--output", default="gate_viz", help="Output directory")
    parser.add_argument("--num-images", type=int, default=10, help="Number of images to visualize")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K detections to visualize per image")
    parser.add_argument("--score-thresh", type=float, default=0.3, help="Score threshold")
    parser.add_argument("--annotations", default=None, help="COCO annotation JSON for GT overlay")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    print("Loading model...")
    cfg = LazyConfig.load(args.config)
    model = instantiate(cfg.model)
    model.to(device)
    DetectionCheckpointer(model).load(args.checkpoint)
    model.eval()
    print("Model loaded.")
    
    # Check OGS layers
    ogs_layers = get_ogs_layers(model)
    print(f"Found {len(ogs_layers)} OGS decoder layers")
    if not ogs_layers:
        print("ERROR: No OGS layers found. Is this a DINO model?")
        return
    
    # Prepare output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get image list
    img_dir = Path(args.images)
    if img_dir.is_file():
        image_paths = [img_dir]
    else:
        image_paths = sorted(img_dir.glob("*.jpg"))[:args.num_images]
    
    print(f"Processing {len(image_paths)} images...")
    
    for img_idx, img_path in enumerate(image_paths):
        print(f"\n  [{img_idx+1}/{len(image_paths)}] {img_path.name}")
        
        # Read image
        image = read_image(str(img_path), format="BGR")
        h, w = image.shape[:2]
        
        # Run inference
        outputs, gate_data = run_inference_with_gates(model, image, device)
        
        if not gate_data:
            print("    No gate data collected, skipping")
            continue
        
        # Get predictions
        instances = outputs[0]["instances"]
        scores = instances.scores.cpu().numpy()
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        
        # Filter by score
        mask = scores >= args.score_thresh
        scores = scores[mask]
        boxes = boxes[mask]
        
        # Top-K
        if len(scores) > args.top_k:
            topk_idx = np.argsort(scores)[-args.top_k:]
            scores = scores[topk_idx]
            boxes = boxes[topk_idx]
        
        print(f"    {len(scores)} detections above threshold {args.score_thresh}")
        
        stem = img_path.stem
        
        # 1. Gate distribution across layers
        create_gate_distribution_plot(gate_data, os.path.join(args.output, f"{stem}_gate_dist.png"))
        
        # 2. For each high-confidence detection, show sampling points
        for det_idx in range(min(len(boxes), args.top_k)):
            # Find closest query to this detection box center
            box = boxes[det_idx]
            box_center = np.array([(box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h])
            
            # Use last layer's gate data
            last_gate = gate_data[-1]
            vis_w = last_gate["vis_weights"][0]  # (nq, H, L, P)
            samp_l = last_gate["sampling_locs"][0] if last_gate["sampling_locs"] is not None else None
            
            if samp_l is None:
                continue
            
            # Heatmap
            query_idx = det_idx  # Approximation: top predictions ≈ first queries after NMS
            if query_idx < vis_w.shape[0]:
                # Sampling points visualization
                sp_viz = create_sampling_point_viz(image, vis_w, samp_l, query_idx, h, w, pred_box=box)
                save_p = os.path.join(args.output, f"{stem}_det{det_idx}_points.png")
                cv2.imwrite(save_p, sp_viz)
                
                # Gate heatmap
                hm_viz = create_gate_heatmap(image, vis_w, samp_l, query_idx, h, w)
                # Add detection box
                x1, y1, x2, y2 = [int(c) for c in box]
                cv2.rectangle(hm_viz, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(hm_viz, f"score={scores[det_idx]:.2f}", (x1, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                save_h = os.path.join(args.output, f"{stem}_det{det_idx}_heatmap.png")
                cv2.imwrite(save_h, hm_viz)
                
                # Per-layer comparison
                create_per_layer_comparison(
                    image, gate_data, query_idx, h, w,
                    os.path.join(args.output, f"{stem}_det{det_idx}_layers.png")
                )
                
                # Gate stats for this detection
                mean_gate = vis_w[query_idx].mean()
                min_gate = vis_w[query_idx].min()
                print(f"    Det {det_idx}: score={scores[det_idx]:.3f}, "
                      f"mean_gate={mean_gate:.3f}, min_gate={min_gate:.3f}")
        
        # 3. Overview: all detections with boxes
        overview = image.copy()
        for det_idx in range(len(boxes)):
            box = boxes[det_idx]
            x1, y1, x2, y2 = [int(c) for c in box]
            # Color by mean gate value of corresponding query
            last_vis = gate_data[-1]["vis_weights"][0]
            if det_idx < last_vis.shape[0]:
                mean_g = last_vis[det_idx].mean()
                green = int(mean_g * 255)
                red = int((1 - mean_g) * 255)
                color = (0, green, red)
            else:
                color = (255, 255, 0)
            cv2.rectangle(overview, (x1, y1), (x2, y2), color, 2)
        
        cv2.imwrite(os.path.join(args.output, f"{stem}_overview.png"), overview)
        
        # Load GT if annotation file provided
        if args.annotations and os.path.exists(args.annotations):
            fboxes, vboxes = load_gt_annotations(args.annotations, img_path.stem)
            if fboxes is not None:
                gt_img = image.copy()
                gt_img = draw_boxes(gt_img, fboxes, (0, 255, 0), 2, "fbox")
                if vboxes is not None:
                    gt_img = draw_boxes(gt_img, vboxes, (0, 255, 255), 1, "vbox")
                cv2.imwrite(os.path.join(args.output, f"{stem}_gt.png"), gt_img)
    
    print(f"\nVisualization complete! Output in: {args.output}")
    print(f"Key figures for paper:")
    print(f"  - *_gate_dist.png — Gate value distributions (Fig. for Discussion)")
    print(f"  - *_heatmap.png  — Gate heatmaps overlaid on image")
    print(f"  - *_points.png   — Sampling points colored by gate value")
    print(f"  - *_layers.png   — Cross-layer gate comparison")


if __name__ == "__main__":
    main()
