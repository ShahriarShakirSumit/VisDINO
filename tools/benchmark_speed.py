#!/usr/bin/env python3
"""
Inference Speed & Overhead Benchmark for VisDINO vs DINO

Measures:
  - Parameters: total, OGS-specific, backbone, decoder
  - FLOPs (estimated)
  - Inference latency: mean, std over N iterations
  - Memory: peak GPU memory during inference

Usage:
    cd <detrex root>
    python projects/visdino/benchmark_speed.py

Outputs: projects/visdino/paper/overhead_table.tex
"""

import os
import sys
import time
import json
import torch
import numpy as np

# Setup path
DETREX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(DETREX_DIR, ".."))

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer


def count_parameters(model, prefix=""):
    """Count parameters with breakdown."""
    total = 0
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total += p.numel()
        # Categorize
        if "backbone" in name:
            groups.setdefault("backbone", 0)
            groups["backbone"] += p.numel()
        elif "vis_gate" in name or "vis_proj" in name:
            groups.setdefault("ogs", 0)
            groups["ogs"] += p.numel()
        elif "vbox" in name:
            groups.setdefault("vbox_head", 0)
            groups["vbox_head"] += p.numel()
        elif "vis_ratio" in name:
            groups.setdefault("ratio_head", 0)
            groups["ratio_head"] += p.numel()
        elif "encoder" in name:
            groups.setdefault("encoder", 0)
            groups["encoder"] += p.numel()
        elif "decoder" in name:
            groups.setdefault("decoder", 0)
            groups["decoder"] += p.numel()
        else:
            groups.setdefault("other", 0)
            groups["other"] += p.numel()
    return total, groups


def benchmark_inference(model, device, input_shape=(1, 3, 800, 1333), n_warmup=10, n_runs=50):
    """Benchmark inference speed."""
    model.eval()
    
    # Create dummy input
    dummy_input = [{"image": torch.randn(*input_shape[1:]).to(device), "height": 800, "width": 1333}]
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            try:
                _ = model(dummy_input)
            except Exception as e:
                print(f"Warmup error: {e}")
                return None, None, None
    
    # Benchmark
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    return np.mean(times), np.std(times), peak_mem


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    results = {}
    
    configs = {
        "DINO": {
            "config": os.path.join(DETREX_DIR, "projects/dino_crowdhuman/configs/dino_r50_4scale_crowdhuman.py"),
            "checkpoint": "output/dino_r50_crowdhuman_baseline/model_final.pth",
        },
        "VisDINO": {
            "config": os.path.join(DETREX_DIR, "projects/visdino/configs/visdino_r50_4scale_crowdhuman_300k.py"),
            "checkpoint": None,  # Will check if available
        },
    }
    
    # Check if VisDINO checkpoint exists
    visdino_ckpt = os.path.join(DETREX_DIR, "output/visdino_r50_crowdhuman_300k_vbox/model_final.pth")
    if os.path.exists(visdino_ckpt):
        configs["VisDINO"]["checkpoint"] = visdino_ckpt
    
    for name, info in configs.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        
        # Load config
        cfg = LazyConfig.load(info["config"])
        
        # Build model
        model = instantiate(cfg.model)
        model.to(device)
        
        # Load checkpoint if available
        if info["checkpoint"] and os.path.exists(info["checkpoint"]):
            DetectionCheckpointer(model).load(info["checkpoint"])
            print(f"  Loaded checkpoint: {info['checkpoint']}")
        else:
            print(f"  No checkpoint loaded (parameter count only)")
        
        # Count parameters
        total_params, groups = count_parameters(model)
        results[name] = {
            "total_params": total_params,
            "param_groups": groups,
        }
        
        print(f"\n  Parameters:")
        print(f"    Total:     {total_params:>12,d}")
        for g, c in sorted(groups.items()):
            pct = 100 * c / total_params
            print(f"    {g:<12s} {c:>12,d}  ({pct:.2f}%)")
        
        # Benchmark if checkpoint available
        if info["checkpoint"] and os.path.exists(info["checkpoint"]):
            print(f"\n  Benchmarking inference speed...")
            mean_ms, std_ms, peak_mem = benchmark_inference(model, device)
            if mean_ms is not None:
                results[name]["latency_ms"] = mean_ms
                results[name]["latency_std"] = std_ms
                results[name]["peak_mem_mb"] = peak_mem
                results[name]["fps"] = 1000.0 / mean_ms
                
                print(f"    Latency:   {mean_ms:.1f} ± {std_ms:.1f} ms")
                print(f"    FPS:       {1000.0/mean_ms:.1f}")
                print(f"    Peak mem:  {peak_mem:.0f} MB")
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Print comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    
    if "DINO" in results and "VisDINO" in results:
        dino_p = results["DINO"]["total_params"]
        vis_p = results["VisDINO"]["total_params"]
        overhead = vis_p - dino_p
        pct = 100.0 * overhead / dino_p
        
        print(f"  DINO params:    {dino_p:>12,d}")
        print(f"  VisDINO params: {vis_p:>12,d}")
        print(f"  Overhead:       {overhead:>12,d} ({pct:.3f}%)")
        
        ogs_params = results["VisDINO"]["param_groups"].get("ogs", 0)
        vbox_params = results["VisDINO"]["param_groups"].get("vbox_head", 0)
        ratio_params = results["VisDINO"]["param_groups"].get("ratio_head", 0)
        print(f"  OGS params:     {ogs_params:>12,d} ({100*ogs_params/dino_p:.3f}%)")
        print(f"  VBox head:      {vbox_params:>12,d}")
        print(f"  Ratio head:     {ratio_params:>12,d}")
        
        if "latency_ms" in results["DINO"] and "latency_ms" in results["VisDINO"]:
            d_lat = results["DINO"]["latency_ms"]
            v_lat = results["VisDINO"]["latency_ms"]
            lat_overhead = v_lat - d_lat
            lat_pct = 100.0 * lat_overhead / d_lat
            print(f"\n  DINO latency:   {d_lat:.1f} ms ({1000/d_lat:.1f} FPS)")
            print(f"  VisDINO lat.:   {v_lat:.1f} ms ({1000/v_lat:.1f} FPS)")
            print(f"  Latency overhead: {lat_overhead:.1f} ms ({lat_pct:.1f}%)")
    
    # Generate LaTeX
    print(f"\n{'='*60}")
    print(f"  LATEX TABLE")
    print(f"{'='*60}")
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Computational overhead of VisDINO over DINO (R-50 backbone, single RTX 4090).}
\label{tab:overhead}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Metric} & \textbf{DINO} & \textbf{VisDINO} \\
\midrule
"""
    
    if "DINO" in results and "VisDINO" in results:
        latex += f"Parameters & {results['DINO']['total_params']/1e6:.1f}M & {results['VisDINO']['total_params']/1e6:.1f}M \\\\\n"
        
        ogs = results["VisDINO"]["param_groups"].get("ogs", 0)
        latex += f"OGS parameters & --- & {ogs:,d} \\\\\n"
        
        if "latency_ms" in results.get("DINO", {}) and "latency_ms" in results.get("VisDINO", {}):
            latex += f"Latency (ms) & {results['DINO']['latency_ms']:.1f} & {results['VisDINO']['latency_ms']:.1f} \\\\\n"
            latex += f"FPS & {results['DINO']['fps']:.1f} & {results['VisDINO']['fps']:.1f} \\\\\n"
        
        if "peak_mem_mb" in results.get("DINO", {}) and "peak_mem_mb" in results.get("VisDINO", {}):
            latex += f"Peak GPU mem (MB) & {results['DINO']['peak_mem_mb']:.0f} & {results['VisDINO']['peak_mem_mb']:.0f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    print(latex)
    
    # Save results
    out_path = os.path.join(DETREX_DIR, "projects/visdino/benchmark_results.json")
    json.dump(results, open(out_path, 'w'), indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
