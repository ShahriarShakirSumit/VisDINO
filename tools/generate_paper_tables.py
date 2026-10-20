#!/usr/bin/env python3
"""
Generate LaTeX tables from experiment results for VisDINO paper.
Run this after all experiments complete and eval_all.sh has been run on each.

Usage:
    python generate_paper_tables.py > tables.tex
"""

import json
import os
import sys
from pathlib import Path


DETREX_DIR = Path(__file__).resolve().parent.parent.parent
OUTER_OUTPUT = DETREX_DIR.parent / "output"
INNER_OUTPUT = DETREX_DIR / "output"

# Experiment directories and their display names
EXPERIMENTS = {
    "DINO (180k)": {
        "dir": OUTER_OUTPUT / "dino_r50_crowdhuman_baseline",
        "is_baseline": True,
    },
    "DINO (300k)": {
        "dir": INNER_OUTPUT / "dino_r50_crowdhuman_continued_300k",
        "is_baseline": True,
    },
    "VisDINO": {
        "dir": INNER_OUTPUT / "visdino_r50_crowdhuman_300k_vbox",
        "is_ours": True,
    },
    "VisDINO (seed 2)": {
        "dir": INNER_OUTPUT / "visdino_r50_crowdhuman_300k_vbox_seed2",
        "is_ours": True,
    },
    "w/o OGS": {
        "dir": INNER_OUTPUT / "ablation_no_ogs_300k",
        "is_ablation": True,
    },
    "w/o $\\mathcal{L}_{\\text{point}}$": {
        "dir": INNER_OUTPUT / "ablation_no_lpoint_300k",
        "is_ablation": True,
    },
}


def load_metrics(exp_dir):
    """Load all available metrics from an experiment directory."""
    metrics = {}
    
    # COCO AP from log.txt
    log_file = exp_dir / "log.txt"
    if log_file.exists():
        with open(log_file) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "copypaste: AP,AP50,AP75,APs,APm,APl" in line:
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # Find the copypaste values line
                    for j in range(i + 1, min(i + 3, len(lines))):
                        if "copypaste:" in lines[j] and "," in lines[j]:
                            parts = lines[j].strip().split()
                            vals_str = parts[-1]
                            try:
                                vals = [float(v) for v in vals_str.split(",")]
                                if len(vals) == 6:
                                    metrics["AP"] = vals[0]
                                    metrics["AP50"] = vals[1]
                                    metrics["AP75"] = vals[2]
                            except:
                                pass
    
    # AP/MR from CrowdDet eval
    ap_mr_file = exp_dir / "ap_mr.json"
    if ap_mr_file.exists():
        data = json.load(open(ap_mr_file))
        metrics["MR-2"] = data.get("MR-2")
        metrics["AP@0.5_crowddet"] = data.get("AP@0.5")
    
    # JI
    ji_file = exp_dir / "ji.json"
    if ji_file.exists():
        data = json.load(open(ji_file))
        metrics["JI"] = data.get("JI")
    
    # Recall
    recall_file = exp_dir / "recall.json"
    if recall_file.exists():
        data = json.load(open(recall_file))
        metrics["Recall"] = data.get("Recall@0.5")
    
    # Per-occlusion
    occ_file = exp_dir / "per_occlusion.json"
    if occ_file.exists():
        data = json.load(open(occ_file))
        metrics["per_occlusion"] = data
    
    # Summary
    summary_file = exp_dir / "eval_summary.json"
    if summary_file.exists():
        data = json.load(open(summary_file))
        if "COCO_AP" in data:
            metrics.setdefault("AP", data["COCO_AP"].get("AP"))
            metrics.setdefault("AP50", data["COCO_AP"].get("AP50"))
            metrics.setdefault("AP75", data["COCO_AP"].get("AP75"))
    
    return metrics


def fmt(val, best_val=None, is_mr=False):
    """Format a value, bold if it's the best."""
    if val is None:
        return "---"
    s = f"{val:.2f}"
    if best_val is not None:
        if is_mr and val <= best_val:
            return f"\\best{{{s}}}"
        elif not is_mr and val >= best_val:
            return f"\\best{{{s}}}"
    return s


def generate_main_table(all_metrics):
    """Generate Table 1: Main CrowdHuman results."""
    print(r"""
%% ===== TABLE 1: Main CrowdHuman Results =====
\begin{table}[t]
\centering
\caption{CrowdHuman validation results (R-50 backbone). DINO (300k) provides a fair baseline with identical training budget. MR$^{-2}$: lower is better. All others: higher is better.}
\label{tab:crowdhuman_main}
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Method} & \textbf{AP} & \textbf{AP\textsubscript{75}} & \textbf{MR$^{-2}$} $\downarrow$ & \textbf{JI} & \textbf{Recall} \\
\midrule""")
    
    # Collect metrics for best comparison
    methods = ["DINO (180k)", "DINO (300k)", "VisDINO"]
    metrics_keys = ["AP", "AP75", "MR-2", "JI", "Recall"]
    
    # Find best values
    best = {}
    for key in metrics_keys:
        vals = [all_metrics.get(m, {}).get(key) for m in methods if all_metrics.get(m, {}).get(key) is not None]
        if vals:
            best[key] = min(vals) if key == "MR-2" else max(vals)
    
    for method in methods:
        m = all_metrics.get(method, {})
        row = [method]
        for key in metrics_keys:
            val = m.get(key)
            row.append(fmt(val, best.get(key), is_mr=(key == "MR-2")))
        print(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} \\\\")
        if method == "DINO (300k)":
            print(r"\midrule")
    
    # Delta row
    dino300 = all_metrics.get("DINO (300k)", {})
    visdino = all_metrics.get("VisDINO", {})
    if dino300 and visdino:
        print(r"\midrule")
        deltas = []
        for key in metrics_keys:
            d = dino300.get(key)
            v = visdino.get(key)
            if d is not None and v is not None:
                diff = v - d
                if key == "MR-2":
                    sign = "" if diff >= 0 else "$-$"
                    deltas.append(f"\\improve{{{sign}{abs(diff):.2f}}}" if diff < 0 else f"\\worse{{+{diff:.2f}}}")
                else:
                    sign = "+" if diff > 0 else "$-$"
                    deltas.append(f"\\improve{{{sign}{abs(diff):.2f}}}" if diff > 0 else f"\\worse{{{sign}{abs(diff):.2f}}}")
            else:
                deltas.append("---")
        print(f"$\\Delta$ (vs DINO 300k) & {' & '.join(deltas)} \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def generate_ablation_table(all_metrics):
    """Generate ablation study table."""
    print(r"""
%% ===== TABLE: Ablation Study =====
\begin{table}[t]
\centering
\caption{Ablation study on CrowdHuman (R-50 backbone, 300k iterations). Each row removes one component from the full VisDINO model. OGS: Occlusion Guidance Signal gating. $\mathcal{L}_{\text{point}}$: gate supervision loss.}
\label{tab:ablation}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Configuration} & \textbf{AP} & \textbf{MR$^{-2}$} $\downarrow$ & \textbf{JI} & \textbf{Recall} \\
\midrule""")
    
    methods = ["DINO (300k)", "w/o OGS", r"w/o $\mathcal{L}_{\text{point}}$", "VisDINO"]
    labels = {
        "DINO (300k)": "DINO baseline (300k)",
        "w/o OGS": "+ vbox/ratio heads (no OGS)",
        r"w/o $\mathcal{L}_{\text{point}}$": "+ OGS (no gate supervision)",
        "VisDINO": "VisDINO (full model)",
    }
    
    for method in methods:
        m = all_metrics.get(method, {})
        label = labels.get(method, method)
        ap = fmt(m.get("AP"))
        mr = fmt(m.get("MR-2"))
        ji = fmt(m.get("JI"))
        recall = fmt(m.get("Recall"))
        
        if method == "VisDINO":
            print(r"\midrule")
        print(f"{label} & {ap} & {mr} & {ji} & {recall} \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def generate_per_occlusion_table(all_metrics):
    """Generate per-occlusion comparison table."""
    print(r"""
%% ===== TABLE: Per-Occlusion Analysis =====
\begin{table}[t]
\centering
\caption{Per-occlusion recall (\%) and AP@0.5 on CrowdHuman. Occlusion bins: Heavy ($r_\text{vis} < 0.50$), Partial ($0.50 \leq r_\text{vis} < 0.85$), None ($r_\text{vis} \geq 0.85$).}
\label{tab:per_occ}
\begin{tabular}{@{}lcccccc@{}}
\toprule
& \multicolumn{3}{c}{\textbf{Recall}} & \multicolumn{3}{c}{\textbf{AP@0.5}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
\textbf{Method} & \textbf{Heavy} & \textbf{Partial} & \textbf{None} & \textbf{Heavy} & \textbf{Partial} & \textbf{None} \\
\midrule""")
    
    methods = ["DINO (300k)", "VisDINO"]
    
    for method in methods:
        m = all_metrics.get(method, {})
        occ = m.get("per_occlusion", {})
        
        h = occ.get("heavy", {})
        p = occ.get("partial", {})
        n = occ.get("none", {})
        
        hr = fmt(h.get("Recall") and h["Recall"] * 100)
        pr = fmt(p.get("Recall") and p["Recall"] * 100)
        nr = fmt(n.get("Recall") and n["Recall"] * 100)
        ha = fmt(h.get("AP") and h["AP"] * 100)
        pa = fmt(p.get("AP") and p["AP"] * 100)
        na = fmt(n.get("AP") and n["AP"] * 100)
        
        print(f"{method} & {hr} & {pr} & {nr} & {ha} & {pa} & {na} \\\\")
    
    # Delta
    dino = all_metrics.get("DINO (300k)", {}).get("per_occlusion", {})
    visd = all_metrics.get("VisDINO", {}).get("per_occlusion", {})
    if dino and visd:
        print(r"\midrule")
        parts = []
        for level in ["heavy", "partial", "none"]:
            for metric in ["Recall", "AP"]:
                d_val = dino.get(level, {}).get(metric)
                v_val = visd.get(level, {}).get(metric)
                if d_val is not None and v_val is not None:
                    diff = (v_val - d_val) * 100
                    sign = "+" if diff > 0 else "$-$"
                    parts.append(f"\\improve{{{sign}{abs(diff):.1f}}}" if diff > 0 else f"\\worse{{{sign}{abs(diff):.1f}}}")
                else:
                    parts.append("---")
        print(f"$\\Delta$ & {' & '.join(parts)} \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def generate_multi_seed_summary(all_metrics):
    """Generate multi-seed statistical summary if available."""
    visdino = all_metrics.get("VisDINO", {})
    visdino2 = all_metrics.get("VisDINO (seed 2)", {})
    
    if not visdino or not visdino2:
        print("% Multi-seed results not yet available")
        return
    
    import statistics
    
    print(r"""
%% ===== Multi-seed Statistical Summary =====
% Two seeds provide range rather than confidence intervals""")
    
    for key in ["AP", "MR-2", "JI"]:
        v1 = visdino.get(key)
        v2 = visdino2.get(key)
        if v1 is not None and v2 is not None:
            mean = (v1 + v2) / 2
            std = abs(v1 - v2) / 2
            print(f"% {key}: {mean:.2f} ± {std:.2f} (seed 1: {v1:.2f}, seed 2: {v2:.2f})")


def main():
    print("% Auto-generated tables from VisDINO experiments")
    print(f"% Generated by {os.path.basename(__file__)}")
    print()
    
    # Load all metrics
    all_metrics = {}
    for name, info in EXPERIMENTS.items():
        exp_dir = info["dir"]
        if exp_dir.exists():
            metrics = load_metrics(exp_dir)
            if metrics:
                all_metrics[name] = metrics
                print(f"% Loaded {name}: {list(metrics.keys())}")
            else:
                print(f"% {name}: no metrics found in {exp_dir}")
        else:
            print(f"% {name}: directory not found ({exp_dir})")
    
    print()
    
    # Generate tables
    generate_main_table(all_metrics)
    generate_ablation_table(all_metrics)
    generate_per_occlusion_table(all_metrics)
    generate_multi_seed_summary(all_metrics)


if __name__ == "__main__":
    main()
