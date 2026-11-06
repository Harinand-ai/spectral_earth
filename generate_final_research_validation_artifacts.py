"""
ConformAL-HSI: Final Research Validation Artifact Generator
Processes screening, confirmation, and diagnostic experimental results,
computes statistical aggregations, ANOVA / interaction effects,
generates master CSV tables, and produces 10 publication figures at 300 DPI.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['lines.markersize'] = 6

OUT_DIR = "results/final_research_validation"
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {
    "Random Sampling": "#7f7f7f",            # Neutral gray
    "Margin Sampling": "#1f77b4",            # Strong blue
    "Margin + Cosine Diversity": "#2ca02c",  # Forest green
    "Adaptive Class-Coverage AL": "#d62728", # Crimson red
}

HEAD_PALETTE = {
    "Head A (Original 256)": "#8c564b",
    "Head B (Improved 512)": "#1f77b4",
    "Head C (Wider 1024)": "#ff7f0e",
    "Head D (Deeper 1024-512-256)": "#9467bd",
}

HEAD_LABELS = {
    "head_a": "Head A (Original 256)",
    "head_b": "Head B (Improved 512)",
    "head_c": "Head C (Wider 1024)",
    "head_d": "Head D (Deeper 1024-512-256)",
}


def load_and_combine_data():
    """Load screening, confirmation, and diagnostic files."""
    # Find all classifier_head_seed_results_*.csv
    csv_files = glob.glob(os.path.join(OUT_DIR, "classifier_head_seed_results_*.csv"))
    dfs = []
    for f in csv_files:
        if "test_smoke" in f:
            continue
        df = pd.read_csv(f)
        dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError("No seed results CSVs found in " + OUT_DIR)
    
    combined = pd.concat(dfs, ignore_index=True)
    # Deduplicate on dataset, classifier_head, strategy, seed, budget, norm_type
    if "norm_type" not in combined.columns:
        combined["norm_type"] = "batchnorm"
    combined = combined.drop_duplicates(subset=["dataset", "classifier_head", "strategy", "seed", "budget", "norm_type"])
    
    # Save unified seed results
    out_master_seed = os.path.join(OUT_DIR, "classifier_head_seed_results.csv")
    combined.to_csv(out_master_seed, index=False)
    print(f"[Combined Seed Results] Saved {len(combined)} records to {out_master_seed}")
    return combined


def compute_aggregations(df_seeds):
    """Compute mean and std across seeds for each combination."""
    group_cols = ["dataset", "classifier_head", "classifier_code", "strategy", "strategy_code", "norm_type", "budget", "parameter_count"]
    metric_cols = ["oa", "aa", "kappa", "macro_f1", "ece", "val_ece", "train_loss", "train_acc", "train_time_s", "acq_time_s", "inf_time_s", "peak_vram_mb"]
    
    agg_dict = {}
    for c in metric_cols:
        agg_dict[f"{c}_mean"] = (c, "mean")
        agg_dict[f"{c}_std"] = (c, "std")
    agg_dict["n_seeds"] = ("seed", "count")
    
    agg = df_seeds.groupby(group_cols).agg(**agg_dict).reset_index()
    
    # Round metrics appropriately
    for c in metric_cols:
        agg[f"{c}_mean"] = agg[f"{c}_mean"].round(4)
        agg[f"{c}_std"] = agg[f"{c}_std"].round(4)
        
    out_agg = os.path.join(OUT_DIR, "classifier_head_aggregated_results.csv")
    agg.to_csv(out_agg, index=False)
    print(f"[Aggregated Results] Saved {len(agg)} configurations to {out_agg}")
    return agg


def compute_strategy_rankings(agg_df):
    """Rank strategies 1 to 4 within each (dataset, classifier_head, norm_type, budget)."""
    rank_rows = []
    groups = agg_df.groupby(["dataset", "classifier_head", "norm_type", "budget"])
    for (ds, head, norm, b), grp in groups:
        # Rank by OA descending
        grp_sorted_oa = grp.sort_values(by="oa_mean", ascending=False).reset_index(drop=True)
        grp_sorted_aa = grp.sort_values(by="aa_mean", ascending=False).reset_index(drop=True)
        grp_sorted_f1 = grp.sort_values(by="macro_f1_mean", ascending=False).reset_index(drop=True)
        
        oa_ranks = {row["strategy"]: r + 1 for r, row in grp_sorted_oa.iterrows()}
        aa_ranks = {row["strategy"]: r + 1 for r, row in grp_sorted_aa.iterrows()}
        f1_ranks = {row["strategy"]: r + 1 for r, row in grp_sorted_f1.iterrows()}
        
        for _, row in grp.iterrows():
            strat = row["strategy"]
            rank_rows.append({
                "dataset": ds,
                "classifier_head": head,
                "norm_type": norm,
                "budget": b,
                "strategy": strat,
                "oa_mean": row["oa_mean"],
                "oa_std": row["oa_std"],
                "oa_rank": oa_ranks[strat],
                "aa_mean": row["aa_mean"],
                "aa_rank": aa_ranks[strat],
                "macro_f1_mean": row["macro_f1_mean"],
                "macro_f1_rank": f1_ranks[strat],
            })
            
    rank_df = pd.DataFrame(rank_rows)
    out_rank = os.path.join(OUT_DIR, "classifier_strategy_rankings.csv")
    rank_df.to_csv(out_rank, index=False)
    print(f"[Strategy Rankings] Saved {len(rank_df)} rank entries to {out_rank}")
    return rank_df


def compute_interaction_effects(agg_df, df_seeds):
    """
    Compute interaction effects:
    1. Delta OA over Random for each strategy across classifier heads.
    2. Two-way ANOVA (Head x Strategy) on OA for each dataset.
    """
    # Only analyze batchnorm norm_type for the main interaction study
    agg_bn = agg_df[agg_df["norm_type"] == "batchnorm"].copy()
    
    inter_rows = []
    groups = agg_bn.groupby(["dataset", "classifier_head", "budget"])
    for (ds, head, b), grp in groups:
        random_row = grp[grp["strategy"] == "Random Sampling"]
        if random_row.empty:
            continue
        rnd_oa = random_row["oa_mean"].values[0]
        rnd_f1 = random_row["macro_f1_mean"].values[0]
        
        for _, row in grp.iterrows():
            strat = row["strategy"]
            delta_oa = row["oa_mean"] - rnd_oa
            rel_gain_oa = (delta_oa / rnd_oa * 100.0) if rnd_oa > 0 else 0.0
            delta_f1 = row["macro_f1_mean"] - rnd_f1
            inter_rows.append({
                "dataset": ds,
                "classifier_head": head,
                "budget": b,
                "strategy": strat,
                "oa_mean": row["oa_mean"],
                "random_oa_mean": rnd_oa,
                "delta_oa_vs_random": round(delta_oa, 3),
                "rel_gain_pct": round(rel_gain_oa, 2),
                "macro_f1_mean": row["macro_f1_mean"],
                "random_f1_mean": rnd_f1,
                "delta_f1_vs_random": round(delta_f1, 4),
            })
            
    inter_df = pd.DataFrame(inter_rows)
    out_inter = os.path.join(OUT_DIR, "classifier_interaction_results.csv")
    inter_df.to_csv(out_inter, index=False)
    print(f"[Interaction Effects] Saved {len(inter_df)} interaction entries to {out_inter}")
    
    # Compute Two-Way ANOVA stats
    import scipy.stats as stats
    print("\n--- Two-Way ANOVA Summary (Classifier Head x Acquisition Strategy) ---")
    seeds_bn = df_seeds[df_seeds["norm_type"] == "batchnorm"]
    for ds in ["Indian Pines", "Salinas", "Pavia University"]:
        sub = seeds_bn[seeds_bn["dataset"] == ds]
        heads = sub["classifier_head"].unique()
        strats = sub["strategy"].unique()
        
        # Grand mean
        grand_mean = sub["oa"].mean()
        N = len(sub)
        
        # Head effect
        ss_head = sum(len(sub[sub["classifier_head"] == h]) * ((sub[sub["classifier_head"] == h]["oa"].mean() - grand_mean) ** 2) for h in heads)
        df_head = len(heads) - 1
        
        # Strategy effect
        ss_strat = sum(len(sub[sub["strategy"] == s]) * ((sub[sub["strategy"] == s]["oa"].mean() - grand_mean) ** 2) for s in strats)
        df_strat = len(strats) - 1
        
        # Cell means for interaction
        ss_cell = 0.0
        for h in heads:
            for s in strats:
                cell_sub = sub[(sub["classifier_head"] == h) & (sub["strategy"] == s)]
                if len(cell_sub) > 0:
                    ss_cell += len(cell_sub) * ((cell_sub["oa"].mean() - grand_mean) ** 2)
                    
        ss_inter = ss_cell - ss_head - ss_strat
        df_inter = df_head * df_strat
        
        # Residual error
        ss_error = 0.0
        for h in heads:
            for s in strats:
                cell_sub = sub[(sub["classifier_head"] == h) & (sub["strategy"] == s)]
                if len(cell_sub) > 0:
                    cell_mean = cell_sub["oa"].mean()
                    ss_error += sum((x - cell_mean) ** 2 for x in cell_sub["oa"])
        df_error = N - (len(heads) * len(strats))
        
        ms_head = ss_head / max(1, df_head)
        ms_strat = ss_strat / max(1, df_strat)
        ms_inter = ss_inter / max(1, df_inter)
        ms_error = ss_error / max(1, df_error)
        
        f_head = ms_head / max(1e-8, ms_error)
        f_strat = ms_strat / max(1e-8, ms_error)
        f_inter = ms_inter / max(1e-8, ms_error)
        
        p_head = 1.0 - stats.f.cdf(f_head, df_head, df_error)
        p_strat = 1.0 - stats.f.cdf(f_strat, df_strat, df_error)
        p_inter = 1.0 - stats.f.cdf(f_inter, df_inter, df_error)
        
        print(f"Dataset: {ds} (N={N})")
        print(f"  Head Main Effect:      F({df_head}, {df_error}) = {f_head:8.2f}, p = {p_head:.4e}")
        print(f"  Strategy Main Effect:  F({df_strat}, {df_error}) = {f_strat:8.2f}, p = {p_strat:.4e}")
        print(f"  Head x Strat Inter:    F({df_inter}, {df_error}) = {f_inter:8.2f}, p = {p_inter:.4e}")
        
    return inter_df


def compute_batchnorm_diagnostics(agg_df, df_seeds):
    """
    Compare BatchNorm vs LayerNorm across heads, specifically focusing on
    budgets 220 vs 260 and testing whether residual batch size (4 samples)
    explains non-monotonic dips.
    """
    # Look for both batchnorm and layernorm records
    bn_ln = agg_df[agg_df["norm_type"].isin(["batchnorm", "layernorm"])].copy()
    
    diag_rows = []
    # For heads that have both norms (head_b, head_c, head_d)
    for ds in ["Indian Pines", "Salinas", "Pavia University"]:
        for head in ["Head B (Improved 512)", "Head C (Wider 1024)", "Head D (Deeper 1024-512-256)"]:
            for strat in ["Random Sampling", "Margin Sampling", "Margin + Cosine Diversity", "Adaptive Class-Coverage AL"]:
                for norm in ["batchnorm", "layernorm"]:
                    sub = bn_ln[(bn_ln["dataset"] == ds) & (bn_ln["classifier_head"] == head) & 
                                (bn_ln["strategy"] == strat) & (bn_ln["norm_type"] == norm)]
                    
                    sub_220 = sub[sub["budget"] == 220]
                    sub_260 = sub[sub["budget"] == 260]
                    
                    if not sub_220.empty and not sub_260.empty:
                        oa_220 = sub_220["oa_mean"].values[0]
                        std_220 = sub_220["oa_std"].values[0]
                        oa_260 = sub_260["oa_mean"].values[0]
                        std_260 = sub_260["oa_std"].values[0]
                        delta_260_220 = oa_260 - oa_220
                        
                        diag_rows.append({
                            "dataset": ds,
                            "classifier_head": head,
                            "strategy": strat,
                            "norm_type": norm,
                            "oa_budget_220": oa_220,
                            "oa_std_220": std_220,
                            "oa_budget_260": oa_260,
                            "oa_std_260": std_260,
                            "delta_260_minus_220": round(delta_260_220, 3),
                            "residual_batch_samples_220": 28,  # 220 % 64
                            "residual_batch_samples_260": 4,   # 260 % 64
                            "dip_observed": delta_260_220 < 0,
                        })
                        
    diag_df = pd.DataFrame(diag_rows)
    out_diag = os.path.join(OUT_DIR, "batchnorm_diagnostic_results.csv")
    diag_df.to_csv(out_diag, index=False)
    print(f"[BatchNorm Diagnostic] Saved {len(diag_df)} diagnostic entries to {out_diag}")
    return diag_df


def compute_final_label_efficiency_thresholds(agg_df):
    """
    Computes exact discrete label savings for target thresholds:
    tau in [50%, 55%, 60%, 65%, 66%, 70%, 75%]
    """
    thresholds = [50.0, 55.0, 60.0, 65.0, 66.0, 70.0, 75.0]
    budgets = [100, 140, 180, 220, 260]
    
    thresh_rows = []
    # Primary evaluation on Head B (Improved 512) and other heads
    for ds in ["Indian Pines", "Salinas", "Pavia University"]:
        for head in ["Head A (Original 256)", "Head B (Improved 512)", "Head C (Wider 1024)", "Head D (Deeper 1024-512-256)"]:
            sub_head = agg_df[(agg_df["dataset"] == ds) & (agg_df["classifier_head"] == head) & (agg_df["norm_type"] == "batchnorm")]
            if sub_head.empty:
                continue
            
            # Find random threshold budgets
            rnd_sub = sub_head[sub_head["strategy"] == "Random Sampling"].sort_values(by="budget")
            
            for tau in thresholds:
                # Find minimum budget where random reaches tau
                rnd_hit = rnd_sub[rnd_sub["oa_mean"] >= tau]
                rnd_b = int(rnd_hit["budget"].iloc[0]) if not rnd_hit.empty else None
                rnd_oa = float(rnd_hit["oa_mean"].iloc[0]) if not rnd_hit.empty else None
                
                for strat in ["Random Sampling", "Margin Sampling", "Margin + Cosine Diversity", "Adaptive Class-Coverage AL"]:
                    strat_sub = sub_head[sub_head["strategy"] == strat].sort_values(by="budget")
                    strat_hit = strat_sub[strat_sub["oa_mean"] >= tau]
                    strat_b = int(strat_hit["budget"].iloc[0]) if not strat_hit.empty else None
                    strat_oa = float(strat_hit["oa_mean"].iloc[0]) if not strat_hit.empty else None
                    
                    # Savings calculation
                    if rnd_b is not None and strat_b is not None:
                        savings_pct = round(((rnd_b - strat_b) / rnd_b) * 100.0, 2)
                        labels_saved = rnd_b - strat_b
                        status = "Achieved" if strat_b <= rnd_b else "Random Superior"
                    elif strat_b is not None and rnd_b is None:
                        savings_pct = "> 0.0"
                        labels_saved = f"> {260 - strat_b}"
                        status = "AL Solely Achieved"
                    elif strat_b is None and rnd_b is not None:
                        savings_pct = "< 0.0"
                        labels_saved = f"< -{260 - rnd_b}"
                        status = "Random Solely Achieved"
                    else:
                        savings_pct = "N/A"
                        labels_saved = "N/A"
                        status = "Neither Reached"
                        
                    thresh_rows.append({
                        "dataset": ds,
                        "classifier_head": head,
                        "strategy": strat,
                        "target_threshold_oa": tau,
                        "budget_required": strat_b if strat_b is not None else ">260",
                        "actual_oa_at_budget": round(strat_oa, 2) if strat_oa is not None else (round(strat_sub["oa_mean"].max(), 2) if not strat_sub.empty else 0.0),
                        "random_budget_required": rnd_b if rnd_b is not None else ">260",
                        "random_actual_oa": round(rnd_oa, 2) if rnd_oa is not None else (round(rnd_sub["oa_mean"].max(), 2) if not rnd_sub.empty else 0.0),
                        "label_savings_pct": savings_pct,
                        "labels_saved": labels_saved,
                        "status": status,
                    })
                    
    thresh_df = pd.DataFrame(thresh_rows)
    out_thresh = os.path.join(OUT_DIR, "final_label_efficiency_thresholds.csv")
    thresh_df.to_csv(out_thresh, index=False)
    print(f"[Threshold Results] Saved {len(thresh_df)} threshold entries to {out_thresh}")
    return thresh_df


def compile_per_class_and_runtime(df_seeds):
    """Aggregate per-class metrics and runtime/memory records."""
    # Find per-class CSVs
    pc_files = glob.glob(os.path.join(OUT_DIR, "classifier_head_per_class_*.csv"))
    if pc_files:
        pc_dfs = [pd.read_csv(f) for f in pc_files if "test_smoke" not in f]
        if pc_dfs:
            combined_pc = pd.concat(pc_dfs, ignore_index=True)
            combined_pc = combined_pc.drop_duplicates()
            out_pc = os.path.join(OUT_DIR, "final_per_class_metrics.csv")
            combined_pc.to_csv(out_pc, index=False)
            print(f"[Per-Class Metrics] Saved {len(combined_pc)} rows to {out_pc}")
            
    # Find runtime memory CSVs
    rt_files = glob.glob(os.path.join(OUT_DIR, "classifier_head_runtime_memory_*.csv"))
    if rt_files:
        rt_dfs = [pd.read_csv(f) for f in rt_files if "test_smoke" not in f]
        if rt_dfs:
            combined_rt = pd.concat(rt_dfs, ignore_index=True)
            combined_rt = combined_rt.drop_duplicates()
            
            # Aggregate by dataset, head, strategy, budget
            rt_agg = combined_rt.groupby(["dataset", "classifier_head", "strategy", "budget"]).agg(
                param_count=("parameter_count", "first"),
                train_time_mean=("train_time_s", "mean"),
                train_time_std=("train_time_s", "std"),
                acq_time_mean=("acq_time_s", "mean"),
                inf_time_mean=("inf_time_s", "mean"),
                peak_vram_mean=("peak_vram_mb", "mean"),
                peak_vram_max=("peak_vram_mb", "max")
            ).reset_index()
            
            for c in ["train_time_mean", "train_time_std", "acq_time_mean", "inf_time_mean", "peak_vram_mean", "peak_vram_max"]:
                rt_agg[c] = rt_agg[c].round(3)
                
            out_rt = os.path.join(OUT_DIR, "final_runtime_memory_results.csv")
            rt_agg.to_csv(out_rt, index=False)
            print(f"[Runtime & Memory Results] Saved {len(rt_agg)} rows to {out_rt}")


# ==============================================================================
# PUBLICATION FIGURES (10 FIGURES AT 300 DPI)
# ==============================================================================

def plot_fig1_oa_vs_budget_by_classifier(agg_df):
    """Figure 1: OA vs Budget by Classifier Head across the 3 Datasets."""
    datasets = ["Indian Pines", "Salinas", "Pavia University"]
    heads = ["Head A (Original 256)", "Head B (Improved 512)", "Head C (Wider 1024)", "Head D (Deeper 1024-512-256)"]
    strats = ["Random Sampling", "Margin Sampling", "Margin + Cosine Diversity", "Adaptive Class-Coverage AL"]
    markers = {"Random Sampling": "o", "Margin Sampling": "s", "Margin + Cosine Diversity": "^", "Adaptive Class-Coverage AL": "D"}

    fig, axes = plt.subplots(3, 4, figsize=(20, 14), sharex=True)
    plt.subplots_adjust(hspace=0.25, wspace=0.2)

    for i, ds in enumerate(datasets):
        for j, head in enumerate(heads):
            ax = axes[i, j]
            sub = agg_df[(agg_df["dataset"] == ds) & (agg_df["classifier_head"] == head) & (agg_df["norm_type"] == "batchnorm")]
            
            for strat in strats:
                s_sub = sub[sub["strategy"] == strat].sort_values(by="budget")
                if not s_sub.empty:
                    ax.plot(s_sub["budget"], s_sub["oa_mean"], marker=markers[strat], color=PALETTE[strat],
                            label=strat, linewidth=2, markersize=6)
                    ax.fill_between(s_sub["budget"], 
                                     s_sub["oa_mean"] - s_sub["oa_std"], 
                                     s_sub["oa_mean"] + s_sub["oa_std"], 
                                     color=PALETTE[strat], alpha=0.15)
            
            ax.set_title(f"{head}\n[{ds}]", fontsize=11, fontweight="bold")
            ax.set_xticks([100, 140, 180, 220, 260])
            ax.grid(True, linestyle="--", alpha=0.6)
            if j == 0:
                ax.set_ylabel(f"{ds}\nOverall Accuracy (%)", fontweight="bold", fontsize=11)
            if i == 2:
                ax.set_xlabel("Label Budget (N)", fontweight="bold", fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.99), frameon=True, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_path = os.path.join(OUT_DIR, "oa_vs_budget_by_classifier.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Figure 1 Generated] {out_path}")


def plot_fig2_aa_vs_budget_by_classifier(agg_df):
    """Figure 2: Average Accuracy (AA) vs Budget by Classifier Head."""
    datasets = ["Indian Pines", "Salinas", "Pavia University"]
    heads = ["Head A (Original 256)", "Head B (Improved 512)", "Head C (Wider 1024)", "Head D (Deeper 1024-512-256)"]
    strats = ["Random Sampling", "Margin Sampling", "Margin + Cosine Diversity", "Adaptive Class-Coverage AL"]
    markers = {"Random Sampling": "o", "Margin Sampling": "s", "Margin + Cosine Diversity": "^", "Adaptive Class-Coverage AL": "D"}

    fig, axes = plt.subplots(3, 4, figsize=(20, 14), sharex=True)
    plt.subplots_adjust(hspace=0.25, wspace=0.2)

    for i, ds in enumerate(datasets):
        for j, head in enumerate(heads):
            ax = axes[i, j]
            sub = agg_df[(agg_df["dataset"] == ds) & (agg_df["classifier_head"] == head) & (agg_df["norm_type"] == "batchnorm")]
            
            for strat in strats:
                s_sub = sub[sub["strategy"] == strat].sort_values(by="budget")
                if not s_sub.empty:
                    ax.plot(s_sub["budget"], s_sub["aa_mean"], marker=markers[strat], color=PALETTE[strat],
                            label=strat, linewidth=2, markersize=6)
                    ax.fill_between(s_sub["budget"], 
                                     s_sub["aa_mean"] - s_sub["aa_std"], 
                                     s_sub["aa_mean"] + s_sub["aa_std"], 
                                     color=PALETTE[strat], alpha=0.15)
            
            ax.set_title(f"{head}\n[{ds}]", fontsize=11, fontweight="bold")
            ax.set_xticks([100, 140, 180, 220, 260])
            ax.grid(True, linestyle="--", alpha=0.6)
            if j == 0:
                ax.set_ylabel(f"{ds}\nAverage Accuracy (%)", fontweight="bold", fontsize=11)
            if i == 2:
                ax.set_xlabel("Label Budget (N)", fontweight="bold", fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.99), frameon=True, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_path = os.path.join(OUT_DIR, "aa_vs_budget_by_classifier.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Figure 2 Generated] {out_path}")


def plot_fig3_macro_f1_vs_budget_by_classifier(agg_df):
    """Figure 3: Macro-F1 vs Budget by Classifier Head."""
    datasets = ["Indian Pines", "Salinas", "Pavia University"]
    heads = ["Head A (Original 256)", "Head B (Improved 512)", "Head C (Wider 1024)", "Head D (Deeper 1024-512-256)"]
    strats = ["Random Sampling", "Margin Sampling", "Margin + Cosine Diversity", "Adaptive Class-Coverage AL"]
    markers = {"Random Sampling": "o", "Margin Sampling": "s", "Margin + Cosine Diversity": "^", "Adaptive Class-Coverage AL": "D"}

    fig, axes = plt.subplots(3, 4, figsize=(20, 14), sharex=True)
    plt.subplots_adjust(hspace=0.25, wspace=0.2)

    for i, ds in enumerate(datasets):
        for j, head in enumerate(heads):
            ax = axes[i, j]
            sub = agg_df[(agg_df["dataset"] == ds) & (agg_df["classifier_head"] == head) & (agg_df["norm_type"] == "batchnorm")]
            
            for strat in strats:
                s_sub = sub[sub["strategy"] == strat].sort_values(by="budget")
                if not s_sub.empty:
                    ax.plot(s_sub["budget"], s_sub["macro_f1_mean"], marker=markers[strat], color=PALETTE[strat],
                            label=strat, linewidth=2, markersize=6)
            
            ax.set_title(f"{head}\n[{ds}]", fontsize=11, fontweight="bold")
            ax.set_xticks([100, 140, 180, 220, 260])
            ax.grid(True, linestyle="--", alpha=0.6)
            if j == 0:
                ax.set_ylabel(f"{ds}\nMacro-F1 Score", fontweight="bold", fontsize=11)
            if i == 2:
                ax.set_xlabel("Label Budget (N)", fontweight="bold", fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.99), frameon=True, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_path = os.path.join(OUT_DIR, "macro_f1_vs_budget_by_classifier.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Figure 3 Generated] {out_path}")


def plot_fig4_strategy_ranking_across_heads(rank_df):
    """Figure 4: Average Rank of Each Strategy across Classifier Architectures."""
    # Compute mean rank across budgets for each (dataset, head, strategy)
    rank_bn = rank_df[rank_df["norm_type"] == "batchnorm"]
    mean_ranks = rank_bn.groupby(["dataset", "classifier_head", "strategy"])["oa_rank"].mean().reset_index()
    
    datasets = ["Indian Pines", "Salinas", "Pavia University"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    heads = ["Head A (Original 256)", "Head B (Improved 512)", "Head C (Wider 1024)", "Head D (Deeper 1024-512-256)"]
    strats = ["Margin Sampling", "Margin + Cosine Diversity", "Adaptive Class-Coverage AL", "Random Sampling"]
    
    bar_width = 0.2
    x = np.arange(len(heads))
    
    for i, ds in enumerate(datasets):
        ax = axes[i]
        ds_ranks = mean_ranks[mean_ranks["dataset"] == ds]
        
        for k, strat in enumerate(strats):
            vals = []
            for h in heads:
                val = ds_ranks[(ds_ranks["classifier_head"] == h) & (ds_ranks["strategy"] == strat)]["oa_rank"].values
                vals.append(val[0] if len(val) > 0 else 4.0)
            ax.bar(x + k * bar_width, vals, width=bar_width, label=strat, color=PALETTE[strat], edgecolor="black", alpha=0.9)
            
        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels(["Head A\n(0.53M)", "Head B\n(1.06M)", "Head C\n(2.12M)", "Head D\n(2.76M)"], fontsize=10)
        ax.set_title(f"Strategy Ranking ({ds})", fontweight="bold", fontsize=12)
        ax.set_ylabel("Average Rank (Lower is Better)", fontweight="bold")
        ax.set_ylim(0.8, 4.3)
        ax.set_yticks([1, 2, 3, 4])
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)
            
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "strategy_ranking_across_heads.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Figure 4 Generated] {out_path}")


def plot_fig5_classifier_capacity_vs_oa(agg_df):
    """Figure 5: Parameter Count vs OA at Budget N=220."""
    agg_bn = agg_df[(agg_df["norm_type"] == "batchnorm") & (agg_df["budget"] == 220)].copy()
    datasets = ["Indian Pines", "Salinas", "Pavia University"]
    strats = ["Random Sampling", "Margin Sampling", "Margin + Cosine Diversity", "Adaptive Class-Coverage AL"]
    markers = {"Random Sampling": "o", "Margin Sampling": "s", "Margin + Cosine Diversity": "^", "Adaptive Class-Coverage AL": "D"}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    for i, ds in enumerate(datasets):
        ax = axes[i]
        ds_sub = agg_bn[agg_bn["dataset"] == ds]
        
        for strat in strats:
            s_sub = ds_sub[ds_sub["strategy"] == strat].sort_values(by="parameter_count")
            param_m = s_sub["parameter_count"] / 1e6
            ax.plot(param_m, s_sub["oa_mean"], marker=markers[strat], color=PALETTE[strat],
                    label=strat, linewidth=2.2, markersize=8)
            ax.fill_between(param_m, s_sub["oa_mean"] - s_sub["oa_std"], s_sub["oa_mean"] + s_sub["oa_std"],
                            color=PALETTE[strat], alpha=0.12)
            
        ax.set_title(f"Classifier Capacity vs. OA (N=220)\n[{ds}]", fontweight="bold", fontsize=12)
        ax.set_xlabel("Classifier Parameters (Millions)", fontweight="bold")
        ax.set_ylabel("Overall Accuracy (%)", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        if i == 0:
            ax.legend(loc="lower right", fontsize=10)
            
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "classifier_capacity_vs_oa.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Figure 5 Generated] {out_path}")


def plot_fig6_classifier_acquisition_interaction(inter_df):
    """Figure 6: Interaction Effect Delta OA (Strategy minus Random) across Heads."""
    datasets = ["Indian Pines", "Salinas", "Pavia University"]
    heads = ["Head A (Original 256)", "Head B (Improved 512)", "Head C (Wider 1024)", "Head D (Deeper 1024-512-256)"]
    strats = ["Margin Sampling", "Margin + Cosine Diversity", "Adaptive Class-Coverage AL"]
    
    # Average delta OA across budgets 140, 180, 220 for each (dataset, head, strategy)
    sub = inter_df[inter_df["budget"].isin([140, 180, 220])]
    mean_delta = sub.groupby(["dataset", "classifier_head", "strategy"])["delta_oa_vs_random"].mean().reset_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    bar_width = 0.25
    x = np.arange(len(heads))
    
    for i, ds in enumerate(datasets):
        ax = axes[i]
        ds_delta = mean_delta[mean_delta["dataset"] == ds]
        
        for k, strat in enumerate(strats):
            vals = []
            for h in heads:
                val = ds_delta[(ds_delta["classifier_head"] == h) & (ds_delta["strategy"] == strat)]["delta_oa_vs_random"].values
                vals.append(val[0] if len(val) > 0 else 0.0)
            ax.bar(x + k * bar_width, vals, width=bar_width, label=strat, color=PALETTE[strat], edgecolor="black", alpha=0.9)
            
        ax.axhline(0, color="black", linestyle="-", linewidth=1.0)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(["Head A\n(0.53M)", "Head B\n(1.06M)", "Head C\n(2.12M)", "Head D\n(2.76M)"], fontsize=10)
        ax.set_title(f"AL Advantage over Random (ΔOA)\n[{ds}]", fontweight="bold", fontsize=12)
        ax.set_ylabel("Mean ΔOA over Random (% points)", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        if i == 0:
            ax.legend(loc="upper right", fontsize=10)
            
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "classifier_acquisition_interaction.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Figure 6 Generated] {out_path}")


def plot_fig7_per_class_recall(pc_df=None):
    """Figure 7: Per-Class Recall comparisons for tail and dominant classes."""
    if pc_df is None or pc_df.empty:
        # Load from file if available
        pc_file = os.path.join(OUT_DIR, "final_per_class_metrics.csv")
        if os.path.exists(pc_file):
            pc_df = pd.read_csv(pc_file)
        else:
            print("[Warning] Per-class CSV not found, skipping Fig 7")
            return

    # Focus on Indian Pines at Budget N=220 on Head B
    sub = pc_df[(pc_df["dataset"] == "Indian Pines") & 
                (pc_df["classifier_head"].str.contains("Head B")) & 
                (pc_df["budget"] == 220)]
    
    if sub.empty:
        # Try generic head_b match
        sub = pc_df[(pc_df["dataset"] == "Indian Pines") & (pc_df["budget"] == 220)]
        
    if sub.empty:
        print("[Warning] No matching rows for per-class Fig 7")
        return

    # Average recall across seeds
    rec_agg = sub.groupby(["strategy", "class_id"])["recall"].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    classes = sorted(rec_agg["class_id"].unique())
    x = np.arange(len(classes))
    strats = ["Random Sampling", "Margin Sampling", "Margin + Cosine Diversity", "Adaptive Class-Coverage AL"]
    bar_width = 0.2
    
    for k, strat in enumerate(strats):
        s_data = rec_agg[rec_agg["strategy"] == strat]
        vals = [s_data[s_data["class_id"] == c]["recall"].values[0] if len(s_data[s_data["class_id"] == c]) > 0 else 0.0 for c in classes]
        ax.bar(x + k * bar_width, vals, width=bar_width, label=strat, color=PALETTE[strat], edgecolor="black", alpha=0.85)
        
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels([f"C{c+1}" for c in classes], fontsize=10)
    ax.set_xlabel("Indian Pines Classes (C1 to C16)", fontweight="bold")
    ax.set_ylabel("Test Recall (%)", fontweight="bold")
    ax.set_title("Per-Class Test Recall on Indian Pines at N=220 (Head B - Improved 512)", fontweight="bold", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "per_class_recall_comparisons.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Figure 7 Generated] {out_path}")


def plot_fig8_batchnorm_vs_layernorm_diagnostic(diag_df):
    """Figure 8: BatchNorm vs. LayerNorm Diagnostic on N=220 vs N=260."""
    if diag_df.empty:
        print("[Warning] Diagnostic dataframe empty, skipping Fig 8")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    datasets = ["Indian Pines", "Salinas", "Pavia University"]
    
    for i, ds in enumerate(datasets):
        ax = axes[i]
        ds_sub = diag_df[diag_df["dataset"] == ds]
        
        # Compare delta (260 - 220) between batchnorm and layernorm across heads
        agg_delta = ds_sub.groupby(["classifier_head", "norm_type"])["delta_260_minus_220"].mean().reset_index()
        heads = ["Head B (Improved 512)", "Head C (Wider 1024)", "Head D (Deeper 1024-512-256)"]
        x = np.arange(len(heads))
        bar_width = 0.35
        
        bn_vals = [agg_delta[(agg_delta["classifier_head"] == h) & (agg_delta["norm_type"] == "batchnorm")]["delta_260_minus_220"].values[0] 
                   if len(agg_delta[(agg_delta["classifier_head"] == h) & (agg_delta["norm_type"] == "batchnorm")]) > 0 else 0.0 for h in heads]
        ln_vals = [agg_delta[(agg_delta["classifier_head"] == h) & (agg_delta["norm_type"] == "layernorm")]["delta_260_minus_220"].values[0] 
                   if len(agg_delta[(agg_delta["classifier_head"] == h) & (agg_delta["norm_type"] == "layernorm")]) > 0 else 0.0 for h in heads]
        
        ax.bar(x - bar_width/2, bn_vals, width=bar_width, label="BatchNorm (Residual 4)", color="#d62728", edgecolor="black", alpha=0.85)
        ax.bar(x + bar_width/2, ln_vals, width=bar_width, label="LayerNorm (Sample-wise)", color="#2ca02c", edgecolor="black", alpha=0.85)
        
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(["Head B (512)", "Head C (1024)", "Head D (Deeper)"], fontsize=10)
        ax.set_ylabel("ΔOA: Budget 260 minus Budget 220 (% points)", fontweight="bold")
        ax.set_title(f"Residual Mini-Batch Effect ({ds})\nBudget 220 (res=28) vs Budget 260 (res=4)", fontweight="bold", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        if i == 0:
            ax.legend(loc="upper left", fontsize=10)
            
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "batchnorm_vs_layernorm_diagnostic.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Figure 8 Generated] {out_path}")


def plot_fig9_label_efficiency_thresholds(thresh_df):
    """Figure 9: Discrete Label Budget required to reach target accuracies."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    datasets = ["Indian Pines", "Salinas", "Pavia University"]
    strats = ["Random Sampling", "Margin Sampling", "Margin + Cosine Diversity", "Adaptive Class-Coverage AL"]
    markers = {"Random Sampling": "o", "Margin Sampling": "s", "Margin + Cosine Diversity": "^", "Adaptive Class-Coverage AL": "D"}
    
    for i, ds in enumerate(datasets):
        ax = axes[i]
        # Filter for Head B (Improved 512)
        ds_sub = thresh_df[(thresh_df["dataset"] == ds) & (thresh_df["classifier_head"].str.contains("Head B"))]
        
        for strat in strats:
            s_data = ds_sub[ds_sub["strategy"] == strat].copy()
            # Convert budget_required to numeric, replacing '>260' with 300 for plotting
            s_data["b_num"] = s_data["budget_required"].apply(lambda v: 300 if str(v).startswith(">") else float(v))
            s_data = s_data.sort_values(by="target_threshold_oa")
            ax.plot(s_data["target_threshold_oa"], s_data["b_num"], marker=markers[strat], color=PALETTE[strat],
                    label=strat, linewidth=2.2, markersize=8)
            
        ax.set_title(f"Label Budget vs. Target Threshold\n[{ds} | Head B]", fontweight="bold", fontsize=12)
        ax.set_xlabel("Target Overall Accuracy Threshold (%)", fontweight="bold")
        ax.set_ylabel("Discrete Budget Required (N)", fontweight="bold")
        ax.set_yticks([100, 140, 180, 220, 260, 300])
        ax.set_yticklabels(["100", "140", "180", "220", "260", ">260"], fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        if i == 0:
            ax.legend(loc="upper left", fontsize=10)
            
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "label_efficiency_threshold_comparisons.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Figure 9 Generated] {out_path}")


def plot_fig10_runtime_memory_comparison(agg_df):
    """Figure 10: Training Runtime and Peak VRAM across Classifier Architectures."""
    # Head-level runtime and memory at budget N=220
    sub = agg_df[(agg_df["budget"] == 220) & (agg_df["norm_type"] == "batchnorm")]
    head_stats = sub.groupby("classifier_head").agg(
        param_count=("parameter_count", "first"),
        train_time=("train_time_s_mean", "mean"),
        peak_vram=("peak_vram_mb_mean", "mean")
    ).reset_index()
    
    heads = ["Head A (Original 256)", "Head B (Improved 512)", "Head C (Wider 1024)", "Head D (Deeper 1024-512-256)"]
    labels = ["Head A\n(256)", "Head B\n(512)", "Head C\n(1024)", "Head D\n(1024-512-256)"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Training time
    train_times = [head_stats[head_stats["classifier_head"] == h]["train_time"].values[0] for h in heads]
    params = [head_stats[head_stats["classifier_head"] == h]["param_count"].values[0] / 1e6 for h in heads]
    vrams = [head_stats[head_stats["classifier_head"] == h]["peak_vram"].values[0] for h in heads]
    
    x = np.arange(len(heads))
    bars1 = ax1.bar(x, train_times, color=["#8c564b", "#1f77b4", "#ff7f0e", "#9467bd"], edgecolor="black", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Training Time per Checkpoint (Seconds)", fontweight="bold")
    ax1.set_title("Training Latency across Architectures (40 Epochs, N=220)", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    # Peak VRAM
    bars2 = ax2.bar(x, vrams, color=["#8c564b", "#1f77b4", "#ff7f0e", "#9467bd"], edgecolor="black", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel("Peak VRAM Allocation (MB)", fontweight="bold")
    ax2.set_title("Peak GPU Memory Consumption (RTX 2050 4GB)", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "runtime_memory_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Figure 10 Generated] {out_path}")


def main():
    print("=" * 80)
    print("CONFORMAL-HSI: FINAL RESEARCH VALIDATION ARTIFACT GENERATOR")
    print("=" * 80)
    
    df_seeds = load_and_combine_data()
    agg_df = compute_aggregations(df_seeds)
    rank_df = compute_strategy_rankings(agg_df)
    inter_df = compute_interaction_effects(agg_df, df_seeds)
    diag_df = compute_batchnorm_diagnostics(agg_df, df_seeds)
    thresh_df = compute_final_label_efficiency_thresholds(agg_df)
    compile_per_class_and_runtime(df_seeds)
    
    print("\n--- Generating Publication Figures (300 DPI) ---")
    plot_fig1_oa_vs_budget_by_classifier(agg_df)
    plot_fig2_aa_vs_budget_by_classifier(agg_df)
    plot_fig3_macro_f1_vs_budget_by_classifier(agg_df)
    plot_fig4_strategy_ranking_across_heads(rank_df)
    plot_fig5_classifier_capacity_vs_oa(agg_df)
    plot_fig6_classifier_acquisition_interaction(inter_df)
    plot_fig7_per_class_recall()
    plot_fig8_batchnorm_vs_layernorm_diagnostic(diag_df)
    plot_fig9_label_efficiency_thresholds(thresh_df)
    plot_fig10_runtime_memory_comparison(agg_df)
    
    print("\n[Done] All CSV artifacts and figures successfully created in " + OUT_DIR)


if __name__ == "__main__":
    main()
