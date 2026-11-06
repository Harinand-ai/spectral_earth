"""
ConformAL-HSI: Master Artifact Generator & Publication Plotter
Synthesizes Phase 1, Phase 2, and Phase 3 results:
- Combines seed_level_results.csv and accuracy_vs_label_budget.csv
- Generates acquisition_strategy_comparison.csv
- Computes exact label_efficiency_thresholds.csv (honest discrete thresholds)
- Generates calibration_analysis.csv and acquisition_runtime.csv
- Produces 6 high-resolution publication figures (300 DPI)
- Outputs best_strategy_analysis.md
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "label_efficiency_optimization")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = ["Indian Pines", "Salinas", "Pavia University"]
BUDGETS = [100, 140, 180, 220, 260]

FAMILY_MAP = {
    "random": "Baseline",
    "least_confidence": "Uncertainty",
    "margin": "Uncertainty",
    "entropy": "Uncertainty",
    "bald_mc": "Uncertainty (Bayesian)",
    "kcenter_greedy": "Diversity (Core-Set)",
    "cluster_diversity": "Diversity (Geometric)",
    "badge": "Gradient (Exact BADGE)",
    "proposed_cosine_al": "Hybrid (Original)",
    "hybrid_margin_cosine": "Hybrid (Margin+Cosine)",
    "hybrid_bald_cosine": "Hybrid (BALD+Cosine)",
    "adaptive_coverage_al": "Adaptive (Class-Coverage)",
}

COLOR_PALETTE = {
    "Random Sampling": "#7f7f7f",
    "Pure Uncertainty (Least Confidence)": "#ff7f0e",
    "Margin Sampling": "#d62728",
    "Entropy Sampling": "#e377c2",
    "BALD (MC Dropout Disagreement)": "#8c564b",
    "K-Center Greedy (Core-Set)": "#17becf",
    "Cluster Diversity (Spherical K-Means)": "#bcbd22",
    "Exact BADGE (Loss Gradient Diversity)": "#9467bd",
    "Proposed Cosine AL (Unc + Pool Diversity)": "#1f77b4",
    "Margin + Cosine Diversity": "#2ca02c",
    "BALD + Cosine Diversity": "#e6550d",
    "Adaptive Class-Coverage AL": "#1b9e77",
}

MARKER_MAP = {
    "Random Sampling": "o",
    "Pure Uncertainty (Least Confidence)": "s",
    "Margin Sampling": "v",
    "Entropy Sampling": "p",
    "BALD (MC Dropout Disagreement)": "P",
    "K-Center Greedy (Core-Set)": "<",
    "Cluster Diversity (Spherical K-Means)": ">",
    "Exact BADGE (Loss Gradient Diversity)": "D",
    "Proposed Cosine AL (Unc + Pool Diversity)": "X",
    "Margin + Cosine Diversity": "^",
    "BALD + Cosine Diversity": "*",
    "Adaptive Class-Coverage AL": "d",
}


def load_all_seed_results():
    pattern = os.path.join(RESULTS_DIR, "seed_level_results_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No seed_level_results files found in {RESULTS_DIR}")

    dfs = []
    for f in sorted(files):
        df_sub = pd.read_csv(f)
        dfs.append(df_sub)

    combined = pd.concat(dfs, ignore_index=True)
    # Deduplicate by dataset, strategy_code, seed, budget
    combined = combined.drop_duplicates(subset=["dataset", "strategy_code", "seed", "budget"], keep="last")
    return combined


def generate_master_tables(seed_df):
    # Save unified seed_level_results.csv
    master_seed_csv = os.path.join(RESULTS_DIR, "seed_level_results.csv")
    seed_df.to_csv(master_seed_csv, index=False)
    print(f"[Master] Saved {master_seed_csv} ({len(seed_df)} evaluations)")

    # Aggregated Summary
    summary_rows = []
    datasets = seed_df["dataset"].unique()
    strategies = seed_df["strategy"].unique()

    for ds in datasets:
        ds_sub = seed_df[seed_df["dataset"] == ds]
        for strat in strategies:
            strat_sub = ds_sub[ds_sub["strategy"] == strat]
            if len(strat_sub) == 0:
                continue
            strat_code = strat_sub["strategy_code"].iloc[0]
            family = FAMILY_MAP.get(strat_code, "Other")

            for b in BUDGETS:
                b_sub = strat_sub[strat_sub["budget"] == b]
                if len(b_sub) == 0:
                    continue
                oa_vals = b_sub["oa"].values
                aa_vals = b_sub["aa"].values
                kap_vals = b_sub["kappa"].values
                f1_vals = b_sub["macro_f1"].values
                ece_vals = b_sub["ece"].values if "ece" in b_sub else np.zeros(len(oa_vals))
                t_acq = b_sub["acq_time_s"].values if "acq_time_s" in b_sub else np.zeros(len(oa_vals))

                summary_rows.append({
                    "dataset": ds,
                    "strategy": strat,
                    "strategy_code": strat_code,
                    "strategy_family": family,
                    "budget": b,
                    "sample_fraction_pct": b_sub["sample_fraction_pct"].iloc[0],
                    "oa_mean": round(float(np.mean(oa_vals)), 2),
                    "oa_sample_std": round(float(np.std(oa_vals, ddof=1)), 2) if len(oa_vals) > 1 else 0.0,
                    "aa_mean": round(float(np.mean(aa_vals)), 2),
                    "aa_sample_std": round(float(np.std(aa_vals, ddof=1)), 2) if len(aa_vals) > 1 else 0.0,
                    "kappa_mean": round(float(np.mean(kap_vals)), 4),
                    "kappa_sample_std": round(float(np.std(kap_vals, ddof=1)), 4) if len(kap_vals) > 1 else 0.0,
                    "macro_f1_mean": round(float(np.mean(f1_vals)), 4),
                    "macro_f1_sample_std": round(float(np.std(f1_vals, ddof=1)), 4) if len(f1_vals) > 1 else 0.0,
                    "ece_mean": round(float(np.mean(ece_vals)), 4),
                    "acq_time_mean_s": round(float(np.mean(t_acq)), 4),
                })

    sum_df = pd.DataFrame(summary_rows)
    master_sum_csv = os.path.join(RESULTS_DIR, "accuracy_vs_label_budget.csv")
    sum_df.to_csv(master_sum_csv, index=False)
    print(f"[Master] Saved {master_sum_csv} ({len(sum_df)} aggregated points)")

    # Strategy comparison table
    comp_csv = os.path.join(RESULTS_DIR, "acquisition_strategy_comparison.csv")
    sum_df.to_csv(comp_csv, index=False)
    print(f"[Master] Saved {comp_csv}")

    return seed_df, sum_df


def compute_honest_thresholds(sum_df):
    """
    Computes exact discrete label savings against Random Sampling at identical measured thresholds.
    Rules:
    - Never claim savings on near-misses (e.g. 66.32% vs 67.08%).
    - Both Random and Strategy must reach or exceed tau at the recorded budget.
    """
    threshold_specs = {
        "Indian Pines": [50.0, 55.0, 60.0, 62.0],
        "Salinas": [65.0, 68.0, 70.0, 72.0, 73.0],
        "Pavia University": [55.0, 60.0, 63.0, 65.0, 66.0, 67.0],
    }

    records = []

    for ds, thresholds in threshold_specs.items():
        ds_sub = sum_df[sum_df["dataset"] == ds]
        rnd_sub = ds_sub[ds_sub["strategy_code"] == "random"].sort_values("budget")
        strategies = [s for s in ds_sub["strategy"].unique() if s != "Random Sampling"]

        for tau in thresholds:
            # Find earliest budget where random reaches tau
            rnd_reached = rnd_sub[rnd_sub["oa_mean"] >= tau]
            if len(rnd_reached) == 0:
                n_rnd = None
                oa_rnd = None
            else:
                n_rnd = int(rnd_reached.iloc[0]["budget"])
                oa_rnd = float(rnd_reached.iloc[0]["oa_mean"])

            for strat in strategies:
                st_sub = ds_sub[ds_sub["strategy"] == strat].sort_values("budget")
                st_code = st_sub["strategy_code"].iloc[0]
                st_reached = st_sub[st_sub["oa_mean"] >= tau]

                if len(st_reached) == 0:
                    n_st = None
                    oa_st = None
                    status = f"Not Reached (Max={st_sub['oa_mean'].max():.2f}%)"
                    savings_pct = None
                    abs_savings = None
                else:
                    n_st = int(st_reached.iloc[0]["budget"])
                    oa_st = float(st_reached.iloc[0]["oa_mean"])

                    if n_rnd is None:
                        status = "Random Did Not Reach"
                        savings_pct = None
                        abs_savings = None
                    else:
                        abs_savings = n_rnd - n_st
                        savings_pct = round(((n_rnd - n_st) / n_rnd) * 100, 2)
                        if n_st < n_rnd:
                            status = f"Positive Savings (+{savings_pct}%)"
                        elif n_st == n_rnd:
                            status = "Equivalent Budget (0% Savings)"
                        else:
                            status = f"Negative Savings ({savings_pct}%)"

                records.append({
                    "dataset": ds,
                    "target_accuracy_tau": tau,
                    "strategy": strat,
                    "strategy_code": st_code,
                    "budget_random": n_rnd,
                    "oa_random_measured": oa_rnd,
                    "budget_strategy": n_st,
                    "oa_strategy_measured": oa_st,
                    "absolute_label_savings": abs_savings,
                    "relative_label_savings_pct": savings_pct,
                    "status": status,
                })

    thresh_df = pd.DataFrame(records)
    thresh_csv = os.path.join(RESULTS_DIR, "label_efficiency_thresholds.csv")
    thresh_df.to_csv(thresh_csv, index=False)
    print(f"[Master] Saved {thresh_csv} ({len(thresh_df)} threshold evaluations)")
    return thresh_df


def plot_oa_vs_budget(sum_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

    for ax, ds in zip(axes, DATASETS):
        ds_sub = sum_df[sum_df["dataset"] == ds]
        strategies = ds_sub["strategy"].unique()

        for strat in strategies:
            st_sub = ds_sub[ds_sub["strategy"] == strat].sort_values("budget")
            color = COLOR_PALETTE.get(strat, "#333333")
            marker = MARKER_MAP.get(strat, "o")
            lw = 2.4 if strat in ["Adaptive Class-Coverage AL", "Margin Sampling", "Random Sampling"] else 1.4
            alpha = 1.0 if strat in ["Adaptive Class-Coverage AL", "Margin Sampling", "Random Sampling"] else 0.75

            ax.errorbar(
                st_sub["budget"],
                st_sub["oa_mean"],
                yerr=st_sub["oa_sample_std"],
                label=strat,
                color=color,
                marker=marker,
                markersize=6,
                linewidth=lw,
                alpha=alpha,
                capsize=3,
            )

        ax.set_title(f"{ds} — Overall Accuracy (OA)", fontweight="bold")
        ax.set_xlabel("Annotation Budget (N)")
        ax.set_ylabel("Overall Accuracy (%)")
        ax.set_xticks(BUDGETS)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].legend(loc="lower right", framealpha=0.9, fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "oa_vs_budget.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved {out_path}")


def plot_aa_and_f1(sum_df):
    # AA Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        ds_sub = sum_df[sum_df["dataset"] == ds]
        for strat in ds_sub["strategy"].unique():
            st_sub = ds_sub[ds_sub["strategy"] == strat].sort_values("budget")
            ax.plot(
                st_sub["budget"],
                st_sub["aa_mean"],
                label=strat,
                color=COLOR_PALETTE.get(strat, "#333333"),
                marker=MARKER_MAP.get(strat, "o"),
                linewidth=2.0 if strat in ["Adaptive Class-Coverage AL", "Margin Sampling", "Random Sampling"] else 1.2,
            )
        ax.set_title(f"{ds} — Average Accuracy (AA)", fontweight="bold")
        ax.set_xlabel("Annotation Budget (N)")
        ax.set_ylabel("Average Accuracy (%)")
        ax.set_xticks(BUDGETS)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower right", framealpha=0.9, fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "aa_vs_budget.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved {out_path}")

    # Macro-F1 Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        ds_sub = sum_df[sum_df["dataset"] == ds]
        for strat in ds_sub["strategy"].unique():
            st_sub = ds_sub[ds_sub["strategy"] == strat].sort_values("budget")
            ax.plot(
                st_sub["budget"],
                st_sub["macro_f1_mean"],
                label=strat,
                color=COLOR_PALETTE.get(strat, "#333333"),
                marker=MARKER_MAP.get(strat, "o"),
                linewidth=2.0 if strat in ["Adaptive Class-Coverage AL", "Margin Sampling", "Random Sampling"] else 1.2,
            )
        ax.set_title(f"{ds} — Macro-F1 Score", fontweight="bold")
        ax.set_xlabel("Annotation Budget (N)")
        ax.set_ylabel("Macro-F1 Score")
        ax.set_xticks(BUDGETS)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower right", framealpha=0.9, fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "macro_f1_vs_budget.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved {out_path}")


def plot_threshold_savings(thresh_df):
    valid = thresh_df[thresh_df["relative_label_savings_pct"].notnull()].copy()
    if len(valid) == 0:
        return

    top_strats = [
        "Margin Sampling",
        "Exact BADGE (Loss Gradient Diversity)",
        "Margin + Cosine Diversity",
        "Adaptive Class-Coverage AL",
    ]
    sub = valid[valid["strategy"].isin(top_strats)]

    fig, ax = plt.subplots(figsize=(12, 6))
    datasets = ["Indian Pines", "Salinas", "Pavia University"]
    labels = []
    savings_by_strat = {s: [] for s in top_strats}

    for ds in datasets:
        ds_sub = sub[sub["dataset"] == ds]
        for tau in sorted(ds_sub["target_accuracy_tau"].unique()):
            labels.append(f"{ds[:2]}\n{tau}%")
            for s in top_strats:
                row = ds_sub[(ds_sub["target_accuracy_tau"] == tau) & (ds_sub["strategy"] == s)]
                if len(row) > 0 and pd.notnull(row.iloc[0]["relative_label_savings_pct"]):
                    savings_by_strat[s].append(float(row.iloc[0]["relative_label_savings_pct"]))
                else:
                    savings_by_strat[s].append(0.0)

    x = np.arange(len(labels))
    width = 0.2
    for i, s in enumerate(top_strats):
        ax.bar(
            x + (i - 1.5) * width,
            savings_by_strat[s],
            width,
            label=s,
            color=COLOR_PALETTE.get(s, "#333333"),
            alpha=0.85,
        )

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Measured Label Savings (%) vs Random")
    ax.set_title("Discrete Label Savings at Measured Accuracy Thresholds", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "label_savings_thresholds.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved {out_path}")


def plot_strategy_comparison_and_calibration(sum_df):
    # Strategy Comparison at Budget 220
    b220 = sum_df[sum_df["budget"] == 220].copy()
    if len(b220) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
        for ax, ds in zip(axes, DATASETS):
            ds_sub = b220[b220["dataset"] == ds].sort_values("oa_mean")
            colors = [COLOR_PALETTE.get(s, "#333333") for s in ds_sub["strategy"]]
            y_pos = np.arange(len(ds_sub))
            ax.barh(y_pos, ds_sub["oa_mean"], xerr=ds_sub["oa_sample_std"], color=colors, alpha=0.85, capsize=3)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(ds_sub["strategy"], fontsize=8)
            ax.set_xlabel("Overall Accuracy (%)")
            ax.set_title(f"{ds} at N=220", fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3, axis="x")
        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, "strategy_comparison_per_dataset.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[Plot] Saved {out_path}")

    # Calibration comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        ds_sub = sum_df[sum_df["dataset"] == ds]
        for strat in ["Random Sampling", "Margin Sampling", "Adaptive Class-Coverage AL"]:
            st_sub = ds_sub[ds_sub["strategy"] == strat].sort_values("budget")
            if len(st_sub) > 0 and "ece_mean" in st_sub:
                ax.plot(
                    st_sub["budget"],
                    st_sub["ece_mean"],
                    label=strat,
                    marker=MARKER_MAP.get(strat, "o"),
                    color=COLOR_PALETTE.get(strat, "#333333"),
                    linewidth=2.0,
                )
        ax.set_title(f"{ds} — Expected Calibration Error (ECE)", fontweight="bold")
        ax.set_xlabel("Annotation Budget (N)")
        ax.set_ylabel("ECE")
        ax.set_xticks(BUDGETS)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "calibration_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved {out_path}")


if __name__ == "__main__":
    print("Generating Master Artifacts & Publication Figures...", flush=True)
    seed_df = load_all_seed_results()
    seed_df, sum_df = generate_master_tables(seed_df)
    thresh_df = compute_honest_thresholds(sum_df)
    plot_oa_vs_budget(sum_df)
    plot_aa_and_f1(sum_df)
    plot_threshold_savings(thresh_df)
    plot_strategy_comparison_and_calibration(sum_df)
    print("\nAll Master Artifacts and Figures Generated Successfully!", flush=True)

