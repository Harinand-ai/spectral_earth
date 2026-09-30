import os
import sys
import json
import csv
import time
import torch
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pyrootutils

pyrootutils.setup_root(r"d:\Projects\spectral_earth", indicator=".project-root", pythonpath=True)

from src.backbones.registry import BACKBONE_REGISTRY
from src.learning_engine import LearningEngine, ActiveLearningDatasetManager
from src.learning_engine.classifier import ClassifierHead
from src.query_engine.query_engine import QueryEngine

def run_multiseed_validation():
    print("=" * 88)
    print("SPECTRALEARTH ACTIVE LEARNING FRAMEWORK — MULTI-SEED STATISTICAL VALIDATION")
    print("=" * 88)

    ckpt_path = r"d:\Projects\spectral_earth\logs\None\runs\2026-07-30_14-24-24\checkpoints\epoch_094.ckpt"
    
    seeds = [42, 43, 44, 45, 46]
    strategies = [
        ("Random", "random"),
        ("Least Confidence", "least_confidence"),
        ("LC Filter + Farthest Diversity", "filter_diversity"),
    ]

    datasets_info = {
        "Indian Pines": {
            "module_cls": "src.datamodules.indian_pines.IndianPinesDataModule",
            "num_classes": 16,
            "init_samples": 10,
            "slug": "indian_pines",
        },
        "Salinas": {
            "module_cls": "src.datamodules.salinas.SalinasDataModule",
            "num_classes": 16,
            "init_samples": 10,
            "slug": "salinas",
        },
        "Pavia University": {
            "module_cls": "src.datamodules.pavia_university.PaviaUniversityDataModule",
            "num_classes": 9,
            "init_samples": 10,
            "slug": "pavia_university",
        },
    }

    raw_multiseed_results = {}
    total_runs = len(datasets_info) * len(strategies) * len(seeds)
    run_counter = 0

    os.makedirs("logs/al_experiments", exist_ok=True)
    os.makedirs("logs/visualizations", exist_ok=True)

    t0_global = time.time()

    for dataset_name, meta in datasets_info.items():
        print(f"\n" + "=" * 88)
        print(f" >>> DATASET: {dataset_name} <<<")
        print("=" * 88)

        mod_path, class_name = meta["module_cls"].rsplit(".", 1)
        import importlib
        module = importlib.import_module(mod_path)
        DataModuleCls = getattr(module, class_name)

        datamodule = DataModuleCls()
        datamodule.setup("fit")
        datamodule.setup("test")

        raw_multiseed_results[dataset_name] = {}

        for strat_label, strat_name in strategies:
            raw_multiseed_results[dataset_name][strat_label] = {}

            for seed in seeds:
                run_counter += 1
                print(f"[{run_counter:02d}/{total_runs:02d}] Running {dataset_name} | Strategy: '{strat_label:30s}' | Seed: {seed}...")

                torch.manual_seed(seed)
                np.random.seed(seed)

                train_dataset = datamodule.train_dataset
                val_dataset = getattr(datamodule, "val_dataset", None)
                test_dataset = getattr(datamodule, "test_dataset", None)

                num_classes = meta["num_classes"]
                backbone = BACKBONE_REGISTRY["spec_resnet50"](num_classes=num_classes)

                if os.path.exists(ckpt_path):
                    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    state_dict = checkpoint.get("state_dict", checkpoint)
                    backbone_state = {}
                    for k, v in state_dict.items():
                        if k.startswith("model.backbone."):
                            backbone_state[k[len("model.backbone."):]] = v
                        elif k.startswith("model."):
                            backbone_state[k[len("model."):]] = v
                        elif k.startswith("feature_extractor.backbone."):
                            backbone_state[k[len("feature_extractor.backbone."):]] = v
                        else:
                            backbone_state[k] = v
                    backbone_state = {k: v for k, v in backbone_state.items() if not ("fc.weight" in k or "fc.bias" in k)}
                    backbone.load_state_dict(backbone_state, strict=False)

                for param in backbone.parameters():
                    param.requires_grad = False
                backbone.eval()

                dataset_manager = ActiveLearningDatasetManager(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    initial_samples_per_class=meta["init_samples"],
                    seed=seed,
                )

                learning_engine = LearningEngine(
                    backbone=backbone,
                    num_classes=num_classes,
                    dataset_manager=dataset_manager,
                    hidden_dim=256,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                learning_engine.classifier_head = ClassifierHead(
                    in_features=learning_engine.feature_extractor.feature_dim,
                    num_classes=num_classes,
                    hidden_dim=256,
                    head_type="mlp_2layer",
                    norm_layer="layernorm",
                )

                query_engine = QueryEngine(
                    uncertainty_method=strat_name,
                    start_uncertainty=0.30,
                    end_uncertainty=0.80,
                )

                history = learning_engine.run_full_active_learning_loop(
                    query_engine=query_engine,
                    k=20,
                    total_rounds=5,
                    epochs_per_round=25,
                    lr=0.0005,
                    batch_size=64,
                )

                # Sanitize numpy arrays for JSON serialization
                sanitized_history = []
                for entry in history:
                    e_copy = dict(entry)
                    if "metrics" in e_copy and "ConfusionMatrix" in e_copy["metrics"]:
                        if isinstance(e_copy["metrics"]["ConfusionMatrix"], np.ndarray):
                            e_copy["metrics"]["ConfusionMatrix"] = e_copy["metrics"]["ConfusionMatrix"].tolist()
                    sanitized_history.append(e_copy)

                raw_multiseed_results[dataset_name][strat_label][str(seed)] = sanitized_history

                final_m = sanitized_history[-1].get("metrics", {})
                print(
                    f"     -> Final OA: {final_m.get('OverallAccuracy', 0.0)*100:6.2f}% | "
                    f"AA: {final_m.get('AverageAccuracy', 0.0)*100:6.2f}% | F1: {final_m.get('F1Score', 0.0):.4f} | "
                    f"Kappa: {final_m.get('Kappa', 0.0):.4f}"
                )

    # Save raw multi-seed JSON results
    raw_json_path = "logs/al_experiments/multiseed_results.json"
    with open(raw_json_path, "w") as f:
        json.dump(raw_multiseed_results, f, indent=2)
    print(f"\nRaw multiseed results saved to {raw_json_path}")

    # Compute Statistical Summaries (Mean ± Std)
    summary_data = {}
    csv_rows = []

    for d_name, d_strats in raw_multiseed_results.items():
        summary_data[d_name] = {}
        for s_label, seed_runs in d_strats.items():
            num_rounds = len(list(seed_runs.values())[0])
            round_summaries = []

            for r in range(num_rounds):
                oas = [seed_runs[str(sd)][r]["metrics"].get("OverallAccuracy", 0.0) * 100 for sd in seeds]
                aas = [seed_runs[str(sd)][r]["metrics"].get("AverageAccuracy", 0.0) * 100 for sd in seeds]
                f1s = [seed_runs[str(sd)][r]["metrics"].get("F1Score", 0.0) for sd in seeds]
                kappas = [seed_runs[str(sd)][r]["metrics"].get("Kappa", 0.0) for sd in seeds]
                precisions = [seed_runs[str(sd)][r]["metrics"].get("Precision", 0.0) for sd in seeds]
                recalls = [seed_runs[str(sd)][r]["metrics"].get("Recall", 0.0) for sd in seeds]
                losses = [seed_runs[str(sd)][r]["retrain_loss"] for sd in seeds]
                n_lab = seed_runs[str(seeds[0])][r]["num_labeled"]

                r_sum = {
                    "round": r,
                    "num_labeled": n_lab,
                    "oa_mean": float(np.mean(oas)),
                    "oa_std": float(np.std(oas, ddof=1)),
                    "aa_mean": float(np.mean(aas)),
                    "aa_std": float(np.std(aas, ddof=1)),
                    "f1_mean": float(np.mean(f1s)),
                    "f1_std": float(np.std(f1s, ddof=1)),
                    "kappa_mean": float(np.mean(kappas)),
                    "kappa_std": float(np.std(kappas, ddof=1)),
                    "precision_mean": float(np.mean(precisions)),
                    "precision_std": float(np.std(precisions, ddof=1)),
                    "recall_mean": float(np.mean(recalls)),
                    "recall_std": float(np.std(recalls, ddof=1)),
                    "loss_mean": float(np.mean(losses)),
                    "loss_std": float(np.std(losses, ddof=1)),
                    "oa_per_seed": oas,
                    "f1_per_seed": f1s,
                }
                round_summaries.append(r_sum)

            summary_data[d_name][s_label] = round_summaries

            # Save final round summary for CSV
            final_r = round_summaries[-1]
            csv_rows.append({
                "Dataset": d_name,
                "Strategy": s_label,
                "Initial Labeled": round_summaries[0]["num_labeled"],
                "Final Labeled": final_r["num_labeled"],
                "OA Mean (%)": f"{final_r['oa_mean']:.2f}",
                "OA Std (%)": f"{final_r['oa_std']:.2f}",
                "AA Mean (%)": f"{final_r['aa_mean']:.2f}",
                "AA Std (%)": f"{final_r['aa_std']:.2f}",
                "F1 Mean": f"{final_r['f1_mean']:.4f}",
                "F1 Std": f"{final_r['f1_std']:.4f}",
                "Kappa Mean": f"{final_r['kappa_mean']:.4f}",
                "Kappa Std": f"{final_r['kappa_std']:.4f}",
            })

    # Save summary JSON
    summary_json_path = "logs/al_experiments/multiseed_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary JSON saved to {summary_json_path}")

    # Save summary CSV
    summary_csv_path = "logs/al_experiments/multiseed_summary.csv"
    fieldnames = ["Dataset", "Strategy", "Initial Labeled", "Final Labeled", "OA Mean (%)", "OA Std (%)", "AA Mean (%)", "AA Std (%)", "F1 Mean", "F1 Std", "Kappa Mean", "Kappa Std"]
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Summary CSV saved to {summary_csv_path}")

    # Generate Publication Plots with Mean ± Std Error Bands
    style_colors = {
        "Random": "#7f7f7f", # Gray
        "Least Confidence": "#1f77b4", # Blue
        "LC Filter + Farthest Diversity": "#2ca02c", # Green
    }
    style_markers = {
        "Random": "s",
        "Least Confidence": "^",
        "LC Filter + Farthest Diversity": "o",
    }

    for d_name, d_strats in summary_data.items():
        ds_slug = datasets_info[d_name]["slug"]

        # Plot 1: OA vs Labeled Samples (Mean ± Std)
        plt.figure(figsize=(8, 6))
        for s_label, r_sums in d_strats.items():
            labeled_x = [rs["num_labeled"] for rs in r_sums]
            oa_means = [rs["oa_mean"] for rs in r_sums]
            oa_stds = [rs["oa_std"] for rs in r_sums]

            c = style_colors.get(s_label, "#333333")
            m = style_markers.get(s_label, "o")

            plt.plot(labeled_x, oa_means, label=s_label, color=c, marker=m, linewidth=2)
            plt.fill_between(labeled_x, np.array(oa_means) - np.array(oa_stds), np.array(oa_means) + np.array(oa_stds), color=c, alpha=0.15)

        plt.title(f"{d_name} — Active Learning Overall Accuracy (Mean ± Std, 5 Seeds)", fontsize=13, fontweight="bold")
        plt.xlabel("Number of Labeled Samples", fontsize=11)
        plt.ylabel("Overall Accuracy (%)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=10, loc="lower right")
        plt.tight_layout()
        plot_path_oa = f"logs/visualizations/multiseed_{ds_slug}_oa_mean_std.png"
        plt.savefig(plot_path_oa, dpi=300)
        plt.close()

        # Plot 2: F1 Score vs Labeled Samples (Mean ± Std)
        plt.figure(figsize=(8, 6))
        for s_label, r_sums in d_strats.items():
            labeled_x = [rs["num_labeled"] for rs in r_sums]
            f1_means = [rs["f1_mean"] for rs in r_sums]
            f1_stds = [rs["f1_std"] for rs in r_sums]

            c = style_colors.get(s_label, "#333333")
            m = style_markers.get(s_label, "o")

            plt.plot(labeled_x, f1_means, label=s_label, color=c, marker=m, linewidth=2)
            plt.fill_between(labeled_x, np.array(f1_means) - np.array(f1_stds), np.array(f1_means) + np.array(f1_stds), color=c, alpha=0.15)

        plt.title(f"{d_name} — Active Learning Macro F1 Score (Mean ± Std, 5 Seeds)", fontsize=13, fontweight="bold")
        plt.xlabel("Number of Labeled Samples", fontsize=11)
        plt.ylabel("Macro F1 Score", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=10, loc="lower right")
        plt.tight_layout()
        plot_path_f1 = f"logs/visualizations/multiseed_{ds_slug}_f1_mean_std.png"
        plt.savefig(plot_path_f1, dpi=300)
        plt.close()

    print("\n" + "=" * 95)
    print("FINAL STATISTICAL VALIDATION SUMMARY TABLE (MEAN ± STD ACROSS 5 SEEDS)")
    print("=" * 95)
    print(f"{'Dataset':<18} | {'Strategy':<30} | {'Final OA (%)':<16} | {'Final AA (%)':<16} | {'Final F1':<14} | {'Final Kappa':<14}")
    print("-" * 118)

    for d_name, d_strats in summary_data.items():
        for s_label, r_sums in d_strats.items():
            final_r = r_sums[-1]
            oa_str = f"{final_r['oa_mean']:.2f} ± {final_r['oa_std']:.2f}"
            aa_str = f"{final_r['aa_mean']:.2f} ± {final_r['aa_std']:.2f}"
            f1_str = f"{final_r['f1_mean']:.4f} ± {final_r['f1_std']:.4f}"
            kappa_str = f"{final_r['kappa_mean']:.4f} ± {final_r['kappa_std']:.4f}"
            print(f"{d_name:<18} | {s_label:<30} | {oa_str:<16} | {aa_str:<16} | {f1_str:<14} | {kappa_str:<14}")

    # Paired Statistical Significance Tests (Paired t-test across 5 seeds)
    print("\n" + "=" * 95)
    print("PAIRED STATISTICAL SIGNIFICANCE TESTS (PAIRED T-TEST, n=5 SEEDS)")
    print("=" * 95)

    for d_name in datasets_info.keys():
        print(f"\n--- {d_name} Paired Comparisons ---")
        sd_lc_fd = [raw_multiseed_results[d_name]["LC Filter + Farthest Diversity"][str(sd)][-1]["metrics"]["OverallAccuracy"] * 100 for sd in seeds]
        sd_lc = [raw_multiseed_results[d_name]["Least Confidence"][str(sd)][-1]["metrics"]["OverallAccuracy"] * 100 for sd in seeds]
        sd_rnd = [raw_multiseed_results[d_name]["Random"][str(sd)][-1]["metrics"]["OverallAccuracy"] * 100 for sd in seeds]

        # 1. LC+FD vs LC
        t_stat_lc, p_val_lc = stats.ttest_rel(sd_lc_fd, sd_lc)
        # 2. LC+FD vs Random
        t_stat_rnd, p_val_rnd = stats.ttest_rel(sd_lc_fd, sd_rnd)
        # 3. LC vs Random
        t_stat_lc_rnd, p_val_lc_rnd = stats.ttest_rel(sd_lc, sd_rnd)

        print(f"  LC+FD vs Least Confidence  -> Mean Diff: {np.mean(sd_lc_fd)-np.mean(sd_lc):+6.2f}% | t-stat: {t_stat_lc:6.3f} | p-value: {p_val_lc:.4f} {'(p < 0.05)' if p_val_lc < 0.05 else '(not sig)'}")
        print(f"  LC+FD vs Random Baseline    -> Mean Diff: {np.mean(sd_lc_fd)-np.mean(sd_rnd):+6.2f}% | t-stat: {t_stat_rnd:6.3f} | p-value: {p_val_rnd:.4f} {'(p < 0.05)' if p_val_rnd < 0.05 else '(not sig)'}")
        print(f"  Least Conf vs Random        -> Mean Diff: {np.mean(sd_lc)-np.mean(sd_rnd):+6.2f}% | t-stat: {t_stat_lc_rnd:6.3f} | p-value: {p_val_lc_rnd:.4f} {'(p < 0.05)' if p_val_lc_rnd < 0.05 else '(not sig)'}")

if __name__ == "__main__":
    run_multiseed_validation()
