import os
import sys
import json
import csv
import time
import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pyrootutils

pyrootutils.setup_root(r"d:\Projects\spectral_earth", indicator=".project-root", pythonpath=True)

from src.backbones.registry import BACKBONE_REGISTRY
from src.learning_engine import LearningEngine, ActiveLearningDatasetManager
from src.learning_engine.classifier import ClassifierHead
from src.learning_engine.feature_cache import FeatureCache
from src.learning_engine.temperature_scaling import TemperatureScaler
from src.query_engine.query_engine import QueryEngine
from src.utils.metrics_engine import compute_detailed_metrics, compute_aulc, compute_ece
from src.utils.spectral_adapter_1d import SpectralResampler1D

def load_frozen_backbone(num_classes, ckpt_path):
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
    return backbone

def get_datamodule(ds_name, meta):
    mod_path, class_name = meta["module_cls"].rsplit(".", 1)
    import importlib
    module = importlib.import_module(mod_path)
    DataModuleCls = getattr(module, class_name)
    dm = DataModuleCls()
    dm.setup("fit")
    dm.setup("test")
    return dm

def run_master_pipeline():
    print("=" * 88)
    print("SPECTRALEARTH ACTIVE LEARNING FRAMEWORK — MASTER PIPELINE EXECUTION")
    print("=" * 88)

    os.makedirs("logs/al_experiments", exist_ok=True)
    os.makedirs("logs/al_experiments/figures", exist_ok=True)

    ckpt_path = r"d:\Projects\spectral_earth\logs\None\runs\2026-07-30_14-24-24\checkpoints\epoch_094.ckpt"
    seeds = [42, 43, 44, 45, 46]

    datasets_meta = {
        "Indian Pines": {
            "module_cls": "src.datamodules.indian_pines.IndianPinesDataModule",
            "num_classes": 16,
            "bands": 200,
            "init_samples": 10,
            "slug": "indian_pines",
        },
        "Salinas": {
            "module_cls": "src.datamodules.salinas.SalinasDataModule",
            "num_classes": 16,
            "bands": 204,
            "init_samples": 10,
            "slug": "salinas",
        },
        "Pavia University": {
            "module_cls": "src.datamodules.pavia_university.PaviaUniversityDataModule",
            "num_classes": 9,
            "bands": 103,
            "init_samples": 10,
            "slug": "pavia_university",
        },
    }

    strategies = [
        ("Random", "random", False, 0.0, False),
        ("Least Confidence", "least_confidence", False, 0.0, False),
        ("Entropy", "entropy", False, 0.0, False),
        ("Margin", "margin", False, 0.0, False),
        ("Diversity", "diversity", False, 0.0, False),
        ("Adaptive", "adaptive", False, 0.0, False),
        ("Hybrid", "hybrid", False, 0.0, False),
        ("LC + Farthest Diversity", "filter_diversity", False, 0.0, False),
        ("Labeled-Pool Diversity", "filter_diversity_labeled", False, 0.0, False),
        ("Calibrated LC + Diversity", "filter_diversity", True, 0.0, False),
        ("Calibrated LC + Labeled-Pool Diversity", "filter_diversity_labeled", True, 0.0, False),
        ("SCUD-AL Candidate (Drop 0.1)", "filter_diversity_labeled", True, 0.1, False),
        ("Class-Balanced Baseline", "filter_diversity", False, 0.0, True),
    ]

    master_results = {}
    master_rows = []
    calibration_records = []
    ablation_records = []

    t0_pipeline = time.time()

    for ds_name, meta in datasets_meta.items():
        print(f"\n" + "=" * 88)
        print(f" >>> RUNNING DATASET BENCHMARK & ABLATIONS: {ds_name} <<<")
        print("=" * 88)

        datamodule = get_datamodule(ds_name, meta)
        train_ds = datamodule.train_dataset
        val_ds = getattr(datamodule, "val_dataset", None)
        test_ds = getattr(datamodule, "test_dataset", None)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Precomputing 2048-dim feature cache for {ds_name} on {device}...")
        t_cache0 = time.time()
        backbone_init = load_frozen_backbone(meta["num_classes"], ckpt_path)
        from src.learning_engine.feature_extractor import FeatureExtractor
        fe_init = FeatureExtractor(backbone=backbone_init).to(device)

        def preextract(ds):
            if ds is None or len(ds) == 0:
                return torch.empty((0, 2048)), torch.empty((0,), dtype=torch.long)
            loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
            all_f, all_l = [], []
            with torch.no_grad():
                for b in loader:
                    x = b["image"].to(device) if isinstance(b, dict) else b[0].to(device)
                    y = b["label"] if isinstance(b, dict) else b[1]
                    f = fe_init(x)
                    all_f.append(f.cpu())
                    all_l.append(y.cpu())
            return torch.cat(all_f, dim=0), torch.cat(all_l, dim=0)

        train_feats, train_lbls = preextract(train_ds)
        val_feats, val_lbls = preextract(val_ds)
        test_feats, test_lbls = preextract(test_ds)
        t_cache = time.time() - t_cache0
        print(f"  Feature cache precomputed in {t_cache:.2f}s! (Train={len(train_feats)}, Val={len(val_feats)}, Test={len(test_feats)})")

        ds_results = {}

        for strat_label, q_method, use_calib, drop_rate, use_cb in strategies:
            print(f"\n  Strategy: '{strat_label}' (Calib={use_calib}, Drop={drop_rate}, ClassBal={use_cb})...")
            seed_runs = []

            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                dataset_manager = ActiveLearningDatasetManager(
                    train_dataset=train_ds,
                    val_dataset=val_ds,
                    test_dataset=test_ds,
                    initial_samples_per_class=meta["init_samples"],
                    seed=seed,
                )

                classifier_head = ClassifierHead(
                    in_features=2048,
                    num_classes=meta["num_classes"],
                    hidden_dim=256,
                    dropout=drop_rate,
                    head_type="mlp_2layer",
                    norm_layer="layernorm",
                ).to(device)

                query_engine = QueryEngine(uncertainty_method=q_method)
                total_rounds = 5
                k_per_round = 20
                run_history = []

                def train_head_cached(head, feats, lbls, epochs=25, lr=0.0005, use_cb_flag=False):
                    head.train()
                    if use_cb_flag and len(lbls) > 0:
                        counts = np.bincount(lbls.numpy(), minlength=meta["num_classes"]).astype(np.float32)
                        weights = float(len(lbls)) / (meta["num_classes"] * np.maximum(counts, 1.0))
                        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
                    else:
                        class_weights = None

                    ds_t = torch.utils.data.TensorDataset(feats.to(device), lbls.to(device))
                    loader_t = torch.utils.data.DataLoader(ds_t, batch_size=64, shuffle=True)
                    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
                    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=1e-5)
                    crit = nn.CrossEntropyLoss(weight=class_weights)

                    l_loss = 0.0
                    for _ in range(epochs):
                        tot_l, tot_s = 0.0, 0
                        for x_b, y_b in loader_t:
                            logits_b = head(x_b)
                            loss_b = crit(logits_b, y_b.long())
                            opt.zero_grad()
                            loss_b.backward()
                            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
                            opt.step()
                            tot_l += loss_b.item() * len(x_b)
                            tot_s += len(x_b)
                        sch.step()
                        l_loss = tot_l / max(1, tot_s)
                    return l_loss

                def predict_probs_c(head, feats, temperature=1.0):
                    head.eval()
                    if len(feats) == 0:
                        return np.empty((0, meta["num_classes"]))
                    ds_t = torch.utils.data.TensorDataset(feats.to(device))
                    loader_t = torch.utils.data.DataLoader(ds_t, batch_size=64, shuffle=False)
                    all_p = []
                    with torch.no_grad():
                        for (x_b,) in loader_t:
                            p_b = head.predict_probabilities(x_b, temperature=temperature)
                            all_p.append(p_b.cpu())
                    return torch.cat(all_p, dim=0).numpy()

                def evaluate_c(head, feats, lbls):
                    if len(feats) == 0:
                        return {}
                    probs_np = predict_probs_c(head, feats, temperature=1.0)
                    preds_np = np.argmax(probs_np, axis=1)
                    return compute_detailed_metrics(lbls.numpy(), preds_np, probs_np, num_classes=meta["num_classes"])

                # Initial seed training
                t_tr0 = time.time()
                init_lbl_idx = dataset_manager.labeled_indices
                init_loss = train_head_cached(
                    classifier_head,
                    train_feats[init_lbl_idx],
                    train_lbls[init_lbl_idx],
                    epochs=25,
                    lr=0.0005,
                    use_cb_flag=use_cb,
                )
                tr_time_init = time.time() - t_tr0

                uncalib_ece, calib_ece, fitted_temp = 0.0, 0.0, 1.0
                if len(val_feats) > 0:
                    val_probs_np = predict_probs_c(classifier_head, val_feats, temperature=1.0)
                    scaler = TemperatureScaler()
                    val_logits_t = torch.tensor(val_probs_np).log().clamp(min=-100)
                    fitted_temp = scaler.fit(val_logits_t, val_lbls)
                    uncalib_ece = compute_ece(val_probs_np, val_lbls.numpy())
                    calib_probs_np = scaler.calibrate_probs(val_logits_t).numpy()
                    calib_ece = compute_ece(calib_probs_np, val_lbls.numpy())

                init_metrics = evaluate_c(classifier_head, test_feats, test_lbls)
                run_history.append({
                    "round": 0,
                    "num_labeled": dataset_manager.num_labeled,
                    "num_unlabeled": dataset_manager.num_unlabeled,
                    "query_time": 0.0,
                    "train_time": tr_time_init,
                    "retrain_loss": init_loss,
                    "metrics": init_metrics,
                    "temperature": fitted_temp,
                    "uncalib_ece": uncalib_ece,
                    "calib_ece": calib_ece,
                })

                # AL Rounds 1..5
                for r in range(1, total_rounds + 1):
                    if dataset_manager.num_unlabeled == 0:
                        break

                    t_q0 = time.time()
                    unlabeled_idx = dataset_manager.unlabeled_indices
                    labeled_idx = dataset_manager.labeled_indices

                    temp_apply = fitted_temp if use_calib else 1.0
                    unlabeled_embeds = train_feats[unlabeled_idx]
                    probs_np = predict_probs_c(classifier_head, unlabeled_embeds, temperature=temp_apply)

                    labeled_embeds_np = None
                    if q_method == "filter_diversity_labeled":
                        labeled_embeds_np = train_feats[labeled_idx].numpy()

                    selected_indices = query_engine.select(
                        embeddings=unlabeled_embeds.numpy(),
                        probabilities=probs_np,
                        k=min(k_per_round, len(unlabeled_embeds)),
                        current_round=r - 1,
                        total_rounds=total_rounds,
                        labeled_embeddings=labeled_embeds_np,
                    )
                    query_time = time.time() - t_q0

                    _ = dataset_manager.update_pools(selected_indices)

                    t_tr = time.time()
                    new_lbl_idx = dataset_manager.labeled_indices
                    retrain_loss = train_head_cached(
                        classifier_head,
                        train_feats[new_lbl_idx],
                        train_lbls[new_lbl_idx],
                        epochs=25,
                        lr=0.0005,
                        use_cb_flag=use_cb,
                    )
                    train_time = time.time() - t_tr

                    test_metrics = evaluate_c(classifier_head, test_feats, test_lbls)
                    run_history.append({
                        "round": r,
                        "num_labeled": dataset_manager.num_labeled,
                        "num_unlabeled": dataset_manager.num_unlabeled,
                        "query_time": query_time,
                        "train_time": train_time,
                        "retrain_loss": retrain_loss,
                        "metrics": test_metrics,
                        "temperature": fitted_temp,
                    })

                seed_runs.append(run_history)

            ds_results[strat_label] = seed_runs

            # Compute 5-seed statistics for final round
            final_oas = [sr[-1]["metrics"].get("OverallAccuracy", 0.0) * 100 for sr in seed_runs]
            final_aas = [sr[-1]["metrics"].get("AverageAccuracy", 0.0) * 100 for sr in seed_runs]
            final_f1s = [sr[-1]["metrics"].get("F1Score", 0.0) for sr in seed_runs]
            final_kappas = [sr[-1]["metrics"].get("Kappa", 0.0) for sr in seed_runs]
            final_eces = [sr[-1]["metrics"].get("ECE", 0.0) for sr in seed_runs]
            
            # Compute AULCs across rounds
            aulc_oas = []
            aulc_aas = []
            aulc_f1s = []
            for sr in seed_runs:
                counts = [r["num_labeled"] for r in sr]
                o_vals = [r["metrics"].get("OverallAccuracy", 0.0) * 100 for r in sr]
                a_vals = [r["metrics"].get("AverageAccuracy", 0.0) * 100 for r in sr]
                f_vals = [r["metrics"].get("F1Score", 0.0) for r in sr]
                aulc_oas.append(compute_aulc(counts, o_vals))
                aulc_aas.append(compute_aulc(counts, a_vals))
                aulc_f1s.append(compute_aulc(counts, f_vals))

            init_labeled = seed_runs[0][0]["num_labeled"]
            final_labeled = seed_runs[0][-1]["num_labeled"]

            mean_oa, std_oa = float(np.mean(final_oas)), float(np.std(final_oas, ddof=1))
            mean_aa, std_aa = float(np.mean(final_aas)), float(np.std(final_aas, ddof=1))
            mean_f1, std_f1 = float(np.mean(final_f1s)), float(np.std(final_f1s, ddof=1))
            mean_kappa, std_kappa = float(np.mean(final_kappas)), float(np.std(final_kappas, ddof=1))
            mean_ece, std_ece = float(np.mean(final_eces)), float(np.std(final_eces, ddof=1))
            mean_aulc_oa = float(np.mean(aulc_oas))

            print(
                f"    -> Summary [{strat_label:36s}] | OA: {mean_oa:6.2f} ± {std_oa:.2f}% | "
                f"AA: {mean_aa:6.2f} ± {std_aa:.2f}% | F1: {mean_f1:.4f} ± {std_f1:.4f} | "
                f"AULC(OA): {mean_aulc_oa:6.2f}%"
            )

            master_rows.append({
                "Dataset": ds_name,
                "Strategy": strat_label,
                "Seed count": len(seeds),
                "Initial labels": init_labeled,
                "Final labels": final_labeled,
                "OA Mean (%)": f"{mean_oa:.2f}",
                "OA Std (%)": f"{std_oa:.2f}",
                "AA Mean (%)": f"{mean_aa:.2f}",
                "AA Std (%)": f"{std_aa:.2f}",
                "Precision Mean": f"{np.mean([sr[-1]['metrics'].get('Precision', 0.0) for sr in seed_runs]):.4f}",
                "Recall Mean": f"{np.mean([sr[-1]['metrics'].get('Recall', 0.0) for sr in seed_runs]):.4f}",
                "Macro F1 Mean": f"{mean_f1:.4f}",
                "Macro F1 Std": f"{std_f1:.4f}",
                "Kappa Mean": f"{mean_kappa:.4f}",
                "Kappa Std": f"{std_kappa:.4f}",
                "ECE Mean": f"{mean_ece:.4f}",
                "AULC OA Mean (%)": f"{mean_aulc_oa:.2f}",
                "Training time (s)": f"{np.mean([sr[-1]['train_time'] for sr in seed_runs]):.3f}",
                "Query time (s)": f"{np.mean([sr[-1]['query_time'] for sr in seed_runs]):.3f}",
            })

            if use_calib:
                uncal_m = seed_runs[0][0]["uncalib_ece"]
                cal_m = seed_runs[0][0]["calib_ece"]
                calibration_records.append({
                    "Dataset": ds_name,
                    "Strategy": strat_label,
                    "Fitted Temperature": f"{seed_runs[0][0]['temperature']:.3f}",
                    "ECE Uncalibrated": f"{uncal_m:.4f}",
                    "ECE Calibrated": f"{cal_m:.4f}",
                    "ECE Delta": f"{cal_m - uncal_m:.4f}",
                })

        master_results[ds_name] = ds_results

    # Save FINAL_MASTER_TABLE.csv
    master_csv_path = "logs/al_experiments/FINAL_MASTER_TABLE.csv"
    with open(master_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(master_rows[0].keys()))
        writer.writeheader()
        writer.writerows(master_rows)
    print(f"\nSaved master summary table to {master_csv_path}")

    # Save final_results.json
    final_json_path = "logs/al_experiments/final_results.json"
    sanitized_master = {}
    for d_name, d_strats in master_results.items():
        sanitized_master[d_name] = {}
        for s_label, seed_runs in d_strats.items():
            sanitized_master[d_name][s_label] = []
            for sr in seed_runs:
                sr_clean = []
                for entry in sr:
                    e_copy = dict(entry)
                    if "metrics" in e_copy and "ConfusionMatrix" in e_copy["metrics"]:
                        if isinstance(e_copy["metrics"]["ConfusionMatrix"], np.ndarray):
                            e_copy["metrics"]["ConfusionMatrix"] = e_copy["metrics"]["ConfusionMatrix"].tolist()
                    sr_clean.append(e_copy)
                sanitized_master[d_name][s_label].append(sr_clean)

    with open(final_json_path, "w") as f:
        json.dump(sanitized_master, f, indent=2)
    print(f"Saved final results JSON to {final_json_path}")

    # Save calibration_results.csv
    if calibration_records:
        calib_csv_path = "logs/al_experiments/calibration_results.csv"
        with open(calib_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(calibration_records[0].keys()))
            writer.writeheader()
            writer.writerows(calibration_records)
        print(f"Saved calibration summary CSV to {calib_csv_path}")

    # Paired Statistical Significance & Ablation Table Generation
    print("\n" + "=" * 88)
    print("STATISTICAL SIGNIFICANCE TESTS (PAIRED T-TEST & WILCOXON)")
    print("=" * 88)

    ablation_rows = []
    for ds_name in datasets_meta.keys():
        print(f"\n--- Statistical Tests for {ds_name} ---")
        base_runs = master_results[ds_name]["LC + Farthest Diversity"]
        scud_runs = master_results[ds_name]["SCUD-AL Candidate (Drop 0.1)"]
        rnd_runs = master_results[ds_name]["Random"]
        lc_runs = master_results[ds_name]["Least Confidence"]

        oas_base = [sr[-1]["metrics"]["OverallAccuracy"] * 100 for sr in base_runs]
        oas_scud = [sr[-1]["metrics"]["OverallAccuracy"] * 100 for sr in scud_runs]
        oas_rnd = [sr[-1]["metrics"]["OverallAccuracy"] * 100 for sr in rnd_runs]
        oas_lc = [sr[-1]["metrics"]["OverallAccuracy"] * 100 for sr in lc_runs]

        # Paired t-test SCUD vs Base
        t_stat, p_val_t = stats.ttest_rel(oas_scud, oas_base)
        # Wilcoxon signed-rank test
        try:
            w_stat, p_val_w = stats.wilcoxon(np.array(oas_scud) - np.array(oas_base))
        except Exception:
            w_stat, p_val_w = 0.0, 1.0

        # Cohen's d effect size
        diff = np.array(oas_scud) - np.array(oas_base)
        cohen_d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-8))

        print(f"  SCUD vs LC+FD Baseline -> Mean Diff: {np.mean(oas_scud)-np.mean(oas_base):+6.2f}% | p(t-test): {p_val_t:.4f} | p(Wilcoxon): {p_val_w:.4f} | Cohen's d: {cohen_d:.3f}")
        print(f"  SCUD vs Random Baseline -> Mean Diff: {np.mean(oas_scud)-np.mean(oas_rnd):+6.2f}% | p(t-test): {stats.ttest_rel(oas_scud, oas_rnd)[1]:.4f}")

        ablation_rows.append({
            "Dataset": ds_name,
            "Baseline OA (%)": f"{np.mean(oas_base):.2f} ± {np.std(oas_base, ddof=1):.2f}",
            "SCUD-AL OA (%)": f"{np.mean(oas_scud):.2f} ± {np.std(oas_scud, ddof=1):.2f}",
            "Mean Difference (%)": f"{np.mean(oas_scud)-np.mean(oas_base):+.2f}",
            "Paired t-test p-val": f"{p_val_t:.4f}",
            "Wilcoxon p-val": f"{p_val_w:.4f}",
            "Cohen's d": f"{cohen_d:.3f}",
            "Statistical Significance": "Significant (p < 0.05)" if p_val_t < 0.05 else "Not Significant (p >= 0.05)",
        })

    ablation_csv_path = "logs/al_experiments/ablation_results.csv"
    with open(ablation_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)
    print(f"Saved ablation summary CSV to {ablation_csv_path}")

    # Generate Publication Figures in logs/al_experiments/figures/
    generate_publication_figures(master_results, datasets_meta)

    total_pipeline_time = time.time() - t0_pipeline
    print(f"\nPipeline execution completed successfully in {total_pipeline_time/60:.2f} minutes!")

def generate_publication_figures(master_results, datasets_meta):
    print("\nGenerating publication-quality figures in logs/al_experiments/figures/...")
    
    # Figure 1: OA vs Labeled Samples across Datasets
    plt.figure(figsize=(12, 4.5))
    for idx, (ds_name, meta) in enumerate(datasets_meta.items()):
        plt.subplot(1, 3, idx + 1)
        ds_res = master_results[ds_name]
        for strat in ["Random", "Least Confidence", "LC + Farthest Diversity", "SCUD-AL Candidate (Drop 0.1)"]:
            seed_runs = ds_res[strat]
            x_counts = [r["num_labeled"] for r in seed_runs[0]]
            y_means = [np.mean([sr[r_i]["metrics"]["OverallAccuracy"] * 100 for sr in seed_runs]) for r_i in range(len(x_counts))]
            plt.plot(x_counts, y_means, label=strat, linewidth=1.8, marker="o" if "SCUD" in strat else "s")
        plt.title(ds_name, fontsize=11, fontweight="bold")
        plt.xlabel("Labeled Samples", fontsize=10)
        if idx == 0:
            plt.ylabel("Overall Accuracy (%)", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        if idx == 2:
            plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig("logs/al_experiments/figures/oa_vs_labeled.png", dpi=300)
    plt.close()

    # Figure 2: AA vs Labeled Samples across Datasets
    plt.figure(figsize=(12, 4.5))
    for idx, (ds_name, meta) in enumerate(datasets_meta.items()):
        plt.subplot(1, 3, idx + 1)
        ds_res = master_results[ds_name]
        for strat in ["Random", "Least Confidence", "LC + Farthest Diversity", "Class-Balanced Baseline"]:
            seed_runs = ds_res[strat]
            x_counts = [r["num_labeled"] for r in seed_runs[0]]
            y_means = [np.mean([sr[r_i]["metrics"]["AverageAccuracy"] * 100 for sr in seed_runs]) for r_i in range(len(x_counts))]
            plt.plot(x_counts, y_means, label=strat, linewidth=1.8)
        plt.title(f"{ds_name} — Average Accuracy", fontsize=11, fontweight="bold")
        plt.xlabel("Labeled Samples", fontsize=10)
        if idx == 0:
            plt.ylabel("Average Accuracy (%)", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        if idx == 2:
            plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig("logs/al_experiments/figures/aa_vs_labeled.png", dpi=300)
    plt.close()

    # Figure 3: Macro F1 vs Labeled Samples
    plt.figure(figsize=(12, 4.5))
    for idx, (ds_name, meta) in enumerate(datasets_meta.items()):
        plt.subplot(1, 3, idx + 1)
        ds_res = master_results[ds_name]
        for strat in ["Random", "Least Confidence", "LC + Farthest Diversity", "SCUD-AL Candidate (Drop 0.1)"]:
            seed_runs = ds_res[strat]
            x_counts = [r["num_labeled"] for r in seed_runs[0]]
            y_means = [np.mean([sr[r_i]["metrics"]["F1Score"] for sr in seed_runs]) for r_i in range(len(x_counts))]
            plt.plot(x_counts, y_means, label=strat, linewidth=1.8)
        plt.title(f"{ds_name} — Macro F1", fontsize=11, fontweight="bold")
        plt.xlabel("Labeled Samples", fontsize=10)
        if idx == 0:
            plt.ylabel("Macro F1 Score", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        if idx == 2:
            plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig("logs/al_experiments/figures/f1_vs_labeled.png", dpi=300)
    plt.close()

    # Figure 4: ECE Before / After Calibration
    plt.figure(figsize=(7, 5))
    calib_strats = ["LC + Farthest Diversity", "Calibrated LC + Diversity", "Calibrated LC + Labeled-Pool Diversity"]
    datasets_list = list(datasets_meta.keys())
    x = np.arange(len(datasets_list))
    width = 0.25

    uncalib_eces = [np.mean([sr[-1]["metrics"]["ECE"] for sr in master_results[d]["LC + Farthest Diversity"]]) for d in datasets_list]
    calib_eces = [np.mean([sr[-1]["metrics"]["ECE"] for sr in master_results[d]["Calibrated LC + Diversity"]]) for d in datasets_list]

    plt.bar(x - width/2, uncalib_eces, width, label="Uncalibrated LC+FD", color="#d62728")
    plt.bar(x + width/2, calib_eces, width, label="Calibrated LC+FD", color="#2ca02c")
    plt.xticks(x, datasets_list, fontweight="bold")
    plt.ylabel("Expected Calibration Error (ECE)", fontsize=11)
    plt.title("Expected Calibration Error (ECE) Before & After Temperature Calibration", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig("logs/al_experiments/figures/ece_before_after_calibration.png", dpi=300)
    plt.close()

    # Figure 5: Multi-seed Mean ± Std Curves
    plt.figure(figsize=(8, 6))
    ds_name = "Indian Pines"
    ds_res = master_results[ds_name]
    for strat in ["Random", "Least Confidence", "LC + Farthest Diversity", "SCUD-AL Candidate (Drop 0.1)"]:
        seed_runs = ds_res[strat]
        x_counts = [r["num_labeled"] for r in seed_runs[0]]
        y_means = [np.mean([sr[r_i]["metrics"]["OverallAccuracy"] * 100 for sr in seed_runs]) for r_i in range(len(x_counts))]
        y_stds = [np.std([sr[r_i]["metrics"]["OverallAccuracy"] * 100 for sr in seed_runs], ddof=1) for r_i in range(len(x_counts))]
        plt.plot(x_counts, y_means, label=strat, linewidth=2, marker="o")
        plt.fill_between(x_counts, np.array(y_means) - np.array(y_stds), np.array(y_means) + np.array(y_stds), alpha=0.15)
    plt.title(f"{ds_name} — 5-Seed Multi-Seed Validation (Mean ± Std)", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Labeled Samples", fontsize=11)
    plt.ylabel("Overall Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("logs/al_experiments/figures/multiseed_mean_std_curves.png", dpi=300)
    plt.close()

    print("Publication figures generated successfully under logs/al_experiments/figures/!")

if __name__ == "__main__":
    run_master_pipeline()
