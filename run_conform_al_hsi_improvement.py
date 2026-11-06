"""
ConformAL-HSI Diagnostic & Systematic Improvement Validation Suite

Empirically validates:
1. Original Architecture (Frozen 3D-ResNet50 + 256-dim MLP + Euclidean Diversity)
2. Improved Architecture (Frozen 3D-ResNet50 + 512-dim Tuned MLP + Cosine Annealing)
3. Improved Active Learning Query Engine (Cosine-Space Farthest Traversal relative to Labeled Pool + Representativeness)
4. Calibrated High-Confidence Pseudo-Labeling (tau >= 0.95, weight 0.20, class capped)
"""

import os
import sys
import time
import json
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.learning_engine.classifier import ClassifierHead
from src.learning_engine.improved_classifier import ImprovedClassifierHead
from src.utils.metrics_engine import compute_detailed_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_CONFIGS = {
    "Indian Pines": {
        "slug": "indian_pines",
        "num_classes": 16,
        "budgets": [100, 250, 645, 980],
        "init_seed_per_class": 5,
    },
    "Salinas": {
        "slug": "salinas",
        "num_classes": 16,
        "budgets": [100, 250, 645, 980],
        "init_seed_per_class": 5,
    },
    "Pavia University": {
        "slug": "pavia_university",
        "num_classes": 9,
        "budgets": [100, 250, 645, 980],
        "init_seed_per_class": 5,
    },
}

CONFIGURATIONS = [
    ("Baseline (Original Head + Random)", "orig_random"),
    ("Baseline (Original Head + Uncertainty)", "orig_unc"),
    ("Baseline (Original Head + Unc+Div)", "orig_unc_div"),
    ("Improved Head + Random", "imp_random"),
    ("Improved Head + Uncertainty", "imp_unc"),
    ("Improved Head + Improved Active Learning", "imp_al"),
    ("Improved Head + Improved AL + Pseudo-Labeling", "imp_al_pseudo"),
]


def load_dataset_features(dataset_slug):
    cache_dir = os.path.join(os.path.dirname(__file__), "cache", "preextracted_features")
    train_path = os.path.join(cache_dir, f"{dataset_slug}_train_feats.pt")
    test_path = os.path.join(cache_dir, f"{dataset_slug}_test_feats.pt")

    train_data = torch.load(train_path, map_location="cpu", weights_only=False)
    test_data = torch.load(test_path, map_location="cpu", weights_only=False)

    return {
        "train_feats": train_data["feats"].float(),
        "train_lbls": train_data["lbls"].long(),
        "test_feats": test_data["feats"].float(),
        "test_lbls": test_data["lbls"].long(),
    }


def sample_initial_seed(labels, num_classes, samples_per_class, seed):
    rng = np.random.RandomState(seed)
    labeled_indices = []
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)

    for c in range(num_classes):
        c_indices = np.where(labels_np == c)[0]
        if len(c_indices) == 0:
            continue
        n_select = min(samples_per_class, len(c_indices))
        selected = rng.choice(c_indices, size=n_select, replace=False)
        labeled_indices.extend(selected.tolist())

    labeled_indices = np.array(sorted(list(set(labeled_indices))), dtype=np.int64)
    all_indices = np.arange(len(labels_np), dtype=np.int64)
    unlabeled_mask = np.ones(len(labels_np), dtype=bool)
    unlabeled_mask[labeled_indices] = False
    unlabeled_indices = all_indices[unlabeled_mask]

    return labeled_indices, unlabeled_indices


def select_samples(
    config_code,
    k,
    unlabeled_idx,
    train_feats,
    train_feats_l2,
    probabilities,
    labeled_idx,
    density_scores,
    rng,
):
    if len(unlabeled_idx) <= k:
        return unlabeled_idx

    # 1. Random Selection
    if "random" in config_code:
        chosen = rng.choice(len(unlabeled_idx), size=k, replace=False)
        return unlabeled_idx[chosen]

    # Uncertainty calculation (Least Confidence)
    max_probs = np.max(probabilities, axis=1)
    uncertainty_scores = 1.0 - max_probs

    # 2. Pure Uncertainty Selection
    if config_code in {"orig_unc", "imp_unc"}:
        top_k = np.argsort(-uncertainty_scores, kind="stable")[:k]
        return unlabeled_idx[top_k]

    # 3. Original Uncertainty + Diversity (Raw Euclidean)
    if config_code == "orig_unc_div":
        cand_size = min(len(unlabeled_idx), max(4 * k, 50))
        top_cand = np.argsort(-uncertainty_scores, kind="stable")[:cand_size]
        cand_raw = train_feats[unlabeled_idx[top_cand]].to(DEVICE)

        with torch.no_grad():
            sel_cand = [0]
            dists = torch.norm(cand_raw - cand_raw[0:1], dim=1)
            for _ in range(1, k):
                farthest = torch.argmax(dists).item()
                sel_cand.append(farthest)
                new_dists = torch.norm(cand_raw - cand_raw[farthest : farthest + 1], dim=1)
                dists = torch.minimum(dists, new_dists)

        return unlabeled_idx[top_cand[sel_cand]]

    # 4. Improved Active Learning (Cosine Diversity relative to Labeled Pool + Density Prior)
    if config_code in {"imp_al", "imp_al_pseudo"}:
        # Combine uncertainty with density prior
        dens = density_scores[unlabeled_idx]
        dens_norm = (dens - dens.min()) / (dens.max() - dens.min() + 1e-8)
        # Moderate density weighting (alpha=0.3) prevents boundary outlier trapping
        acq_scores = uncertainty_scores * (dens_norm ** 0.3)

        cand_size = min(len(unlabeled_idx), max(4 * k, 100))
        top_cand = np.argsort(-acq_scores, kind="stable")[:cand_size]

        cand_l2 = train_feats_l2[unlabeled_idx[top_cand]].to(DEVICE)
        lab_l2 = train_feats_l2[labeled_idx].to(DEVICE)

        with torch.no_grad():
            # Initial distance is distance to nearest already-labeled sample in Cosine space
            dots_to_lab = torch.matmul(cand_l2, lab_l2.t())
            max_dot, _ = torch.max(dots_to_lab, dim=1)
            dists = 1.0 - max_dot

            sel_cand = [torch.argmax(dists).item()]
            for _ in range(1, k):
                new_dots = torch.matmul(cand_l2, cand_l2[sel_cand[-1] : sel_cand[-1] + 1].t()).squeeze(1)
                dists = torch.minimum(dists, 1.0 - new_dots)
                sel_cand.append(torch.argmax(dists).item())

        return unlabeled_idx[top_cand[sel_cand]]

    raise ValueError(f"Unknown config: {config_code}")


def train_model(
    config_code,
    train_feats,
    train_feats_l2,
    train_lbls,
    labeled_idx,
    num_classes,
    pseudo_idx=None,
    pseudo_lbls=None,
    epochs=None,
):
    is_improved = "imp_" in config_code
    if is_improved:
        model = ImprovedClassifierHead(
            in_features=2048,
            num_classes=num_classes,
            hidden_dim=512,
            dropout=0.1,
            norm_type="batchnorm",
        ).to(DEVICE)
        actual_epochs = epochs or 45
        lr = 2e-3
        weight_decay = 1e-3
        use_cosine_sched = True
        feats_to_use = train_feats_l2
    else:
        model = ClassifierHead(
            in_features=2048,
            num_classes=num_classes,
            hidden_dim=256,
            dropout=0.0,
            head_type="mlp_2layer",
            norm_layer="layernorm",
        ).to(DEVICE)
        actual_epochs = epochs or 25
        lr = 1e-3
        weight_decay = 1e-4
        use_cosine_sched = False
        feats_to_use = train_feats

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=actual_epochs) if use_cosine_sched else None
    crit = nn.CrossEntropyLoss()

    lab_x = feats_to_use[labeled_idx].to(DEVICE)
    lab_y = train_lbls[labeled_idx].to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(lab_x, lab_y),
        batch_size=64,
        shuffle=True,
        drop_last=(len(lab_x) > 64 and len(lab_x) % 64 == 1),
    )

    has_pseudo = pseudo_idx is not None and len(pseudo_idx) > 1
    if has_pseudo:
        p_x = feats_to_use[pseudo_idx].to(DEVICE)
        p_y = torch.as_tensor(pseudo_lbls, dtype=torch.long, device=DEVICE)
        p_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(p_x, p_y),
            batch_size=64,
            shuffle=True,
            drop_last=(len(p_x) > 64 and len(p_x) % 64 == 1),
        )
        p_iter = iter(p_loader)

    t0 = time.time()
    model.train()
    for _ in range(actual_epochs):
        for bx, by in loader:
            if bx.size(0) <= 1:
                continue
            opt.zero_grad()
            out = model(bx, normalize_input=False) if is_improved else model(bx)
            loss = crit(out, by)

            if has_pseudo:
                try:
                    px_b, py_b = next(p_iter)
                except StopIteration:
                    p_iter = iter(p_loader)
                    px_b, py_b = next(p_iter)
                if px_b.size(0) > 1:
                    out_p = model(px_b, normalize_input=False) if is_improved else model(px_b)
                    loss = loss + 0.20 * crit(out_p, py_b)

            loss.backward()
            opt.step()
        if sched:
            sched.step()

    train_time = time.time() - t0
    return model, train_time


def evaluate_model(model, test_feats, test_feats_l2, test_lbls, num_classes, is_improved):
    model.eval()
    t0 = time.time()
    feats = test_feats_l2 if is_improved else test_feats
    with torch.no_grad():
        test_x = feats.to(DEVICE)
        logits = model(test_x, normalize_input=False) if is_improved else model(test_x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
    inference_time = time.time() - t0

    y_true = test_lbls.numpy()
    metrics = compute_detailed_metrics(
        y_true=y_true,
        y_pred=preds,
        y_prob=probs,
        num_classes=num_classes,
    )
    return metrics, probs, preds, inference_time


def run_conform_al_hsi_study(seeds=(42, 43, 44), out_dir="results/improved_framework"):
    os.makedirs(out_dir, exist_ok=True)
    all_results = []
    best_confusion_matrices = {}

    print("=" * 96, flush=True)
    print("CONFORMAL-HSI: SYSTEMATIC ACTIVE LEARNING DIAGNOSIS & IMPROVEMENT BENCHMARK", flush=True)
    print(f"Device: {DEVICE} | Seeds: {seeds} | Datasets: {list(DATASET_CONFIGS.keys())}", flush=True)
    print("=" * 96, flush=True)

    for ds_name, cfg in DATASET_CONFIGS.items():
        print(f"\n=======================================================", flush=True)
        print(f">>> EVALUATING DATASET: {ds_name.upper()} (C={cfg['num_classes']}) <<<", flush=True)
        print(f"=======================================================", flush=True)

        feats_dict = load_dataset_features(cfg["slug"])
        train_feats = feats_dict["train_feats"]
        train_lbls = feats_dict["train_lbls"]
        test_feats = feats_dict["test_feats"]
        test_lbls = feats_dict["test_lbls"]

        # Precompute L2-normalized representations
        train_feats_l2 = train_feats / torch.clamp(torch.norm(train_feats, dim=1, keepdim=True), min=1e-8)
        test_feats_l2 = test_feats / torch.clamp(torch.norm(test_feats, dim=1, keepdim=True), min=1e-8)

        # Precompute dataset centroid and density scores in Cosine space
        centroid = torch.mean(train_feats_l2, dim=0, keepdim=True)
        centroid = centroid / torch.norm(centroid, dim=1, keepdim=True)
        density_scores = torch.matmul(train_feats_l2, centroid.t()).squeeze(1).numpy()

        best_oa_per_config = {}

        for config_label, config_code in CONFIGURATIONS:
            print(f"\n  --- Running Configuration: {config_label} ---", flush=True)
            is_improved = "imp_" in config_code

            for seed in seeds:
                rng = np.random.RandomState(seed)
                torch.manual_seed(seed)

                labeled_idx, unlabeled_idx = sample_initial_seed(
                    labels=train_lbls,
                    num_classes=cfg["num_classes"],
                    samples_per_class=cfg["init_seed_per_class"],
                    seed=seed,
                )

                for target_budget in cfg["budgets"]:
                    needed = target_budget - len(labeled_idx)
                    if needed > 0:
                        # Extract predictions on unlabeled pool
                        temp_model, _ = train_model(
                            config_code=config_code,
                            train_feats=train_feats,
                            train_feats_l2=train_feats_l2,
                            train_lbls=train_lbls,
                            labeled_idx=labeled_idx,
                            num_classes=cfg["num_classes"],
                            epochs=20,
                        )
                        temp_model.eval()
                        with torch.no_grad():
                            u_x = (train_feats_l2 if is_improved else train_feats)[unlabeled_idx].to(DEVICE)
                            u_logits = temp_model(u_x, normalize_input=False) if is_improved else temp_model(u_x)
                            u_probs = torch.softmax(u_logits, dim=-1).cpu().numpy()

                        new_selected = select_samples(
                            config_code=config_code,
                            k=min(needed, len(unlabeled_idx)),
                            unlabeled_idx=unlabeled_idx,
                            train_feats=train_feats,
                            train_feats_l2=train_feats_l2,
                            probabilities=u_probs,
                            labeled_idx=labeled_idx,
                            density_scores=density_scores,
                            rng=rng,
                        )

                        labeled_idx = np.concatenate([labeled_idx, new_selected])
                        mask = np.ones(len(unlabeled_idx), dtype=bool)
                        set_new = set(new_selected)
                        for i, u in enumerate(unlabeled_idx):
                            if u in set_new:
                                mask[i] = False
                        unlabeled_idx = unlabeled_idx[mask]

                    # Optional Pseudo-Labeling
                    pseudo_idx = None
                    pseudo_lbls = None
                    num_pseudo = 0
                    if config_code == "imp_al_pseudo" and len(unlabeled_idx) > 0:
                        eval_model, _ = train_model(
                            config_code=config_code,
                            train_feats=train_feats,
                            train_feats_l2=train_feats_l2,
                            train_lbls=train_lbls,
                            labeled_idx=labeled_idx,
                            num_classes=cfg["num_classes"],
                            epochs=25,
                        )
                        eval_model.eval()
                        with torch.no_grad():
                            rem_x = train_feats_l2[unlabeled_idx].to(DEVICE)
                            rem_probs = torch.softmax(eval_model(rem_x, normalize_input=False), dim=-1).cpu().numpy()

                        max_conf = np.max(rem_probs, axis=1)
                        pred_c = np.argmax(rem_probs, axis=1)
                        p_mask = max_conf >= 0.95

                        if np.any(p_mask):
                            cand_p_idx = unlabeled_idx[p_mask]
                            cand_p_lbl = pred_c[p_mask]
                            final_p_idx, final_p_lbl = [], []
                            for c in range(cfg["num_classes"]):
                                c_mask = cand_p_lbl == c
                                if np.any(c_mask):
                                    sub_p = cand_p_idx[c_mask][:20]
                                    final_p_idx.extend(sub_p.tolist())
                                    final_p_lbl.extend([c] * len(sub_p))
                            if final_p_idx:
                                pseudo_idx = np.array(final_p_idx, dtype=np.int64)
                                pseudo_lbls = np.array(final_p_lbl, dtype=np.int64)
                                num_pseudo = len(pseudo_idx)

                    # Retrain final model for this milestone
                    final_model, train_time = train_model(
                        config_code=config_code,
                        train_feats=train_feats,
                        train_feats_l2=train_feats_l2,
                        train_lbls=train_lbls,
                        labeled_idx=labeled_idx,
                        num_classes=cfg["num_classes"],
                        pseudo_idx=pseudo_idx,
                        pseudo_lbls=pseudo_lbls,
                        epochs=45 if is_improved else 25,
                    )

                    metrics, probs, preds, inf_time = evaluate_model(
                        final_model, test_feats, test_feats_l2, test_lbls, cfg["num_classes"], is_improved
                    )

                    oa = metrics["OverallAccuracy"] * 100.0
                    aa = metrics["AverageAccuracy"] * 100.0
                    kappa = metrics["Kappa"]
                    f1 = metrics["F1Score"]

                    record = {
                        "dataset": ds_name,
                        "configuration": config_label,
                        "config_code": config_code,
                        "is_improved_head": is_improved,
                        "seed": seed,
                        "target_budget": target_budget,
                        "genuine_labeled": len(labeled_idx),
                        "pseudo_labeled": num_pseudo,
                        "overall_accuracy": oa,
                        "average_accuracy": aa,
                        "kappa": kappa,
                        "macro_f1": f1,
                        "train_time_ms": train_time * 1000.0,
                        "inference_time_ms": inf_time * 1000.0,
                    }
                    all_results.append(record)

                    print(
                        f"    [Seed {seed}] Budget {len(labeled_idx):3d} (Pseudo: {num_pseudo:2d}) -> "
                        f"OA: {oa:5.2f}% | AA: {aa:5.2f}% | Kappa: {kappa:.4f} | F1: {f1:.4f} | Train: {train_time*1000:.0f}ms",
                        flush=True,
                    )

                    if target_budget == cfg["budgets"][-1]:
                        if config_label not in best_oa_per_config or oa > best_oa_per_config[config_label]["oa"]:
                            best_oa_per_config[config_label] = {
                                "oa": oa,
                                "cm": metrics["ConfusionMatrix"],
                                "dataset": ds_name,
                            }

        best_confusion_matrices[ds_name] = best_oa_per_config

    save_outputs(all_results, best_confusion_matrices, out_dir)
    return all_results, best_confusion_matrices


def save_outputs(all_results, best_confusion_matrices, out_dir):
    json_path = os.path.join(out_dir, "improved_framework_results.json")
    csv_path = os.path.join(out_dir, "improved_framework_results.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    fieldnames = [
        "dataset", "configuration", "config_code", "is_improved_head", "seed",
        "target_budget", "genuine_labeled", "pseudo_labeled", "overall_accuracy",
        "average_accuracy", "kappa", "macro_f1", "train_time_ms", "inference_time_ms"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n[Saved] CSV: {csv_path}", flush=True)
    print(f"[Saved] JSON: {json_path}", flush=True)

    plot_comparison_figures(all_results, best_confusion_matrices, out_dir)


def plot_comparison_figures(all_results, best_confusion_matrices, out_dir):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    datasets = list(DATASET_CONFIGS.keys())

    # Combined 3-Panel Scaling Plot
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), dpi=300)

    color_scheme = {
        "Baseline (Original Head + Random)": ("#7f7f7f", "--", "o"),
        "Baseline (Original Head + Uncertainty)": ("#d62728", "--", "s"),
        "Baseline (Original Head + Unc+Div)": ("#9467bd", "--", "^"),
        "Improved Head + Random": ("#1f77b4", "-", "o"),
        "Improved Head + Uncertainty": ("#ff7f0e", "-", "s"),
        "Improved Head + Improved Active Learning": ("#2ca02c", "-", "^"),
        "Improved Head + Improved AL + Pseudo-Labeling": ("#008080", "-", "D"),
    }

    for idx, ds_name in enumerate(datasets):
        ax = axes[idx]
        ds_records = [r for r in all_results if r["dataset"] == ds_name]
        budgets = DATASET_CONFIGS[ds_name]["budgets"]

        for config_label, _ in CONFIGURATIONS:
            strat_records = [r for r in ds_records if r["configuration"] == config_label]
            means = [np.mean([r["overall_accuracy"] for r in strat_records if r["target_budget"] == b]) for b in budgets]
            stds = [np.std([r["overall_accuracy"] for r in strat_records if r["target_budget"] == b]) for b in budgets]
            means, stds = np.array(means), np.array(stds)

            color, style, marker = color_scheme.get(config_label, ("black", "-", "o"))
            is_imp_al = "Improved Active Learning" in config_label
            lw = 2.6 if is_imp_al else 1.8
            ms = 8 if is_imp_al else 6

            ax.plot(budgets, means, label=config_label, color=color, linestyle=style, marker=marker, linewidth=lw, markersize=ms)
            if is_imp_al or "Random" in config_label:
                ax.fill_between(budgets, means - stds, means + stds, color=color, alpha=0.12)

        ax.set_title(f"{ds_name}: Architecture & Sampling Scaling", fontsize=12, fontweight="bold")
        ax.set_xlabel("Genuine Labeled Samples", fontsize=11)
        ax.set_ylabel("Overall Accuracy (%)", fontsize=11)
        ax.set_xticks(budgets)
        ax.grid(True, linestyle="--", alpha=0.55)
        if idx == 0:
            ax.legend(frameon=True, fontsize=7.5, loc="lower right")

    plt.tight_layout()
    comb_path = os.path.join(out_dir, "accuracy_vs_budget_comparison_all_datasets.png")
    plt.savefig(comb_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Combined Comparison Graph: {comb_path}", flush=True)

    # Plot Confusion Matrices for Improved Head + Improved Active Learning at Budget 980
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    for idx, ds_name in enumerate(datasets):
        ax = axes[idx]
        cfg_dict = best_confusion_matrices.get(ds_name, {})
        target = "Improved Head + Improved Active Learning"
        if target not in cfg_dict and cfg_dict:
            target = list(cfg_dict.keys())[-1]

        cm = np.array(cfg_dict[target]["cm"])
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype("float") / row_sums

        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Greens)
        ax.set_title(f"{ds_name} (Budget 980)\n{target}\n(OA: {cfg_dict[target]['oa']:.1f}%)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=9)
        ax.set_ylabel("True Class", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrices_improved_active_learning.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Confusion Matrices: {cm_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--out_dir", type=str, default="results/improved_framework", help="Output directory")
    args = parser.parse_args()

    run_conform_al_hsi_study(seeds=args.seeds, out_dir=args.out_dir)

