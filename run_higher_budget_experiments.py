"""
SpectralEarth Higher Label-Budget Active Learning Validation Suite
Evaluates classification performance across increasing annotation budgets:
- Baseline (160 for Indian Pines/Salinas, 90 for Pavia University)
- 100 labeled samples
- 250 labeled samples
- 645 labeled samples
- 980 labeled samples

Compares:
1. Random Sampling
2. Uncertainty Sampling (Least Confidence)
3. Uncertainty + Diversity Sampling (Proposed Candidate Farthest)
4. Proposed + Pseudo-Labeling (tau >= 0.95, weight = 0.20)
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
from src.utils.metrics_engine import compute_detailed_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_CONFIGS = {
    "Indian Pines": {
        "slug": "indian_pines",
        "num_classes": 16,
        "init_seed_per_class": 5,  # 16 * 5 = 80 initial seed
        "budgets": [100, 160, 250, 645, 980],
        "baseline_budget": 160,
    },
    "Salinas": {
        "slug": "salinas",
        "num_classes": 16,
        "init_seed_per_class": 5,  # 16 * 5 = 80 initial seed
        "budgets": [100, 160, 250, 645, 980],
        "baseline_budget": 160,
    },
    "Pavia University": {
        "slug": "pavia_university",
        "num_classes": 9,
        "init_seed_per_class": 5,  # 9 * 5 = 45 initial seed
        "budgets": [100, 190, 250, 645, 980],
        "baseline_budget": 190,
    },
}

STRATEGIES = [
    ("Random", "random"),
    ("Uncertainty (Least Confidence)", "uncertainty"),
    ("Proposed (Uncertainty + Diversity)", "proposed"),
    ("Proposed + Pseudo-Labeling", "proposed_pseudolabel"),
]


def load_dataset_features(dataset_slug):
    cache_dir = os.path.join(os.path.dirname(__file__), "cache", "preextracted_features")
    train_path = os.path.join(cache_dir, f"{dataset_slug}_train_feats.pt")
    test_path = os.path.join(cache_dir, f"{dataset_slug}_test_feats.pt")
    val_path = os.path.join(cache_dir, f"{dataset_slug}_val_feats.pt")

    train_data = torch.load(train_path, map_location="cpu", weights_only=False)
    test_data = torch.load(test_path, map_location="cpu", weights_only=False)
    val_data = torch.load(val_path, map_location="cpu", weights_only=False) if os.path.exists(val_path) else None

    return {
        "train_feats": train_data["feats"],
        "train_lbls": train_data["lbls"],
        "test_feats": test_data["feats"],
        "test_lbls": test_data["lbls"],
        "val_feats": val_data["feats"] if val_data else None,
        "val_lbls": val_data["lbls"] if val_data else None,
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


def select_query_batch(
    strategy,
    k,
    unlabeled_indices,
    unlabeled_embeddings,
    probabilities,
    labeled_embeddings=None,
    rng=None,
):
    if len(unlabeled_indices) <= k:
        return unlabeled_indices

    if strategy == "random":
        chosen_sub_indices = rng.choice(len(unlabeled_indices), size=k, replace=False)
        return unlabeled_indices[chosen_sub_indices]

    # Uncertainty scores (Least Confidence)
    max_probs = np.max(probabilities, axis=1)
    uncertainty_scores = 1.0 - max_probs

    if strategy == "uncertainty":
        top_k_sub = np.argsort(-uncertainty_scores, kind="stable")[:k]
        return unlabeled_indices[top_k_sub]

    if strategy in {"proposed", "proposed_pseudolabel"}:
        # Candidate filter + Farthest diversity traversal (GPU Accelerated)
        cand_pool_size = min(len(unlabeled_indices), max(4 * k, 50))
        cand_sub_indices = np.argsort(-uncertainty_scores, kind="stable")[:cand_pool_size]
        cand_embeddings = unlabeled_embeddings[cand_sub_indices]

        # Fast GPU-accelerated Farthest First Traversal
        with torch.no_grad():
            cand_t = torch.as_tensor(cand_embeddings, device=DEVICE, dtype=torch.float32)
            selected_cand_idx = [0]
            dists = torch.norm(cand_t - cand_t[0:1], dim=1)

            for _ in range(1, k):
                farthest_cand = torch.argmax(dists).item()
                selected_cand_idx.append(farthest_cand)
                new_dists = torch.norm(cand_t - cand_t[farthest_cand : farthest_cand + 1], dim=1)
                dists = torch.minimum(dists, new_dists)

        selected_sub = cand_sub_indices[selected_cand_idx]
        return unlabeled_indices[selected_sub]

    raise ValueError(f"Unknown strategy: {strategy}")


def train_classifier(
    train_feats,
    train_lbls,
    labeled_indices,
    num_classes,
    pseudo_indices=None,
    pseudo_labels=None,
    pseudo_weight=0.20,
    epochs=25,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=64,
):
    model = ClassifierHead(
        in_features=2048,
        num_classes=num_classes,
        hidden_dim=256,
        dropout=0.0,
        head_type="mlp_2layer",
        norm_layer="layernorm",
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Labeled dataset
    lab_x = train_feats[labeled_indices].to(DEVICE)
    lab_y = train_lbls[labeled_indices].to(DEVICE)
    lab_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(lab_x, lab_y),
        batch_size=batch_size,
        shuffle=True,
    )

    # Optional pseudo-labeled dataset
    has_pseudo = pseudo_indices is not None and len(pseudo_indices) > 0
    if has_pseudo:
        p_x = train_feats[pseudo_indices].to(DEVICE)
        p_y = torch.as_tensor(pseudo_labels, dtype=torch.long, device=DEVICE)
        pseudo_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(p_x, p_y),
            batch_size=batch_size,
            shuffle=True,
        )
        p_iter = iter(pseudo_loader)

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for bx, by in lab_loader:
            optimizer.zero_grad()
            out_lab = model(bx)
            loss = criterion(out_lab, by)

            if has_pseudo:
                try:
                    px_batch, py_batch = next(p_iter)
                except StopIteration:
                    p_iter = iter(pseudo_loader)
                    px_batch, py_batch = next(p_iter)
                out_p = model(px_batch)
                loss_p = criterion(out_p, py_batch)
                loss = loss + pseudo_weight * loss_p

            loss.backward()
            optimizer.step()

    train_time = time.time() - t0
    return model, train_time


def predict_pool_probabilities(model, feats, indices, batch_size=2048):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_feats = feats[batch_idx].to(DEVICE)
            probs = torch.softmax(model(batch_feats), dim=-1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, model.num_classes))
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        test_x = test_feats.to(DEVICE)
        logits = model(test_x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
    inference_time = time.time() - t0

    y_true = test_lbls.numpy() if isinstance(test_lbls, torch.Tensor) else np.array(test_lbls)
    metrics = compute_detailed_metrics(
        y_true=y_true,
        y_pred=preds,
        y_prob=probs,
        num_classes=num_classes,
    )
    return metrics, probs, preds, inference_time


def evaluate_model(model, test_feats, test_lbls, num_classes):
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        test_x = test_feats.to(DEVICE)
        logits = model(test_x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
    inference_time = time.time() - t0

    y_true = test_lbls.numpy() if isinstance(test_lbls, torch.Tensor) else np.array(test_lbls)
    metrics = compute_detailed_metrics(
        y_true=y_true,
        y_pred=preds,
        y_prob=probs,
        num_classes=num_classes,
    )
    return metrics, probs, preds, inference_time


def run_experiments(seeds=(42, 43, 44), out_dir="results/higher_budget"):
    os.makedirs(out_dir, exist_ok=True)
    all_results = []
    best_confusion_matrices = {}

    print("=" * 90, flush=True)
    print("SPECTRALEARTH HIGHER LABEL-BUDGET EXPERIMENTAL VALIDATION SUITE", flush=True)
    print(f"Device: {DEVICE} | Seeds: {seeds} | Datasets: {list(DATASET_CONFIGS.keys())}", flush=True)
    print("=" * 90, flush=True)

    for ds_name, cfg in DATASET_CONFIGS.items():
        print(f"\n>>> PROCESSING DATASET: {ds_name} (Classes={cfg['num_classes']}) <<<", flush=True)
        feats_dict = load_dataset_features(cfg["slug"])
        train_feats = feats_dict["train_feats"]
        train_lbls = feats_dict["train_lbls"]
        test_feats = feats_dict["test_feats"]
        test_lbls = feats_dict["test_lbls"]

        best_oa_per_strat = {}

        for strat_label, strat_code in STRATEGIES:
            print(f"\n  --- Strategy: {strat_label} ---", flush=True)

            for seed in seeds:
                rng = np.random.RandomState(seed)
                torch.manual_seed(seed)

                # Initialize stratified seed
                labeled_idx, unlabeled_idx = sample_initial_seed(
                    labels=train_lbls,
                    num_classes=cfg["num_classes"],
                    samples_per_class=cfg["init_seed_per_class"],
                    seed=seed,
                )

                # Progressively query to meet each target budget
                for target_budget in cfg["budgets"]:
                    needed = target_budget - len(labeled_idx)
                    if needed > 0:
                        # Extract unlabelled pool predictions
                        temp_model, _ = train_classifier(
                            train_feats=train_feats,
                            train_lbls=train_lbls,
                            labeled_indices=labeled_idx,
                            num_classes=cfg["num_classes"],
                            epochs=15,
                        )
                        u_probs = predict_pool_probabilities(temp_model, train_feats, unlabeled_idx)

                        new_selected = select_query_batch(
                            strategy=strat_code,
                            k=min(needed, len(unlabeled_idx)),
                            unlabeled_indices=unlabeled_idx,
                            unlabeled_embeddings=train_feats[unlabeled_idx].numpy(),
                            probabilities=u_probs,
                            rng=rng,
                        )

                        labeled_idx = np.concatenate([labeled_idx, new_selected])
                        mask = np.ones(len(unlabeled_idx), dtype=bool)
                        selected_set = set(new_selected)
                        for i, u_idx in enumerate(unlabeled_idx):
                            if u_idx in selected_set:
                                mask[i] = False
                        unlabeled_idx = unlabeled_idx[mask]

                    # Pseudo-label extraction if applicable
                    pseudo_idx = None
                    pseudo_lbl = None
                    num_pseudo = 0
                    if strat_code == "proposed_pseudolabel" and len(unlabeled_idx) > 0:
                        # Infer on remaining unlabelled pool
                        eval_model, _ = train_classifier(
                            train_feats=train_feats,
                            train_lbls=train_lbls,
                            labeled_indices=labeled_idx,
                            num_classes=cfg["num_classes"],
                            epochs=20,
                        )
                        rem_probs = predict_pool_probabilities(eval_model, train_feats, unlabeled_idx)

                        max_conf = np.max(rem_probs, axis=1)
                        pred_class = np.argmax(rem_probs, axis=1)
                        p_mask = max_conf >= 0.95

                        if np.any(p_mask):
                            cand_p_idx = unlabeled_idx[p_mask]
                            cand_p_lbl = pred_class[p_mask]
                            final_p_idx = []
                            final_p_lbl = []
                            for c in range(cfg["num_classes"]):
                                c_mask = cand_p_lbl == c
                                if np.any(c_mask):
                                    sub_p_idx = cand_p_idx[c_mask][:20]
                                    final_p_idx.extend(sub_p_idx.tolist())
                                    final_p_lbl.extend([c] * len(sub_p_idx))
                            if final_p_idx:
                                pseudo_idx = np.array(final_p_idx, dtype=np.int64)
                                pseudo_lbl = np.array(final_p_lbl, dtype=np.int64)
                                num_pseudo = len(pseudo_idx)

                    # Final retrain for this budget milestone
                    final_model, train_time = train_classifier(
                        train_feats=train_feats,
                        train_lbls=train_lbls,
                        labeled_indices=labeled_idx,
                        num_classes=cfg["num_classes"],
                        pseudo_indices=pseudo_idx,
                        pseudo_labels=pseudo_lbl,
                        epochs=25,
                    )

                    # Evaluate on test set
                    metrics, probs, preds, inf_time = evaluate_model(
                        final_model, test_feats, test_lbls, cfg["num_classes"]
                    )

                    oa = metrics["OverallAccuracy"] * 100.0
                    aa = metrics["AverageAccuracy"] * 100.0
                    kappa = metrics["Kappa"]

                    record = {
                        "dataset": ds_name,
                        "strategy": strat_label,
                        "strategy_code": strat_code,
                        "seed": seed,
                        "target_budget": target_budget,
                        "genuine_labeled": len(labeled_idx),
                        "pseudo_labeled": num_pseudo,
                        "overall_accuracy": oa,
                        "average_accuracy": aa,
                        "kappa": kappa,
                        "ece": metrics["ECE"],
                        "train_time_ms": train_time * 1000.0,
                        "inference_time_ms": inf_time * 1000.0,
                    }
                    all_results.append(record)

                    print(
                        f"    [Seed {seed}] Budget {len(labeled_idx):3d} (Pseudo: {num_pseudo:2d}) -> "
                        f"OA: {oa:5.2f}% | AA: {aa:5.2f}% | Kappa: {kappa:.4f} | Train: {train_time*1000:.0f}ms",
                        flush=True,
                    )

                    # Track best confusion matrix at max budget
                    if target_budget == cfg["budgets"][-1]:
                        if strat_label not in best_oa_per_strat or oa > best_oa_per_strat[strat_label]["oa"]:
                            best_oa_per_strat[strat_label] = {
                                "oa": oa,
                                "cm": metrics["ConfusionMatrix"],
                                "dataset": ds_name,
                            }

        best_confusion_matrices[ds_name] = best_oa_per_strat

    # Save CSV and JSON
    save_outputs(all_results, best_confusion_matrices, out_dir)
    return all_results, best_confusion_matrices


def save_outputs(all_results, best_confusion_matrices, out_dir):
    json_path = os.path.join(out_dir, "higher_budget_results.json")
    csv_path = os.path.join(out_dir, "higher_budget_results.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    fieldnames = [
        "dataset", "strategy", "strategy_code", "seed", "target_budget",
        "genuine_labeled", "pseudo_labeled", "overall_accuracy", "average_accuracy",
        "kappa", "ece", "train_time_ms", "inference_time_ms"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n[Saved] CSV: {csv_path}")
    print(f"[Saved] JSON: {json_path}")

    # Plot Accuracy vs Budget graphs
    plot_accuracy_curves(all_results, out_dir)
    plot_confusion_matrices(best_confusion_matrices, out_dir)


def plot_accuracy_curves(all_results, out_dir):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    datasets = list(DATASET_CONFIGS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    color_map = {
        "Random": "#7f7f7f",
        "Uncertainty (Least Confidence)": "#d62728",
        "Proposed (Uncertainty + Diversity)": "#1f77b4",
        "Proposed + Pseudo-Labeling": "#2ca02c",
    }
    marker_map = {
        "Random": "o",
        "Uncertainty (Least Confidence)": "s",
        "Proposed (Uncertainty + Diversity)": "^",
        "Proposed + Pseudo-Labeling": "D",
    }

    for idx, ds_name in enumerate(datasets):
        ax = axes[idx]
        ds_records = [r for r in all_results if r["dataset"] == ds_name]
        budgets = DATASET_CONFIGS[ds_name]["budgets"]

        for strat_label, _ in STRATEGIES:
            strat_records = [r for r in ds_records if r["strategy"] == strat_label]
            means = []
            stds = []

            for b in budgets:
                b_records = [r["overall_accuracy"] for r in strat_records if r["target_budget"] == b]
                means.append(np.mean(b_records) if b_records else 0)
                stds.append(np.std(b_records) if b_records else 0)

            means = np.array(means)
            stds = np.array(stds)

            c = color_map.get(strat_label, "black")
            m = marker_map.get(strat_label, "o")

            ax.plot(budgets, means, label=strat_label, color=c, marker=m, linewidth=2.2, markersize=7)
            ax.fill_between(budgets, means - stds, means + stds, color=c, alpha=0.15)

        ax.set_title(f"{ds_name} — Accuracy vs. Label Budget", fontsize=13, fontweight="bold")
        ax.set_xlabel("Total Genuine Labeled Samples", fontsize=11)
        ax.set_ylabel("Overall Accuracy (%)", fontsize=11)
        ax.set_xticks(budgets)
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax.legend(frameon=True, fontsize=9, loc="lower right")

    plt.tight_layout()
    all_fig_path = os.path.join(out_dir, "accuracy_vs_budget_all_datasets.png")
    plt.savefig(all_fig_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Combined Figure: {all_fig_path}")

    # Also save individual figures
    for ds_name in datasets:
        fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)
        ds_records = [r for r in all_results if r["dataset"] == ds_name]
        budgets = DATASET_CONFIGS[ds_name]["budgets"]

        for strat_label, _ in STRATEGIES:
            strat_records = [r for r in ds_records if r["strategy"] == strat_label]
            means = [np.mean([r["overall_accuracy"] for r in strat_records if r["target_budget"] == b]) for b in budgets]
            stds = [np.std([r["overall_accuracy"] for r in strat_records if r["target_budget"] == b]) for b in budgets]
            means, stds = np.array(means), np.array(stds)
            c = color_map.get(strat_label, "black")
            m = marker_map.get(strat_label, "o")
            ax.plot(budgets, means, label=strat_label, color=c, marker=m, linewidth=2.2, markersize=7)
            ax.fill_between(budgets, means - stds, means + stds, color=c, alpha=0.15)

        ax.set_title(f"{ds_name}: Classification Scaling Trajectory", fontsize=12, fontweight="bold")
        ax.set_xlabel("Total Genuine Labeled Samples", fontsize=11)
        ax.set_ylabel("Overall Accuracy (%)", fontsize=11)
        ax.set_xticks(budgets)
        ax.legend(frameon=True, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.6)
        slug = DATASET_CONFIGS[ds_name]["slug"]
        single_path = os.path.join(out_dir, f"accuracy_vs_budget_{slug}.png")
        plt.savefig(single_path, bbox_inches="tight")
        plt.close()
        print(f"[Saved] Dataset Figure: {single_path}")


def plot_confusion_matrices(best_confusion_matrices, out_dir):
    datasets = list(DATASET_CONFIGS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    for i, ds_name in enumerate(datasets):
        ax = axes[i]
        strat_dict = best_confusion_matrices.get(ds_name, {})
        # Pick Proposed (Uncertainty + Diversity) or best strategy
        target_strat = "Proposed (Uncertainty + Diversity)"
        if target_strat not in strat_dict and strat_dict:
            target_strat = list(strat_dict.keys())[0]

        cm_data = strat_dict[target_strat]["cm"]
        cm = np.array(cm_data)
        # Row-normalize for clear percentage visualization
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype("float") / row_sums

        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{ds_name} (Budget 980)\n{target_strat} (OA: {strat_dict[target_strat]['oa']:.1f}%)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrices_best_configurations.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Confusion Matrices: {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--out_dir", type=str, default="results/higher_budget", help="Output directory")
    args = parser.parse_args()

    run_experiments(seeds=args.seeds, out_dir=args.out_dir)
