"""
ConformAL-HSI: Controlled Three-Way Active Learning Comparison
=============================================================
Strictly controlled comparison between:
  1. Random Sampling (baseline)
  2. Least Confidence + Farthest-First Diversity (LC + FD) - Original method
  3. Margin Sampling + Cosine Diversity (Margin + Cosine) - Updated method

Experimental Controls:
  - Architecture: Original ClassifierHead (2048 -> LayerNorm -> Linear(2048, 256) -> LayerNorm(256) -> GELU -> Linear(256, C))
  - Backbone: Pretrained SpectralEarth 3D-ResNet50 frozen embeddings (2048-dim)
  - Initial pool: 10 samples per class, shared identically across methods per (dataset, seed)
  - Acquisition: 20 samples per round (batch size k=20)
  - Training: AdamW (lr=0.0005, weight_decay=1e-4, CosineAnnealingLR 25 epochs, batch_size=64)
  - Seeds: 42, 43, 44, 45, 46
  - Datasets: Indian Pines, Salinas, Pavia University
  - Checkpoint metric: Exact labels required to cross 60% OA
"""

import os
import sys
import time
import json
import csv
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.learning_engine.classifier import ClassifierHead
from src.utils.metrics_engine import compute_detailed_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "original_vs_updated_active_learning")
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_CONFIGS = {
    "Indian Pines": {
        "slug": "indian_pines",
        "num_classes": 16,
        "init_samples_per_class": 10,
        "num_rounds": 10,
        "k_per_round": 20,
    },
    "Salinas": {
        "slug": "salinas",
        "num_classes": 16,
        "init_samples_per_class": 10,
        "num_rounds": 10,
        "k_per_round": 20,
    },
    "Pavia University": {
        "slug": "pavia_university",
        "num_classes": 9,
        "init_samples_per_class": 10,
        "num_rounds": 10,
        "k_per_round": 20,
    },
}

STRATEGIES = [
    ("Random Sampling", "random"),
    ("Least Confidence + Farthest Diversity (LC + FD)", "lc_fd"),
    ("Margin Sampling + Cosine Diversity (Margin + Cosine)", "margin_cosine"),
]

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


# ==============================================================================
# FEATURE LOADING & PAIRED SEED INITIALIZATION
# ==============================================================================

def load_dataset_features(slug):
    cache_dir = os.path.join(os.path.dirname(__file__), "cache", "preextracted_features")
    train_data = torch.load(os.path.join(cache_dir, f"{slug}_train_feats.pt"), map_location="cpu", weights_only=False)
    val_data = torch.load(os.path.join(cache_dir, f"{slug}_val_feats.pt"), map_location="cpu", weights_only=False)
    test_data = torch.load(os.path.join(cache_dir, f"{slug}_test_feats.pt"), map_location="cpu", weights_only=False)

    return {
        "train_feats": train_data["feats"].float(),
        "train_lbls": train_data["lbls"].long(),
        "val_feats": val_data["feats"].float(),
        "val_lbls": val_data["lbls"].long(),
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


# ==============================================================================
# ORIGINAL CLASSIFIER HEAD TRAINING & EVALUATION
# ==============================================================================

def train_original_head(train_feats, train_lbls, labeled_idx, num_classes, epochs=25, lr=0.0005):
    """
    Original ClassifierHead training protocol:
    in_features=2048, hidden_dim=256, dropout=0.0, head_type="mlp_2layer", norm_layer="layernorm"
    AdamW (lr=0.0005, weight_decay=1e-4), CosineAnnealingLR, batch_size=64.
    """
    model = ClassifierHead(
        in_features=2048,
        num_classes=num_classes,
        hidden_dim=256,
        dropout=0.0,
        head_type="mlp_2layer",
        norm_layer="layernorm",
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=1e-5)
    crit = nn.CrossEntropyLoss()

    lab_x = train_feats[labeled_idx].to(DEVICE)
    lab_y = train_lbls[labeled_idx].to(DEVICE)

    ds = torch.utils.data.TensorDataset(lab_x, lab_y)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    t0 = time.time()
    model.train()
    epoch_loss = 0.0

    for _ in range(epochs):
        epoch_loss = 0.0
        for bx, by in loader:
            opt.zero_grad()
            out = model(bx)
            loss = crit(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item() * bx.size(0)
        sched.step()

    train_time = time.time() - t0
    avg_loss = epoch_loss / max(1, len(lab_x))
    return model, train_time, avg_loss


def evaluate_original_head(model, test_feats, test_lbls, num_classes):
    model.eval()
    t0 = time.time()
    probs_list = []
    with torch.no_grad():
        for bi in range(0, len(test_feats), 2048):
            bx = test_feats[bi : bi + 2048].to(DEVICE)
            logits = model(bx)
            probs_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(probs_list, axis=0)
    preds = np.argmax(probs, axis=-1)
    inf_time = time.time() - t0

    metrics = compute_detailed_metrics(test_lbls.numpy(), preds, probs, num_classes)
    return metrics, probs, preds, inf_time


# ==============================================================================
# THREE ACQUISITION STRATEGIES
# ==============================================================================

def select_samples_three_way(
    strat_code,
    k,
    unlabeled_idx,
    train_feats,
    train_feats_l2,
    train_lbls,
    labeled_idx,
    model,
    rng,
):
    if len(unlabeled_idx) <= k:
        return unlabeled_idx, 0.0

    t0 = time.time()

    # 1. Random Sampling
    if strat_code == "random":
        chosen = rng.choice(len(unlabeled_idx), size=k, replace=False)
        return unlabeled_idx[chosen], time.time() - t0

    # Extract model probabilities on unlabeled pool
    model.eval()
    u_probs_list = []
    with torch.no_grad():
        for bi in range(0, len(unlabeled_idx), 2048):
            bx = train_feats[unlabeled_idx[bi : bi + 2048]].to(DEVICE)
            logits = model(bx)
            u_probs_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(u_probs_list, axis=0)

    # 2. Least Confidence + Farthest-First Diversity (LC + FD) - Original Method
    if strat_code == "lc_fd":
        max_probs = np.max(probs, axis=1)
        unc_scores = 1.0 - max_probs

        cand_size = min(len(unlabeled_idx), max(4 * k, 50))
        top_cand = np.argsort(-unc_scores, kind="stable")[:cand_size]
        cand_raw = train_feats[unlabeled_idx[top_cand]].to(DEVICE)

        with torch.no_grad():
            sel_cand = [0]
            dists = torch.norm(cand_raw - cand_raw[0:1], dim=1)
            for _ in range(1, k):
                farthest = torch.argmax(dists).item()
                sel_cand.append(farthest)
                new_dists = torch.norm(cand_raw - cand_raw[farthest : farthest + 1], dim=1)
                dists = torch.minimum(dists, new_dists)

        return unlabeled_idx[top_cand[sel_cand]], time.time() - t0

    # 3. Margin Sampling + Cosine Diversity (Margin + Cosine) - Updated Method
    if strat_code == "margin_cosine":
        sorted_probs = np.sort(probs, axis=1)
        margin_unc = 1.0 - (sorted_probs[:, -1] - sorted_probs[:, -2])

        cand_size = min(len(unlabeled_idx), max(4 * k, 200))
        top_cand = np.argsort(-margin_unc, kind="stable")[:cand_size]

        cand_l2 = train_feats_l2[unlabeled_idx[top_cand]].to(DEVICE)
        lab_l2 = train_feats_l2[labeled_idx].to(DEVICE)

        with torch.no_grad():
            dots_to_lab = torch.matmul(cand_l2, lab_l2.t())
            max_dot, _ = torch.max(dots_to_lab, dim=1)
            dists = 1.0 - max_dot

            sel_cand = [torch.argmax(dists).item()]
            dists[sel_cand[-1]] = -1.0
            for _ in range(1, k):
                new_dots = torch.matmul(cand_l2, cand_l2[sel_cand[-1] : sel_cand[-1] + 1].t()).squeeze(1)
                dists = torch.minimum(dists, 1.0 - new_dots)
                dists[sel_cand[-1]] = -1.0
                sel_cand.append(torch.argmax(dists).item())

        return unlabeled_idx[top_cand[sel_cand]], time.time() - t0

    raise ValueError(f"Unknown strategy code: {strat_code}")


# ==============================================================================
# MAIN BENCHMARK EXECUTION
# ==============================================================================

def run_controlled_comparison(seeds=DEFAULT_SEEDS):
    print("=" * 110, flush=True)
    print("CONFORMAL-HSI: CONTROLLED THREE-WAY ACTIVE LEARNING COMPARISON", flush=True)
    print("Methods: 1. Random Sampling | 2. Original LC + FD | 3. Updated Margin + Cosine", flush=True)
    print(f"Classifier: Original Head (2048 -> LayerNorm -> Linear(256) -> LayerNorm -> GELU -> Classes)", flush=True)
    print(f"Seeds: {seeds} | Batch Size per Round k=20 | Evaluation Milestone: 60% OA Crossing", flush=True)
    print("=" * 110, flush=True)

    csv_seed_path = os.path.join(OUT_DIR, "three_way_al_seed_results.csv")
    csv_per_class_path = os.path.join(OUT_DIR, "three_way_al_per_class_metrics.csv")
    csv_runtime_path = os.path.join(OUT_DIR, "three_way_al_runtime.csv")

    existing_runs = set()
    if os.path.exists(csv_seed_path):
        df_ex = pd.read_csv(csv_seed_path)
        for _, r in df_ex.iterrows():
            existing_runs.add((r["dataset"], r["strategy_code"], int(r["seed"]), int(r["round"])))

    for ds_name, cfg in DATASET_CONFIGS.items():
        print(f"\n==================================================================", flush=True)
        print(f">>> DATASET: {ds_name} (Classes={cfg['num_classes']}, Init Samples/Class={cfg['init_samples_per_class']}) <<<", flush=True)
        print(f"==================================================================", flush=True)

        feats = load_dataset_features(cfg["slug"])
        train_x = feats["train_feats"]
        train_y = feats["train_lbls"]
        test_x = feats["test_feats"]
        test_y = feats["test_lbls"]
        num_classes = cfg["num_classes"]

        # Precompute L2 normalized features for Cosine space
        train_x_l2 = train_x / torch.clamp(torch.norm(train_x, dim=1, keepdim=True), min=1e-8)

        for seed in seeds:
            print(f"\n--- [Dataset: {ds_name} | Seed {seed}] ---", flush=True)
            # Sample shared initial labeled set for strict paired control
            init_labeled, init_unlabeled = sample_initial_seed(
                train_y, num_classes, cfg["init_samples_per_class"], seed
            )
            init_count = len(init_labeled)

            for strat_name, strat_code in STRATEGIES:
                print(f"  >>> Strategy: {strat_name} ({strat_code}) <<<", flush=True)
                rng = np.random.RandomState(seed)

                # Reset to shared initial labeled pool
                labeled_idx = np.array(init_labeled, copy=True)
                unlabeled_idx = np.array(init_unlabeled, copy=True)

                for r in range(cfg["num_rounds"] + 1):
                    # Check if already completed
                    if (ds_name, strat_code, seed, r) in existing_runs:
                        continue

                    acq_time = 0.0

                    # Acquire additional samples starting from round 1
                    if r > 0:
                        # Train temporary model on current labeled set to compute query scores
                        temp_model, _, _ = train_original_head(
                            train_feats=train_x,
                            train_lbls=train_y,
                            labeled_idx=labeled_idx,
                            num_classes=num_classes,
                            epochs=25,
                        )
                        new_samples, acq_time = select_samples_three_way(
                            strat_code=strat_code,
                            k=cfg["k_per_round"],
                            unlabeled_idx=unlabeled_idx,
                            train_feats=train_x,
                            train_feats_l2=train_x_l2,
                            train_lbls=train_y,
                            labeled_idx=labeled_idx,
                            model=temp_model,
                            rng=rng,
                        )
                        labeled_idx = np.concatenate([labeled_idx, new_samples])
                        set_new = set(new_samples)
                        unlabeled_idx = np.array([u for u in unlabeled_idx if u not in set_new])

                    # Train final model on current labeled set
                    final_model, train_time, train_loss = train_original_head(
                        train_feats=train_x,
                        train_lbls=train_y,
                        labeled_idx=labeled_idx,
                        num_classes=num_classes,
                        epochs=25,
                    )

                    # Evaluate on strictly held-out test set
                    metrics, probs, preds, inf_time = evaluate_original_head(
                        model=final_model,
                        test_feats=test_x,
                        test_lbls=test_y,
                        num_classes=num_classes,
                    )

                    oa = metrics["OverallAccuracy"] * 100.0
                    aa = metrics["AverageAccuracy"] * 100.0
                    macro_f1 = metrics["F1Score"]
                    kappa = metrics["Kappa"]
                    cur_labels_count = len(labeled_idx)

                    # Check 60% OA threshold status
                    reached_60 = oa >= 60.0

                    status_str = f"Round {r:>2} | N={cur_labels_count:>3} | OA={oa:5.2f}% | AA={aa:5.2f}% | F1={macro_f1:6.4f} | Loss={train_loss:5.3f} | Acq={acq_time:4.2f}s"
                    if reached_60:
                        status_str += " [* >= 60% OA]"
                    print(f"    {status_str}", flush=True)

                    rec = {
                        "dataset": ds_name,
                        "strategy": strat_name,
                        "strategy_code": strat_code,
                        "seed": seed,
                        "round": r,
                        "initial_labels": init_count,
                        "acquired_labels": cur_labels_count - init_count,
                        "total_labels": cur_labels_count,
                        "oa": round(oa, 2),
                        "aa": round(aa, 2),
                        "kappa": round(kappa, 4),
                        "macro_f1": round(macro_f1, 4),
                        "train_loss": round(train_loss, 4),
                        "train_time_s": round(train_time, 3),
                        "acq_time_s": round(acq_time, 3),
                        "inf_time_s": round(inf_time, 3),
                        "reached_60_oa": reached_60,
                    }

                    # Write seed results
                    file_exists = os.path.exists(csv_seed_path)
                    with open(csv_seed_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(rec.keys()))
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(rec)

                    existing_runs.add((ds_name, strat_code, seed, r))

                    # Append per-class metrics
                    pc_dict = metrics.get("PerClass", {})
                    pc_rows = []
                    for c_name, c_m in pc_dict.items():
                        pc_rows.append({
                            "dataset": ds_name,
                            "strategy": strat_name,
                            "strategy_code": strat_code,
                            "seed": seed,
                            "round": r,
                            "total_labels": cur_labels_count,
                            "class_name": c_name,
                            "precision": round(c_m["precision"], 4),
                            "recall": round(c_m["recall"], 4),
                            "f1": round(c_m["f1"], 4),
                            "support": c_m["support"],
                        })
                    pc_file_exists = os.path.exists(csv_per_class_path)
                    pd.DataFrame(pc_rows).to_csv(csv_per_class_path, mode="a", header=not pc_file_exists, index=False)

                    # Append runtime
                    rt_rec = {
                        "dataset": ds_name,
                        "strategy": strat_name,
                        "strategy_code": strat_code,
                        "seed": seed,
                        "round": r,
                        "total_labels": cur_labels_count,
                        "train_time_s": round(train_time, 3),
                        "acq_time_s": round(acq_time, 3),
                        "inf_time_s": round(inf_time, 3),
                    }
                    rt_file_exists = os.path.exists(csv_runtime_path)
                    pd.DataFrame([rt_rec]).to_csv(csv_runtime_path, mode="a", header=not rt_file_exists, index=False)

    print(f"\n[Completed Benchmark Runs] Output saved to {OUT_DIR}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Controlled Three-Way Active Learning Comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds")
    args = parser.parse_args()

    run_controlled_comparison(seeds=args.seeds)

