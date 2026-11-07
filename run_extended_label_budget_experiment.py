"""
ConformAL-HSI: Extended Label-Budget Accuracy Experiment
=========================================================
Investigates whether increasing the labeled training pixels (from N=220 up to N=2000)
allows top-performing Active Learning methods to achieve 90-99% accuracy on:
  - Indian Pines (Selected: Margin + Cosine Diversity)
  - Salinas (Selected: Margin + Cosine Diversity)
  - Pavia University (Selected: Margin Sampling)

Followed by Phase B: Random Sampling comparison at identical budgets.
Budgets: 220, 300, 400, 500, 700, 1000, 1500, 2000.
Seeds: 42, 43, 44, 45, 46.
Classifier: Head B (Improved 512: 2048 -> 512 -> BatchNorm -> GELU -> Dropout(0.1) -> Classes).
Optimizer: AdamW (lr=2e-3, weight_decay=1e-3, CosineAnnealingLR, 45 epochs, batch_size=64).
Backbone: Frozen SpectralEarth 3D-ResNet50 (2048-d L2-normalized embeddings, requires_grad=False).
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

from src.learning_engine.temperature_scaling import TemperatureScaler
from src.utils.metrics_engine import compute_detailed_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "extended_label_budget_study")
os.makedirs(OUT_DIR, exist_ok=True)

# Datasets and their assigned best active learning strategy
DATASET_CONFIGS = {
    "Indian Pines": {
        "slug": "indian_pines",
        "num_classes": 16,
        "init_seed_per_class": 5,
        "total_pixels": 10249,
        "train_pool_size": 7174,
        "al_strategy_name": "Margin + Cosine Diversity",
        "al_strategy_code": "hybrid_margin_cosine",
    },
    "Salinas": {
        "slug": "salinas",
        "num_classes": 16,
        "init_seed_per_class": 5,
        "total_pixels": 56975,
        "train_pool_size": 37890,
        "al_strategy_name": "Margin + Cosine Diversity",
        "al_strategy_code": "hybrid_margin_cosine",
    },
    "Pavia University": {
        "slug": "pavia_university",
        "num_classes": 9,
        "init_seed_per_class": 5,
        "total_pixels": 42776,
        "train_pool_size": 29943,
        "al_strategy_name": "Margin Sampling",
        "al_strategy_code": "margin",
    },
}

ACQUISITION_BUDGET_STEPS = [100, 140, 180, 220, 300, 400, 500, 700, 1000, 1500, 2000]
EVALUATION_BUDGETS = [220, 300, 400, 500, 700, 1000, 1500, 2000]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


# ==============================================================================
# CLASSIFIER HEAD B (IMPROVED 512)
# ==============================================================================

class HeadB_Improved512(nn.Module):
    """Head B: Improved 512 units (2048 -> 512 -> BatchNorm -> GELU -> Dropout(0.1) -> Classes)"""
    def __init__(self, in_features=2048, num_classes=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(512, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, normalize_input=True):
        if normalize_input:
            x = x / torch.clamp(torch.norm(x, p=2, dim=-1, keepdim=True), min=1e-8)
        # Drop last batch guard: if single sample during train, evaluate cleanly
        if self.training and x.dim() > 1 and x.size(0) == 1:
            self.eval()
            out = self.net(x)
            self.train()
            return out
        return self.net(x)


# ==============================================================================
# FEATURE LOADING & INITIAL SEED SELECTION
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
# MODEL TRAINING & EVALUATION
# ==============================================================================

def train_classifier(
    train_feats_l2,
    train_lbls,
    labeled_idx,
    num_classes,
    epochs=45,
    lr=2e-3,
    weight_decay=1e-3,
):
    model = HeadB_Improved512(2048, num_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    lab_x = train_feats_l2[labeled_idx].to(DEVICE)
    lab_y = train_lbls[labeled_idx].to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(lab_x, lab_y),
        batch_size=64,
        shuffle=True,
        drop_last=(len(lab_x) > 64 and len(lab_x) % 64 == 1),
    )

    t0 = time.time()
    model.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0

    for _ in range(epochs):
        epoch_loss = 0.0
        for bx, by in loader:
            if bx.size(0) <= 1:
                continue
            opt.zero_grad()
            out = model(bx, normalize_input=False)
            loss = crit(out, by)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * bx.size(0)
            preds = torch.argmax(out, dim=1)
            correct_train += (preds == by).sum().item()
            total_train += bx.size(0)
        sched.step()

    train_time = time.time() - t0
    avg_train_loss = epoch_loss / max(1, len(lab_x))
    train_acc = (correct_train / max(1, total_train)) * 100.0

    return model, train_time, avg_train_loss, train_acc


def evaluate_model(model, test_feats_l2, test_lbls, num_classes, val_feats_l2=None, val_lbls=None):
    model.eval()
    t0 = time.time()
    probs_list = []
    with torch.no_grad():
        for bi in range(0, len(test_feats_l2), 2048):
            bx = test_feats_l2[bi : bi + 2048].to(DEVICE)
            logits = model(bx, normalize_input=False)
            probs_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(probs_list, axis=0)
    preds = np.argmax(probs, axis=-1)
    inf_time = time.time() - t0

    metrics = compute_detailed_metrics(test_lbls.numpy(), preds, probs, num_classes)

    # Validation calibration
    val_ece = 0.0
    if val_feats_l2 is not None and val_lbls is not None:
        with torch.no_grad():
            val_logits_list = []
            for bi in range(0, len(val_feats_l2), 2048):
                bx = val_feats_l2[bi : bi + 2048].to(DEVICE)
                val_logits_list.append(model(bx, normalize_input=False))
            val_logits = torch.cat(val_logits_list, dim=0)
        scaler = TemperatureScaler().to(DEVICE)
        learned_T = scaler.fit(val_logits, val_lbls.to(DEVICE))
        with torch.no_grad():
            calib_val_probs = scaler.calibrate_probs(val_logits).cpu().numpy()
        m_val = compute_detailed_metrics(val_lbls.numpy(), np.argmax(calib_val_probs, axis=-1), calib_val_probs, num_classes)
        val_ece = float(m_val.get("ECE", 0.0))

    return metrics, probs, preds, inf_time, val_ece


# ==============================================================================
# ACQUISITION SELECTION FUNCTION
# ==============================================================================

def select_query_batch(
    strategy_code,
    k,
    unlabeled_idx,
    train_feats_l2,
    train_lbls,
    labeled_idx,
    model,
    num_classes,
    rng,
):
    if len(unlabeled_idx) <= k:
        return unlabeled_idx, 0.0

    t0 = time.time()

    # 1. Random Sampling
    if strategy_code == "random":
        chosen = rng.choice(len(unlabeled_idx), size=k, replace=False)
        return unlabeled_idx[chosen], time.time() - t0

    # Extract model probabilities on unlabeled pool
    model.eval()
    u_probs_list = []
    with torch.no_grad():
        for bi in range(0, len(unlabeled_idx), 2048):
            bx = train_feats_l2[unlabeled_idx[bi : bi + 2048]].to(DEVICE)
            logits = model(bx, normalize_input=False)
            u_probs_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(u_probs_list, axis=0)

    # 2. Margin Sampling: 1 - (p1 - p2)
    if strategy_code == "margin":
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        margin_unc = 1.0 - margin
        top_k = np.argsort(-margin_unc)[:k]
        return unlabeled_idx[top_k], time.time() - t0

    # 3. Margin + Cosine Diversity
    if strategy_code == "hybrid_margin_cosine":
        sorted_probs = np.sort(probs, axis=1)
        margin_unc = 1.0 - (sorted_probs[:, -1] - sorted_probs[:, -2])
        # Candidate pool for diversity selection
        cand_size = min(len(unlabeled_idx), max(4 * k, 200))
        top_cand = np.argsort(-margin_unc)[:cand_size]

        cand_l2 = train_feats_l2[unlabeled_idx[top_cand]].to(DEVICE)
        lab_l2 = train_feats_l2[labeled_idx].to(DEVICE)

        with torch.no_grad():
            dots = torch.matmul(cand_l2, lab_l2.t())
            dists = 1.0 - torch.max(dots, dim=1)[0]
            sel = [torch.argmax(dists).item()]
            dists[sel[-1]] = -1.0
            for _ in range(1, k):
                new_dots = torch.matmul(cand_l2, cand_l2[sel[-1] : sel[-1] + 1].t()).squeeze(1)
                dists = torch.minimum(dists, 1.0 - new_dots)
                dists[sel[-1]] = -1.0
                sel.append(torch.argmax(dists).item())

        return unlabeled_idx[top_cand[sel]], time.time() - t0

    raise ValueError(f"Unknown strategy code: {strategy_code}")


# ==============================================================================
# BENCHMARK SUITE RUNNER
# ==============================================================================

def run_extended_budget_suite(phase="all", seeds=DEFAULT_SEEDS):
    """
    Executes Phase A (Active Learning) and Phase B (Random Sampling).
    """
    print("=" * 110, flush=True)
    print(f"CONFORMAL-HSI: EXTENDED LABEL-BUDGET ACCURACY EXPERIMENT (N=220 to 2000)", flush=True)
    print(f"Phase Mode: {phase.upper()} | Seeds: {seeds} | Budgets: {EVALUATION_BUDGETS}", flush=True)
    print(f"Classifier: Head B (Improved 512, ~1.06M params, BatchNorm1d, GELU, Dropout 0.1)", flush=True)
    print(f"Backbone: Frozen SpectralEarth 3D-ResNet50 (2048-dim features, requires_grad=False)", flush=True)
    print("=" * 110, flush=True)

    csv_seed_path = os.path.join(OUT_DIR, "extended_budget_seed_results.csv")
    csv_per_class_path = os.path.join(OUT_DIR, "extended_budget_per_class_metrics.csv")
    csv_runtime_path = os.path.join(OUT_DIR, "extended_budget_runtime_memory.csv")

    existing_runs = set()
    if os.path.exists(csv_seed_path):
        df_ex = pd.read_csv(csv_seed_path)
        for _, r in df_ex.iterrows():
            existing_runs.add((r["dataset"], r["strategy_code"], int(r["seed"]), int(r["budget"])))

    phases_to_run = []
    if phase in ["A", "all"]:
        phases_to_run.append("A")
    if phase in ["B", "all"]:
        phases_to_run.append("B")

    for cur_phase in phases_to_run:
        phase_name = "PHASE A: ACTIVE LEARNING SCALING" if cur_phase == "A" else "PHASE B: RANDOM SAMPLING COMPARISON"
        print(f"\n{'#' * 110}", flush=True)
        print(f"STARTING {phase_name}", flush=True)
        print(f"{'#' * 110}\n", flush=True)

        for ds_name, cfg in DATASET_CONFIGS.items():
            if cur_phase == "A":
                strat_name = cfg["al_strategy_name"]
                strat_code = cfg["al_strategy_code"]
            else:
                strat_name = "Random Sampling"
                strat_code = "random"

            print(f"\n==================================================================", flush=True)
            print(f">>> DATASET: {ds_name} | STRATEGY: {strat_name} ({strat_code}) <<<", flush=True)
            print(f"    Train Pool: {cfg['train_pool_size']:,} | Total Pixels: {cfg['total_pixels']:,}", flush=True)
            print(f"==================================================================", flush=True)

            feats = load_dataset_features(cfg["slug"])
            train_x = feats["train_feats"]
            train_y = feats["train_lbls"]
            val_x = feats["val_feats"]
            val_y = feats["val_lbls"]
            test_x = feats["test_feats"]
            test_y = feats["test_lbls"]
            num_classes = cfg["num_classes"]

            # L2 Normalize
            train_x_l2 = train_x / torch.clamp(torch.norm(train_x, dim=1, keepdim=True), min=1e-8)
            val_x_l2 = val_x / torch.clamp(torch.norm(val_x, dim=1, keepdim=True), min=1e-8)
            test_x_l2 = test_x / torch.clamp(torch.norm(test_x, dim=1, keepdim=True), min=1e-8)

            for seed in seeds:
                rng = np.random.RandomState(seed)
                torch.manual_seed(seed)

                # Initialize initial labeled set
                labeled_idx, unlabeled_idx = sample_initial_seed(
                    train_y, num_classes, cfg["init_seed_per_class"], seed
                )
                init_seed_count = len(labeled_idx)

                # Step through all budgets cumulatively
                for target_budget in ACQUISITION_BUDGET_STEPS:
                    needed = target_budget - len(labeled_idx)
                    acq_time = 0.0

                    if needed > 0:
                        # Warm-up model to compute uncertainty
                        temp_model, _, _, _ = train_classifier(
                            train_feats_l2=train_x_l2,
                            train_lbls=train_y,
                            labeled_idx=labeled_idx,
                            num_classes=num_classes,
                            epochs=20,
                        )
                        new_selected, acq_time = select_query_batch(
                            strategy_code=strat_code,
                            k=min(needed, len(unlabeled_idx)),
                            unlabeled_idx=unlabeled_idx,
                            train_feats_l2=train_x_l2,
                            train_lbls=train_y,
                            labeled_idx=labeled_idx,
                            model=temp_model,
                            num_classes=num_classes,
                            rng=rng,
                        )
                        labeled_idx = np.concatenate([labeled_idx, new_selected])
                        set_new = set(new_selected)
                        unlabeled_idx = np.array([u for u in unlabeled_idx if u not in set_new])

                    # Only evaluate and record if target_budget is in EVALUATION_BUDGETS
                    if target_budget not in EVALUATION_BUDGETS:
                        continue

                    if (ds_name, strat_code, seed, target_budget) in existing_runs:
                        print(f"  [Skip] {ds_name} | {strat_code} | Seed {seed} | Budget {target_budget} already completed.", flush=True)
                        continue

                    # Reset peak memory stats
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    # Final training for 45 epochs
                    final_model, train_time, train_loss, train_acc = train_classifier(
                        train_feats_l2=train_x_l2,
                        train_lbls=train_y,
                        labeled_idx=labeled_idx,
                        num_classes=num_classes,
                        epochs=45,
                    )

                    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

                    # Evaluate on strictly held-out test set
                    metrics, probs, preds, inf_time, val_ece = evaluate_model(
                        model=final_model,
                        test_feats_l2=test_x_l2,
                        test_lbls=test_y,
                        num_classes=num_classes,
                        val_feats_l2=val_x_l2,
                        val_lbls=val_y,
                    )

                    oa = metrics["OverallAccuracy"] * 100.0
                    aa = metrics["AverageAccuracy"] * 100.0
                    kappa = metrics["Kappa"]
                    macro_f1 = metrics["F1Score"]
                    ece = metrics["ECE"]
                    sample_pct = (len(labeled_idx) / cfg["total_pixels"]) * 100.0

                    print(f"  [{ds_name[:8]}] {strat_code[:12]:<12} | S{seed} | N={target_budget:>4} ({sample_pct:>5.2f}%) | OA={oa:6.2f}% | AA={aa:6.2f}% | F1={macro_f1:6.4f} | Loss={train_loss:6.4f} | {train_time:4.2f}s", flush=True)

                    seed_rec = {
                        "dataset": ds_name,
                        "strategy": strat_name,
                        "strategy_code": strat_code,
                        "seed": seed,
                        "budget": target_budget,
                        "initial_labels": init_seed_count,
                        "acquired_labels": len(labeled_idx) - init_seed_count,
                        "total_labels": len(labeled_idx),
                        "sample_fraction_pct": round(sample_pct, 2),
                        "oa": round(oa, 2),
                        "aa": round(aa, 2),
                        "kappa": round(kappa, 4),
                        "macro_f1": round(macro_f1, 4),
                        "ece": round(ece, 4),
                        "val_ece": round(val_ece, 4),
                        "train_loss": round(train_loss, 4),
                        "train_acc": round(train_acc, 2),
                        "train_time_s": round(train_time, 3),
                        "acq_time_s": round(acq_time, 3),
                        "inf_time_s": round(inf_time, 3),
                        "peak_vram_mb": round(peak_vram_mb, 2),
                    }

                    # Append to seed CSV immediately for safe incremental logging
                    file_exists = os.path.exists(csv_seed_path)
                    with open(csv_seed_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(seed_rec.keys()))
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(seed_rec)

                    existing_runs.add((ds_name, strat_code, seed, target_budget))

                    # Append per-class metrics
                    pc_dict = metrics.get("PerClass", {})
                    pc_rows = []
                    for c_name, c_m in pc_dict.items():
                        pc_rows.append({
                            "dataset": ds_name,
                            "strategy": strat_name,
                            "strategy_code": strat_code,
                            "seed": seed,
                            "budget": target_budget,
                            "class_name": c_name,
                            "precision": round(c_m["precision"], 4),
                            "recall": round(c_m["recall"], 4),
                            "f1": round(c_m["f1"], 4),
                            "support": c_m["support"],
                        })
                    pc_file_exists = os.path.exists(csv_per_class_path)
                    pd.DataFrame(pc_rows).to_csv(csv_per_class_path, mode="a", header=not pc_file_exists, index=False)

                    # Append runtime memory records
                    rt_rec = {
                        "dataset": ds_name,
                        "strategy": strat_name,
                        "strategy_code": strat_code,
                        "seed": seed,
                        "budget": target_budget,
                        "train_time_s": round(train_time, 3),
                        "acq_time_s": round(acq_time, 3),
                        "inf_time_s": round(inf_time, 3),
                        "peak_vram_mb": round(peak_vram_mb, 2),
                    }
                    rt_file_exists = os.path.exists(csv_runtime_path)
                    pd.DataFrame([rt_rec]).to_csv(csv_runtime_path, mode="a", header=not rt_file_exists, index=False)

        print(f"\n[Completed {phase_name}] Results saved to {OUT_DIR}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ConformAL-HSI Extended Label-Budget Experiment")
    parser.add_argument("--phase", type=str, default="all", choices=["A", "B", "all"], help="Phase to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds")
    args = parser.parse_args()

    run_extended_budget_suite(phase=args.phase, seeds=args.seeds)

