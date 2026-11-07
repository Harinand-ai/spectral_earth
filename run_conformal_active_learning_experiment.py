"""
ConformAL-HSI: Master Conformal-Guided Active Learning Experiment Runner.

Evaluates:
1. Random Sampling (random)
2. Original LC + FD (lc_fd)
3. Margin Sampling (margin)
4. Margin + Cosine Diversity (margin_cosine)
5. Conformal Uncertainty Only (conformal_unc)
6. Conformal + Cosine Diversity (conformal_cosine) [Proposed ConformAL-HSI]
7. Conformal + Cosine + Class Coverage (conformal_cov)

Under strictly controlled identical initial pools, 5 random seeds (42-46),
and strictly isolated validation calibration and test sets.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any

from src.conformal.split_conformal import SplitConformalPredictor
from src.query_engine.class_coverage import ClassCoverageController

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = "cache/preextracted_features"
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
OUT_DIR = "results/master_research_consolidation"

DATASET_CONFIGS = {
    "indian_pines": {
        "slug": "indian_pines",
        "name": "Indian Pines",
        "num_classes": 16,
        "init_per_class": 10,
        "k_per_round": 20,
        "num_rounds": 10
    },
    "salinas": {
        "slug": "salinas",
        "name": "Salinas",
        "num_classes": 16,
        "init_per_class": 10,
        "k_per_round": 20,
        "num_rounds": 10
    },
    "pavia_university": {
        "slug": "pavia_university",
        "name": "Pavia University",
        "num_classes": 9,
        "init_per_class": 10,
        "k_per_round": 20,
        "num_rounds": 10
    }
}

STRATEGIES = [
    ("Random Sampling", "random"),
    ("Original LC + FD", "lc_fd"),
    ("Margin Sampling", "margin"),
    ("Margin + Cosine Diversity", "margin_cosine"),
    ("Conformal Uncertainty", "conformal_unc"),
    ("Conformal + Cosine Diversity", "conformal_cosine"),
    ("Conformal + Cosine + Class Coverage", "conformal_cov")
]

# -------------------------------------------------------------------------
# CLASSIFIER HEAD
# -------------------------------------------------------------------------
class StandardizedClassifierHead(nn.Module):
    """
    Standardized Head A: 2048 -> LayerNorm(2048) -> Linear(2048, 256) -> LayerNorm(256) -> GELU -> Linear(256, C).
    Provides optimal calibration and reproducible deterministic inference.
    """
    def __init__(self, in_features=2048, num_classes=16):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, 256)
        self.ln = nn.LayerNorm(256)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(256, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.input_norm(x)
        return self.fc2(self.act(self.ln(self.fc1(x))))


def train_classifier(train_feats, train_lbls, labeled_idx, num_classes, epochs=15, lr=5e-4):
    model = StandardizedClassifierHead(2048, num_classes).to(DEVICE)
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item() * bx.size(0)
        sched.step()

    train_time = time.time() - t0
    avg_loss = epoch_loss / max(1, len(lab_x))
    return model, train_time, avg_loss


def evaluate_model(model, feats, lbls, num_classes):
    model.eval()
    t0 = time.time()
    probs_list = []
    with torch.no_grad():
        for bi in range(0, len(feats), 2048):
            bx = feats[bi : bi + 2048].to(DEVICE)
            probs_list.append(F.softmax(model(bx), dim=-1).cpu().numpy())
    probs = np.concatenate(probs_list, axis=0)
    preds = np.argmax(probs, axis=-1)
    inf_time = time.time() - t0

    y_true = lbls.numpy()
    oa = float((y_true == preds).mean() * 100.0)
    
    recalls = []
    per_class = {}
    from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
    prec, rec, f1s, sup = precision_recall_fscore_support(y_true, preds, labels=list(range(num_classes)), zero_division=0)
    for c in range(num_classes):
        recalls.append(rec[c])
        per_class[f"class_{c}"] = {
            "precision": float(prec[c]),
            "recall": float(rec[c]),
            "f1": float(f1s[c]),
            "support": int(sup[c])
        }
    aa = float(np.mean(recalls) * 100.0)
    macro_f1 = float(np.mean(f1s) * 100.0)
    kappa = float(cohen_kappa_score(y_true, preds) * 100.0)
    
    return {
        "oa": oa,
        "aa": aa,
        "macro_f1": macro_f1,
        "kappa": kappa,
        "inf_time_s": inf_time,
        "per_class": per_class
    }, probs, preds


# -------------------------------------------------------------------------
# ACQUISITION CONTROLLER
# -------------------------------------------------------------------------
def acquire_samples(
    strat_code: str,
    k: int,
    unlabeled_idx: np.ndarray,
    train_feats: torch.Tensor,
    train_feats_l2: torch.Tensor,
    train_lbls: torch.Tensor,
    labeled_idx: np.ndarray,
    model: nn.Module,
    conformal_predictor: SplitConformalPredictor,
    coverage_controller: ClassCoverageController,
    num_classes: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, float]:
    if len(unlabeled_idx) <= k:
        return unlabeled_idx, 0.0

    t0 = time.time()

    # 1. Random Sampling
    if strat_code == "random":
        chosen = rng.choice(len(unlabeled_idx), size=k, replace=False)
        return unlabeled_idx[chosen], time.time() - t0

    # Extract model probabilities on candidate pool
    model.eval()
    u_probs_list = []
    with torch.no_grad():
        for bi in range(0, len(unlabeled_idx), 2048):
            bx = train_feats[unlabeled_idx[bi : bi + 2048]].to(DEVICE)
            u_probs_list.append(F.softmax(model(bx), dim=-1).cpu().numpy())
    probs = np.concatenate(u_probs_list, axis=0)

    # 2. Least Confidence + Farthest-First Diversity (Original LC+FD)
    if strat_code == "lc_fd":
        max_p = np.max(probs, axis=1)
        unc = 1.0 - max_p
        cand_size = min(len(unlabeled_idx), max(4 * k, 50))
        top_cand = np.argsort(-unc, kind="stable")[:cand_size]
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

    # 3. Pure Margin Sampling
    if strat_code == "margin":
        sorted_p = np.sort(probs, axis=1)
        margin_unc = 1.0 - (sorted_p[:, -1] - sorted_p[:, -2])
        top_cand = np.argsort(-margin_unc, kind="stable")[:k]
        return unlabeled_idx[top_cand], time.time() - t0

    # 4. Margin + Cosine Diversity
    if strat_code == "margin_cosine":
        sorted_p = np.sort(probs, axis=1)
        margin_unc = 1.0 - (sorted_p[:, -1] - sorted_p[:, -2])
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

    # 5. Conformal Uncertainty Only
    if strat_code == "conformal_unc":
        conf_unc = conformal_predictor.compute_conformal_uncertainty(probs)
        top_cand = np.argsort(-conf_unc, kind="stable")[:k]
        return unlabeled_idx[top_cand], time.time() - t0

    # 6. Conformal + Cosine Diversity (Proposed ConformAL-HSI)
    if strat_code == "conformal_cosine":
        conf_unc = conformal_predictor.compute_conformal_uncertainty(probs)
        cand_size = min(len(unlabeled_idx), max(4 * k, 200))
        top_cand = np.argsort(-conf_unc, kind="stable")[:cand_size]

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

    # 7. Conformal + Cosine + Class Coverage
    if strat_code == "conformal_cov":
        conf_unc = conformal_predictor.compute_conformal_uncertainty(probs)
        labeled_labels = train_lbls[labeled_idx].numpy()
        # Apply class-coverage multiplier
        adj_unc = coverage_controller.apply_coverage_adjustment(
            conf_unc, probs, labeled_labels, num_classes
        )
        cand_size = min(len(unlabeled_idx), max(4 * k, 200))
        top_cand = np.argsort(-adj_unc, kind="stable")[:cand_size]

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


# -------------------------------------------------------------------------
# MAIN BENCHMARK RUNNER
# -------------------------------------------------------------------------
def run_benchmark(seeds=DEFAULT_SEEDS):
    os.makedirs(OUT_DIR, exist_ok=True)
    all_seed_records = []
    all_class_records = []
    all_conformal_records = []

    print("=" * 80)
    print("CONFORMAL-HSI: MASTER ACTIVE LEARNING BENCHMARK (SEEDS 42-46)")
    print(f"Device: {DEVICE} | Datasets: {list(DATASET_CONFIGS.keys())}")
    print(f"Strategies: {[s[1] for s in STRATEGIES]}")
    print("=" * 80)

    for ds_key, cfg in DATASET_CONFIGS.items():
        slug = cfg["slug"]
        ds_name = cfg["name"]
        num_classes = cfg["num_classes"]
        init_pc = cfg["init_per_class"]
        k = cfg["k_per_round"]
        n_rounds = cfg["num_rounds"]

        print(f"\n>>> Processing Dataset: {ds_name} (Classes: {num_classes}) <<<")
        tr_data = torch.load(f"{CACHE_DIR}/{slug}_train_feats.pt", map_location="cpu", weights_only=False)
        va_data = torch.load(f"{CACHE_DIR}/{slug}_val_feats.pt", map_location="cpu", weights_only=False)
        te_data = torch.load(f"{CACHE_DIR}/{slug}_test_feats.pt", map_location="cpu", weights_only=False)

        tr_feats, tr_lbls = tr_data["feats"], tr_data["lbls"]
        va_feats, va_lbls = va_data["feats"], va_data["lbls"]
        te_feats, te_lbls = te_data["feats"], te_data["lbls"]

        tr_feats_l2 = F.normalize(tr_feats, p=2, dim=-1)

        for seed in seeds:
            print(f"  --- Running Seed {seed} ---")
            rng_init = np.random.RandomState(seed)

            # Construct identical initial labeled set for all strategies
            initial_labeled = []
            for c in range(num_classes):
                c_idx = np.where(tr_lbls.numpy() == c)[0]
                rng_init.shuffle(c_idx)
                take = min(len(c_idx), init_pc)
                initial_labeled.extend(c_idx[:take])
            initial_labeled = np.array(initial_labeled, dtype=np.int64)
            n_initial = len(initial_labeled)

            for strat_name, strat_code in STRATEGIES:
                labeled_idx = initial_labeled.copy()
                mask = np.ones(len(tr_lbls), dtype=bool)
                mask[labeled_idx] = False
                unlabeled_idx = np.where(mask)[0]

                rng_strat = np.random.RandomState(seed + 100)
                conformal_predictor = SplitConformalPredictor(alpha=0.10)
                coverage_controller = ClassCoverageController(lambda_cov=0.5)

                for r in range(n_rounds + 1):
                    total_labels = len(labeled_idx)
                    acquired_labels = total_labels - n_initial

                    # Train classifier
                    model, train_time, train_loss = train_classifier(
                        tr_feats, tr_lbls, labeled_idx, num_classes, epochs=15, lr=5e-4
                    )

                    # Conformal Calibration on Validation Set
                    val_metrics, val_probs, _ = evaluate_model(model, va_feats, va_lbls, num_classes)
                    q_hat = conformal_predictor.calibrate(val_probs, va_lbls.numpy())

                    # Evaluate on strictly held-out Test Set
                    test_metrics, test_probs, test_preds = evaluate_model(model, te_feats, te_lbls, num_classes)
                    conf_eval = conformal_predictor.evaluate_coverage(test_probs, te_lbls.numpy())

                    # Log per-class metrics
                    for c_key, c_val in test_metrics["per_class"].items():
                        all_class_records.append({
                            "dataset": ds_name,
                            "strategy": strat_name,
                            "strategy_code": strat_code,
                            "seed": seed,
                            "round": r,
                            "total_labels": total_labels,
                            "class_name": c_key,
                            "precision": c_val["precision"],
                            "recall": c_val["recall"],
                            "f1": c_val["f1"],
                            "support": c_val["support"]
                        })

                    # Log conformal diagnostic record
                    all_conformal_records.append({
                        "dataset": ds_name,
                        "strategy": strat_name,
                        "strategy_code": strat_code,
                        "seed": seed,
                        "round": r,
                        "total_labels": total_labels,
                        "empirical_coverage_pct": conf_eval["empirical_coverage_pct"],
                        "target_coverage_pct": conf_eval["target_coverage_pct"],
                        "mean_set_size": conf_eval["mean_set_size"],
                        "singleton_pct": conf_eval["singleton_pct"],
                        "multi_pct": conf_eval["multi_pct"],
                        "q_hat": q_hat
                    })

                    # Acquisition step (if not final round)
                    acq_time = 0.0
                    if r < n_rounds:
                        selected_global, acq_time = acquire_samples(
                            strat_code=strat_code,
                            k=k,
                            unlabeled_idx=unlabeled_idx,
                            train_feats=tr_feats,
                            train_feats_l2=tr_feats_l2,
                            train_lbls=tr_lbls,
                            labeled_idx=labeled_idx,
                            model=model,
                            conformal_predictor=conformal_predictor,
                            coverage_controller=coverage_controller,
                            num_classes=num_classes,
                            rng=rng_strat
                        )

                        # Update pools
                        labeled_idx = np.concatenate([labeled_idx, selected_global])
                        rem_mask = ~np.isin(unlabeled_idx, selected_global)
                        unlabeled_idx = unlabeled_idx[rem_mask]

                    # Record seed result
                    all_seed_records.append({
                        "dataset": ds_name,
                        "strategy": strat_name,
                        "strategy_code": strat_code,
                        "seed": seed,
                        "round": r,
                        "initial_labels": n_initial,
                        "acquired_labels": acquired_labels,
                        "total_labels": total_labels,
                        "oa": test_metrics["oa"],
                        "aa": test_metrics["aa"],
                        "kappa": test_metrics["kappa"],
                        "macro_f1": test_metrics["macro_f1"],
                        "empirical_coverage_pct": conf_eval["empirical_coverage_pct"],
                        "mean_set_size": conf_eval["mean_set_size"],
                        "train_loss": train_loss,
                        "train_time_s": train_time,
                        "acq_time_s": acq_time,
                        "inf_time_s": test_metrics["inf_time_s"]
                    })

            print(f"    Completed seed {seed} for all strategies.")

    # Save raw seed files
    df_seeds = pd.DataFrame(all_seed_records)
    df_seeds.to_csv(os.path.join(OUT_DIR, "conformal_al_seed_results.csv"), index=False)

    df_class = pd.DataFrame(all_class_records)
    df_class.to_csv(os.path.join(OUT_DIR, "conformal_al_per_class_metrics.csv"), index=False)

    df_conf = pd.DataFrame(all_conformal_records)
    df_conf.to_csv(os.path.join(OUT_DIR, "conformal_al_diagnostics.csv"), index=False)

    print("\nMaster benchmark run complete. Saved raw CSVs to", OUT_DIR)


if __name__ == "__main__":
    run_benchmark(DEFAULT_SEEDS)

