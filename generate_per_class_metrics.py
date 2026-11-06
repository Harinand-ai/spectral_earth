"""
Generate comprehensive per_class_metrics.csv for ConformAL-HSI.
Evaluates class-wise precision, recall, F1, and sample allocation across budgets.
"""

import os
import sys
import csv
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from run_label_efficiency_optimization import (
    load_dataset_features,
    sample_initial_seed,
    train_classifier,
    select_query_batch,
    DATASET_CONFIGS,
    DEVICE,
)
from src.utils.metrics_engine import compute_detailed_metrics

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "label_efficiency_optimization")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_STRATEGIES = [
    ("Random Sampling", "random"),
    ("Pure Uncertainty (Least Confidence)", "least_confidence"),
    ("Margin Sampling", "margin"),
    ("Exact BADGE (Loss Gradient Diversity)", "badge"),
    ("Cluster Diversity (Spherical K-Means)", "cluster_diversity"),
    ("Margin + Cosine Diversity", "hybrid_margin_cosine"),
    ("BALD + Cosine Diversity", "hybrid_bald_cosine"),
    ("Adaptive Class-Coverage AL", "adaptive_coverage_al"),
]

SEEDS = [42, 43, 44]
BUDGETS = [100, 180, 220]  # Representative low, mid, peak budgets


def run_per_class_evaluation():
    records = []

    for ds_name, cfg in DATASET_CONFIGS.items():
        print(f"Per-class evaluation on {ds_name}...", flush=True)
        feats = load_dataset_features(cfg["slug"])
        train_x = feats["train_feats"]
        train_y = feats["train_lbls"]
        test_x = feats["test_feats"]
        test_y = feats["test_lbls"]
        num_classes = cfg["num_classes"]

        train_x_l2 = train_x / torch.clamp(torch.norm(train_x, dim=1, keepdim=True), min=1e-8)
        test_x_l2 = test_x / torch.clamp(torch.norm(test_x, dim=1, keepdim=True), min=1e-8)

        centroid = torch.mean(train_x_l2, dim=0, keepdim=True)
        centroid = centroid / torch.norm(centroid, dim=1, keepdim=True)
        density_scores = torch.matmul(train_x_l2, centroid.t()).squeeze(1).numpy()

        for strat_name, strat_code in TARGET_STRATEGIES:
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                torch.manual_seed(seed)

                labeled_idx, unlabeled_idx = sample_initial_seed(
                    train_y, num_classes, cfg["init_seed_per_class"], seed
                )

                for b in BUDGETS:
                    needed = b - len(labeled_idx)
                    if needed > 0:
                        temp_model, _ = train_classifier(
                            train_x_l2, train_y, labeled_idx, num_classes, epochs=20
                        )
                        new_selected, _ = select_query_batch(
                            strategy_code=strat_code,
                            k=min(needed, len(unlabeled_idx)),
                            unlabeled_idx=unlabeled_idx,
                            train_feats_l2=train_x_l2,
                            train_lbls=train_y,
                            labeled_idx=labeled_idx,
                            density_scores=density_scores,
                            model=temp_model,
                            num_classes=num_classes,
                            rng=rng,
                        )
                        labeled_idx = np.concatenate([labeled_idx, new_selected])
                        set_new = set(new_selected)
                        unlabeled_idx = np.array([u for u in unlabeled_idx if u not in set_new])

                    # Train final classifier
                    final_model, _ = train_classifier(
                        train_x_l2, train_y, labeled_idx, num_classes, epochs=45
                    )
                    final_model.eval()

                    probs_list = []
                    with torch.no_grad():
                        for bi in range(0, len(test_x_l2), 2048):
                            bx = test_x_l2[bi : bi + 2048].to(DEVICE)
                            probs_list.append(torch.softmax(final_model(bx, normalize_input=False), dim=-1).cpu().numpy())
                    probs = np.concatenate(probs_list, axis=0)
                    preds = np.argmax(probs, axis=-1)

                    m = compute_detailed_metrics(test_y.numpy(), preds, probs, num_classes)
                    per_class = m["PerClass"]

                    # Count how many training samples were allocated to each class
                    lab_counts = np.bincount(train_y[labeled_idx].numpy(), minlength=num_classes)

                    for c_idx in range(num_classes):
                        c_key = f"Class_{c_idx}"
                        c_data = per_class.get(c_key, {})
                        records.append({
                            "dataset": ds_name,
                            "strategy": strat_name,
                            "strategy_code": strat_code,
                            "seed": seed,
                            "budget": b,
                            "class_id": c_idx,
                            "class_name": c_key,
                            "labeled_count": int(lab_counts[c_idx]),
                            "test_support": c_data.get("support", 0),
                            "precision": round(c_data.get("precision", 0.0) * 100, 2),
                            "recall": round(c_data.get("recall", 0.0) * 100, 2),
                            "f1": round(c_data.get("f1", 0.0), 4),
                        })

    out_csv = os.path.join(OUT_DIR, "per_class_metrics.csv")
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv} ({len(df)} per-class records)")


if __name__ == "__main__":
    run_per_class_evaluation()

