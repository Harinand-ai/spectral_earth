import os
import sys
import time
import json
import csv
import torch
import torch.nn as nn
import numpy as np

import pyrootutils
pyrootutils.setup_root(r"d:\Projects\spectral_earth", indicator=".project-root", pythonpath=True)

from src.learning_engine import ActiveLearningDatasetManager, ConfidencePseudoLabeler
from src.learning_engine.classifier import ClassifierHead
from src.query_engine.query_engine import QueryEngine
from src.utils.metrics_engine import compute_detailed_metrics

DATASETS_META = {
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

def load_cached_feats(ds_name, split):
    slug = DATASETS_META[ds_name]["slug"]
    cache_path = os.path.join(r"d:\Projects\spectral_earth\cache\preextracted_features", f"{slug}_{split}_feats.pt")
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    return data["feats"], data["lbls"]

def train_classifier(head, genuine_feats, genuine_lbls, pseudo_feats=None, pseudo_lbls=None, pseudo_weight=0.2, epochs=25, lr=0.0005, device="cpu"):
    head.train()
    is_pseudo_list = [False] * len(genuine_feats)
    all_feats = [genuine_feats]
    all_lbls = [genuine_lbls]

    has_pseudo = (pseudo_feats is not None and len(pseudo_feats) > 0 and pseudo_weight > 0.0)
    if has_pseudo:
        all_feats.append(pseudo_feats)
        all_lbls.append(pseudo_lbls)
        is_pseudo_list.extend([True] * len(pseudo_feats))

    cat_feats = torch.cat(all_feats, dim=0).to(device)
    cat_lbls = torch.cat(all_lbls, dim=0).to(device)
    is_pseudo = torch.tensor(is_pseudo_list, dtype=torch.bool, device=device)

    ds = torch.utils.data.TensorDataset(cat_feats, cat_lbls, is_pseudo)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    last_loss = 0.0
    for _ in range(epochs):
        tot_loss, tot_samples = 0.0, 0
        for x_b, y_b, is_p_b in loader:
            logits_b = head(x_b)
            gen_mask = ~is_p_b
            gen_loss = criterion(logits_b[gen_mask], y_b[gen_mask].long()) if gen_mask.any() else torch.tensor(0.0, device=device)
            p_loss = criterion(logits_b[is_p_b], y_b[is_p_b].long()) if is_p_b.any() else torch.tensor(0.0, device=device)
            loss = gen_loss + pseudo_weight * p_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()

            tot_loss += loss.item() * len(x_b)
            tot_samples += len(x_b)
        scheduler.step()
        last_loss = tot_loss / max(1, tot_samples)
    return last_loss

def predict_probabilities(head, feats, device="cpu"):
    head.eval()
    if len(feats) == 0:
        return np.empty((0, head.num_classes))
    ds = torch.utils.data.TensorDataset(feats.to(device))
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    all_p = []
    with torch.no_grad():
        for (x_b,) in loader:
            p_b = head.predict_probabilities(x_b)
            all_p.append(p_b.cpu())
    return torch.cat(all_p, dim=0).numpy()

def run_scientific_evaluation():
    print("=" * 80)
    print("SPECTRALEARTH — CONFIDENCE-BASED PSEUDO-LABELING CONTROLLED SCIENTIFIC AUDIT")
    print("=" * 80)

    seeds = [42, 43, 44, 45, 46]
    total_rounds = 5
    k_per_round = 20
    tau = 0.95
    lambda_p = 0.2
    class_cap = 15

    datasets = ["Indian Pines", "Salinas", "Pavia University"]
    overall_summary = {}

    for ds_name in datasets:
        print(f"\nEvaluating dataset: {ds_name}...")
        meta = DATASETS_META[ds_name]
        num_classes = meta["num_classes"]
        init_samples = meta["init_samples"]

        # Instantiate datamodule for dataset structure
        mod_path, class_name = meta["module_cls"].rsplit(".", 1)
        import importlib
        module = importlib.import_module(mod_path)
        dm = getattr(module, class_name)()
        dm.setup("fit")
        dm.setup("test")

        train_ds = dm.train_dataset
        test_ds = getattr(dm, "test_dataset", None)

        train_feats, train_lbls = load_cached_feats(ds_name, "train")
        test_feats, test_lbls = load_cached_feats(ds_name, "test")

        pseudo_labeler = ConfidencePseudoLabeler(
            confidence_threshold=tau,
            max_pseudo_per_class=class_cap,
            pseudo_weight=lambda_p,
        )

        for use_pseudo in [False, True]:
            mode_label = "Proposed (+ Pseudo-Labeling)" if use_pseudo else "Baseline (LC+FD Only)"
            print(f"  --- Running {mode_label} across 5 seeds ---")

            seed_runs = []
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                dataset_manager = ActiveLearningDatasetManager(
                    train_dataset=train_ds,
                    test_dataset=test_ds,
                    initial_samples_per_class=init_samples,
                    seed=seed,
                )

                head = ClassifierHead(
                    in_features=2048,
                    num_classes=num_classes,
                    hidden_dim=256,
                    dropout=0.0,
                    head_type="mlp_2layer",
                    norm_layer="layernorm",
                )

                qe = QueryEngine(uncertainty_method="filter_diversity")
                history = []

                # Round 0
                t0 = time.time()
                init_lbl_idx = dataset_manager.labeled_indices
                train_classifier(head, train_feats[init_lbl_idx], train_lbls[init_lbl_idx], epochs=25, lr=0.0005)

                p_stats = {"count": 0, "per_class": {}, "mean_confidence": 0.0}
                if use_pseudo:
                    u_idx = dataset_manager.unlabeled_indices
                    probs_u = predict_probabilities(head, train_feats[u_idx])
                    p_res = pseudo_labeler.select_pseudo_labels(
                        probabilities=probs_u,
                        unlabeled_indices=u_idx,
                        num_classes=num_classes,
                        labeled_indices=dataset_manager.labeled_indices,
                    )
                    if p_res["total_accepted"] > 0:
                        p_feats = train_feats[p_res["pseudo_indices"]]
                        p_lbls = torch.tensor(p_res["pseudo_labels"], dtype=torch.long)
                        train_classifier(head, train_feats[init_lbl_idx], train_lbls[init_lbl_idx],
                                         pseudo_feats=p_feats, pseudo_lbls=p_lbls,
                                         pseudo_weight=lambda_p, epochs=25, lr=0.0005)
                        p_stats = {
                            "count": p_res["total_accepted"],
                            "per_class": p_res["per_class_counts"],
                            "mean_confidence": p_res["mean_confidence"],
                        }

                test_probs = predict_probabilities(head, test_feats)
                test_preds = np.argmax(test_probs, axis=1)
                init_metrics = compute_detailed_metrics(test_lbls.numpy(), test_preds, test_probs, num_classes=num_classes)

                history.append({
                    "round": 0,
                    "num_genuine_labeled": dataset_manager.num_labeled,
                    "num_pseudo": p_stats["count"],
                    "pseudo_stats": p_stats,
                    "metrics": init_metrics,
                    "time_s": time.time() - t0,
                })

                for r in range(1, total_rounds + 1):
                    tr0 = time.time()
                    u_idx = dataset_manager.unlabeled_indices
                    if len(u_idx) == 0:
                        break

                    probs_u = predict_probabilities(head, train_feats[u_idx])
                    selected_rel = qe.select(
                        embeddings=train_feats[u_idx].numpy(),
                        probabilities=probs_u,
                        k=min(k_per_round, len(u_idx)),
                        current_round=r - 1,
                        total_rounds=total_rounds,
                    )

                    dataset_manager.update_pools(selected_rel)

                    # Retrain genuine
                    curr_lbl_idx = dataset_manager.labeled_indices
                    train_classifier(head, train_feats[curr_lbl_idx], train_lbls[curr_lbl_idx], epochs=25, lr=0.0005)

                    # Score unlabeled pool for pseudo-labeling
                    round_p_stats = {"count": 0, "per_class": {}, "mean_confidence": 0.0}
                    if use_pseudo and dataset_manager.num_unlabeled > 0:
                        rem_u_idx = dataset_manager.unlabeled_indices
                        rem_probs_u = predict_probabilities(head, train_feats[rem_u_idx])
                        p_res = pseudo_labeler.select_pseudo_labels(
                            probabilities=rem_probs_u,
                            unlabeled_indices=rem_u_idx,
                            num_classes=num_classes,
                            labeled_indices=dataset_manager.labeled_indices,
                        )
                        if p_res["total_accepted"] > 0:
                            p_feats = train_feats[p_res["pseudo_indices"]]
                            p_lbls = torch.tensor(p_res["pseudo_labels"], dtype=torch.long)
                            train_classifier(head, train_feats[curr_lbl_idx], train_lbls[curr_lbl_idx],
                                             pseudo_feats=p_feats, pseudo_lbls=p_lbls,
                                             pseudo_weight=lambda_p, epochs=25, lr=0.0005)
                            round_p_stats = {
                                "count": p_res["total_accepted"],
                                "per_class": p_res["per_class_counts"],
                                "mean_confidence": p_res["mean_confidence"],
                            }

                    test_probs = predict_probabilities(head, test_feats)
                    test_preds = np.argmax(test_probs, axis=1)
                    r_metrics = compute_detailed_metrics(test_lbls.numpy(), test_preds, test_probs, num_classes=num_classes)

                    history.append({
                        "round": r,
                        "num_genuine_labeled": dataset_manager.num_labeled,
                        "num_pseudo": round_p_stats["count"],
                        "pseudo_stats": round_p_stats,
                        "metrics": r_metrics,
                        "time_s": time.time() - tr0,
                    })

                seed_runs.append(history)

            key = f"{ds_name}__pseudo_{use_pseudo}"
            overall_summary[key] = seed_runs

    # Compile Final Comparative Metrics
    os.makedirs(r"d:\Projects\spectral_earth\logs\al_experiments", exist_ok=True)
    out_json = r"d:\Projects\spectral_earth\logs\al_experiments\pseudolabel_controlled_ablation.json"

    # Export structured metrics
    export_data = {}
    print("\n" + "=" * 80)
    print("FINAL 5-SEED SCIENTIFIC SUMMARY (Round 5 - Fixed Genuine Budget)")
    print("=" * 80)
    print(f"{'Dataset':<18} | {'Method':<20} | {'OA (%)':<15} | {'AA (%)':<15} | {'Macro F1':<10} | {'Kappa':<10} | {'Pseudo Count'}")
    print("-" * 105)

    for ds_name in datasets:
        base_runs = overall_summary[f"{ds_name}__pseudo_False"]
        prop_runs = overall_summary[f"{ds_name}__pseudo_True"]

        b_oas = [r[-1]["metrics"]["OverallAccuracy"] * 100 for r in base_runs]
        b_aas = [r[-1]["metrics"]["AverageAccuracy"] * 100 for r in base_runs]
        b_f1s = [r[-1]["metrics"]["F1Score"] for r in base_runs]
        b_kps = [r[-1]["metrics"]["Kappa"] for r in base_runs]

        p_oas = [r[-1]["metrics"]["OverallAccuracy"] * 100 for r in prop_runs]
        p_aas = [r[-1]["metrics"]["AverageAccuracy"] * 100 for r in prop_runs]
        p_f1s = [r[-1]["metrics"]["F1Score"] for r in prop_runs]
        p_kps = [r[-1]["metrics"]["Kappa"] for r in prop_runs]
        p_cnts = [r[-1]["num_pseudo"] for r in prop_runs]

        export_data[ds_name] = {
            "baseline": {
                "OA_mean": float(np.mean(b_oas)), "OA_std": float(np.std(b_oas)),
                "AA_mean": float(np.mean(b_aas)), "AA_std": float(np.std(b_aas)),
                "F1_mean": float(np.mean(b_f1s)), "F1_std": float(np.std(b_f1s)),
                "Kappa_mean": float(np.mean(b_kps)), "Kappa_std": float(np.std(b_kps)),
            },
            "proposed": {
                "OA_mean": float(np.mean(p_oas)), "OA_std": float(np.std(p_oas)),
                "AA_mean": float(np.mean(p_aas)), "AA_std": float(np.std(p_aas)),
                "F1_mean": float(np.mean(p_f1s)), "F1_std": float(np.std(p_f1s)),
                "Kappa_mean": float(np.mean(p_kps)), "Kappa_std": float(np.std(p_kps)),
                "pseudo_count_mean": float(np.mean(p_cnts)), "pseudo_count_std": float(np.std(p_cnts)),
            },
            "delta": {
                "OA": float(np.mean(p_oas) - np.mean(b_oas)),
                "AA": float(np.mean(p_aas) - np.mean(b_aas)),
                "F1": float(np.mean(p_f1s) - np.mean(b_f1s)),
                "Kappa": float(np.mean(p_kps) - np.mean(b_kps)),
            }
        }

        print(f"{ds_name:<18} | {'Baseline (LC+FD)':<20} | {np.mean(b_oas):.2f} ± {np.std(b_oas):.2f}% | {np.mean(b_aas):.2f} ± {np.std(b_aas):.2f}% | {np.mean(b_f1s):.4f}     | {np.mean(b_kps):.4f}    | 0")
        print(f"{'':<18} | {'Proposed (+PL)':<20}   | {np.mean(p_oas):.2f} ± {np.std(p_oas):.2f}% | {np.mean(p_aas):.2f} ± {np.std(p_aas):.2f}% | {np.mean(p_f1s):.4f}     | {np.mean(p_kps):.4f}    | {np.mean(p_cnts):.1f}")
        print(f"{'':<18} | {'Delta':<20}           | {np.mean(p_oas)-np.mean(b_oas):+.2f}%           | {np.mean(p_aas)-np.mean(b_aas):+.2f}%           | {np.mean(p_f1s)-np.mean(b_f1s):+.4f}     | {np.mean(p_kps)-np.mean(b_kps):+.4f}    | +{np.mean(p_cnts):.1f}")
        print("-" * 105)

    with open(out_json, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"\nDetailed ablation data successfully written to {out_json}")

if __name__ == "__main__":
    run_scientific_evaluation()
