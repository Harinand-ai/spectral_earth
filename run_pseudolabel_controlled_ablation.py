import os
import sys
import time
import json
import csv
import torch
import torch.nn as nn
import numpy as np
import scipy.stats

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
    """
    Train classifier head with sample-weighted loss:
        loss = (sample_weights * per_sample_losses).mean()
    where sample_weights is 1.0 for genuine labeled samples and pseudo_weight (0.2)
    for pseudo-labeled samples.
    
    This ensures that every pseudo-labeled sample contributes exactly lambda_p (0.2)
    times the weight of a genuine sample across mixed, genuine-only, and pseudo-only batches.
    """
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
    criterion = nn.CrossEntropyLoss(reduction="none")

    last_loss = 0.0
    for _ in range(epochs):
        tot_loss, tot_samples = 0.0, 0
        for x_b, y_b, is_p_b in loader:
            logits_b = head(x_b)
            sample_losses = criterion(logits_b, y_b.long())

            # Assign sample weights: 1.0 for genuine, pseudo_weight (0.2) for pseudo-labels
            sample_weights = torch.where(
                is_p_b,
                torch.as_tensor(pseudo_weight, dtype=torch.float32, device=device),
                torch.tensor(1.0, dtype=torch.float32, device=device),
            )
            loss = (sample_weights * sample_losses).mean()

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
            print(f"  --- Running {mode_label} across {len(seeds)} seeds ---")

            seed_runs = []
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                # Reset pseudo-labeler history for each seed to ensure clean bookkeeping
                pseudo_labeler.reset_history()

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

                # Round 0 (Initial Seed Fit)
                t0 = time.time()
                init_lbl_idx = dataset_manager.labeled_indices
                train_classifier(head, train_feats[init_lbl_idx], train_lbls[init_lbl_idx], epochs=25, lr=0.0005)

                p_stats = {
                    "round_pseudo_count": 0,
                    "cumulative_pseudo_count": 0,
                    "per_class": {c: 0 for c in range(num_classes)},
                    "mean_confidence": 0.0,
                }
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
                            "round_pseudo_count": p_res["round_pseudo_count"],
                            "cumulative_pseudo_count": p_res["cumulative_pseudo_count"],
                            "per_class": p_res["per_class_counts"],
                            "mean_confidence": p_res["mean_confidence"],
                        }

                # Evaluate on strictly held-out test split
                test_probs = predict_probabilities(head, test_feats)
                test_preds = np.argmax(test_probs, axis=1)
                init_metrics = compute_detailed_metrics(test_lbls.numpy(), test_preds, test_probs, num_classes=num_classes)

                history.append({
                    "round": 0,
                    "seed": seed,
                    "num_genuine_labeled": dataset_manager.num_labeled,
                    "round_pseudo_count": p_stats["round_pseudo_count"],
                    "cumulative_pseudo_count": p_stats["cumulative_pseudo_count"],
                    "per_class_pseudo_counts": p_stats["per_class"],
                    "mean_confidence": p_stats["mean_confidence"],
                    "metrics": init_metrics,
                    "runtime_s": time.time() - t0,
                })

                # Rounds 1 through 5
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

                    # Retrain genuine base
                    curr_lbl_idx = dataset_manager.labeled_indices
                    train_classifier(head, train_feats[curr_lbl_idx], train_lbls[curr_lbl_idx], epochs=25, lr=0.0005)

                    # Pseudo-labeling from remaining unlabeled pool
                    round_p_stats = {
                        "round_pseudo_count": 0,
                        "cumulative_pseudo_count": len(pseudo_labeler.seen_pseudo_indices),
                        "per_class": {c: 0 for c in range(num_classes)},
                        "mean_confidence": 0.0,
                    }
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
                                "round_pseudo_count": p_res["round_pseudo_count"],
                                "cumulative_pseudo_count": p_res["cumulative_pseudo_count"],
                                "per_class": p_res["per_class_counts"],
                                "mean_confidence": p_res["mean_confidence"],
                            }

                    # Evaluate on strictly held-out test split
                    test_probs = predict_probabilities(head, test_feats)
                    test_preds = np.argmax(test_probs, axis=1)
                    r_metrics = compute_detailed_metrics(test_lbls.numpy(), test_preds, test_probs, num_classes=num_classes)

                    history.append({
                        "round": r,
                        "seed": seed,
                        "num_genuine_labeled": dataset_manager.num_labeled,
                        "round_pseudo_count": round_p_stats["round_pseudo_count"],
                        "cumulative_pseudo_count": round_p_stats["cumulative_pseudo_count"],
                        "per_class_pseudo_counts": round_p_stats["per_class"],
                        "mean_confidence": round_p_stats["mean_confidence"],
                        "metrics": r_metrics,
                        "runtime_s": time.time() - tr0,
                    })

                seed_runs.append(history)

            key = f"{ds_name}__pseudo_{use_pseudo}"
            overall_summary[key] = seed_runs

    # Compile Final Comparative Metrics
    os.makedirs(r"d:\Projects\spectral_earth\logs\al_experiments", exist_ok=True)
    out_json = r"d:\Projects\spectral_earth\logs\al_experiments\pseudolabel_controlled_ablation.json"

    export_data = {
        "metadata": {
            "title": "SpectralEarth Pseudo-Labeling Controlled Ablation Study",
            "loss_formulation": "weighted_per_sample_loss_mean",
            "loss_equation": "loss = (sample_weights * per_sample_losses).mean(), with w_genuine = 1.0 and w_pseudo = lambda_p (0.2)",
            "std_convention": "sample_standard_deviation (ddof=1)",
            "statistical_test": "two_sided_paired_student_t_test (scipy.stats.ttest_rel)",
            "configuration": {
                "seeds": seeds,
                "total_rounds": total_rounds,
                "k_per_round": k_per_round,
                "tau_confidence_threshold": tau,
                "pseudo_loss_weight": lambda_p,
                "per_class_pseudo_cap": class_cap,
                "classifier": "ClassifierHead (mlp_2layer, hidden_dim=256, norm_layer=layernorm)",
                "feature_dim": 2048,
                "backbone": "SpectralEarth (frozen)",
            }
        },
        "datasets": {}
    }

    print("\n" + "=" * 115)
    print("FINAL 5-SEED SCIENTIFIC SUMMARY (Round 5 - Fixed Genuine Budget: 10/class + 5x20 active queries)")
    print("All standard deviations use sample standard deviation (ddof=1). Paired t-test significance: * (p < 0.05)")
    print("=" * 115)
    header = f"{'Dataset':<16} | {'Method':<18} | {'OA (%)':<16} | {'AA (%)':<16} | {'Macro F1':<13} | {'Kappa':<13} | {'Pseudo (R5/Cum)'}"
    print(header)
    print("-" * 115)

    for ds_name in datasets:
        base_runs = overall_summary[f"{ds_name}__pseudo_False"]
        prop_runs = overall_summary[f"{ds_name}__pseudo_True"]

        # Round 5 metrics
        b_oas = np.array([r[-1]["metrics"]["OverallAccuracy"] * 100 for r in base_runs])
        b_aas = np.array([r[-1]["metrics"]["AverageAccuracy"] * 100 for r in base_runs])
        b_f1s = np.array([r[-1]["metrics"]["F1Score"] for r in base_runs])
        b_kps = np.array([r[-1]["metrics"]["Kappa"] for r in base_runs])

        p_oas = np.array([r[-1]["metrics"]["OverallAccuracy"] * 100 for r in prop_runs])
        p_aas = np.array([r[-1]["metrics"]["AverageAccuracy"] * 100 for r in prop_runs])
        p_f1s = np.array([r[-1]["metrics"]["F1Score"] for r in prop_runs])
        p_kps = np.array([r[-1]["metrics"]["Kappa"] for r in prop_runs])

        p_r5_cnts = np.array([r[-1]["round_pseudo_count"] for r in prop_runs])
        p_cum_cnts = np.array([r[-1]["cumulative_pseudo_count"] for r in prop_runs])

        # Paired per-seed differences
        delta_oa = p_oas - b_oas
        delta_aa = p_aas - b_aas
        delta_f1 = p_f1s - b_f1s
        delta_kp = p_kps - b_kps

        # Paired Student's t-test
        t_oa, pval_oa = scipy.stats.ttest_rel(p_oas, b_oas)
        t_aa, pval_aa = scipy.stats.ttest_rel(p_aas, b_aas)
        t_f1, pval_f1 = scipy.stats.ttest_rel(p_f1s, b_f1s)
        t_kp, pval_kp = scipy.stats.ttest_rel(p_kps, b_kps)

        export_data["datasets"][ds_name] = {
            "round_5_summary": {
                "baseline": {
                    "OA_mean": float(np.mean(b_oas)), "OA_std": float(np.std(b_oas, ddof=1)),
                    "AA_mean": float(np.mean(b_aas)), "AA_std": float(np.std(b_aas, ddof=1)),
                    "F1_mean": float(np.mean(b_f1s)), "F1_std": float(np.std(b_f1s, ddof=1)),
                    "Kappa_mean": float(np.mean(b_kps)), "Kappa_std": float(np.std(b_kps, ddof=1)),
                    "per_seed_OA": [float(v) for v in b_oas],
                    "per_seed_AA": [float(v) for v in b_aas],
                    "per_seed_F1": [float(v) for v in b_f1s],
                    "per_seed_Kappa": [float(v) for v in b_kps],
                },
                "proposed": {
                    "OA_mean": float(np.mean(p_oas)), "OA_std": float(np.std(p_oas, ddof=1)),
                    "AA_mean": float(np.mean(p_aas)), "AA_std": float(np.std(p_aas, ddof=1)),
                    "F1_mean": float(np.mean(p_f1s)), "F1_std": float(np.std(p_f1s, ddof=1)),
                    "Kappa_mean": float(np.mean(p_kps)), "Kappa_std": float(np.std(p_kps, ddof=1)),
                    "final_round_pseudo_count_mean": float(np.mean(p_r5_cnts)),
                    "final_round_pseudo_count_std": float(np.std(p_r5_cnts, ddof=1)),
                    "cumulative_pseudo_count_mean": float(np.mean(p_cum_cnts)),
                    "cumulative_pseudo_count_std": float(np.std(p_cum_cnts, ddof=1)),
                    "per_seed_OA": [float(v) for v in p_oas],
                    "per_seed_AA": [float(v) for v in p_aas],
                    "per_seed_F1": [float(v) for v in p_f1s],
                    "per_seed_Kappa": [float(v) for v in p_kps],
                    "per_seed_final_round_pseudo": [int(v) for v in p_r5_cnts],
                    "per_seed_cumulative_pseudo": [int(v) for v in p_cum_cnts],
                },
                "paired_differences": {
                    "delta_OA": {
                        "mean": float(np.mean(delta_oa)), "std": float(np.std(delta_oa, ddof=1)),
                        "per_seed": [float(v) for v in delta_oa],
                        "t_statistic": float(t_oa), "p_value": float(pval_oa),
                        "statistically_significant": bool(pval_oa < 0.05),
                    },
                    "delta_AA": {
                        "mean": float(np.mean(delta_aa)), "std": float(np.std(delta_aa, ddof=1)),
                        "per_seed": [float(v) for v in delta_aa],
                        "t_statistic": float(t_aa), "p_value": float(pval_aa),
                        "statistically_significant": bool(pval_aa < 0.05),
                    },
                    "delta_F1": {
                        "mean": float(np.mean(delta_f1)), "std": float(np.std(delta_f1, ddof=1)),
                        "per_seed": [float(v) for v in delta_f1],
                        "t_statistic": float(t_f1), "p_value": float(pval_f1),
                        "statistically_significant": bool(pval_f1 < 0.05),
                    },
                    "delta_Kappa": {
                        "mean": float(np.mean(delta_kp)), "std": float(np.std(delta_kp, ddof=1)),
                        "per_seed": [float(v) for v in delta_kp],
                        "t_statistic": float(t_kp), "p_value": float(pval_kp),
                        "statistically_significant": bool(pval_kp < 0.05),
                    },
                }
            },
            "per_seed_trajectories": {
                "baseline": [
                    {"seed": seeds[i], "rounds": base_runs[i]} for i in range(len(seeds))
                ],
                "proposed": [
                    {"seed": seeds[i], "rounds": prop_runs[i]} for i in range(len(seeds))
                ],
            }
        }

        # Terminal printing
        b_str_oa = f"{np.mean(b_oas):.2f} ± {np.std(b_oas, ddof=1):.2f}%"
        b_str_aa = f"{np.mean(b_aas):.2f} ± {np.std(b_aas, ddof=1):.2f}%"
        b_str_f1 = f"{np.mean(b_f1s):.4f} ± {np.std(b_f1s, ddof=1):.4f}"
        b_str_kp = f"{np.mean(b_kps):.4f} ± {np.std(b_kps, ddof=1):.4f}"

        p_str_oa = f"{np.mean(p_oas):.2f} ± {np.std(p_oas, ddof=1):.2f}%"
        p_str_aa = f"{np.mean(p_aas):.2f} ± {np.std(p_aas, ddof=1):.2f}%"
        p_str_f1 = f"{np.mean(p_f1s):.4f} ± {np.std(p_f1s, ddof=1):.4f}"
        p_str_kp = f"{np.mean(p_kps):.4f} ± {np.std(p_kps, ddof=1):.4f}"
        p_cnt_str = f"{np.mean(p_r5_cnts):.0f} / {np.mean(p_cum_cnts):.0f}"

        sig_oa = "*" if pval_oa < 0.05 else " "
        sig_aa = "*" if pval_aa < 0.05 else " "
        sig_f1 = "*" if pval_f1 < 0.05 else " "
        sig_kp = "*" if pval_kp < 0.05 else " "

        d_str_oa = f"{np.mean(delta_oa):+.2f}% ({sig_oa}p={pval_oa:.3f})"
        d_str_aa = f"{np.mean(delta_aa):+.2f}% ({sig_aa}p={pval_aa:.3f})"
        d_str_f1 = f"{np.mean(delta_f1):+.4f} ({sig_f1}p={pval_f1:.3f})"
        d_str_kp = f"{np.mean(delta_kp):+.4f} ({sig_kp}p={pval_kp:.3f})"

        print(f"{ds_name:<16} | {'Baseline (LC+FD)':<18} | {b_str_oa:<16} | {b_str_aa:<16} | {b_str_f1:<13} | {b_str_kp:<13} | 0 / 0")
        print(f"{'':<16} | {'Proposed (+PL)':<18} | {p_str_oa:<16} | {p_str_aa:<16} | {p_str_f1:<13} | {p_str_kp:<13} | {p_cnt_str}")
        print(f"{'':<16} | {'Paired Delta':<18} | {d_str_oa:<16} | {d_str_aa:<16} | {d_str_f1:<13} | {d_str_kp:<13} | +{np.mean(p_r5_cnts):.0f} / +{np.mean(p_cum_cnts):.0f}")
        print("-" * 115)

    with open(out_json, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"\nDetailed per-seed ablation data and aggregate summary written to:\n  {out_json}")

if __name__ == "__main__":
    run_scientific_evaluation()
