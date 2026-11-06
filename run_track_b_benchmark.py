"""
ConformAL-HSI: Track B — External Model Benchmark Execution Suite

Empirically benchmarks:
1. Linear Probe (Single Linear Layer on Frozen SpectralEarth)
2. SVM (RBF Kernel, C=10.0, gamma='scale')
3. Random Forest (200 trees)
4. ConformAL-HSI: Random Sampling (Frozen 3D-ResNet50 + 512-dim Head)
5. ConformAL-HSI: Proposed Cosine AL (Frozen 3D-ResNet50 + 512-dim Head + Cosine Pool Diversity)

Protocol:
- Datasets: Indian Pines, Salinas, Pavia University
- Budgets: [100, 260, 980]
- Seeds: [42, 43, 44]
- Splits: Identical pre-extracted train/val/test feature caches
- Output: results/track_b/
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
from scipy.special import softmax
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.learning_engine.improved_classifier import ImprovedClassifierHead
from src.utils.metrics_engine import compute_detailed_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_CONFIGS = {
    "Indian Pines": {
        "slug": "indian_pines",
        "num_classes": 16,
        "budgets": [100, 260, 980],
        "init_seed_per_class": 5,
        "total_pixels": 10249,
    },
    "Salinas": {
        "slug": "salinas",
        "num_classes": 16,
        "budgets": [100, 260, 980],
        "init_seed_per_class": 5,
        "total_pixels": 56975,
    },
    "Pavia University": {
        "slug": "pavia_university",
        "num_classes": 9,
        "budgets": [100, 260, 980],
        "init_seed_per_class": 5,
        "total_pixels": 42776,
    },
}

MODELS_TO_BENCHMARK = [
    ("Linear Probe (Frozen SpectralEarth)", "linear_probe", 2048, "Single Linear Layer (32.8K params)"),
    ("Support Vector Machine (RBF Kernel)", "svm_rbf", None, "Non-linear RBF Kernel (C=10.0, gamma=scale)"),
    ("Random Forest (200 Trees)", "random_forest", None, "Ensemble of 200 Decorrelated Decision Trees"),
    ("ConformAL-HSI (Random Sampling)", "conform_al_random", 1182224, "Frozen 3D-ResNet50 + 512-dim Head + Random"),
    ("ConformAL-HSI (Proposed Cosine AL)", "conform_al_cosine", 1182224, "Frozen 3D-ResNet50 + 512-dim Head + Cosine AL"),
]


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


def run_cosine_al_query(k, unlabeled_idx, train_feats_l2, labeled_idx, density_scores, model):
    model.eval()
    u_probs_list = []
    with torch.no_grad():
        for bi in range(0, len(unlabeled_idx), 2048):
            batch_idx = unlabeled_idx[bi : bi + 2048]
            u_x = train_feats_l2[batch_idx].to(DEVICE)
            u_logits = model(u_x, normalize_input=False)
            u_probs_list.append(torch.softmax(u_logits, dim=-1).cpu().numpy())
    u_probs = np.concatenate(u_probs_list, axis=0)

    uncertainty = 1.0 - np.max(u_probs, axis=1)
    dens = density_scores[unlabeled_idx]
    dens_norm = (dens - dens.min()) / (dens.max() - dens.min() + 1e-8)
    acq_scores = uncertainty * (dens_norm ** 0.3)

    cand_size = min(len(unlabeled_idx), max(4 * k, 100))
    top_cand = np.argsort(-acq_scores)[:cand_size]

    cand_l2 = train_feats_l2[unlabeled_idx[top_cand]].to(DEVICE)
    lab_l2 = train_feats_l2[labeled_idx].to(DEVICE)

    with torch.no_grad():
        dots = torch.matmul(cand_l2, lab_l2.t())
        dists = 1.0 - torch.max(dots, dim=1)[0]
        sel = [torch.argmax(dists).item()]
        for _ in range(1, k):
            new_dots = torch.matmul(cand_l2, cand_l2[sel[-1] : sel[-1] + 1].t()).squeeze(1)
            dists = torch.minimum(dists, 1.0 - new_dots)
            sel.append(torch.argmax(dists).item())

    return unlabeled_idx[top_cand[sel]]


def run_track_b_benchmark(seeds=(42, 43, 44), out_dir="results/track_b"):
    os.makedirs(out_dir, exist_ok=True)
    all_results = []
    per_class_results = []

    print("=" * 100, flush=True)
    print("TRACK B: NORMALIZED EXTERNAL MODEL BENCHMARK (CONFORMAL-HSI vs BASELINES)", flush=True)
    print(f"Device: {DEVICE} | Seeds: {seeds} | Budgets: [100, 260, 980]", flush=True)
    print("Strict Controls: Identical pre-extracted features, identical labeled subsets across models", flush=True)
    print("=" * 100, flush=True)

    for ds_name, cfg in DATASET_CONFIGS.items():
        print(f"\n==================================================================", flush=True)
        print(f">>> DATASET: {ds_name.upper()} (Classes={cfg['num_classes']}, Total Pixels={cfg['total_pixels']:,}) <<<", flush=True)
        print(f"==================================================================", flush=True)

        feats = load_dataset_features(cfg["slug"])
        train_x = feats["train_feats"]
        train_y = feats["train_lbls"]
        test_x = feats["test_feats"]
        test_y = feats["test_lbls"]
        num_classes = cfg["num_classes"]

        # Unit hypersphere normalized features
        train_x_l2 = train_x / torch.clamp(torch.norm(train_x, dim=1, keepdim=True), min=1e-8)
        test_x_l2 = test_x / torch.clamp(torch.norm(test_x, dim=1, keepdim=True), min=1e-8)

        # Density scores for Cosine AL
        centroid = torch.mean(train_x_l2, dim=0, keepdim=True)
        centroid = centroid / torch.norm(centroid, dim=1, keepdim=True)
        density_scores = torch.matmul(train_x_l2, centroid.t()).squeeze(1).numpy()

        for seed in seeds:
            rng = np.random.RandomState(seed)
            torch.manual_seed(seed)

            # Generate sample pools for this seed
            # 1. Random sample pools at 100, 260, 980
            init_labeled, unlabeled = sample_initial_seed(train_y, num_classes, cfg["init_seed_per_class"], seed)

            # Cumulative random subsets
            rand_pool_100 = np.concatenate([init_labeled, rng.choice(unlabeled, size=100 - len(init_labeled), replace=False)])
            rem_unlab_100 = np.array([u for u in unlabeled if u not in set(rand_pool_100)])
            rand_pool_260 = np.concatenate([rand_pool_100, rng.choice(rem_unlab_100, size=260 - len(rand_pool_100), replace=False)])
            rem_unlab_260 = np.array([u for u in unlabeled if u not in set(rand_pool_260)])
            rand_pool_980 = np.concatenate([rand_pool_260, rng.choice(rem_unlab_260, size=980 - len(rand_pool_260), replace=False)])

            rand_pools = {100: rand_pool_100, 260: rand_pool_260, 980: rand_pool_980}

            # 2. Cumulative Active Learning pool using intermediate models
            al_pool_100 = init_labeled.copy()
            # Query up to 100
            m_temp = ImprovedClassifierHead(2048, num_classes, 512, 0.1).to(DEVICE)
            opt_t = torch.optim.AdamW(m_temp.parameters(), lr=2e-3)
            ld_t = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_x_l2[al_pool_100].to(DEVICE), train_y[al_pool_100].to(DEVICE)),
                batch_size=64, shuffle=True
            )
            m_temp.train()
            for _ in range(25):
                for bx, by in ld_t:
                    if bx.size(0) <= 1:
                        continue
                    opt_t.zero_grad()
                    torch.nn.functional.cross_entropy(m_temp(bx, normalize_input=False), by).backward()
                    opt_t.step()

            unlab_al = unlabeled.copy()
            new_100 = run_cosine_al_query(100 - len(al_pool_100), unlab_al, train_x_l2, al_pool_100, density_scores, m_temp)
            al_pool_100 = np.concatenate([al_pool_100, new_100])
            unlab_al = np.array([u for u in unlab_al if u not in set(new_100)])

            # Query up to 260
            m_temp2 = ImprovedClassifierHead(2048, num_classes, 512, 0.1).to(DEVICE)
            opt_t2 = torch.optim.AdamW(m_temp2.parameters(), lr=2e-3)
            ld_t2 = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_x_l2[al_pool_100].to(DEVICE), train_y[al_pool_100].to(DEVICE)),
                batch_size=64, shuffle=True
            )
            m_temp2.train()
            for _ in range(25):
                for bx, by in ld_t2:
                    if bx.size(0) <= 1:
                        continue
                    opt_t2.zero_grad()
                    torch.nn.functional.cross_entropy(m_temp2(bx, normalize_input=False), by).backward()
                    opt_t2.step()

            new_260 = run_cosine_al_query(260 - len(al_pool_100), unlab_al, train_x_l2, al_pool_100, density_scores, m_temp2)
            al_pool_260 = np.concatenate([al_pool_100, new_260])
            unlab_al = np.array([u for u in unlab_al if u not in set(new_260)])

            # Query up to 980
            m_temp3 = ImprovedClassifierHead(2048, num_classes, 512, 0.1).to(DEVICE)
            opt_t3 = torch.optim.AdamW(m_temp3.parameters(), lr=2e-3)
            ld_t3 = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_x_l2[al_pool_260].to(DEVICE), train_y[al_pool_260].to(DEVICE)),
                batch_size=64, shuffle=True
            )
            m_temp3.train()
            for _ in range(25):
                for bx, by in ld_t3:
                    if bx.size(0) <= 1:
                        continue
                    opt_t3.zero_grad()
                    torch.nn.functional.cross_entropy(m_temp3(bx, normalize_input=False), by).backward()
                    opt_t3.step()

            new_980 = run_cosine_al_query(980 - len(al_pool_260), unlab_al, train_x_l2, al_pool_260, density_scores, m_temp3)
            al_pool_980 = np.concatenate([al_pool_260, new_980])

            al_pools = {100: al_pool_100, 260: al_pool_260, 980: al_pool_980}

            # Now evaluate each model on these exact subsets
            for budget in cfg["budgets"]:
                # --------------------------------------------------------------
                # Model 1: Linear Probe
                # --------------------------------------------------------------
                sub_idx = rand_pools[budget]
                t0 = time.time()
                lin_model = nn.Linear(2048, num_classes).to(DEVICE)
                nn.init.kaiming_normal_(lin_model.weight)
                nn.init.constant_(lin_model.bias, 0.0)
                opt_lin = torch.optim.AdamW(lin_model.parameters(), lr=2e-3, weight_decay=1e-3)
                sched_lin = torch.optim.lr_scheduler.CosineAnnealingLR(opt_lin, T_max=45)
                loader_lin = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(train_x_l2[sub_idx].to(DEVICE), train_y[sub_idx].to(DEVICE)),
                    batch_size=64, shuffle=True
                )
                lin_model.train()
                for _ in range(45):
                    for bx, by in loader_lin:
                        opt_lin.zero_grad()
                        torch.nn.functional.cross_entropy(lin_model(bx), by).backward()
                        opt_lin.step()
                    sched_lin.step()
                train_time_lin = time.time() - t0

                t0 = time.time()
                lin_model.eval()
                with torch.no_grad():
                    probs_lin_list = []
                    for bi in range(0, len(test_x_l2), 2048):
                        bx = test_x_l2[bi : bi + 2048].to(DEVICE)
                        logits_lin = lin_model(bx)
                        probs_lin_list.append(torch.softmax(logits_lin, dim=-1).cpu().numpy())
                    probs_lin = np.concatenate(probs_lin_list, axis=0)
                    preds_lin = np.argmax(probs_lin, axis=-1)
                inf_time_lin = time.time() - t0
                m_lin = compute_detailed_metrics(test_y.numpy(), preds_lin, probs_lin, num_classes)

                record_lin = {
                    "dataset": ds_name, "model": "Linear Probe", "model_category": "Foundation Linear Probe",
                    "seed": seed, "budget": budget, "sample_fraction_pct": round(budget / cfg["total_pixels"] * 100, 2),
                    "oa": round(m_lin["OverallAccuracy"] * 100, 2), "aa": round(m_lin["AverageAccuracy"] * 100, 2),
                    "kappa": round(m_lin["Kappa"], 4), "macro_f1": round(m_lin["F1Score"], 4),
                    "train_time_s": round(train_time_lin, 3), "inf_time_s": round(inf_time_lin, 4),
                    "trainable_params": 2048 * num_classes + num_classes, "backbone_frozen": True,
                }
                all_results.append(record_lin)

                # --------------------------------------------------------------
                # Model 2: Support Vector Machine (SVM-RBF)
                # --------------------------------------------------------------
                t0 = time.time()
                train_sub_np = train_x_l2[sub_idx].numpy()
                train_sub_y = train_y[sub_idx].numpy()
                test_np = test_x_l2.numpy()
                test_y_np = test_y.numpy()

                svm_model = SVC(kernel="rbf", C=10.0, gamma="scale", probability=False, random_state=seed)
                svm_model.fit(train_sub_np, train_sub_y)
                train_time_svm = time.time() - t0

                t0 = time.time()
                preds_svm = svm_model.predict(test_np)
                df_scores = svm_model.decision_function(test_np)
                if df_scores.ndim == 1:
                    probs_svm = np.zeros((len(test_np), 2))
                    probs_svm[:, 1] = 1.0 / (1.0 + np.exp(-df_scores))
                    probs_svm[:, 0] = 1.0 - probs_svm[:, 1]
                else:
                    probs_svm = softmax(df_scores, axis=1)
                inf_time_svm = time.time() - t0
                m_svm = compute_detailed_metrics(test_y_np, preds_svm, probs_svm, num_classes)

                record_svm = {
                    "dataset": ds_name, "model": "SVM (RBF Kernel)", "model_category": "Classical Kernel Machine",
                    "seed": seed, "budget": budget, "sample_fraction_pct": round(budget / cfg["total_pixels"] * 100, 2),
                    "oa": round(m_svm["OverallAccuracy"] * 100, 2), "aa": round(m_svm["AverageAccuracy"] * 100, 2),
                    "kappa": round(m_svm["Kappa"], 4), "macro_f1": round(m_svm["F1Score"], 4),
                    "train_time_s": round(train_time_svm, 3), "inf_time_s": round(inf_time_svm, 4),
                    "trainable_params": len(svm_model.support_), "backbone_frozen": True,
                }
                all_results.append(record_svm)

                # --------------------------------------------------------------
                # Model 3: Random Forest (200 Trees)
                # --------------------------------------------------------------
                t0 = time.time()
                rf_model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
                rf_model.fit(train_sub_np, train_sub_y)
                train_time_rf = time.time() - t0

                t0 = time.time()
                probs_rf = rf_model.predict_proba(test_np)
                if probs_rf.shape[1] < num_classes:
                    full_probs = np.zeros((len(test_np), num_classes))
                    for idx_c, c in enumerate(rf_model.classes_):
                        full_probs[:, c] = probs_rf[:, idx_c]
                    probs_rf = full_probs
                preds_rf = np.argmax(probs_rf, axis=-1)
                inf_time_rf = time.time() - t0
                m_rf = compute_detailed_metrics(test_y_np, preds_rf, probs_rf, num_classes)

                record_rf = {
                    "dataset": ds_name, "model": "Random Forest", "model_category": "Classical Ensemble",
                    "seed": seed, "budget": budget, "sample_fraction_pct": round(budget / cfg["total_pixels"] * 100, 2),
                    "oa": round(m_rf["OverallAccuracy"] * 100, 2), "aa": round(m_rf["AverageAccuracy"] * 100, 2),
                    "kappa": round(m_rf["Kappa"], 4), "macro_f1": round(m_rf["F1Score"], 4),
                    "train_time_s": round(train_time_rf, 3), "inf_time_s": round(inf_time_rf, 4),
                    "trainable_params": sum(tree.tree_.node_count for tree in rf_model.estimators_), "backbone_frozen": True,
                }
                all_results.append(record_rf)

                # --------------------------------------------------------------
                # Model 4: ConformAL-HSI (Random Sampling)
                # --------------------------------------------------------------
                t0 = time.time()
                c_rand_model = ImprovedClassifierHead(2048, num_classes, 512, 0.1).to(DEVICE)
                opt_cr = torch.optim.AdamW(c_rand_model.parameters(), lr=2e-3, weight_decay=1e-3)
                sched_cr = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cr, T_max=45)
                loader_cr = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(train_x_l2[sub_idx].to(DEVICE), train_y[sub_idx].to(DEVICE)),
                    batch_size=64, shuffle=True, drop_last=(len(sub_idx) > 64 and len(sub_idx) % 64 == 1)
                )
                c_rand_model.train()
                for _ in range(45):
                    for bx, by in loader_cr:
                        if bx.size(0) <= 1:
                            continue
                        opt_cr.zero_grad()
                        torch.nn.functional.cross_entropy(c_rand_model(bx, normalize_input=False), by).backward()
                        opt_cr.step()
                    sched_cr.step()
                train_time_cr = time.time() - t0

                t0 = time.time()
                c_rand_model.eval()
                with torch.no_grad():
                    probs_cr_list = []
                    for bi in range(0, len(test_x_l2), 2048):
                        bx = test_x_l2[bi : bi + 2048].to(DEVICE)
                        logits_cr = c_rand_model(bx, normalize_input=False)
                        probs_cr_list.append(torch.softmax(logits_cr, dim=-1).cpu().numpy())
                    probs_cr = np.concatenate(probs_cr_list, axis=0)
                    preds_cr = np.argmax(probs_cr, axis=-1)
                inf_time_cr = time.time() - t0
                m_cr = compute_detailed_metrics(test_y_np, preds_cr, probs_cr, num_classes)

                record_cr = {
                    "dataset": ds_name, "model": "ConformAL-HSI (Random)", "model_category": "Foundation + Trained Head",
                    "seed": seed, "budget": budget, "sample_fraction_pct": round(budget / cfg["total_pixels"] * 100, 2),
                    "oa": round(m_cr["OverallAccuracy"] * 100, 2), "aa": round(m_cr["AverageAccuracy"] * 100, 2),
                    "kappa": round(m_cr["Kappa"], 4), "macro_f1": round(m_cr["F1Score"], 4),
                    "train_time_s": round(train_time_cr, 3), "inf_time_s": round(inf_time_cr, 4),
                    "trainable_params": 1182224, "backbone_frozen": True,
                }
                all_results.append(record_cr)

                # --------------------------------------------------------------
                # Model 5: ConformAL-HSI (Proposed Cosine AL)
                # --------------------------------------------------------------
                al_sub_idx = al_pools[budget]
                t0 = time.time()
                c_al_model = ImprovedClassifierHead(2048, num_classes, 512, 0.1).to(DEVICE)
                opt_cal = torch.optim.AdamW(c_al_model.parameters(), lr=2e-3, weight_decay=1e-3)
                sched_cal = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cal, T_max=45)
                loader_cal = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(train_x_l2[al_sub_idx].to(DEVICE), train_y[al_sub_idx].to(DEVICE)),
                    batch_size=64, shuffle=True, drop_last=(len(al_sub_idx) > 64 and len(al_sub_idx) % 64 == 1)
                )
                c_al_model.train()
                for _ in range(45):
                    for bx, by in loader_cal:
                        if bx.size(0) <= 1:
                            continue
                        opt_cal.zero_grad()
                        torch.nn.functional.cross_entropy(c_al_model(bx, normalize_input=False), by).backward()
                        opt_cal.step()
                    sched_cal.step()
                train_time_cal = time.time() - t0

                t0 = time.time()
                c_al_model.eval()
                with torch.no_grad():
                    probs_cal_list = []
                    for bi in range(0, len(test_x_l2), 2048):
                        bx = test_x_l2[bi : bi + 2048].to(DEVICE)
                        logits_cal = c_al_model(bx, normalize_input=False)
                        probs_cal_list.append(torch.softmax(logits_cal, dim=-1).cpu().numpy())
                    probs_cal = np.concatenate(probs_cal_list, axis=0)
                    preds_cal = np.argmax(probs_cal, axis=-1)
                inf_time_cal = time.time() - t0
                m_cal = compute_detailed_metrics(test_y_np, preds_cal, probs_cal, num_classes)

                record_cal = {
                    "dataset": ds_name, "model": "ConformAL-HSI (Proposed AL)", "model_category": "Foundation + Trained Head + AL",
                    "seed": seed, "budget": budget, "sample_fraction_pct": round(budget / cfg["total_pixels"] * 100, 2),
                    "oa": round(m_cal["OverallAccuracy"] * 100, 2), "aa": round(m_cal["AverageAccuracy"] * 100, 2),
                    "kappa": round(m_cal["Kappa"], 4), "macro_f1": round(m_cal["F1Score"], 4),
                    "train_time_s": round(train_time_cal, 3), "inf_time_s": round(inf_time_cal, 4),
                    "trainable_params": 1182224, "backbone_frozen": True,
                }
                all_results.append(record_cal)

                print(
                    f"  [{ds_name[:6]}] Seed {seed} Budget {budget:3d} | "
                    f"LinProbe: {m_lin['OverallAccuracy']*100:5.2f}% | "
                    f"SVM: {m_svm['OverallAccuracy']*100:5.2f}% | "
                    f"RF: {m_rf['OverallAccuracy']*100:5.2f}% | "
                    f"Ours(Rand): {m_cr['OverallAccuracy']*100:5.2f}% | "
                    f"Ours(AL): {m_cal['OverallAccuracy']*100:5.2f}%",
                    flush=True
                )

                # Record per-class recalls at budget 980
                if budget == 980:
                    for c_id in range(num_classes):
                        c_mask = test_y_np == c_id
                        per_class_results.append({
                            "dataset": ds_name, "seed": seed, "budget": 980, "class_id": c_id,
                            "class_support": int(np.sum(c_mask)),
                            "linear_probe_recall": round(float((preds_lin[c_mask] == c_id).mean() * 100), 2),
                            "svm_rbf_recall": round(float((preds_svm[c_mask] == c_id).mean() * 100), 2),
                            "random_forest_recall": round(float((preds_rf[c_mask] == c_id).mean() * 100), 2),
                            "conform_al_random_recall": round(float((preds_cr[c_mask] == c_id).mean() * 100), 2),
                            "conform_al_cosine_recall": round(float((preds_cal[c_mask] == c_id).mean() * 100), 2),
                        })

    # Save benchmark_results.csv
    raw_csv = os.path.join(out_dir, "benchmark_results.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[Saved] {raw_csv}", flush=True)

    # Save per_class_metrics.csv
    pc_csv = os.path.join(out_dir, "per_class_metrics.csv")
    with open(pc_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class_results[0].keys()))
        writer.writeheader()
        writer.writerows(per_class_results)
    print(f"[Saved] {pc_csv}", flush=True)

    # Generate benchmark_summary.csv (Mean ± Sample Std across seeds)
    res_df = pd.DataFrame(all_results)
    summary_rows = []
    models = res_df["model"].unique()

    for ds_name in DATASET_CONFIGS.keys():
        ds_sub = res_df[res_df["dataset"] == ds_name]
        for m_name in models:
            m_sub = ds_sub[ds_sub["model"] == m_name]
            cat = m_sub["model_category"].iloc[0]
            params = m_sub["trainable_params"].iloc[0]

            for b in [100, 260, 980]:
                b_sub = m_sub[m_sub["budget"] == b]
                oa_vals = b_sub["oa"].values
                aa_vals = b_sub["aa"].values
                kap_vals = b_sub["kappa"].values
                f1_vals = b_sub["macro_f1"].values
                t_train = b_sub["train_time_s"].values
                t_inf = b_sub["inf_time_s"].values

                summary_rows.append({
                    "dataset": ds_name,
                    "model": m_name,
                    "model_category": cat,
                    "budget": b,
                    "sample_fraction_pct": b_sub["sample_fraction_pct"].iloc[0],
                    "trainable_params": params,
                    "oa_mean": round(float(np.mean(oa_vals)), 2),
                    "oa_sample_std": round(float(np.std(oa_vals, ddof=1)), 2),
                    "aa_mean": round(float(np.mean(aa_vals)), 2),
                    "aa_sample_std": round(float(np.std(aa_vals, ddof=1)), 2),
                    "kappa_mean": round(float(np.mean(kap_vals)), 4),
                    "kappa_sample_std": round(float(np.std(kap_vals, ddof=1)), 4),
                    "macro_f1_mean": round(float(np.mean(f1_vals)), 4),
                    "macro_f1_sample_std": round(float(np.std(f1_vals, ddof=1)), 4),
                    "train_time_mean_s": round(float(np.mean(t_train)), 3),
                    "inf_time_mean_s": round(float(np.mean(t_inf)), 4),
                })

    sum_csv = os.path.join(out_dir, "benchmark_summary.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[Saved] {sum_csv}", flush=True)

    # Generate runtime_comparison.csv
    runtime_rows = []
    for ds_name in DATASET_CONFIGS.keys():
        ds_sub = res_df[res_df["dataset"] == ds_name]
        for m_name in models:
            m_sub = ds_sub[(ds_sub["model"] == m_name) & (ds_sub["budget"] == 980)]
            runtime_rows.append({
                "dataset": ds_name,
                "model": m_name,
                "budget": 980,
                "train_time_mean_s": round(float(m_sub["train_time_s"].mean()), 3),
                "train_time_sample_std": round(float(m_sub["train_time_s"].std(ddof=1)), 3),
                "inference_time_mean_s": round(float(m_sub["inf_time_s"].mean()), 4),
                "inference_time_sample_std": round(float(m_sub["inf_time_s"].std(ddof=1)), 4),
                "throughput_samples_per_s": round(float(DATASET_CONFIGS[ds_name]["total_pixels"] / m_sub["inf_time_s"].mean()), 1),
            })

    rt_csv = os.path.join(out_dir, "runtime_comparison.csv")
    with open(rt_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(runtime_rows[0].keys()))
        writer.writeheader()
        writer.writerows(runtime_rows)
    print(f"[Saved] {rt_csv}", flush=True)

    # Generate label_efficiency_comparison.csv (Category 2: Measured Checkpoint Thresholds)
    le_rows = []
    for ds_name in DATASET_CONFIGS.keys():
        ds_sub = pd.DataFrame(summary_rows)
        ds_sub = ds_sub[ds_sub["dataset"] == ds_name]

        for target_oa in [50.0, 60.0, 70.0, 80.0]:
            for m_name in models:
                m_sub = ds_sub[ds_sub["model"] == m_name]
                reach = m_sub[m_sub["oa_mean"] >= target_oa]
                if len(reach) > 0:
                    first_b = int(reach["budget"].iloc[0])
                    first_oa = float(reach["oa_mean"].iloc[0])
                    status = f"Reached at N={first_b} ({first_oa:.2f}%)"
                else:
                    first_b = "Not Reached"
                    first_oa = "N/A"
                    status = "Failed to Reach (< 980)"

                le_rows.append({
                    "dataset": ds_name,
                    "target_oa_threshold": target_oa,
                    "model": m_name,
                    "first_measured_budget_reaching_threshold": first_b,
                    "measured_accuracy_at_threshold": first_oa,
                    "status": status,
                })

    le_csv = os.path.join(out_dir, "label_efficiency_comparison.csv")
    with open(le_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(le_rows[0].keys()))
        writer.writeheader()
        writer.writerows(le_rows)
    print(f"[Saved] {le_csv}", flush=True)

    # Generate comparison plots
    plot_track_b_figures(pd.DataFrame(summary_rows), out_dir)

    return all_results, summary_rows


def plot_track_b_figures(sum_df, out_dir):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    datasets = list(DATASET_CONFIGS.keys())
    budgets = [100, 260, 980]
    models = sum_df["model"].unique()

    color_scheme = {
        "Linear Probe": ("#7f7f7f", ":", "x"),
        "SVM (RBF Kernel)": ("#ff7f0e", "--", "s"),
        "Random Forest": ("#8c564b", "-.", "^"),
        "ConformAL-HSI (Random)": ("#1f77b4", "-", "o"),
        "ConformAL-HSI (Proposed AL)": ("#2ca02c", "-", "D"),
    }

    # 1. OA vs Budget Plot
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), dpi=300)
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        ds_data = sum_df[sum_df["dataset"] == ds]
        for m in models:
            m_data = ds_data[ds_data["model"] == m]
            means = [m_data[m_data["budget"] == b]["oa_mean"].iloc[0] for b in budgets]
            stds = [m_data[m_data["budget"] == b]["oa_sample_std"].iloc[0] for b in budgets]
            color, ls, marker = color_scheme[m]
            ax.plot(budgets, means, label=m, color=color, linestyle=ls, marker=marker, linewidth=2.0, markersize=6)
            ax.fill_between(budgets, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=color, alpha=0.10)

        ax.set_title(f"{ds}: Overall Accuracy", fontsize=12, fontweight="bold")
        ax.set_xlabel("Annotation Budget ($N$)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Overall Accuracy (%)", fontsize=10, fontweight="bold")
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"N={b}" for b in budgets])
        ax.set_ylim(35, 92)
        ax.grid(True, linestyle="--", alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=9.5, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "oa_vs_budget.png"), bbox_inches="tight")
    plt.close()

    # 2. AA vs Budget Plot
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), dpi=300)
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        ds_data = sum_df[sum_df["dataset"] == ds]
        for m in models:
            m_data = ds_data[ds_data["model"] == m]
            means = [m_data[m_data["budget"] == b]["aa_mean"].iloc[0] for b in budgets]
            stds = [m_data[m_data["budget"] == b]["aa_sample_std"].iloc[0] for b in budgets]
            color, ls, marker = color_scheme[m]
            ax.plot(budgets, means, label=m, color=color, linestyle=ls, marker=marker, linewidth=2.0, markersize=6)
            ax.fill_between(budgets, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=color, alpha=0.10)

        ax.set_title(f"{ds}: Average Accuracy", fontsize=12, fontweight="bold")
        ax.set_xlabel("Annotation Budget ($N$)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Average Accuracy (%)", fontsize=10, fontweight="bold")
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"N={b}" for b in budgets])
        ax.set_ylim(35, 92)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=9.5, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "aa_vs_budget.png"), bbox_inches="tight")
    plt.close()

    # 3. Macro F1 vs Budget Plot
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), dpi=300)
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        ds_data = sum_df[sum_df["dataset"] == ds]
        for m in models:
            m_data = ds_data[ds_data["model"] == m]
            means = [m_data[m_data["budget"] == b]["macro_f1_mean"].iloc[0] for b in budgets]
            color, ls, marker = color_scheme[m]
            ax.plot(budgets, means, label=m, color=color, linestyle=ls, marker=marker, linewidth=2.0, markersize=6)

        ax.set_title(f"{ds}: Macro F1 Score", fontsize=12, fontweight="bold")
        ax.set_xlabel("Annotation Budget ($N$)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Macro F1", fontsize=10, fontweight="bold")
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"N={b}" for b in budgets])
        ax.set_ylim(0.35, 0.90)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=9.5, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "macro_f1_vs_budget.png"), bbox_inches="tight")
    plt.close()

    # 4. Runtime Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    m_980 = sum_df[sum_df["budget"] == 980]
    avg_train_times = [m_980[m_980["model"] == m]["train_time_mean_s"].mean() for m in models]
    bars = ax.barh(models, avg_train_times, color=["#7f7f7f", "#ff7f0e", "#8c564b", "#1f77b4", "#2ca02c"])
    ax.set_xlabel("Training Duration at N=980 (Seconds, Log Scale)", fontsize=11, fontweight="bold")
    ax.set_title("Computational Efficiency Comparison across Benchmark Models", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.15, bar.get_y() + bar.get_height() / 2, f"{w:.2f}s", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "runtime_comparison.png"), bbox_inches="tight")
    plt.close()

    # 5. Label Efficiency Comparison Plot (Grouped Bar Chart at Budgets 100, 260, 980)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), dpi=300)
    width = 0.15
    x = np.arange(len(budgets))

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        ds_data = sum_df[sum_df["dataset"] == ds]
        for m_idx, m in enumerate(models):
            m_data = ds_data[ds_data["model"] == m]
            means = [m_data[m_data["budget"] == b]["oa_mean"].iloc[0] for b in budgets]
            stds = [m_data[m_data["budget"] == b]["oa_sample_std"].iloc[0] for b in budgets]
            color = color_scheme[m][0]
            offset = (m_idx - 2) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=m if idx == 0 else "", color=color, alpha=0.85)

        ax.set_title(f"{ds}: Label Efficiency (OA %)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Annotation Budget ($N$)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Overall Accuracy (%)", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={b}" for b in budgets])
        ax.set_ylim(35, 95)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=9.5, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "label_efficiency_comparison.png"), bbox_inches="tight")
    plt.close()

    print(f"[Saved] All 5 Track B Comparison Figures in {out_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--out_dir", type=str, default="results/track_b")
    args = parser.parse_args()

    run_track_b_benchmark(seeds=args.seeds, out_dir=args.out_dir)
