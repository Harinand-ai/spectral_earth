"""
ConformAL-HSI: Low-Label Efficiency Optimization & Advanced Active Learning Suite

Evaluates:
- Phase 1: Low-Budget Baselines (Random, Proposed Cosine AL, Least Confidence, Margin, Entropy)
- Phase 2: Advanced Acquisition (K-Center Greedy, BALD MC Dropout, BADGE, Cluster Diversity)
- Phase 3: Adaptive & Hybrid Strategies (Margin+Cosine, BALD+Cosine, Adaptive Class-Coverage AL)
- Phase 4: Classifier Calibration (Validation Temperature Scaling, ECE, Brier Score)
- Phase 5: Pseudo-Labeling Investigation (Conservative, Class-Specific Thresholds)
- Phase 6: Multi-Seed Confirmation on Top Candidates

Protocols:
- Datasets: Indian Pines, Salinas, Pavia University
- Low-Budget Range: [100, 140, 180, 220, 260]
- Initial Seed: 5 per class (IP/SA: 80 labels, PU: 45 labels)
- Seeds: [42, 43, 44] (exploration), [42, 43, 44, 45, 46] (confirmation)
- Backbone: Strictly Frozen SpectralEarth 3D-ResNet50 (∇θ = 0)
- Classifier Head: Fixed ImprovedClassifierHead (2048 -> 512 -> C)
- Output: results/label_efficiency_optimization/
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
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.learning_engine.improved_classifier import ImprovedClassifierHead
from src.learning_engine.temperature_scaling import TemperatureScaler
from src.utils.metrics_engine import compute_detailed_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_CONFIGS = {
    "Indian Pines": {
        "slug": "indian_pines",
        "num_classes": 16,
        "budgets": [100, 140, 180, 220, 260],
        "init_seed_per_class": 5,
        "total_pixels": 10249,
    },
    "Salinas": {
        "slug": "salinas",
        "num_classes": 16,
        "budgets": [100, 140, 180, 220, 260],
        "init_seed_per_class": 5,
        "total_pixels": 56975,
    },
    "Pavia University": {
        "slug": "pavia_university",
        "num_classes": 9,
        "budgets": [100, 140, 180, 220, 260],
        "init_seed_per_class": 5,
        "total_pixels": 42776,
    },
}

PHASE1_STRATEGIES = [
    ("Random Sampling", "random"),
    ("Proposed Cosine AL (Unc + Pool Diversity)", "proposed_cosine_al"),
    ("Pure Uncertainty (Least Confidence)", "least_confidence"),
    ("Margin Sampling", "margin"),
    ("Entropy Sampling", "entropy"),
]

PHASE2_STRATEGIES = [
    ("K-Center Greedy (Core-Set)", "kcenter_greedy"),
    ("BALD (MC Dropout Disagreement)", "bald_mc"),
    ("Exact BADGE (Loss Gradient Diversity)", "badge"),
    ("Cluster Diversity (Spherical K-Means)", "cluster_diversity"),
]

PHASE3_STRATEGIES = [
    ("Margin + Cosine Diversity", "hybrid_margin_cosine"),
    ("BALD + Cosine Diversity", "hybrid_bald_cosine"),
    ("Adaptive Class-Coverage AL", "adaptive_coverage_al"),
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


def train_classifier(train_feats_l2, train_lbls, labeled_idx, num_classes, epochs=45, lr=2e-3, weight_decay=1e-3):
    model = ImprovedClassifierHead(2048, num_classes, 512, 0.1).to(DEVICE)
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
    for _ in range(epochs):
        for bx, by in loader:
            if bx.size(0) <= 1:
                continue
            opt.zero_grad()
            out = model(bx, normalize_input=False)
            loss = crit(out, by)
            loss.backward()
            opt.step()
        sched.step()
    train_time = time.time() - t0
    return model, train_time


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

    # Validation calibration evaluation (if val data provided)
    calib_metrics = {}
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
        calib_metrics = {
            "optimal_temperature_T": round(learned_T, 4),
            "val_ece_uncalibrated": round(metrics.get("ECE", 0.0), 4),
            "val_ece_calibrated": round(m_val.get("ECE", 0.0), 4),
        }

    return metrics, probs, preds, inf_time, calib_metrics


def select_query_batch(
    strategy_code,
    k,
    unlabeled_idx,
    train_feats_l2,
    train_lbls,
    labeled_idx,
    density_scores,
    model,
    num_classes,
    rng,
):
    if len(unlabeled_idx) <= k:
        return unlabeled_idx

    t0 = time.time()

    # 1. Random Sampling
    if strategy_code == "random":
        chosen = rng.choice(len(unlabeled_idx), size=k, replace=False)
        return unlabeled_idx[chosen], time.time() - t0

    # Extract model probabilities in mini-batches
    model.eval()
    u_probs_list = []
    with torch.no_grad():
        for bi in range(0, len(unlabeled_idx), 2048):
            bx = train_feats_l2[unlabeled_idx[bi : bi + 2048]].to(DEVICE)
            logits = model(bx, normalize_input=False)
            u_probs_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(u_probs_list, axis=0)  # (N_unlab, C)

    # 2. Least Confidence: 1 - max p
    if strategy_code == "least_confidence":
        uncertainty = 1.0 - np.max(probs, axis=1)
        top_k = np.argsort(-uncertainty)[:k]
        return unlabeled_idx[top_k], time.time() - t0

    # 3. Margin Sampling: 1 - (p1 - p2)
    if strategy_code == "margin":
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        margin_unc = 1.0 - margin
        top_k = np.argsort(-margin_unc)[:k]
        return unlabeled_idx[top_k], time.time() - t0

    # 4. Entropy Sampling: - sum p log p / log C
    if strategy_code == "entropy":
        eps = 1e-12
        p_clipped = np.clip(probs, eps, 1.0)
        ent = -np.sum(p_clipped * np.log(p_clipped), axis=1) / np.log(num_classes)
        top_k = np.argsort(-ent)[:k]
        return unlabeled_idx[top_k], time.time() - t0

    # 5. Proposed Cosine AL (Least Confidence * Density^0.3 + Cosine Farthest-First)
    if strategy_code == "proposed_cosine_al":
        uncertainty = 1.0 - np.max(probs, axis=1)
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

        return unlabeled_idx[top_cand[sel]], time.time() - t0

    # 6. K-Center Greedy (Core-Set Facility Location on Unit Hypersphere)
    if strategy_code == "kcenter_greedy":
        # Full-pool Core-Set selection
        unlab_l2 = train_feats_l2[unlabeled_idx].to(DEVICE)
        lab_l2 = train_feats_l2[labeled_idx].to(DEVICE)

        with torch.no_grad():
            dots = torch.matmul(unlab_l2, lab_l2.t())
            dists = 1.0 - torch.max(dots, dim=1)[0]
            sel = [torch.argmax(dists).item()]
            dists[sel[-1]] = -1.0
            for _ in range(1, k):
                new_dots = torch.matmul(unlab_l2, unlab_l2[sel[-1] : sel[-1] + 1].t()).squeeze(1)
                dists = torch.minimum(dists, 1.0 - new_dots)
                dists[sel[-1]] = -1.0
                sel.append(torch.argmax(dists).item())

        return unlabeled_idx[sel], time.time() - t0

    # 7. BALD (Monte Carlo Dropout Disagreement)
    if strategy_code == "bald_mc":
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        T = 10
        chunk_size = 4096
        bald_scores = []
        with torch.no_grad():
            for bi in range(0, len(unlabeled_idx), chunk_size):
                bx = train_feats_l2[unlabeled_idx[bi : bi + chunk_size]].to(DEVICE)
                probs_mc = []
                for _ in range(T):
                    logits_t = model(bx, normalize_input=False)
                    probs_mc.append(torch.softmax(logits_t, dim=-1))
                probs_mc = torch.stack(probs_mc, dim=0)  # (T, chunk, C)
                mean_p = torch.mean(probs_mc, dim=0)
                eps = 1e-12
                total_entropy = -torch.sum(mean_p * torch.log(torch.clamp(mean_p, min=eps)), dim=-1)
                expected_entropy = torch.mean(-torch.sum(probs_mc * torch.log(torch.clamp(probs_mc, min=eps)), dim=-1), dim=0)
                bald = total_entropy - expected_entropy
                bald_scores.append(bald.cpu().numpy())
        bald_scores = np.concatenate(bald_scores, axis=0)
        top_k = np.argsort(-bald_scores)[:k]
        return unlabeled_idx[top_k], time.time() - t0

    # 8. Exact BADGE (Diverse Loss Gradient Embeddings via Tensor Decomposition)
    if strategy_code == "badge":
        # Exact BADGE (Ash et al., 2020) without candidate prefiltering.
        # Tensor decomposition: ||g_i - g_j||^2 = ||r_i||^2 + ||r_j||^2 - 2 (r_i . r_j) * (f_i . f_j)
        y_hat = np.argmax(probs, axis=-1)
        M_unlab = len(unlabeled_idx)

        r_np = probs.copy()
        r_np[np.arange(M_unlab), y_hat] -= 1.0
        r_t = torch.tensor(r_np, dtype=torch.float32, device=DEVICE)  # (N_unlab, C)
        f_t = train_feats_l2[unlabeled_idx].to(DEVICE)                 # (N_unlab, d)

        with torch.no_grad():
            norm_r_sq = torch.sum(r_t ** 2, dim=1)  # ||g_i||^2 = ||r_i||^2 since ||f_i|| = 1

            # First center: sample with highest gradient norm (maximum uncertainty)
            sel = [torch.argmax(norm_r_sq).item()]

            # Initial distances to first center
            r_s0 = r_t[sel[0] : sel[0] + 1]
            f_s0 = f_t[sel[0] : sel[0] + 1]
            dot_r = torch.matmul(r_t, r_s0.t()).squeeze(1)
            dot_f = torch.matmul(f_t, f_s0.t()).squeeze(1)
            min_dists = norm_r_sq + norm_r_sq[sel[0]] - 2.0 * dot_r * dot_f
            min_dists[sel[0]] = -1.0

            # Greedy farthest-first traversal in gradient space
            for _ in range(1, k):
                next_idx = torch.argmax(min_dists).item()
                sel.append(next_idx)
                r_s = r_t[next_idx : next_idx + 1]
                f_s = f_t[next_idx : next_idx + 1]
                dot_r = torch.matmul(r_t, r_s.t()).squeeze(1)
                dot_f = torch.matmul(f_t, f_s.t()).squeeze(1)
                new_dists = norm_r_sq + norm_r_sq[next_idx] - 2.0 * dot_r * dot_f
                min_dists = torch.minimum(min_dists, new_dists)
                min_dists[next_idx] = -1.0

        return unlabeled_idx[sel], time.time() - t0

    # 9. Cluster Diversity (Spherical K-Means / Cosine Centers)
    if strategy_code == "cluster_diversity":
        from sklearn.cluster import MiniBatchKMeans
        cand_size = min(len(unlabeled_idx), 10000)
        cand_local = rng.choice(len(unlabeled_idx), size=cand_size, replace=False) if len(unlabeled_idx) > cand_size else np.arange(len(unlabeled_idx))
        cand_x_np = train_feats_l2[unlabeled_idx[cand_local]].numpy()

        mbk = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256, n_init=3)
        cluster_labels = mbk.fit_predict(cand_x_np)
        centers = mbk.cluster_centers_
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

        sel_local = []
        for c_id in range(k):
            members = np.where(cluster_labels == c_id)[0]
            if len(members) == 0:
                continue
            dots_to_c = np.dot(cand_x_np[members], centers[c_id])
            best_m = members[np.argmax(dots_to_c)]
            sel_local.append(best_m)

        while len(sel_local) < k:
            rem = [idx for idx in range(len(cand_local)) if idx not in set(sel_local)]
            sel_local.append(rem[0])

        return unlabeled_idx[cand_local[sel_local[:k]]], time.time() - t0

    # 10. Margin + Cosine Diversity
    if strategy_code == "hybrid_margin_cosine":
        sorted_probs = np.sort(probs, axis=1)
        margin_unc = 1.0 - (sorted_probs[:, -1] - sorted_probs[:, -2])
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

    # 11. BALD + Cosine Diversity
    if strategy_code == "hybrid_bald_cosine":
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        T = 10
        chunk_size = 4096
        bald_scores = []
        with torch.no_grad():
            for bi in range(0, len(unlabeled_idx), chunk_size):
                bx = train_feats_l2[unlabeled_idx[bi : bi + chunk_size]].to(DEVICE)
                probs_mc = []
                for _ in range(T):
                    logits_t = model(bx, normalize_input=False)
                    probs_mc.append(torch.softmax(logits_t, dim=-1))
                probs_mc = torch.stack(probs_mc, dim=0)
                mean_p = torch.mean(probs_mc, dim=0)
                eps = 1e-12
                total_entropy = -torch.sum(mean_p * torch.log(torch.clamp(mean_p, min=eps)), dim=-1)
                expected_entropy = torch.mean(-torch.sum(probs_mc * torch.log(torch.clamp(probs_mc, min=eps)), dim=-1), dim=0)
                bald = total_entropy - expected_entropy
                bald_scores.append(bald.cpu().numpy())
        bald_scores = np.concatenate(bald_scores, axis=0)

        cand_size = min(len(unlabeled_idx), max(4 * k, 200))
        top_cand = np.argsort(-bald_scores)[:cand_size]

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

    # 12. Adaptive Class-Coverage AL: S(x) = alpha * U(x) + beta * D(x) + gamma * C(x)
    if strategy_code == "adaptive_coverage_al":
        alpha, beta, gamma = 0.50, 0.30, 0.20
        # Margin uncertainty
        sorted_probs = np.sort(probs, axis=1)
        u_score = 1.0 - (sorted_probs[:, -1] - sorted_probs[:, -2])
        u_norm = (u_score - u_score.min()) / (u_score.max() - u_score.min() + 1e-8)

        # Predicted class on unlabeled pool & current labeled training set counts (NEVER test labels)
        pred_c = np.argmax(probs, axis=1)
        cur_labels = train_lbls[labeled_idx].numpy()
        cur_counts = np.bincount(cur_labels, minlength=num_classes).astype(np.float32)

        # Candidate pool: filter top candidates by (alpha * U + gamma * C) to accelerate pairwise distance
        c_score = 1.0 / np.sqrt(cur_counts[pred_c] + 1.0)
        c_norm = (c_score - c_score.min()) / (c_score.max() - c_score.min() + 1e-8)
        pre_score = alpha * u_norm + gamma * c_norm
        cand_size = min(len(unlabeled_idx), max(6 * k, 300))
        top_cand = np.argsort(-pre_score)[:cand_size]

        cand_l2 = train_feats_l2[unlabeled_idx[top_cand]].to(DEVICE)
        lab_l2 = train_feats_l2[labeled_idx].to(DEVICE)
        u_cand = u_norm[top_cand]
        c_cand = c_norm[top_cand]
        pred_c_cand = pred_c[top_cand]

        with torch.no_grad():
            dots = torch.matmul(cand_l2, lab_l2.t())
            dists = 1.0 - torch.max(dots, dim=1)[0].cpu().numpy()
            d_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)

            total_scores = alpha * u_cand + beta * d_norm + gamma * c_cand
            sel = [int(np.argmax(total_scores))]
            total_scores[sel[-1]] = -1e9

            # Dynamic greedy batch acquisition
            for _ in range(1, k):
                # Update distances with newly selected sample
                new_dot = torch.matmul(cand_l2, cand_l2[sel[-1] : sel[-1] + 1].t()).squeeze(1).cpu().numpy()
                dists = np.minimum(dists, 1.0 - new_dot)
                d_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)

                # Increment count of chosen predicted class and update C
                cur_counts[pred_c_cand[sel[-1]]] += 1.0
                c_score = 1.0 / np.sqrt(cur_counts[pred_c_cand] + 1.0)
                c_norm = (c_score - c_score.min()) / (c_score.max() - c_score.min() + 1e-8)

                total_scores = alpha * u_cand + beta * d_norm + gamma * c_cand
                for s in sel:
                    total_scores[s] = -1e9
                sel.append(int(np.argmax(total_scores)))

        return unlabeled_idx[top_cand[sel]], time.time() - t0

    raise ValueError(f"Unknown strategy code: {strategy_code}")


def run_label_efficiency_suite(
    phase=1,
    seeds=(42, 43, 44),
    out_dir="results/label_efficiency_optimization",
):
    os.makedirs(out_dir, exist_ok=True)

    phase_str = str(phase).lower()
    if phase_str == "1":
        strategies_to_run = PHASE1_STRATEGIES
        phase_title = "PHASE 1: LOW-BUDGET BASELINES REPRODUCTION & VERIFICATION"
    elif phase_str == "2":
        strategies_to_run = PHASE2_STRATEGIES
        phase_title = "PHASE 2: ADVANCED ACQUISITION STRATEGIES (CORE-SET, BALD, EXACT BADGE, CLUSTERING)"
    elif phase_str == "3":
        strategies_to_run = PHASE3_STRATEGIES
        phase_title = "PHASE 3: ADAPTIVE & HYBRID ACQUISITION STRATEGIES (CLASS-COVERAGE AL)"
    elif phase_str in ("2_3", "2-3", "2,3"):
        strategies_to_run = PHASE2_STRATEGIES + PHASE3_STRATEGIES
        phase_title = "PHASES 2 & 3: ADVANCED ACQUISITION & HYBRID / ADAPTIVE STRATEGIES"
    elif phase_str in ("all", "1_2_3"):
        strategies_to_run = PHASE1_STRATEGIES + PHASE2_STRATEGIES + PHASE3_STRATEGIES
        phase_title = "FULL BENCHMARK: ALL ACQUISITION STRATEGIES ACROSS PHASES 1-3"
    else:
        raise ValueError(f"Invalid phase: {phase}")

    print("=" * 100, flush=True)
    print(f"CONFORMAL-HSI: LOW-LABEL EFFICIENCY OPTIMIZATION — {phase_title}", flush=True)
    print(f"Device: {DEVICE} | Seeds: {seeds} | Budgets: [100, 140, 180, 220, 260]", flush=True)
    print("Controls: Frozen SpecResNet50 Embeddings, Fixed ImprovedClassifierHead, Strictly Held-Out Test Set", flush=True)
    print("=" * 100, flush=True)

    all_records = []
    runtime_records = []
    calibration_records = []

    for ds_name, cfg in DATASET_CONFIGS.items():
        print(f"\n==================================================================", flush=True)
        print(f">>> DATASET: {ds_name.upper()} (Classes={cfg['num_classes']}, Total Pixels={cfg['total_pixels']:,}) <<<", flush=True)
        print(f"==================================================================", flush=True)

        feats = load_dataset_features(cfg["slug"])
        train_x = feats["train_feats"]
        train_y = feats["train_lbls"]
        val_x = feats["val_feats"]
        val_y = feats["val_lbls"]
        test_x = feats["test_feats"]
        test_y = feats["test_lbls"]
        num_classes = cfg["num_classes"]

        train_x_l2 = train_x / torch.clamp(torch.norm(train_x, dim=1, keepdim=True), min=1e-8)
        val_x_l2 = val_x / torch.clamp(torch.norm(val_x, dim=1, keepdim=True), min=1e-8)
        test_x_l2 = test_x / torch.clamp(torch.norm(test_x, dim=1, keepdim=True), min=1e-8)

        centroid = torch.mean(train_x_l2, dim=0, keepdim=True)
        centroid = centroid / torch.norm(centroid, dim=1, keepdim=True)
        density_scores = torch.matmul(train_x_l2, centroid.t()).squeeze(1).numpy()

        for strat_name, strat_code in strategies_to_run:
            print(f"\n  --- Strategy: {strat_name} ({strat_code}) ---", flush=True)

            for seed in seeds:
                rng = np.random.RandomState(seed)
                torch.manual_seed(seed)

                labeled_idx, unlabeled_idx = sample_initial_seed(
                    train_y, num_classes, cfg["init_seed_per_class"], seed
                )
                init_seed_count = len(labeled_idx)

                for target_budget in cfg["budgets"]:
                    needed = target_budget - len(labeled_idx)
                    acq_time = 0.0

                    if needed > 0:
                        # Train intermediate model for acquisition scoring
                        temp_model, _ = train_classifier(
                            train_x_l2, train_y, labeled_idx, num_classes, epochs=20
                        )

                        new_selected, acq_time = select_query_batch(
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

                    # Train final classifier for this budget
                    final_model, train_time = train_classifier(
                        train_x_l2, train_y, labeled_idx, num_classes, epochs=45
                    )

                    # Evaluate on strictly held-out test set
                    metrics, probs, preds, inf_time, calib = evaluate_model(
                        final_model, test_x_l2, test_y, num_classes, val_x_l2, val_y
                    )

                    record = {
                        "dataset": ds_name,
                        "strategy": strat_name,
                        "strategy_code": strat_code,
                        "seed": seed,
                        "budget": target_budget,
                        "initial_labels": init_seed_count,
                        "acquired_labels": target_budget - init_seed_count,
                        "total_labels": len(labeled_idx),
                        "sample_fraction_pct": round(target_budget / cfg["total_pixels"] * 100, 2),
                        "oa": round(metrics["OverallAccuracy"] * 100, 2),
                        "aa": round(metrics["AverageAccuracy"] * 100, 2),
                        "kappa": round(metrics["Kappa"], 4),
                        "macro_f1": round(metrics["F1Score"], 4),
                        "ece": round(metrics.get("ECE", 0.0), 4),
                        "train_time_s": round(train_time, 3),
                        "acq_time_s": round(acq_time, 4),
                        "inf_time_s": round(inf_time, 4),
                    }
                    all_records.append(record)

                    runtime_records.append({
                        "dataset": ds_name,
                        "strategy": strat_name,
                        "seed": seed,
                        "budget": target_budget,
                        "train_time_s": round(train_time, 3),
                        "acq_time_s": round(acq_time, 4),
                        "inf_time_s": round(inf_time, 4),
                    })

                    if calib:
                        calibration_records.append({
                            "dataset": ds_name,
                            "strategy": strat_name,
                            "seed": seed,
                            "budget": target_budget,
                            **calib,
                        })

                    print(
                        f"    [{ds_name[:6]}] Seed {seed} N={target_budget:3d} | "
                        f"OA: {record['oa']:5.2f}% | AA: {record['aa']:5.2f}% | "
                        f"F1: {record['macro_f1']:6.4f} | AcqTime: {acq_time:6.3f}s",
                        flush=True,
                    )

    # Save seed-level records
    phase_tag = f"phase{phase_str}"
    seed_csv = os.path.join(out_dir, f"seed_level_results_{phase_tag}.csv")
    with open(seed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n[Saved] {seed_csv}", flush=True)

    # Save runtime records
    if runtime_records:
        rt_csv = os.path.join(out_dir, f"acquisition_runtime_{phase_tag}.csv")
        with open(rt_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(runtime_records[0].keys()))
            writer.writeheader()
            writer.writerows(runtime_records)
        print(f"[Saved] {rt_csv}", flush=True)

    # Save calibration records
    if calibration_records:
        cal_csv = os.path.join(out_dir, f"calibration_analysis_{phase_tag}.csv")
        with open(cal_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(calibration_records[0].keys()))
            writer.writeheader()
            writer.writerows(calibration_records)
        print(f"[Saved] {cal_csv}", flush=True)

    # Generate aggregated summary (Mean +/- Sample Std)
    res_df = pd.DataFrame(all_records)
    summary_rows = []
    strategies = res_df["strategy"].unique()
    budgets = [100, 140, 180, 220, 260]

    for ds_name in DATASET_CONFIGS.keys():
        ds_sub = res_df[res_df["dataset"] == ds_name]
        for strat in strategies:
            strat_sub = ds_sub[ds_sub["strategy"] == strat]
            strat_code = strat_sub["strategy_code"].iloc[0]

            for b in budgets:
                b_sub = strat_sub[strat_sub["budget"] == b]
                if len(b_sub) == 0:
                    continue
                oa_vals = b_sub["oa"].values
                aa_vals = b_sub["aa"].values
                kap_vals = b_sub["kappa"].values
                f1_vals = b_sub["macro_f1"].values
                ece_vals = b_sub["ece"].values
                t_acq = b_sub["acq_time_s"].values

                summary_rows.append({
                    "dataset": ds_name,
                    "strategy": strat,
                    "strategy_code": strat_code,
                    "budget": b,
                    "sample_fraction_pct": b_sub["sample_fraction_pct"].iloc[0],
                    "oa_mean": round(float(np.mean(oa_vals)), 2),
                    "oa_sample_std": round(float(np.std(oa_vals, ddof=1)), 2) if len(oa_vals) > 1 else 0.0,
                    "aa_mean": round(float(np.mean(aa_vals)), 2),
                    "aa_sample_std": round(float(np.std(aa_vals, ddof=1)), 2) if len(aa_vals) > 1 else 0.0,
                    "kappa_mean": round(float(np.mean(kap_vals)), 4),
                    "kappa_sample_std": round(float(np.std(kap_vals, ddof=1)), 4) if len(kap_vals) > 1 else 0.0,
                    "macro_f1_mean": round(float(np.mean(f1_vals)), 4),
                    "macro_f1_sample_std": round(float(np.std(f1_vals, ddof=1)), 4) if len(f1_vals) > 1 else 0.0,
                    "ece_mean": round(float(np.mean(ece_vals)), 4),
                    "acq_time_mean_s": round(float(np.mean(t_acq)), 4),
                })

    sum_csv = os.path.join(out_dir, f"accuracy_vs_label_budget_{phase_tag}.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[Saved] {sum_csv}", flush=True)

    return all_records, summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="1", help="1, 2, 3, or all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--out_dir", type=str, default="results/label_efficiency_optimization")
    args = parser.parse_args()

    phase_arg = int(args.phase) if args.phase.isdigit() else args.phase
    run_label_efficiency_suite(phase=phase_arg, seeds=args.seeds, out_dir=args.out_dir)

