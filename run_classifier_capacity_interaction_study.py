"""
ConformAL-HSI: Classifier Capacity Ablation & Active Learning Interaction Study

Evaluates:
1. Head A (Original 256): 2048 -> 256 -> Classes (~0.53M params)
2. Head B (Improved 512): 2048 -> 512 -> BatchNorm -> GELU -> Dropout(0.1) -> Classes (~1.06M params)
3. Head C (Wider 1024): 2048 -> 1024 -> BatchNorm -> GELU -> Dropout(0.2) -> Classes (~2.12M params)
4. Head D (Deeper 1024-512-256): 2048 -> 1024 -> BatchNorm -> GELU -> Dropout(0.2) -> 512 -> BatchNorm -> GELU -> Dropout(0.1) -> 256 -> GELU -> Classes (~2.76M params)

Acquisition Strategies:
- Random Sampling
- Margin Sampling
- Margin + Cosine Diversity
- Adaptive Class-Coverage AL

Datasets: Indian Pines, Salinas, Pavia University
Budgets: [100, 140, 180, 220, 260]
Screening Seeds: [42, 43, 44]
Confirmation Seeds: [42, 43, 44, 45, 46]
Outputs: results/final_research_validation/
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

INTERACTION_STRATEGIES = [
    ("Random Sampling", "random"),
    ("Margin Sampling", "margin"),
    ("Margin + Cosine Diversity", "hybrid_margin_cosine"),
    ("Adaptive Class-Coverage AL", "adaptive_coverage_al"),
]

# ==============================================================================
# CLASSIFIER ARCHITECTURES
# ==============================================================================

class HeadA_Original(nn.Module):
    """Head A: Original baseline (2048 -> 256 -> Classes)"""
    def __init__(self, in_features=2048, num_classes=16, norm_type=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
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
        return self.net(x)


class HeadB_Improved512(nn.Module):
    """Head B: Improved 512 units (2048 -> 512 -> Norm -> GELU -> Dropout(0.1) -> Classes)"""
    def __init__(self, in_features=2048, num_classes=16, norm_type="batchnorm"):
        super().__init__()
        self.norm_type = norm_type.lower()
        norm = nn.BatchNorm1d(512) if self.norm_type == "batchnorm" else nn.LayerNorm(512)
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            norm,
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
        if self.training and self.norm_type == "batchnorm" and x.dim() > 1 and x.size(0) == 1:
            self.eval()
            out = self.net(x)
            self.train()
            return out
        return self.net(x)


class HeadC_Wider1024(nn.Module):
    """Head C: Wider 1024 units (2048 -> 1024 -> Norm -> GELU -> Dropout(0.2) -> Classes)"""
    def __init__(self, in_features=2048, num_classes=16, norm_type="batchnorm"):
        super().__init__()
        self.norm_type = norm_type.lower()
        norm = nn.BatchNorm1d(1024) if self.norm_type == "batchnorm" else nn.LayerNorm(1024)
        self.net = nn.Sequential(
            nn.Linear(in_features, 1024),
            norm,
            nn.GELU(),
            nn.Dropout(p=0.20),
            nn.Linear(1024, num_classes)
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
        if self.training and self.norm_type == "batchnorm" and x.dim() > 1 and x.size(0) == 1:
            self.eval()
            out = self.net(x)
            self.train()
            return out
        return self.net(x)


class HeadD_Deeper(nn.Module):
    """Head D: Deeper (2048 -> 1024 -> Norm -> GELU -> Drop(0.2) -> 512 -> Norm -> GELU -> Drop(0.1) -> 256 -> GELU -> Classes)"""
    def __init__(self, in_features=2048, num_classes=16, norm_type="batchnorm"):
        super().__init__()
        self.norm_type = norm_type.lower()
        norm1 = nn.BatchNorm1d(1024) if self.norm_type == "batchnorm" else nn.LayerNorm(1024)
        norm2 = nn.BatchNorm1d(512) if self.norm_type == "batchnorm" else nn.LayerNorm(512)
        self.net = nn.Sequential(
            nn.Linear(in_features, 1024),
            norm1,
            nn.GELU(),
            nn.Dropout(p=0.20),
            nn.Linear(1024, 512),
            norm2,
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
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
        if self.training and self.norm_type == "batchnorm" and x.dim() > 1 and x.size(0) == 1:
            self.eval()
            out = self.net(x)
            self.train()
            return out
        return self.net(x)


def build_classifier(head_key, num_classes, norm_type="batchnorm"):
    if head_key == "head_a":
        return HeadA_Original(2048, num_classes)
    elif head_key == "head_b":
        return HeadB_Improved512(2048, num_classes, norm_type=norm_type)
    elif head_key == "head_c":
        return HeadC_Wider1024(2048, num_classes, norm_type=norm_type)
    elif head_key == "head_d":
        return HeadD_Deeper(2048, num_classes, norm_type=norm_type)
    else:
        raise ValueError(f"Unknown head_key: {head_key}")


HEAD_REGISTRY = {
    "head_a": {"name": "Head A (Original 256)", "code": "head_a", "capacity": "256 units", "depth": 2},
    "head_b": {"name": "Head B (Improved 512)", "code": "head_b", "capacity": "512 units", "depth": 2},
    "head_c": {"name": "Head C (Wider 1024)", "code": "head_c", "capacity": "1024 units", "depth": 2},
    "head_d": {"name": "Head D (Deeper 1024-512-256)", "code": "head_d", "capacity": "1024-512-256 units", "depth": 4},
}


# ==============================================================================
# DATA LOADING & INITIAL SEEDING
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
    head_key,
    train_feats_l2,
    train_lbls,
    labeled_idx,
    num_classes,
    epochs=45,
    lr=2e-3,
    weight_decay=1e-3,
    norm_type="batchnorm",
):
    model = build_classifier(head_key, num_classes, norm_type=norm_type).to(DEVICE)
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
            if bx.size(0) <= 1 and norm_type == "batchnorm":
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
    density_scores,
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

    # Extract model probabilities
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

    # 4. Adaptive Class-Coverage AL: S(x) = alpha * U(x) + beta * D(x) + gamma * C(x)
    if strategy_code == "adaptive_coverage_al":
        alpha, beta, gamma = 0.50, 0.30, 0.20
        sorted_probs = np.sort(probs, axis=1)
        u_score = 1.0 - (sorted_probs[:, -1] - sorted_probs[:, -2])
        u_norm = (u_score - u_score.min()) / (u_score.max() - u_score.min() + 1e-8)

        # Strictly training pool labels & model predictions
        pred_c = np.argmax(probs, axis=1)
        cur_labels = train_lbls[labeled_idx].numpy()
        cur_counts = np.bincount(cur_labels, minlength=num_classes).astype(np.float32)

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

            for _ in range(1, k):
                new_dot = torch.matmul(cand_l2, cand_l2[sel[-1] : sel[-1] + 1].t()).squeeze(1).cpu().numpy()
                dists = np.minimum(dists, 1.0 - new_dot)
                d_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)

                cur_counts[pred_c_cand[sel[-1]]] += 1.0
                c_score = 1.0 / np.sqrt(cur_counts[pred_c_cand] + 1.0)
                c_norm = (c_score - c_score.min()) / (c_score.max() - c_score.min() + 1e-8)

                total_scores = alpha * u_cand + beta * d_norm + gamma * c_cand
                for s in sel:
                    total_scores[s] = -1e9
                sel.append(int(np.argmax(total_scores)))

        return unlabeled_idx[top_cand[sel]], time.time() - t0

    raise ValueError(f"Unknown strategy code: {strategy_code}")


# ==============================================================================
# MAIN INTERACTION BENCHMARK SUITE
# ==============================================================================

def run_classifier_interaction_suite(
    heads=("head_a", "head_b", "head_c", "head_d"),
    strategies=INTERACTION_STRATEGIES,
    seeds=(42, 43, 44),
    norm_type="batchnorm",
    out_dir="results/final_research_validation",
    tag="interaction",
):
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 110, flush=True)
    print(f"CONFORMAL-HSI: CLASSIFIER CAPACITY ABLATION & ACTIVE LEARNING INTERACTION STUDY [{tag.upper()}]", flush=True)
    print(f"Heads: {heads} | Strategies: {[s[1] for s in strategies]} | Seeds: {seeds} | Norm: {norm_type}", flush=True)
    print("Controls: Frozen SpectralEarth 3D-ResNet50 Embeddings, Fixed AdamW/Cosine Schedule, Strictly Held-Out Test Set", flush=True)
    print("=" * 110, flush=True)

    seed_records = []
    runtime_memory_records = []
    per_class_records = []

    for ds_name, cfg in DATASET_CONFIGS.items():
        print(f"\n==================================================================", flush=True)
        print(f">>> DATASET: {ds_name.upper()} (Classes={cfg['num_classes']}, Pixels={cfg['total_pixels']:,}) <<<", flush=True)
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

        for head_key in heads:
            head_meta = HEAD_REGISTRY[head_key]
            head_name = head_meta["name"]
            dummy_model = build_classifier(head_key, num_classes, norm_type=norm_type)
            param_count = sum(p.numel() for p in dummy_model.parameters())

            print(f"\n  >>> Classifier: {head_name} (Params: {param_count:,}, Norm: {norm_type}) <<<", flush=True)

            for strat_name, strat_code in strategies:
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
                            temp_model, _, _, _ = train_classifier(
                                head_key=head_key,
                                train_feats_l2=train_x_l2,
                                train_lbls=train_y,
                                labeled_idx=labeled_idx,
                                num_classes=num_classes,
                                epochs=20,
                                norm_type=norm_type,
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

                        # Track GPU memory
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()

                        final_model, train_time, train_loss, train_acc = train_classifier(
                            head_key=head_key,
                            train_feats_l2=train_x_l2,
                            train_lbls=train_y,
                            labeled_idx=labeled_idx,
                            num_classes=num_classes,
                            epochs=45,
                            norm_type=norm_type,
                        )

                        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

                        metrics, probs, preds, inf_time, val_ece = evaluate_model(
                            final_model, test_x_l2, test_y, num_classes, val_x_l2, val_y
                        )

                        rec = {
                            "dataset": ds_name,
                            "classifier_head": head_name,
                            "classifier_code": head_key,
                            "norm_type": norm_type,
                            "parameter_count": param_count,
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
                            "val_ece": round(val_ece, 4),
                            "train_loss": round(train_loss, 4),
                            "train_acc": round(train_acc, 2),
                            "train_time_s": round(train_time, 3),
                            "acq_time_s": round(acq_time, 4),
                            "inf_time_s": round(inf_time, 4),
                            "peak_vram_mb": round(peak_vram_mb, 2),
                        }
                        seed_records.append(rec)

                        runtime_memory_records.append({
                            "dataset": ds_name,
                            "classifier_head": head_name,
                            "strategy": strat_name,
                            "seed": seed,
                            "budget": target_budget,
                            "parameter_count": param_count,
                            "train_time_s": round(train_time, 3),
                            "acq_time_s": round(acq_time, 4),
                            "inf_time_s": round(inf_time, 4),
                            "peak_vram_mb": round(peak_vram_mb, 2),
                        })

                        # Record per-class metrics at key budgets (100, 180, 220)
                        if target_budget in [100, 180, 220]:
                            per_class = metrics["PerClass"]
                            lab_counts = np.bincount(train_y[labeled_idx].numpy(), minlength=num_classes)
                            for c_i in range(num_classes):
                                c_meta = per_class.get(f"Class_{c_i}", {})
                                per_class_records.append({
                                    "dataset": ds_name,
                                    "classifier_head": head_name,
                                    "strategy": strat_name,
                                    "seed": seed,
                                    "budget": target_budget,
                                    "class_id": c_i,
                                    "labeled_count": int(lab_counts[c_i]),
                                    "support": c_meta.get("support", 0),
                                    "precision": round(c_meta.get("precision", 0.0) * 100, 2),
                                    "recall": round(c_meta.get("recall", 0.0) * 100, 2),
                                    "f1": round(c_meta.get("f1", 0.0), 4),
                                })

                        print(
                            f"    [{ds_name[:6]}|{head_key}] {strat_code:18s} S{seed} N={target_budget:3d} | "
                            f"OA: {rec['oa']:5.2f}% | AA: {rec['aa']:5.2f}% | F1: {rec['macro_f1']:6.4f} | "
                            f"VRAM: {peak_vram_mb:5.1f}MB | TrainT: {train_time:5.2f}s",
                            flush=True,
                        )

    return seed_records, runtime_memory_records, per_class_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", nargs="+", default=["head_a", "head_b", "head_c", "head_d"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--norm_type", type=str, default="batchnorm")
    parser.add_argument("--out_dir", type=str, default="results/final_research_validation")
    parser.add_argument("--tag", type=str, default="interaction")
    args = parser.parse_args()

    seed_recs, rt_recs, pc_recs = run_classifier_interaction_suite(
        heads=args.heads,
        seeds=args.seeds,
        norm_type=args.norm_type,
        out_dir=args.out_dir,
        tag=args.tag,
    )

    out_csv = os.path.join(args.out_dir, f"classifier_head_seed_results_{args.tag}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_recs[0].keys()))
        writer.writeheader()
        writer.writerows(seed_recs)
    print(f"\n[Saved] {out_csv} ({len(seed_recs)} seed records)", flush=True)

    if rt_recs:
        rt_csv = os.path.join(args.out_dir, f"classifier_head_runtime_memory_{args.tag}.csv")
        with open(rt_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rt_recs[0].keys()))
            writer.writeheader()
            writer.writerows(rt_recs)
        print(f"[Saved] {rt_csv} ({len(rt_recs)} runtime records)", flush=True)

    if pc_recs:
        pc_csv = os.path.join(args.out_dir, f"classifier_head_per_class_{args.tag}.csv")
        with open(pc_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(pc_recs[0].keys()))
            writer.writeheader()
            writer.writerows(pc_recs)
        print(f"[Saved] {pc_csv} ({len(pc_recs)} per-class records)", flush=True)
