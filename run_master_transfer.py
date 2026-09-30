import os
import sys
import json
import csv
import time
import torch
import torch.nn as nn
import numpy as np
import pyrootutils

pyrootutils.setup_root(r"d:\Projects\spectral_earth", indicator=".project-root", pythonpath=True)

from src.backbones.registry import BACKBONE_REGISTRY
from src.learning_engine import LearningEngine, ActiveLearningDatasetManager
from src.learning_engine.classifier import ClassifierHead
from src.learning_engine.temperature_scaling import TemperatureScaler
from src.query_engine.query_engine import QueryEngine
from src.utils.metrics_engine import compute_detailed_metrics, compute_aulc, compute_ece
from src.utils.spectral_adapter_1d import SpectralResampler1D

def load_frozen_backbone(num_classes, ckpt_path):
    backbone = BACKBONE_REGISTRY["spec_resnet50"](num_classes=num_classes)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        backbone_state = {}
        for k, v in state_dict.items():
            if k.startswith("model.backbone."):
                backbone_state[k[len("model.backbone."):]] = v
            elif k.startswith("model."):
                backbone_state[k[len("model."):]] = v
            elif k.startswith("feature_extractor.backbone."):
                backbone_state[k[len("feature_extractor.backbone."):]] = v
            else:
                backbone_state[k] = v
        backbone_state = {k: v for k, v in backbone_state.items() if not ("fc.weight" in k or "fc.bias" in k)}
        backbone.load_state_dict(backbone_state, strict=False)

    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    return backbone

def get_datamodule(ds_name, meta):
    mod_path, class_name = meta["module_cls"].rsplit(".", 1)
    import importlib
    module = importlib.import_module(mod_path)
    DataModuleCls = getattr(module, class_name)
    dm = DataModuleCls()
    dm.setup("fit")
    dm.setup("test")
    return dm

def run_master_transfer():
    print("=" * 88)
    print("SPECTRALEARTH ACTIVE LEARNING FRAMEWORK — CROSS-SENSOR & CROSS-DATASET EVALUATION")
    print("=" * 88)

    os.makedirs("logs/al_experiments", exist_ok=True)
    ckpt_path = r"d:\Projects\spectral_earth\logs\None\runs\2026-07-30_14-24-24\checkpoints\epoch_094.ckpt"

    datasets_meta = {
        "Indian Pines": {
            "module_cls": "src.datamodules.indian_pines.IndianPinesDataModule",
            "num_classes": 16,
            "bands": 200,
            "sensor": "AVIRIS",
            "init_samples": 10,
        },
        "Salinas": {
            "module_cls": "src.datamodules.salinas.SalinasDataModule",
            "num_classes": 16,
            "bands": 204,
            "sensor": "AVIRIS",
            "init_samples": 10,
        },
        "Pavia University": {
            "module_cls": "src.datamodules.pavia_university.PaviaUniversityDataModule",
            "num_classes": 9,
            "bands": 103,
            "sensor": "ROSIS",
            "init_samples": 10,
        },
    }

    transfer_pairs = [
        ("Indian Pines", "Salinas", "Cross-Dataset Transfer (AVIRIS -> AVIRIS)"),
        ("Salinas", "Indian Pines", "Cross-Dataset Transfer (AVIRIS -> AVIRIS)"),
        ("Indian Pines", "Pavia University", "Cross-Sensor Adaptation (AVIRIS -> ROSIS)"),
        ("Salinas", "Pavia University", "Cross-Sensor Adaptation (AVIRIS -> ROSIS)"),
    ]

    summary_rows = []
    full_export = {}

    for src_name, tgt_name, transfer_type in transfer_pairs:
        print(f"\n--- Transfer Evaluation: {src_name} -> {tgt_name} ({transfer_type}) ---")

        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        src_meta = datasets_meta[src_name]
        tgt_meta = datasets_meta[tgt_name]

        # 1. Train Source Model
        src_dm = get_datamodule(src_name, src_meta)
        src_manager = ActiveLearningDatasetManager(
            train_dataset=src_dm.train_dataset,
            val_dataset=getattr(src_dm, "val_dataset", None),
            test_dataset=getattr(src_dm, "test_dataset", None),
            initial_samples_per_class=10,
            seed=seed,
        )
        src_backbone = load_frozen_backbone(src_meta["num_classes"], ckpt_path)
        src_engine = LearningEngine(
            backbone=src_backbone,
            num_classes=src_meta["num_classes"],
            dataset_manager=src_manager,
            hidden_dim=256,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        src_engine.classifier_head = ClassifierHead(
            in_features=src_engine.feature_extractor.feature_dim,
            num_classes=src_meta["num_classes"],
            hidden_dim=256,
            head_type="mlp_2layer",
            norm_layer="layernorm",
        )
        _ = src_engine.trainer.train_classifier_head(
            feature_extractor=src_engine.feature_extractor,
            classifier_head=src_engine.classifier_head,
            labeled_dataloader=src_manager.get_labeled_dataloader(batch_size=64),
            epochs=25,
            lr=0.0005,
        )

        # 2. Evaluate Zero-Shot (only if class counts match)
        zero_shot_metrics = {}
        if src_meta["num_classes"] == tgt_meta["num_classes"]:
            tgt_dm = get_datamodule(tgt_name, tgt_meta)
            tgt_test_loader = torch.utils.data.DataLoader(tgt_dm.test_dataset, batch_size=64, shuffle=False)

            resampler = SpectralResampler1D(target_bands=src_meta["bands"]).to(src_engine.trainer.device)
            src_engine.classifier_head.eval()
            src_engine.feature_extractor.eval()

            all_preds, all_probs, all_targets = [], [], []
            with torch.no_grad():
                for batch in tgt_test_loader:
                    x = batch["image"].to(src_engine.trainer.device) if isinstance(batch, dict) else batch[0].to(src_engine.trainer.device)
                    y = batch["label"] if isinstance(batch, dict) else batch[1]
                    x_res = resampler(x)
                    feats = src_engine.feature_extractor(x_res)
                    probs = src_engine.classifier_head.predict_probabilities(feats)
                    preds = torch.argmax(probs, dim=1)
                    all_preds.append(preds.cpu())
                    all_probs.append(probs.cpu())
                    all_targets.append(y)

            zero_shot_metrics = compute_detailed_metrics(
                torch.cat(all_targets).numpy(),
                torch.cat(all_preds).numpy(),
                torch.cat(all_probs).numpy(),
                num_classes=src_meta["num_classes"],
            )

            print(f"  Zero-Shot Transfer   -> OA: {zero_shot_metrics['OverallAccuracy']*100:6.2f}% | AA: {zero_shot_metrics['AverageAccuracy']*100:6.2f}% | F1: {zero_shot_metrics['F1Score']:.4f}")

            summary_rows.append({
                "Transfer Pair": f"{src_name} -> {tgt_name}",
                "Transfer Category": transfer_type,
                "Mode": "Zero-Shot Direct Transfer",
                "Final Target Labels": 0,
                "OA (%)": f"{zero_shot_metrics['OverallAccuracy']*100:.2f}",
                "AA (%)": f"{zero_shot_metrics['AverageAccuracy']*100:.2f}",
                "Macro F1": f"{zero_shot_metrics['F1Score']:.4f}",
                "Kappa": f"{zero_shot_metrics['Kappa']:.4f}",
                "ECE": f"{zero_shot_metrics['ECE']:.4f}",
            })

        # 3. Target-Only AL Baseline (No Transfer)
        tgt_dm = get_datamodule(tgt_name, tgt_meta)
        tgt_manager = ActiveLearningDatasetManager(
            train_dataset=tgt_dm.train_dataset,
            val_dataset=getattr(tgt_dm, "val_dataset", None),
            test_dataset=getattr(tgt_dm, "test_dataset", None),
            initial_samples_per_class=tgt_meta["init_samples"],
            seed=seed,
        )
        tgt_backbone = load_frozen_backbone(tgt_meta["num_classes"], ckpt_path)
        tgt_engine_scratch = LearningEngine(
            backbone=tgt_backbone,
            num_classes=tgt_meta["num_classes"],
            dataset_manager=tgt_manager,
            hidden_dim=256,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        tgt_engine_scratch.classifier_head = ClassifierHead(
            in_features=tgt_engine_scratch.feature_extractor.feature_dim,
            num_classes=tgt_meta["num_classes"],
            hidden_dim=256,
            head_type="mlp_2layer",
            norm_layer="layernorm",
        )
        hist_scratch = tgt_engine_scratch.run_full_active_learning_loop(
            query_engine=QueryEngine(uncertainty_method="filter_diversity_labeled"),
            k=20,
            total_rounds=5,
            epochs_per_round=25,
            lr=0.0005,
        )
        scratch_metrics = hist_scratch[-1]["metrics"]
        print(f"  Target-Only AL (No XFER) -> OA: {scratch_metrics['OverallAccuracy']*100:6.2f}% | AA: {scratch_metrics['AverageAccuracy']*100:6.2f}% | F1: {scratch_metrics['F1Score']:.4f}")

        summary_rows.append({
            "Transfer Pair": f"{src_name} -> {tgt_name}",
            "Transfer Category": transfer_type,
            "Mode": "Target-Only AL (No Transfer Baseline)",
            "Final Target Labels": hist_scratch[-1]["num_labeled"],
            "OA (%)": f"{scratch_metrics['OverallAccuracy']*100:.2f}",
            "AA (%)": f"{scratch_metrics['AverageAccuracy']*100:.2f}",
            "Macro F1": f"{scratch_metrics['F1Score']:.4f}",
            "Kappa": f"{scratch_metrics['Kappa']:.4f}",
            "ECE": f"{scratch_metrics.get('ECE', 0.0):.4f}",
        })

        # 4. Few-Shot Target AL Adaptation with Source FC1 Representation Transfer
        tgt_dm_xfer = get_datamodule(tgt_name, tgt_meta)
        tgt_manager_xfer = ActiveLearningDatasetManager(
            train_dataset=tgt_dm_xfer.train_dataset,
            val_dataset=getattr(tgt_dm_xfer, "val_dataset", None),
            test_dataset=getattr(tgt_dm_xfer, "test_dataset", None),
            initial_samples_per_class=tgt_meta["init_samples"],
            seed=seed,
        )
        tgt_backbone_xfer = load_frozen_backbone(src_meta["num_classes"], ckpt_path)
        resampler = SpectralResampler1D(target_bands=src_meta["bands"])

        tgt_engine_xfer = LearningEngine(
            backbone=tgt_backbone_xfer,
            num_classes=tgt_meta["num_classes"],
            dataset_manager=tgt_manager_xfer,
            hidden_dim=256,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        tgt_engine_xfer.classifier_head = ClassifierHead(
            in_features=tgt_engine_xfer.feature_extractor.feature_dim,
            num_classes=tgt_meta["num_classes"],
            hidden_dim=256,
            head_type="mlp_2layer",
            norm_layer="layernorm",
        )
        # Copy FC1 representation weights
        tgt_engine_xfer.classifier_head.head[0].weight.data.copy_(src_engine.classifier_head.head[0].weight.data)
        tgt_engine_xfer.classifier_head.head[0].bias.data.copy_(src_engine.classifier_head.head[0].bias.data)

        orig_fe_fwd = tgt_engine_xfer.feature_extractor.forward
        def resample_fwd(x):
            return orig_fe_fwd(resampler.to(x.device)(x))
        tgt_engine_xfer.feature_extractor.forward = resample_fwd

        hist_xfer = tgt_engine_xfer.run_full_active_learning_loop(
            query_engine=QueryEngine(uncertainty_method="filter_diversity_labeled"),
            k=20,
            total_rounds=5,
            epochs_per_round=25,
            lr=0.0005,
        )
        xfer_metrics = hist_xfer[-1]["metrics"]
        print(f"  Few-Shot XFER AL Adaptation -> OA: {xfer_metrics['OverallAccuracy']*100:6.2f}% | AA: {xfer_metrics['AverageAccuracy']*100:6.2f}% | F1: {xfer_metrics['F1Score']:.4f}")

        summary_rows.append({
            "Transfer Pair": f"{src_name} -> {tgt_name}",
            "Transfer Category": transfer_type,
            "Mode": "Few-Shot AL Adaptation + Feature Alignment",
            "Final Target Labels": hist_xfer[-1]["num_labeled"],
            "OA (%)": f"{xfer_metrics['OverallAccuracy']*100:.2f}",
            "AA (%)": f"{xfer_metrics['AverageAccuracy']*100:.2f}",
            "Macro F1": f"{xfer_metrics['F1Score']:.4f}",
            "Kappa": f"{xfer_metrics['Kappa']:.4f}",
            "ECE": f"{xfer_metrics.get('ECE', 0.0):.4f}",
        })

    # Export CSV summary
    csv_out = "logs/al_experiments/cross_sensor_results.csv"
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved cross-sensor and cross-dataset summary table to {csv_out}")

if __name__ == "__main__":
    run_master_transfer()
