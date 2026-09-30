import os
import sys
import json
import csv
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pyrootutils

pyrootutils.setup_root(r"d:\Projects\spectral_earth", indicator=".project-root", pythonpath=True)

from src.backbones.registry import BACKBONE_REGISTRY
from src.learning_engine import LearningEngine, ActiveLearningDatasetManager
from src.learning_engine.classifier import ClassifierHead
from src.query_engine.query_engine import QueryEngine
from src.utils.spectral_adapter_1d import SpectralResampler1D

def run_cross_sensor_evaluation():
    print("=" * 88)
    print("SPECTRALEARTH ACTIVE LEARNING FRAMEWORK — CROSS-SENSOR / CROSS-DATASET TRANSFER EVALUATION")
    print("=" * 88)

    ckpt_path = r"d:\Projects\spectral_earth\logs\None\runs\2026-07-30_14-24-24\checkpoints\epoch_094.ckpt"
    os.makedirs("logs/al_experiments", exist_ok=True)
    os.makedirs("logs/visualizations/cross_sensor", exist_ok=True)

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

    # Helper function to instantiate backbone with state dict
    def load_frozen_backbone(num_classes):
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

    # Helper function to instantiate datamodule
    def get_datamodule(ds_name):
        mod_path, class_name = datasets_meta[ds_name]["module_cls"].rsplit(".", 1)
        import importlib
        module = importlib.import_module(mod_path)
        DataModuleCls = getattr(module, class_name)
        dm = DataModuleCls()
        dm.setup("fit")
        dm.setup("test")
        return dm

    transfer_pairs = [
        ("Indian Pines", "Salinas", "Cross-Dataset Transfer (AVIRIS -> AVIRIS)"),
        ("Salinas", "Indian Pines", "Cross-Dataset Transfer (AVIRIS -> AVIRIS)"),
        ("Indian Pines", "Pavia University", "Cross-Sensor & Cross-Domain Transfer (AVIRIS -> ROSIS)"),
        ("Salinas", "Pavia University", "Cross-Sensor & Cross-Domain Transfer (AVIRIS -> ROSIS)"),
    ]

    zero_shot_results = {}
    few_shot_results = {}

    # =========================================================================
    # EXPERIMENT 1: ZERO-SHOT TRANSFER EVALUATION
    # =========================================================================
    print("\n" + "=" * 88)
    print(" >>> EXPERIMENT 1: ZERO-SHOT CROSS-DOMAIN TRANSFER EVALUATION <<<")
    print("=" * 88)

    for src_name, tgt_name, pair_desc in transfer_pairs:
        if datasets_meta[src_name]["num_classes"] != datasets_meta[tgt_name]["num_classes"]:
            print(f"Skipping Zero-Shot for {src_name} -> {tgt_name} (Class count mismatch: {datasets_meta[src_name]['num_classes']} vs {datasets_meta[tgt_name]['num_classes']})")
            continue

        print(f"\nEvaluating Zero-Shot Transfer: {src_name} -> {tgt_name} ({pair_desc})...")
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 1. Train classifier on Source Dataset
        src_dm = get_datamodule(src_name)
        src_manager = ActiveLearningDatasetManager(
            train_dataset=src_dm.train_dataset,
            val_dataset=getattr(src_dm, "val_dataset", None),
            test_dataset=getattr(src_dm, "test_dataset", None),
            initial_samples_per_class=10,
            seed=seed,
        )
        src_backbone = load_frozen_backbone(datasets_meta[src_name]["num_classes"])
        src_engine = LearningEngine(
            backbone=src_backbone,
            num_classes=datasets_meta[src_name]["num_classes"],
            dataset_manager=src_manager,
            hidden_dim=256,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        src_engine.classifier_head = ClassifierHead(
            in_features=src_engine.feature_extractor.feature_dim,
            num_classes=datasets_meta[src_name]["num_classes"],
            hidden_dim=256,
            head_type="mlp_2layer",
            norm_layer="layernorm",
        )
        
        # Train Source Classifier Head on initial pool
        print(f"  Training Source Classifier Head on {src_name}...")
        _ = src_engine.trainer.train_classifier_head(
            feature_extractor=src_engine.feature_extractor,
            classifier_head=src_engine.classifier_head,
            labeled_dataloader=src_manager.get_labeled_dataloader(batch_size=64),
            epochs=25,
            lr=0.0005,
            reset_head=False,
        )

        # 2. Evaluate Zero-Shot on Target Dataset Test Set with 1D Spectral Alignment
        tgt_dm = get_datamodule(tgt_name)
        tgt_manager = ActiveLearningDatasetManager(
            train_dataset=tgt_dm.train_dataset,
            val_dataset=getattr(tgt_dm, "val_dataset", None),
            test_dataset=getattr(tgt_dm, "test_dataset", None),
            initial_samples_per_class=10,
            seed=seed,
        )

        # Create isolated 1D resampler mapping Target Bands -> Source Bands
        resampler = SpectralResampler1D(target_bands=datasets_meta[src_name]["bands"])
        
        # Custom zero-shot evaluation on target test dataset
        tgt_test_loader = tgt_manager.get_test_dataloader(batch_size=64)
        src_engine.classifier_head.eval()
        src_engine.feature_extractor.eval()

        device = src_engine.trainer.device
        resampler = resampler.to(device)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tgt_test_loader:
                if isinstance(batch, dict):
                    batch_x = batch["image"].to(device)
                    batch_y = batch["label"].to(device)
                else:
                    batch_x, batch_y = batch[0].to(device), batch[1].to(device)
                
                # Apply 1D spectral alignment target_bands -> source_bands
                batch_x_resampled = resampler(batch_x)
                
                # Forward through frozen SpectralEarth and trained Source Head
                feats = src_engine.feature_extractor(batch_x_resampled)
                logits = src_engine.classifier_head(feats)
                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds.cpu())
                all_targets.append(batch_y.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Compute Metrics via TorchMetrics
        from torchmetrics import Accuracy, F1Score, CohenKappa, Precision, Recall
        nc = datasets_meta[src_name]["num_classes"]
        acc_metric = Accuracy(task="multiclass", num_classes=nc, average="micro")
        aa_metric = Accuracy(task="multiclass", num_classes=nc, average="macro")
        f1_metric = F1Score(task="multiclass", num_classes=nc, average="macro")
        kappa_metric = CohenKappa(task="multiclass", num_classes=nc)
        prec_metric = Precision(task="multiclass", num_classes=nc, average="macro")
        rec_metric = Recall(task="multiclass", num_classes=nc, average="macro")

        oa_val = float(acc_metric(all_preds, all_targets)) * 100
        aa_val = float(aa_metric(all_preds, all_targets)) * 100
        f1_val = float(f1_metric(all_preds, all_targets))
        kappa_val = float(kappa_metric(all_preds, all_targets))
        prec_val = float(prec_metric(all_preds, all_targets))
        rec_val = float(rec_metric(all_preds, all_targets))

        pair_key = f"{src_name} -> {tgt_name}"
        zero_shot_results[pair_key] = {
            "source_dataset": src_name,
            "target_dataset": tgt_name,
            "description": pair_desc,
            "oa": oa_val,
            "aa": aa_val,
            "f1": f1_val,
            "kappa": kappa_val,
            "precision": prec_val,
            "recall": rec_val,
        }

        print(f"  Zero-Shot Result ({pair_key}): OA = {oa_val:6.2f}% | AA = {aa_val:6.2f}% | F1 = {f1_val:.4f} | Kappa = {kappa_val:.4f}")

    # =========================================================================
    # EXPERIMENT 2: FEW-SHOT TARGET ADAPTATION VIA ACTIVE LEARNING
    # =========================================================================
    print("\n" + "=" * 88)
    print(" >>> EXPERIMENT 2: FEW-SHOT TARGET ADAPTATION VIA ACTIVE LEARNING <<<")
    print("=" * 88)

    for src_name, tgt_name, pair_desc in transfer_pairs:
        pair_key = f"{src_name} -> {tgt_name}"
        print(f"\nRunning Few-Shot Target AL Adaptation: {pair_key} ({pair_desc})...")
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 1. Train Source Model
        src_dm = get_datamodule(src_name)
        src_manager = ActiveLearningDatasetManager(
            train_dataset=src_dm.train_dataset,
            val_dataset=getattr(src_dm, "val_dataset", None),
            test_dataset=getattr(src_dm, "test_dataset", None),
            initial_samples_per_class=10,
            seed=seed,
        )
        src_backbone = load_frozen_backbone(datasets_meta[src_name]["num_classes"])
        src_engine = LearningEngine(
            backbone=src_backbone,
            num_classes=datasets_meta[src_name]["num_classes"],
            dataset_manager=src_manager,
            hidden_dim=256,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        src_engine.classifier_head = ClassifierHead(
            in_features=src_engine.feature_extractor.feature_dim,
            num_classes=datasets_meta[src_name]["num_classes"],
            hidden_dim=256,
            head_type="mlp_2layer",
            norm_layer="layernorm",
        )
        print(f"  Training Source Classifier Head on {src_name} (Classes: {datasets_meta[src_name]['num_classes']}, Bands: {datasets_meta[src_name]['bands']})...")
        _ = src_engine.trainer.train_classifier_head(
            feature_extractor=src_engine.feature_extractor,
            classifier_head=src_engine.classifier_head,
            labeled_dataloader=src_manager.get_labeled_dataloader(batch_size=64),
            epochs=25,
            lr=0.0005,
            reset_head=False,
        )

        src_weight_sum = float(torch.sum(src_engine.classifier_head.head[0].weight).item())
        print(f"  Source Head FC1 Weight Sum Signature: {src_weight_sum:.6f}")

        # 2. Setup Target Dataset and Resampler
        tgt_dm = get_datamodule(tgt_name)
        tgt_manager = ActiveLearningDatasetManager(
            train_dataset=tgt_dm.train_dataset,
            val_dataset=getattr(tgt_dm, "val_dataset", None),
            test_dataset=getattr(tgt_dm, "test_dataset", None),
            initial_samples_per_class=datasets_meta[tgt_name]["init_samples"],
            seed=seed,
        )

        tgt_num_classes = datasets_meta[tgt_name]["num_classes"]
        tgt_backbone = load_frozen_backbone(datasets_meta[src_name]["num_classes"]) # Expects Source Bands

        resampler = SpectralResampler1D(target_bands=datasets_meta[src_name]["bands"])

        tgt_engine = LearningEngine(
            backbone=tgt_backbone,
            num_classes=tgt_num_classes,
            dataset_manager=tgt_manager,
            hidden_dim=256,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        tgt_engine.classifier_head = ClassifierHead(
            in_features=tgt_engine.feature_extractor.feature_dim,
            num_classes=tgt_num_classes,
            hidden_dim=256,
            head_type="mlp_2layer",
            norm_layer="layernorm",
        )

        # Transfer source hidden representation (FC1 weights) to target head
        print(f"  Transferring Source Hidden Representation (FC1) to Target Head (Target Classes: {tgt_num_classes})...")
        tgt_engine.classifier_head.head[0].weight.data.copy_(src_engine.classifier_head.head[0].weight.data)
        tgt_engine.classifier_head.head[0].bias.data.copy_(src_engine.classifier_head.head[0].bias.data)

        # Override feature extractor forward for resampled input
        original_fe_forward = tgt_engine.feature_extractor.forward
        def resampled_fe_forward(x):
            x_resampler = resampler.to(x.device)
            x_resampled = x_resampler(x)
            return original_fe_forward(x_resampled)
        tgt_engine.feature_extractor.forward = resampled_fe_forward

        query_engine = QueryEngine(uncertainty_method="filter_diversity")

        history = tgt_engine.run_full_active_learning_loop(
            query_engine=query_engine,
            k=20,
            total_rounds=5,
            epochs_per_round=25,
            lr=0.0005,
            batch_size=64,
        )

        sanitized_history = []
        for entry in history:
            e_copy = dict(entry)
            if "metrics" in e_copy and "ConfusionMatrix" in e_copy["metrics"]:
                if isinstance(e_copy["metrics"]["ConfusionMatrix"], np.ndarray):
                    e_copy["metrics"]["ConfusionMatrix"] = e_copy["metrics"]["ConfusionMatrix"].tolist()
            sanitized_history.append(e_copy)

        few_shot_results[pair_key] = {
            "source_dataset": src_name,
            "target_dataset": tgt_name,
            "description": pair_desc,
            "history": sanitized_history,
        }

        final_m = sanitized_history[-1]["metrics"]
        print(
            f"  Few-Shot Final Result ({pair_key}): OA = {final_m.get('OverallAccuracy', 0.0)*100:6.2f}% | "
            f"AA = {final_m.get('AverageAccuracy', 0.0)*100:6.2f}% | F1 = {final_m.get('F1Score', 0.0):.4f} | "
            f"Kappa = {final_m.get('Kappa', 0.0):.4f}"
        )

    # Save JSON and CSV exports
    cross_sensor_export = {
        "zero_shot_results": zero_shot_results,
        "few_shot_results": few_shot_results,
    }

    json_path = "logs/al_experiments/cross_sensor_results.json"
    with open(json_path, "w") as f:
        json.dump(cross_sensor_export, f, indent=2)
    print(f"\nCross-sensor results saved to {json_path}")

    # Summary CSV Export
    csv_path = "logs/al_experiments/cross_sensor_summary.csv"
    csv_rows = []

    for pair_key, z_res in zero_shot_results.items():
        csv_rows.append({
            "Transfer Pair": pair_key,
            "Transfer Type": z_res["description"],
            "Evaluation Mode": "Zero-Shot Transfer",
            "Final Labeled Samples": 0,
            "OA (%)": f"{z_res['oa']:.2f}",
            "AA (%)": f"{z_res['aa']:.2f}",
            "Macro F1": f"{z_res['f1']:.4f}",
            "Kappa": f"{z_res['kappa']:.4f}",
        })

    for pair_key, f_res in few_shot_results.items():
        final_m = f_res["history"][-1]["metrics"]
        final_l = f_res["history"][-1]["num_labeled"]
        csv_rows.append({
            "Transfer Pair": pair_key,
            "Transfer Type": f_res["description"],
            "Evaluation Mode": "Few-Shot AL Adaptation",
            "Final Labeled Samples": final_l,
            "OA (%)": f"{final_m.get('OverallAccuracy', 0.0)*100:.2f}",
            "AA (%)": f"{final_m.get('AverageAccuracy', 0.0)*100:.2f}",
            "Macro F1": f"{final_m.get('F1Score', 0.0):.4f}",
            "Kappa": f"{final_m.get('Kappa', 0.0):.4f}",
        })

    fieldnames = ["Transfer Pair", "Transfer Type", "Evaluation Mode", "Final Labeled Samples", "OA (%)", "AA (%)", "Macro F1", "Kappa"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Cross-sensor summary CSV saved to {csv_path}")

    # Save Round-by-Round CSV Export
    round_csv_path = "logs/al_experiments/cross_sensor_round_results.csv"
    round_csv_rows = []

    for pair_key, f_res in few_shot_results.items():
        for r_entry in f_res["history"]:
            m = r_entry["metrics"]
            round_csv_rows.append({
                "Transfer Pair": pair_key,
                "Round": r_entry["round"],
                "Labeled Samples": r_entry["num_labeled"],
                "OA (%)": f"{m.get('OverallAccuracy', 0.0)*100:.2f}",
                "AA (%)": f"{m.get('AverageAccuracy', 0.0)*100:.2f}",
                "Macro F1": f"{m.get('F1Score', 0.0):.4f}",
                "Kappa": f"{m.get('Kappa', 0.0):.4f}",
            })

    round_fieldnames = ["Transfer Pair", "Round", "Labeled Samples", "OA (%)", "AA (%)", "Macro F1", "Kappa"]
    with open(round_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=round_fieldnames)
        writer.writeheader()
        writer.writerows(round_csv_rows)
    print(f"Cross-sensor round CSV saved to {round_csv_path}")

    # Generate Publication Plots
    # Plot 1: Zero-Shot Transfer OA Comparison
    if zero_shot_results:
        plt.figure(figsize=(7, 5))
        pairs_list = list(zero_shot_results.keys())
        oas_list = [zero_shot_results[p]["oa"] for p in pairs_list]
        plt.bar(pairs_list, oas_list, color="#1f77b4", width=0.4)
        plt.title("Zero-Shot Cross-Dataset Transfer Overall Accuracy (%)", fontsize=12, fontweight="bold")
        plt.ylabel("Overall Accuracy (%)", fontsize=11)
        plt.ylim(0, 100)
        plt.grid(True, linestyle="--", alpha=0.5, axis="y")
        plt.tight_layout()
        plt.savefig("logs/visualizations/cross_sensor/zero_shot_transfer_oa.png", dpi=300)
        plt.close()

    # Plot 2: Few-Shot AL Curves OA Comparison
    plt.figure(figsize=(8, 6))
    for pair_key, f_res in few_shot_results.items():
        labeled_x = [e["num_labeled"] for e in f_res["history"]]
        oa_y = [e["metrics"].get("OverallAccuracy", 0.0)*100 for e in f_res["history"]]
        plt.plot(labeled_x, oa_y, label=pair_key, marker="o", linewidth=2)

    plt.title("Few-Shot Target Adaptation — OA vs Labeled Target Samples", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Labeled Target Samples", fontsize=11)
    plt.ylabel("Overall Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig("logs/visualizations/cross_sensor/few_shot_transfer_oa_curves.png", dpi=300)
    plt.close()

    print("\n" + "=" * 90)
    print("CROSS-SENSOR EVALUATION SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Transfer Pair':<30} | {'Mode':<22} | {'OA (%)':<10} | {'AA (%)':<10} | {'Macro F1':<10} | {'Kappa':<10}")
    print("-" * 102)

    for row in csv_rows:
        print(f"{row['Transfer Pair']:<30} | {row['Evaluation Mode']:<22} | {row['OA (%)']:<10} | {row['AA (%)']:<10} | {row['Macro F1']:<10} | {row['Kappa']:<10}")

if __name__ == "__main__":
    run_cross_sensor_evaluation()
