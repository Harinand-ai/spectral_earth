"""
Controller Screening Experiment (Priority 2 & 3).
Evaluates AdaptiveQueryController against Random, Least Confidence, and LC+FD across Indian Pines, Salinas, and Pavia University.
"""
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
from src.query_engine.query_engine import QueryEngine, AdaptiveQueryController
from src.query_engine.query_engine.uncertainty import expected_calibration_error

def run_controller_benchmark(seeds=[42], output_prefix="screening"):
    print("=" * 88)
    print(f"SPECTRALEARTH — ADAPTIVE QUERY CONTROLLER BENCHMARK ({output_prefix.upper()})")
    print(f"Seeds: {seeds}")
    print("=" * 88)

    ckpt_path = r"d:\Projects\spectral_earth\logs\None\runs\2026-07-30_14-24-24\checkpoints\epoch_094.ckpt"
    os.makedirs("logs/al_experiments", exist_ok=True)

    datasets_meta = {
        "Indian Pines": {
            "module_cls": "src.datamodules.indian_pines.IndianPinesDataModule",
            "num_classes": 16,
            "bands": 200,
            "init_samples": 10,
        },
        "Salinas": {
            "module_cls": "src.datamodules.salinas.SalinasDataModule",
            "num_classes": 16,
            "bands": 204,
            "init_samples": 10,
        },
        "Pavia University": {
            "module_cls": "src.datamodules.pavia_university.PaviaUniversityDataModule",
            "num_classes": 9,
            "bands": 103,
            "init_samples": 10,
        },
    }

    strategies = [
        ("Random", "random"),
        ("Least Confidence", "least_confidence"),
        ("LC + Farthest Diversity", "filter_diversity"),
        ("Adaptive Query Controller", "adaptive_controller"),
    ]

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

    def get_datamodule(ds_name):
        mod_path, class_name = datasets_meta[ds_name]["module_cls"].rsplit(".", 1)
        import importlib
        module = importlib.import_module(mod_path)
        DataModuleCls = getattr(module, class_name)
        dm = DataModuleCls()
        dm.setup("fit")
        dm.setup("test")
        return dm

    all_run_results = []
    weight_logs = []

    for ds_name, meta in datasets_meta.items():
        print(f"\n" + "=" * 80)
        print(f" DATASET: {ds_name} (Classes: {meta['num_classes']}, Bands: {meta['bands']})")
        print("=" * 80)

        for strat_label, strat_key in strategies:
            for seed in seeds:
                print(f"\n--- Strategy: {strat_label} | Seed: {seed} ---")
                torch.manual_seed(seed)
                np.random.seed(seed)

                dm = get_datamodule(ds_name)
                manager = ActiveLearningDatasetManager(
                    train_dataset=dm.train_dataset,
                    val_dataset=getattr(dm, "val_dataset", None),
                    test_dataset=getattr(dm, "test_dataset", None),
                    initial_samples_per_class=meta["init_samples"],
                    seed=seed,
                )

                backbone = load_frozen_backbone(meta["num_classes"])
                engine = LearningEngine(
                    backbone=backbone,
                    num_classes=meta["num_classes"],
                    dataset_manager=manager,
                    hidden_dim=256,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                engine.classifier_head = ClassifierHead(
                    in_features=engine.feature_extractor.feature_dim,
                    num_classes=meta["num_classes"],
                    hidden_dim=256,
                    head_type="mlp_2layer",
                    norm_layer="layernorm",
                )

                # Confirm backbone is frozen
                trainable_bb_params = sum(p.numel() for p in engine.feature_extractor.parameters() if p.requires_grad)
                assert trainable_bb_params == 0, "BACKBONE MUST BE 100% FROZEN"

                query_engine = QueryEngine(uncertainty_method=strat_key)

                # Custom loop tracking query time and controller weights
                t_start = time.time()
                history = engine.run_full_active_learning_loop(
                    query_engine=query_engine,
                    k=20,
                    total_rounds=5,
                    epochs_per_round=25,
                    lr=0.0005,
                    batch_size=64,
                )
                total_time = time.time() - t_start

                # Extract round metrics
                oa_curve = [entry["metrics"].get("OverallAccuracy", 0.0) * 100 for entry in history]
                aa_curve = [entry["metrics"].get("AverageAccuracy", 0.0) * 100 for entry in history]
                f1_curve = [entry["metrics"].get("F1Score", 0.0) for entry in history]
                kappa_curve = [entry["metrics"].get("Kappa", 0.0) for entry in history]
                labeled_counts = [entry["num_labeled"] for entry in history]

                final_m = history[-1]["metrics"]
                final_oa = final_m.get("OverallAccuracy", 0.0) * 100
                final_aa = final_m.get("AverageAccuracy", 0.0) * 100
                final_f1 = final_m.get("F1Score", 0.0)
                final_kappa = final_m.get("Kappa", 0.0)

                # Calculate test set metrics via trainer.evaluate
                test_metrics = engine.trainer.evaluate(
                    feature_extractor=engine.feature_extractor,
                    classifier_head=engine.classifier_head,
                    dataloader=manager.get_test_dataloader(batch_size=64),
                    num_classes=meta["num_classes"],
                )
                final_ece = float(test_metrics.get("ECE", 0.0))

                # Calculate AULC via explicit trapezoidal integration
                def trapz_calc(y, x):
                    y_arr = np.asarray(y, dtype=np.float64)
                    x_arr = np.asarray(x, dtype=np.float64)
                    return float(np.sum(0.5 * (y_arr[1:] + y_arr[:-1]) * (x_arr[1:] - x_arr[:-1])))

                dx = labeled_counts[-1] - labeled_counts[0]
                aulc_oa = float(trapz_calc(oa_curve, labeled_counts) / dx) if dx > 0 else float(oa_curve[0])
                aulc_aa = float(trapz_calc(aa_curve, labeled_counts) / dx) if dx > 0 else float(aa_curve[0])

                run_record = {
                    "dataset": ds_name,
                    "strategy": strat_label,
                    "strategy_key": strat_key,
                    "seed": seed,
                    "initial_labeled": labeled_counts[0],
                    "final_labeled": labeled_counts[-1],
                    "final_oa": final_oa,
                    "final_aa": final_aa,
                    "final_f1": final_f1,
                    "final_kappa": final_kappa,
                    "final_ece": final_ece,
                    "aulc_oa": aulc_oa,
                    "aulc_aa": aulc_aa,
                    "total_time_s": total_time,
                    "oa_curve": oa_curve,
                    "labeled_counts": labeled_counts,
                }
                all_run_results.append(run_record)

                print(f"  Result ({strat_label}): Final OA = {final_oa:6.2f}% | AA = {final_aa:6.2f}% | F1 = {final_f1:.4f} | ECE = {final_ece:.4f} | AULC_OA = {aulc_oa:6.2f}%")

                # If adaptive controller, log per-round weights
                if strat_key == "adaptive_controller":
                    controller = AdaptiveQueryController()
                    val_loader = manager.get_unlabeled_dataloader(batch_size=64)
                    _, val_probs = engine.trainer.extract_unlabeled_embeddings_and_probs(
                        feature_extractor=engine.feature_extractor,
                        classifier_head=engine.classifier_head,
                        unlabeled_dataloader=val_loader,
                    )
                    for r_idx in range(5):
                        w_info = controller.compute_controller_weights(
                            probabilities=val_probs,
                            current_round=r_idx,
                            total_rounds=5,
                        )
                        weight_logs.append({
                            "dataset": ds_name,
                            "seed": seed,
                            "round": r_idx,
                            "w_uncertainty": w_info["w_uncertainty"],
                            "w_diversity": w_info["w_diversity"],
                            "w_hard_region": w_info["w_hard_region"],
                        })

    # Save output artifacts
    json_path = f"logs/al_experiments/controller_{output_prefix}_results.json"
    with open(json_path, "w") as f:
        json.dump(all_run_results, f, indent=2)
    print(f"\nSaved screening results to {json_path}")

    # Summary CSV
    csv_path = f"logs/al_experiments/controller_{output_prefix}_summary.csv"
    fieldnames = ["dataset", "strategy", "seed", "initial_labeled", "final_labeled", "final_oa", "final_aa", "final_f1", "final_kappa", "final_ece", "aulc_oa", "total_time_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_run_results:
            writer.writerow({k: f"{r[k]:.4f}" if isinstance(r[k], float) else r[k] for k in fieldnames})
    print(f"Saved summary CSV to {csv_path}")

    # Weights CSV
    if weight_logs:
        w_csv_path = "logs/al_experiments/controller_weights.csv"
        with open(w_csv_path, "w", newline="") as f:
            w_writer = csv.DictWriter(f, fieldnames=["dataset", "seed", "round", "w_uncertainty", "w_diversity", "w_hard_region"])
            w_writer.writeheader()
            w_writer.writerows(weight_logs)
        print(f"Saved controller weights CSV to {w_csv_path}")

    return all_run_results

if __name__ == "__main__":
    # Priority 2: Seed 42 Screening
    run_controller_benchmark(seeds=[42], output_prefix="screening")
