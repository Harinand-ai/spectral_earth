import os
import sys
import json
import time
import torch
import numpy as np
import pyrootutils

pyrootutils.setup_root(r"d:\Projects\spectral_earth", indicator=".project-root", pythonpath=True)

from src.backbones.registry import BACKBONE_REGISTRY
from src.learning_engine import LearningEngine, ActiveLearningDatasetManager
from src.learning_engine.classifier import ClassifierHead
from src.query_engine.query_engine import QueryEngine
from src.utils.visualization import (
    plot_learning_curves,
    plot_strategy_comparison,
    plot_loss_and_calibration,
)

def run_full_benchmark():
    print("=" * 70)
    print("SPECTRALEARTH ACTIVE LEARNING FRAMEWORK — COMPREHENSIVE BENCHMARK")
    print("=" * 70)

    ckpt_path = r"d:\Projects\spectral_earth\logs\None\runs\2026-07-30_14-24-24\checkpoints\epoch_094.ckpt"
    strategies = [
        "least_confidence",
        "entropy",
        "margin",
        "adaptive",
        "hybrid",
        "diversity",
        "random",
    ]
    
    datasets_info = {
        "Indian Pines": {
            "module_cls": "src.datamodules.indian_pines.IndianPinesDataModule",
            "num_classes": 16,
            "num_bands": 200,
            "init_samples": 10,
        },
        "Salinas": {
            "module_cls": "src.datamodules.salinas.SalinasDataModule",
            "num_classes": 16,
            "num_bands": 204,
            "init_samples": 10,
        },
        "Pavia University": {
            "module_cls": "src.datamodules.pavia_university.PaviaUniversityDataModule",
            "num_classes": 9,
            "num_bands": 103,
            "init_samples": 10,
        },
    }

    all_results = {}
    os.makedirs("logs/al_experiments", exist_ok=True)
    os.makedirs("logs/visualizations", exist_ok=True)

    for dataset_name, meta in datasets_info.items():
        print(f"\n" + "=" * 70)
        print(f" >>> BENCHMARKING DATASET: {dataset_name} <<<")
        print("=" * 70)
        
        # Instantiate datamodule dynamically
        mod_path, class_name = meta["module_cls"].rsplit(".", 1)
        import importlib
        module = importlib.import_module(mod_path)
        DataModuleCls = getattr(module, class_name)
        
        datamodule = DataModuleCls()
        datamodule.setup("fit")
        datamodule.setup("test")

        dataset_results = {}

        for strat in strategies:
            print(f"Running strategy '{strat:18s}' on {dataset_name}...")
            torch.manual_seed(42)
            np.random.seed(42)

            train_dataset = datamodule.train_dataset
            val_dataset = getattr(datamodule, "val_dataset", None)
            test_dataset = getattr(datamodule, "test_dataset", None)

            num_classes = meta["num_classes"]

            backbone = BACKBONE_REGISTRY["spec_resnet50"](num_classes=num_classes)
            
            # Load pretrained SpectralEarth backbone weights
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

            # Freeze backbone parameters permanently
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()

            # Active Learning Dataset Manager
            dataset_manager = ActiveLearningDatasetManager(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                initial_samples_per_class=meta["init_samples"],
                seed=42,
            )

            # Model A: Learning Engine with Optimized mlp_2layer Classifier Head
            learning_engine = LearningEngine(
                backbone=backbone,
                num_classes=num_classes,
                dataset_manager=dataset_manager,
                hidden_dim=256,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Assign optimized MLP head
            learning_engine.classifier_head = ClassifierHead(
                in_features=learning_engine.feature_extractor.feature_dim,
                num_classes=num_classes,
                hidden_dim=256,
                head_type="mlp_2layer",
                norm_layer="layernorm",
            )

            # Model B: Query Engine
            query_engine = QueryEngine(
                uncertainty_method=strat,
                start_uncertainty=0.30,
                end_uncertainty=0.80,
            )

            # Run Active Learning loop with optimal hyperparameter setup
            history = learning_engine.run_full_active_learning_loop(
                query_engine=query_engine,
                k=20,
                total_rounds=5,
                epochs_per_round=25,
                lr=0.0005,
                batch_size=64,
            )

            # Sanitize numpy arrays for JSON export
            sanitized_history = []
            for entry in history:
                e_copy = dict(entry)
                if "metrics" in e_copy and "ConfusionMatrix" in e_copy["metrics"]:
                    if isinstance(e_copy["metrics"]["ConfusionMatrix"], np.ndarray):
                        e_copy["metrics"]["ConfusionMatrix"] = e_copy["metrics"]["ConfusionMatrix"].tolist()
                sanitized_history.append(e_copy)

            dataset_results[strat] = sanitized_history

            last_m = sanitized_history[-1].get("metrics", {})
            print(
                f"   Result [{strat:18s}] -> Final OA: {last_m.get('OverallAccuracy', 0.0)*100:6.2f}% | "
                f"AA: {last_m.get('AverageAccuracy', 0.0)*100:6.2f}% | F1: {last_m.get('F1Score', 0.0):.4f} | "
                f"Kappa: {last_m.get('Kappa', 0.0):.4f}"
            )

        all_results[dataset_name] = dataset_results

        # Generate publication figures for this dataset
        ds_slug = dataset_name.lower().replace(" ", "_")
        plot_strategy_comparison(
            dataset_results,
            metric_key="OverallAccuracy",
            save_path=f"logs/visualizations/{ds_slug}_strategy_comparison_oa.png",
            dataset_name=dataset_name,
        )
        best_strat_key = max(dataset_results.keys(), key=lambda s: dataset_results[s][-1]["metrics"].get("OverallAccuracy", 0.0))
        plot_learning_curves(
            dataset_results[best_strat_key],
            save_path=f"logs/visualizations/{ds_slug}_learning_curves.png",
            title_suffix=f"{dataset_name} ({best_strat_key})",
        )
        plot_loss_and_calibration(
            dataset_results[best_strat_key],
            save_path=f"logs/visualizations/{ds_slug}_loss_and_calibration.png",
        )

    # Save full benchmark JSON
    json_path = "logs/al_experiments/full_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull benchmark results saved to {json_path}")

    # Print Final Summary Table
    print("\n" + "=" * 88)
    print("FINAL BENCHMARK SUMMARY TABLE")
    print("=" * 88)
    print(f"{'Dataset':<18} | {'Strategy':<18} | {'Final OA (%)':<12} | {'Final AA (%)':<12} | {'F1 Score':<10} | {'Kappa':<10}")
    print("-" * 88)

    for d_name, d_res in all_results.items():
        for s_name, s_hist in d_res.items():
            final_m = s_hist[-1].get("metrics", {})
            oa = final_m.get("OverallAccuracy", 0.0) * 100
            aa = final_m.get("AverageAccuracy", 0.0) * 100
            f1 = final_m.get("F1Score", 0.0)
            kappa = final_m.get("Kappa", 0.0)
            print(f"{d_name:<18} | {s_name:<18} | {oa:12.2f} | {aa:12.2f} | {f1:10.4f} | {kappa:10.4f}")

if __name__ == "__main__":
    run_full_benchmark()
