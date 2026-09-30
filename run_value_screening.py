"""
Expected Learning Value Seed-42 Screening Runner.
Compares Random, Least Confidence, LC + Farthest Diversity, and Expected Learning Value
across Indian Pines, Salinas, and Pavia University.
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
from src.query_engine.query_engine import QueryEngine, ExpectedLearningValueController
from src.query_engine.query_engine.uncertainty import expected_calibration_error

def run_value_screening(seeds=[42], output_prefix="screening"):
    print("=" * 88)
    print(f"SPECTRALEARTH — EXPECTED LEARNING VALUE BENCHMARK ({output_prefix.upper()})")
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
        ("Expected Learning Value", "expected_learning_value"),
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
    round_curves_rows = []
    components_rows = []
    timing_rows = []

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

                # Strict frozen backbone assertion
                trainable_bb_params = sum(p.numel() for p in engine.feature_extractor.parameters() if p.requires_grad)
                assert trainable_bb_params == 0, "BACKBONE MUST BE 100% FROZEN"

                query_engine = QueryEngine(uncertainty_method=strat_key)

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

                # Extract per-round curves
                oa_curve = [entry["metrics"].get("OverallAccuracy", 0.0) * 100 for entry in history]
                aa_curve = [entry["metrics"].get("AverageAccuracy", 0.0) * 100 for entry in history]
                f1_curve = [entry["metrics"].get("F1Score", 0.0) for entry in history]
                kappa_curve = [entry["metrics"].get("Kappa", 0.0) for entry in history]
                labeled_counts = [entry["num_labeled"] for entry in history]
                query_times = [entry.get("query_time", 0.0) for entry in history]

                for r_idx in range(len(history)):
                    round_curves_rows.append({
                        "dataset": ds_name,
                        "strategy": strat_label,
                        "seed": seed,
                        "round": r_idx,
                        "num_labeled": labeled_counts[r_idx],
                        "oa": oa_curve[r_idx],
                        "aa": aa_curve[r_idx],
                        "macro_f1": f1_curve[r_idx],
                        "kappa": kappa_curve[r_idx],
                        "query_time_s": query_times[r_idx],
                    })

                final_m = history[-1]["metrics"]
                final_oa = final_m.get("OverallAccuracy", 0.0) * 100
                final_aa = final_m.get("AverageAccuracy", 0.0) * 100
                final_f1 = final_m.get("F1Score", 0.0)
                final_kappa = final_m.get("Kappa", 0.0)

                test_metrics = engine.trainer.evaluate(
                    feature_extractor=engine.feature_extractor,
                    classifier_head=engine.classifier_head,
                    dataloader=manager.get_test_dataloader(batch_size=64),
                    num_classes=meta["num_classes"],
                )
                final_ece = float(test_metrics.get("ECE", 0.0))

                def trapz_calc(y, x):
                    y_arr = np.asarray(y, dtype=np.float64)
                    x_arr = np.asarray(x, dtype=np.float64)
                    return float(np.sum(0.5 * (y_arr[1:] + y_arr[:-1]) * (x_arr[1:] - x_arr[:-1])))

                dx = labeled_counts[-1] - labeled_counts[0]
                aulc_oa = float(trapz_calc(oa_curve, labeled_counts) / dx) if dx > 0 else float(oa_curve[0])
                aulc_aa = float(trapz_calc(aa_curve, labeled_counts) / dx) if dx > 0 else float(aa_curve[0])

                mean_query_time = float(np.mean(query_times))

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
                    "mean_query_time_s": mean_query_time,
                    "oa_curve": oa_curve,
                    "labeled_counts": labeled_counts,
                }
                all_run_results.append(run_record)

                timing_rows.append({
                    "dataset": ds_name,
                    "strategy": strat_label,
                    "seed": seed,
                    "total_time_s": total_time,
                    "mean_query_time_s": mean_query_time,
                })

                print(f"  Result ({strat_label}): Final OA = {final_oa:6.2f}% | AA = {final_aa:6.2f}% | F1 = {final_f1:.4f} | ECE = {final_ece:.4f} | AULC_OA = {aulc_oa:6.2f}%")

                # If Expected Learning Value, log components breakdown
                if strat_key == "expected_learning_value":
                    val_loader = manager.get_unlabeled_dataloader(batch_size=64)
                    unlabeled_feats, val_probs = engine.trainer.extract_unlabeled_embeddings_and_probs(
                        feature_extractor=engine.feature_extractor,
                        classifier_head=engine.classifier_head,
                        unlabeled_dataloader=val_loader,
                    )
                    labeled_loader = manager.get_labeled_dataloader(batch_size=64)
                    labeled_feats, _ = engine.trainer.extract_unlabeled_embeddings_and_probs(
                        feature_extractor=engine.feature_extractor,
                        classifier_head=engine.classifier_head,
                        unlabeled_dataloader=labeled_loader,
                    )

                    controller = ExpectedLearningValueController()
                    _, c_details = controller.select_samples(
                        embeddings=unlabeled_feats,
                        probabilities=val_probs,
                        k=20,
                        labeled_embeddings=labeled_feats,
                    )

                    components_rows.append({
                        "dataset": ds_name,
                        "seed": seed,
                        "w_u": c_details["weights"]["w_u"],
                        "w_n": c_details["weights"]["w_n"],
                        "w_g": c_details["weights"]["w_g"],
                        "mean_norm_u": float(np.mean(c_details["norm_u"])),
                        "mean_norm_n": float(np.mean(c_details["norm_n"])),
                        "mean_norm_g": float(np.mean(c_details["norm_g"])),
                    })

    # Save output artifacts
    json_path = f"logs/al_experiments/value_selection_results.json"
    with open(json_path, "w") as f:
        json.dump(all_run_results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    csv_path = f"logs/al_experiments/value_selection_summary.csv"
    fieldnames = ["dataset", "strategy", "seed", "initial_labeled", "final_labeled", "final_oa", "final_aa", "final_f1", "final_kappa", "final_ece", "aulc_oa", "mean_query_time_s", "total_time_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_run_results:
            writer.writerow({k: f"{r[k]:.4f}" if isinstance(r[k], float) else r[k] for k in fieldnames})
    print(f"Saved summary CSV to {csv_path}")

    # Curves CSV
    curves_csv_path = "logs/al_experiments/value_selection_round_curves.csv"
    with open(curves_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "strategy", "seed", "round", "num_labeled", "oa", "aa", "macro_f1", "kappa", "query_time_s"])
        w.writeheader()
        w.writerows(round_curves_rows)
    print(f"Saved round curves CSV to {curves_csv_path}")

    # Components CSV
    if components_rows:
        comp_csv_path = "logs/al_experiments/value_selection_components.csv"
        with open(comp_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "seed", "w_u", "w_n", "w_g", "mean_norm_u", "mean_norm_n", "mean_norm_g"])
            w.writeheader()
            w.writerows(components_rows)
        print(f"Saved components CSV to {comp_csv_path}")

    # Timing CSV
    timing_csv_path = "logs/al_experiments/value_selection_timing.csv"
    with open(timing_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "strategy", "seed", "total_time_s", "mean_query_time_s"])
        w.writeheader()
        w.writerows(timing_rows)
    print(f"Saved timing CSV to {timing_csv_path}")

    return all_run_results

if __name__ == "__main__":
    run_value_screening(seeds=[42], output_prefix="screening")
