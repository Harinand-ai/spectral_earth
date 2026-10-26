import os
import sys
import json
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import pyrootutils

# Set up project root
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
from src.backbones.registry import BACKBONE_REGISTRY
from src.learning_engine import (
    LearningEngine,
    ActiveLearningDatasetManager,
)
from src.query_engine.query_engine import QueryEngine

log = utils.get_pylogger(__name__)


def run_active_learning(cfg: DictConfig):
    """
    Run active learning pipeline connecting Model A (Learning Engine)
    and Model B (Query Engine).
    """
    # 1. Set seed
    seed = cfg.get("seed")
    if seed is None:
        seed = 42
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    log.info("=" * 60)
    log.info("SPECTRALEARTH PHASE 2 ACTIVE LEARNING FRAMEWORK")
    log.info("=" * 60)

    # 2. Instantiate datamodule
    datamodule_cfg = cfg.get("data", {})
    if not datamodule_cfg or "_target_" not in datamodule_cfg:
        # Default fallback to IndianPines
        log.info("No data config specified. Using IndianPinesDataModule by default.")
        from src.datamodules.indian_pines import IndianPinesDataModule
        datamodule = IndianPinesDataModule()
    else:
        log.info(f"Instantiating DataModule: <{datamodule_cfg._target_}>")
        datamodule = hydra.utils.instantiate(datamodule_cfg)

    datamodule.setup("fit")
    datamodule.setup("test")

    train_dataset = datamodule.train_dataset
    val_dataset = getattr(datamodule, "val_dataset", None)
    test_dataset = getattr(datamodule, "test_dataset", None)

    # Determine num_classes and in_channels
    if hasattr(datamodule, "num_classes"):
        num_classes = datamodule.num_classes
    elif hasattr(train_dataset, "labels"):
        num_classes = len(np.unique(train_dataset.labels[train_dataset.labels > 0])) if np.min(train_dataset.labels) >= 0 else len(np.unique(train_dataset.labels))
    else:
        num_classes = getattr(cfg, "num_classes", 16)

    if hasattr(datamodule, "num_bands"):
        in_channels = datamodule.num_bands
    elif hasattr(train_dataset, "image") and hasattr(train_dataset.image, "shape"):
        in_channels = train_dataset.image.shape[-1] if train_dataset.image.ndim == 3 else 200
    else:
        in_channels = getattr(cfg, "in_channels", 200)

    # 3. Instantiate Backbone (SpectralEarth)
    backbone_name = cfg.get("backbone", "spec_resnet50")
    log.info(f"Instantiating Backbone: {backbone_name}")

    if backbone_name in BACKBONE_REGISTRY:
        model_cls = BACKBONE_REGISTRY[backbone_name]
        if "in_channels" in model_cls.__init__.__code__.co_varnames:
            backbone = model_cls(num_classes=num_classes, in_channels=in_channels)
        else:
            backbone = model_cls(num_classes=num_classes)
    else:
        import timm
        backbone = timm.create_model(backbone_name, num_classes=num_classes, in_chans=in_channels)

    # 4. Instantiate Active Learning Dataset Manager (Model A component)
    al_cfg = cfg.get("al", {})
    initial_labeled_ratio = al_cfg.get("initial_labeled_ratio", 0.05)
    initial_samples_per_class = al_cfg.get("initial_samples_per_class", None)
    
    dataset_manager = ActiveLearningDatasetManager(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        initial_labeled_ratio=initial_labeled_ratio,
        initial_samples_per_class=initial_samples_per_class,
        seed=seed,
    )

    log.info(f"Dataset Initialized: Total Train={len(train_dataset)}, Initial Labeled={dataset_manager.num_labeled}, Initial Unlabeled={dataset_manager.num_unlabeled}")

    # 5. Instantiate Model A (Learning Engine)
    use_pseudo_labeling = al_cfg.get("use_pseudo_labeling", False)
    pseudo_threshold = al_cfg.get("pseudo_threshold", 0.95)
    pseudo_weight = al_cfg.get("pseudo_weight", 0.2)
    max_pseudo_per_class = al_cfg.get("max_pseudo_per_class", 15)

    learning_engine = LearningEngine(
        backbone=backbone,
        num_classes=num_classes,
        dataset_manager=dataset_manager,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_pseudo_labeling=use_pseudo_labeling,
        pseudo_threshold=pseudo_threshold,
        pseudo_weight=pseudo_weight,
        max_pseudo_per_class=max_pseudo_per_class,
    )

    # Load pretrained weights if ckpt_path is provided
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path and os.path.exists(ckpt_path):
        log.info(f"Loading pretrained backbone weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        backbone_state = {}
        head_state = {}
        
        for k, v in state_dict.items():
            if k.startswith("model.resnet.fc."):
                head_state[k[len("model.resnet.fc."):]] = v
            elif k.startswith("model.fc."):
                head_state[k[len("model.fc."):]] = v
            elif k.startswith("model.backbone."):
                backbone_state[k[len("model.backbone."):]] = v
            elif k.startswith("model."):
                backbone_state[k[len("model."):]] = v
            elif k.startswith("feature_extractor.backbone."):
                backbone_state[k[len("feature_extractor.backbone."):]] = v
            else:
                backbone_state[k] = v
                
        missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
        log.info(f"Loaded backbone checkpoint. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

        if head_state:
            try:
                head_obj = learning_engine.classifier_head.head
                if isinstance(head_obj, torch.nn.Linear) and head_obj.weight.shape == head_state["weight"].shape:
                    head_obj.load_state_dict(head_state)
                    log.info("Successfully loaded pretrained classifier head weights from checkpoint!")
            except Exception as e:
                log.info(f"Pretrained head load skipped: {e}")

    # Freeze backbone strictly
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    # 6. Instantiate Model B (Query Engine)
    uncertainty_method = al_cfg.get("uncertainty_method", "entropy")
    start_uncertainty = al_cfg.get("start_uncertainty", 0.30)
    end_uncertainty = al_cfg.get("end_uncertainty", 0.80)

    query_engine = QueryEngine(
        uncertainty_method=uncertainty_method,
        start_uncertainty=start_uncertainty,
        end_uncertainty=end_uncertainty,
    )
    log.info(f"Instantiated Model B (QueryEngine): {query_engine}")

    # 7. Run Active Learning Loop
    num_rounds = al_cfg.get("num_rounds", 5)
    k_samples = al_cfg.get("k_samples_per_round", 20)
    epochs_per_round = al_cfg.get("epochs_per_round", 15)
    lr = al_cfg.get("learning_rate", 1e-3)
    batch_size = al_cfg.get("batch_size", 64)

    log.info(f"Starting Active Learning Loop: Rounds={num_rounds}, k={k_samples}/round, Epochs/round={epochs_per_round}")

    history = learning_engine.run_full_active_learning_loop(
        query_engine=query_engine,
        k=k_samples,
        total_rounds=num_rounds,
        epochs_per_round=epochs_per_round,
        lr=lr,
        batch_size=batch_size,
    )

    # 8. Log and Output Final Metrics
    log.info("\n" + "=" * 60)
    log.info("ACTIVE LEARNING EXPERIMENT SUMMARY")
    log.info("=" * 60)
    
    output_records = []

    for entry in history:
        r = entry["round"]
        n_lab = entry["num_labeled"]
        n_unlab = entry["num_unlabeled"]
        loss = entry["retrain_loss"]
        m = entry.get("metrics", {})
        oa = m.get("OverallAccuracy", 0.0)
        aa = m.get("AverageAccuracy", 0.0)
        f1 = m.get("F1Score", 0.0)
        kappa = m.get("Kappa", 0.0)

        log.info(
            f"Round {r:02d} | Labeled: {n_lab:4d} | Unlabeled: {n_unlab:4d} | "
            f"Loss: {loss:.4f} | OA: {oa*100:6.2f}% | AA: {aa*100:6.2f}% | "
            f"F1: {f1:.4f} | Kappa: {kappa:.4f}"
        )

        record = {
            "round": r,
            "num_labeled": n_lab,
            "num_unlabeled": n_unlab,
            "retrain_loss": float(loss),
            "OverallAccuracy": float(oa),
            "AverageAccuracy": float(aa),
            "F1Score": float(f1),
            "Kappa": float(kappa),
        }
        if "ConfusionMatrix" in m and isinstance(m["ConfusionMatrix"], np.ndarray):
            record["ConfusionMatrix"] = m["ConfusionMatrix"].tolist()
            
        output_records.append(record)

    # Save details to file as requested
    os.makedirs("logs/al_experiments", exist_ok=True)
    out_file = "logs/al_experiments/active_learning_run.json"
    with open(out_file, "w") as f:
        json.dump(output_records, f, indent=2)

    log.info(f"Detailed run metrics saved to {out_file}")
    return history


@hydra.main(version_base="1.3", config_path="configs", config_name="downstream.yaml")
def main(cfg: DictConfig):
    # Ensure default AL config parameters exist if not present in downstream.yaml
    with open_dict(cfg):
        if "al" not in cfg or cfg.al is None:
            cfg.al = OmegaConf.create({
                "num_rounds": 5,
                "k_samples_per_round": 20,
                "initial_labeled_ratio": 0.05,
                "uncertainty_method": "entropy",
                "start_uncertainty": 0.30,
                "end_uncertainty": 0.80,
                "epochs_per_round": 15,
                "batch_size": 64,
                "learning_rate": 0.001,
                "seed": 42,
            })
    run_active_learning(cfg)



if __name__ == "__main__":
    main()
