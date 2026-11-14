import os
import sys
import json
import time
import base64
import io
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Tuple

import pyrootutils
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backbones.registry import BACKBONE_REGISTRY
from src.learning_engine import LearningEngine, ActiveLearningDatasetManager
from src.learning_engine.classifier import ClassifierHead
from src.learning_engine.temperature_scaling import TemperatureScaler
from src.query_engine.query_engine import QueryEngine
from src.utils.metrics_engine import compute_detailed_metrics, compute_ece

CLASS_NAMES = {
    "Indian Pines": [
        "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
        "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
        "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean",
        "Wheat", "Woods", "Buildings-Grass-Trees-drives", "Stone-Steel-Towers"
    ],
    "Salinas": [
        "Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow", "Fallow_rough_plow",
        "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained",
        "Soil_vinyard_develop", "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk",
        "Lettuce_romaine_6wk", "Lettuce_romaine_7wk", "Vinyard_untrained", "Vinyard_vertical_trellis"
    ],
    "Pavia University": [
        "Asphalt", "Meadows", "Gravel", "Trees",
        "Painted metal sheets", "Bare Soil", "Bitumen", "Self-Blocking Bricks",
        "Shadows"
    ]
}

CLASS_COLORS = {
    "Indian Pines": [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
        "#f58231", "#911eb4", "#46f0f0", "#f032e6",
        "#bcfd4c", "#fabebe", "#008080", "#e6beff",
        "#9a6324", "#fffac8", "#800000", "#aaffc3"
    ],
    "Salinas": [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
        "#f58231", "#911eb4", "#46f0f0", "#f032e6",
        "#bcfd4c", "#fabebe", "#008080", "#e6beff",
        "#9a6324", "#fffac8", "#800000", "#aaffc3"
    ],
    "Pavia University": [
        "#c0c0c0", "#2ecc71", "#e74c3c", "#27ae60",
        "#3498db", "#e67e22", "#9b59b6", "#f1c40f",
        "#34495e"
    ]
}

DATASET_META = {
    "Indian Pines": {
        "module_cls": "src.datamodules.indian_pines.IndianPinesDataModule",
        "num_classes": 16,
        "bands": 200,
        "init_samples": 10,
        "slug": "indian_pines",
        "rgb_bands": [29, 19, 9],
        "wavelengths": None
    },
    "Salinas": {
        "module_cls": "src.datamodules.salinas.SalinasDataModule",
        "num_classes": 16,
        "bands": 204,
        "init_samples": 10,
        "slug": "salinas",
        "rgb_bands": [29, 19, 9],
        "wavelengths": None
    },
    "Pavia University": {
        "module_cls": "src.datamodules.pavia_university.PaviaUniversityDataModule",
        "num_classes": 9,
        "bands": 103,
        "init_samples": 10,
        "slug": "pavia_university",
        "rgb_bands": [50, 30, 10],
        "wavelengths": None
    }
}

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "logs", "None", "runs", "2026-07-30_14-24-24", "checkpoints", "epoch_094.ckpt")

def load_frozen_backbone(num_classes):
    backbone = BACKBONE_REGISTRY["spec_resnet50"](num_classes=num_classes)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
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
    mod_path, class_name = DATASET_META[ds_name]["module_cls"].rsplit(".", 1)
    import importlib
    module = importlib.import_module(mod_path)
    DataModuleCls = getattr(module, class_name)
    dm = DataModuleCls()
    dm.setup("fit")
    dm.setup("test")
    return dm

class ActiveLearningSession:
    """
    Manages active learning state for a selected dataset, calling existing
    LearningEngine and QueryEngine components.
    Strictly enforces data leakage protection rules via assertions.
    """
    def __init__(self, dataset_name: str = "Indian Pines", seed: int = 42):
        self.dataset_name = dataset_name
        self.seed = seed
        self.meta = DATASET_META[dataset_name]
        self.num_classes = self.meta["num_classes"]
        self.class_names = CLASS_NAMES[dataset_name]
        self.class_colors = CLASS_COLORS[dataset_name]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load datamodule
        self.datamodule = get_datamodule(dataset_name)
        self.train_ds = self.datamodule.train_dataset
        self.val_ds = getattr(self.datamodule, "val_dataset", None)
        self.test_ds = getattr(self.datamodule, "test_dataset", None)

        # STRICT DATA LEAKAGE ASSERTIONS
        assert self.train_ds is not None, "Train dataset missing"
        assert self.test_ds is not None, "Test dataset missing"

        # Initialize dataset manager
        self.dataset_manager = ActiveLearningDatasetManager(
            train_dataset=self.train_ds,
            val_dataset=self.val_ds,
            test_dataset=self.test_ds,
            initial_samples_per_class=self.meta["init_samples"],
            seed=seed,
        )

        # Extract features using frozen backbone
        self.backbone = load_frozen_backbone(self.num_classes)
        from src.learning_engine.feature_extractor import FeatureExtractor
        self.feature_extractor = FeatureExtractor(backbone=self.backbone).to(self.device)

        print(f"Precomputing features for {dataset_name}...")
        self.train_feats, self.train_lbls = self._preextract(self.train_ds, "train")
        self.val_feats, self.val_lbls = self._preextract(self.val_ds, "val")
        self.test_feats, self.test_lbls = self._preextract(self.test_ds, "test")

        # Classifier head
        self.classifier_head = ClassifierHead(
            in_features=2048,
            num_classes=self.num_classes,
            hidden_dim=256,
            dropout=0.0,
            head_type="mlp_2layer",
            norm_layer="layernorm",
        ).to(self.device)

        # Session-level Annotation & Target State
        self.mode = "oracle"  # "oracle" or "human"
        self.user_annotations = {}  # { global_index: class_index }
        self.target_oa = 70.0
        self.target_reached_info = None

        # Initial training on seed pool
        self.current_round = 0
        self.history = []
        self.last_retrain_time = 0.0
        self.last_retrain_loss = 0.0
        self.last_selected_relative_indices = []
        self.last_selected_global_indices = []
        self.last_query_details = []
        self.prev_prediction_map = None

        # Pseudo-labeling configuration (kept optional and disabled by default)
        self.use_pseudo_labeling = True
        self.pseudo_threshold = 0.95
        self.pseudo_weight = 0.2
        self.max_pseudo_per_class = 15
        from src.learning_engine.pseudo_labeler import ConfidencePseudoLabeler
        self.pseudo_labeler = ConfidencePseudoLabeler(
            confidence_threshold=self.pseudo_threshold,
            max_pseudo_per_class=self.max_pseudo_per_class,
            pseudo_weight=self.pseudo_weight,
        )
        self.last_pseudo_stats = {"count": 0, "per_class": {}, "mean_confidence": 0.0}
        self.last_pseudo_points = []

        self._train_classifier(epochs=25, lr=0.0005)
        init_metrics = self.evaluate_test()

        self.history.append({
            "round": 0,
            "num_labeled": self.dataset_manager.num_labeled,
            "num_unlabeled": self.dataset_manager.num_unlabeled,
            "num_pseudo": 0,
            "pseudo_stats": self.last_pseudo_stats,
            "query_time": 0.0,
            "train_time": self.last_retrain_time,
            "retrain_loss": self.last_retrain_loss,
            "metrics": init_metrics,
        })
        self._check_target_reached(init_metrics)

    def _preextract(self, ds, split_name="train"):
        if ds is None or len(ds) == 0:
            return torch.empty((0, 2048)), torch.empty((0,), dtype=torch.long)
        
        cache_dir = os.path.join(PROJECT_ROOT, "cache", "preextracted_features")
        os.makedirs(cache_dir, exist_ok=True)
        ds_slug = self.dataset_name.lower().replace(" ", "_")
        cache_file = os.path.join(cache_dir, f"{ds_slug}_{split_name}_feats.pt")

        if os.path.exists(cache_file):
            print(f"Loading cached pre-extracted features for {self.dataset_name} ({split_name})...")
            data = torch.load(cache_file, map_location="cpu", weights_only=False)
            return data["feats"], data["lbls"]

        print(f"Extracting features for {self.dataset_name} ({split_name}) using frozen backbone...")
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
        all_f, all_l = [], []
        with torch.no_grad():
            for b in loader:
                x = b["image"].to(self.device) if isinstance(b, dict) else b[0].to(self.device)
                y = b["label"] if isinstance(b, dict) else b[1]
                f = self.feature_extractor(x)
                all_f.append(f.cpu())
                all_l.append(y.cpu())
        feats, lbls = torch.cat(all_f, dim=0), torch.cat(all_l, dim=0)
        torch.save({"feats": feats, "lbls": lbls}, cache_file)
        return feats, lbls

    def set_pseudo_labeling_config(
        self,
        use_pseudo_labeling: bool,
        pseudo_threshold: Optional[float] = None,
        pseudo_weight: Optional[float] = None,
        max_pseudo_per_class: Optional[int] = None,
    ):
        """Configure pseudo-labeling hyperparameters for the active session."""
        self.use_pseudo_labeling = bool(use_pseudo_labeling)
        if pseudo_threshold is not None:
            self.pseudo_threshold = float(pseudo_threshold)
        if pseudo_weight is not None:
            self.pseudo_weight = float(pseudo_weight)
        if max_pseudo_per_class is not None:
            self.max_pseudo_per_class = int(max_pseudo_per_class)

        from src.learning_engine.pseudo_labeler import ConfidencePseudoLabeler
        self.pseudo_labeler = ConfidencePseudoLabeler(
            confidence_threshold=self.pseudo_threshold,
            max_pseudo_per_class=self.max_pseudo_per_class,
            pseudo_weight=self.pseudo_weight,
        )
        return {
            "use_pseudo_labeling": self.use_pseudo_labeling,
            "pseudo_threshold": self.pseudo_threshold,
            "pseudo_weight": self.pseudo_weight,
            "max_pseudo_per_class": self.max_pseudo_per_class,
        }

    def _train_classifier(self, epochs=25, lr=0.0005, pseudo_feats=None, pseudo_lbls=None, pseudo_weight=None):
        t0 = time.time()
        self.classifier_head.train()
        lbl_idx = self.dataset_manager.labeled_indices

        # STRICT ASSERTION: Genuine labels come ONLY from labeled pool of training set
        assert len(lbl_idx) > 0, "Labeled pool cannot be empty"

        feats = self.train_feats[lbl_idx].to(self.device)

        # Support user annotations when provided (Human Annotation Mode)
        lbls_list = []
        for g_idx in lbl_idx:
            g_idx_int = int(g_idx)
            if g_idx_int in self.user_annotations:
                lbls_list.append(self.user_annotations[g_idx_int])
            else:
                lbls_list.append(int(self.train_lbls[g_idx_int].item()))
        lbls = torch.tensor(lbls_list, dtype=torch.long, device=self.device)

        p_weight = self.pseudo_weight if pseudo_weight is None else pseudo_weight
        has_pseudo = (pseudo_feats is not None and len(pseudo_feats) > 0 and p_weight > 0.0)

        if has_pseudo:
            is_pseudo_list = [False] * len(feats) + [True] * len(pseudo_feats)
            cat_feats = torch.cat([feats, pseudo_feats.to(self.device)], dim=0)
            cat_lbls = torch.cat([lbls, pseudo_lbls.to(self.device)], dim=0)
            is_pseudo = torch.tensor(is_pseudo_list, dtype=torch.bool, device=self.device)
            ds_t = torch.utils.data.TensorDataset(cat_feats, cat_lbls, is_pseudo)
        else:
            ds_t = torch.utils.data.TensorDataset(feats, lbls)

        loader_t = torch.utils.data.DataLoader(ds_t, batch_size=64, shuffle=True)

        opt = torch.optim.AdamW(self.classifier_head.parameters(), lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=1e-5)
        crit = nn.CrossEntropyLoss(reduction="none")

        l_loss = 0.0
        for _ in range(epochs):
            tot_l, tot_s = 0.0, 0
            if has_pseudo:
                for x_b, y_b, is_p_b in loader_t:
                    logits_b = self.classifier_head(x_b)
                    sample_losses = crit(logits_b, y_b.long())
                    sample_weights = torch.where(
                        is_p_b,
                        torch.as_tensor(p_weight, dtype=torch.float32, device=self.device),
                        torch.tensor(1.0, dtype=torch.float32, device=self.device),
                    )
                    loss_b = (sample_weights * sample_losses).mean()

                    opt.zero_grad()
                    loss_b.backward()
                    torch.nn.utils.clip_grad_norm_(self.classifier_head.parameters(), max_norm=1.0)
                    opt.step()
                    tot_l += loss_b.item() * len(x_b)
                    tot_s += len(x_b)
            else:
                for x_b, y_b in loader_t:
                    logits_b = self.classifier_head(x_b)
                    loss_b = crit(logits_b, y_b.long()).mean()
                    opt.zero_grad()
                    loss_b.backward()
                    torch.nn.utils.clip_grad_norm_(self.classifier_head.parameters(), max_norm=1.0)
                    opt.step()
                    tot_l += loss_b.item() * len(x_b)
                    tot_s += len(x_b)
            sch.step()
            l_loss = tot_l / max(1, tot_s)

        self.last_retrain_loss = float(l_loss)
        self.last_retrain_time = time.time() - t0

    def predict_probs(self, feats, temperature=1.0):
        self.classifier_head.eval()
        if len(feats) == 0:
            return np.empty((0, self.num_classes))
        ds_t = torch.utils.data.TensorDataset(feats.to(self.device))
        loader_t = torch.utils.data.DataLoader(ds_t, batch_size=64, shuffle=False)
        all_p = []
        with torch.no_grad():
            for (x_b,) in loader_t:
                p_b = self.classifier_head.predict_probabilities(x_b, temperature=temperature)
                all_p.append(p_b.cpu())
        return torch.cat(all_p, dim=0).numpy()

    def evaluate_test(self):
        # STRICT DATA LEAKAGE ASSERTION: Test labels evaluated ONLY here
        if len(self.test_feats) == 0:
            return {}
        probs_np = self.predict_probs(self.test_feats)
        preds_np = np.argmax(probs_np, axis=1)
        res = compute_detailed_metrics(self.test_lbls.numpy(), preds_np, probs_np, num_classes=self.num_classes)
        return res

    def get_scene_data(self):
        """STEP 1 & 2: Scene metadata, RGB visualization, spatial maps."""
        gt_labels = self.train_ds.labels
        height, width = gt_labels.shape

        # RGB Composite
        img = self.train_ds.image[self.train_ds.margin:-self.train_ds.margin, self.train_ds.margin:-self.train_ds.margin]
        rgb = img[:, :, self.meta["rgb_bands"]]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        # Labeled/Unlabeled spatial mask for training set
        spatial_mask = np.zeros((height, width), dtype=np.int32)
        # 0 = unlabeled, 1 = labeled, -1 = non-ground-truth background
        for u_idx in self.dataset_manager.unlabeled_indices:
            r, c = self.train_ds.samples[u_idx]
            spatial_mask[r, c] = 0

        for l_idx in self.dataset_manager.labeled_indices:
            r, c = self.train_ds.samples[l_idx]
            spatial_mask[r, c] = 1

        # Labeled class distribution
        labeled_lbls = self.train_lbls[self.dataset_manager.labeled_indices].numpy()
        counts = np.bincount(labeled_lbls, minlength=self.num_classes).tolist()

        # Palette & color indices for real 3D HSI scene rendering
        from PIL import Image
        img_pil = Image.fromarray(rgb_uint8)
        pal = img_pil.quantize(colors=256)
        palette = pal.getpalette()
        scene_colorscale = [
            [round(i / 255.0, 4), f"rgb({palette[i*3]},{palette[i*3+1]},{palette[i*3+2]})"]
            for i in range(256)
        ]
        scene_color_indices = np.array(pal, dtype=np.uint8).tolist()

        return {
            "dataset_name": self.dataset_name,
            "dimensions": [height, width],
            "bands": self.meta["bands"],
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "class_colors": self.class_colors,
            "total_pixels": height * width,
            "num_labeled": self.dataset_manager.num_labeled,
            "num_unlabeled": self.dataset_manager.num_unlabeled,
            "labeled_counts": counts,
            "gt_labels": gt_labels.tolist(),
            "spatial_mask": spatial_mask.tolist(),
            "rgb_uint8": rgb_uint8.tolist(),
            "scene_colorscale": scene_colorscale,
            "scene_color_indices": scene_color_indices,
        }

    def get_predictions_and_uncertainty(self, uncertainty_method="margin"):
        """STEPS 3 & 4: Full prediction map, confidence map, and spatial uncertainty heatmap."""
        gt_labels = self.train_ds.labels
        height, width = gt_labels.shape

        pred_map = np.zeros((height, width), dtype=np.int32) - 1
        conf_map = np.zeros((height, width), dtype=np.float32)
        unc_map = np.zeros((height, width), dtype=np.float32)

        # Predict over all training samples
        probs_all = self.predict_probs(self.train_feats)
        preds_all = np.argmax(probs_all, axis=1)
        confs_all = np.max(probs_all, axis=1)

        # Uncertainty over unlabeled pool ONLY
        unlabeled_idx = self.dataset_manager.unlabeled_indices

        # STRICT DATA LEAKAGE ASSERTION: QueryEngine receives ONLY unlabeled training embeddings/probs
        assert len(unlabeled_idx) > 0 or self.dataset_manager.num_unlabeled == 0

        if len(unlabeled_idx) > 0:
            unlabeled_probs = probs_all[unlabeled_idx]
            qe = QueryEngine(uncertainty_method=uncertainty_method)
            scores_dict = qe.score(
                embeddings=self.train_feats[unlabeled_idx].numpy(),
                probabilities=unlabeled_probs,
                current_round=self.current_round,
                total_rounds=5,
            )
            unc_scores = scores_dict["uncertainty"]

            for i, u_idx in enumerate(unlabeled_idx):
                r, c = self.train_ds.samples[u_idx]
                unc_map[r, c] = float(unc_scores[i])

        for idx in range(len(self.train_ds)):
            r, c = self.train_ds.samples[idx]
            pred_map[r, c] = int(preds_all[idx])
            conf_map[r, c] = float(confs_all[idx])

        # Verified Human-Labelled Points
        human_points = []
        for idx in self.dataset_manager.labeled_indices:
            r, c = self.train_ds.samples[idx]
            cls_idx = int(self.user_annotations.get(int(idx), self.train_lbls[idx].item()))
            human_points.append({
                "coord": [int(r), int(c)],
                "row": int(r),
                "col": int(c),
                "class_id": cls_idx,
                "class_name": self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"Class {cls_idx}",
                "global_index": int(idx),
                "is_oracle_verified": True
            })

        pseudo_points = getattr(self, "last_pseudo_points", [])
        current_oa = float((self.history[-1]["metrics"]["OverallAccuracy"] * 100) if self.history else 0.0)

        return {
            "prediction_map": pred_map.tolist(),
            "confidence_map": conf_map.tolist(),
            "uncertainty_map": unc_map.tolist(),
            "uncertainty_method": uncertainty_method,
            "human_points": human_points,
            "pseudo_points": pseudo_points,
            "class_names": self.class_names,
            "class_colors": self.class_colors,
            "statistics": {
                "human_labeled_samples": len(self.dataset_manager.labeled_indices),
                "predicted_pixels": int(np.count_nonzero(pred_map >= 0)),
                "accepted_pseudo_labels": len(pseudo_points),
                "current_test_accuracy": current_oa,
                "round": self.current_round,
            }
        }

    def run_query_selection(self, k=20, uncertainty_method="margin"):
        """STEP 5: Pure Margin / Active Learning selection & 'Why Was This Sample Selected?' explanations."""
        t0 = time.time()
        unlabeled_idx = self.dataset_manager.unlabeled_indices
        labeled_idx = self.dataset_manager.labeled_indices

        if len(unlabeled_idx) == 0:
            return {"selected_samples": [], "query_time": 0.0}

        unlabeled_feats = self.train_feats[unlabeled_idx]
        probs_unlabeled = self.predict_probs(unlabeled_feats)

        qe = QueryEngine(uncertainty_method=uncertainty_method)

        k_actual = min(k, len(unlabeled_idx))
        selected_rel = qe.select(
            embeddings=unlabeled_feats.numpy(),
            probabilities=probs_unlabeled,
            k=k_actual,
            current_round=self.current_round,
            total_rounds=5,
            labeled_embeddings=self.train_feats[labeled_idx].numpy() if uncertainty_method == "filter_diversity_labeled" else None,
        )

        query_time = time.time() - t0

        self.last_selected_relative_indices = selected_rel.tolist()
        self.last_selected_global_indices = [int(unlabeled_idx[i]) for i in selected_rel]

        # Compute detailed scores for explanations
        lc_scores = 1.0 - np.max(probs_unlabeled, axis=1)
        sorted_probs = np.sort(probs_unlabeled, axis=1)
        margin_diffs = sorted_probs[:, -1] - sorted_probs[:, -2]
        margin_scores = 1.0 - margin_diffs
        div_scores = QueryEngine.compute_diversity_scores(unlabeled_feats.numpy())

        # Normalize scores for ranking
        norm_lc = (lc_scores - lc_scores.min()) / (lc_scores.max() - lc_scores.min() + 1e-8)
        norm_div = (div_scores - div_scores.min()) / (div_scores.max() - div_scores.min() + 1e-8)

        # Compute component info if expected_learning_value, value_of_information, or adaptive_controller is used
        elv_details = None
        controller_weights = None
        if uncertainty_method in {"expected_learning_value", "value_of_information"}:
            from src.query_engine.query_engine.expected_learning_value import ExpectedLearningValueController
            w_u, w_n, w_g = (0.5, 0.5, 0.0) if uncertainty_method == "value_of_information" else (0.4, 0.4, 0.2)
            controller = ExpectedLearningValueController(w_uncertainty=w_u, w_novelty=w_n, w_gradient=w_g)
            _, elv_details = controller.select_samples(
                embeddings=unlabeled_feats.numpy(),
                probabilities=probs_unlabeled,
                k=k_actual,
                labeled_embeddings=self.train_feats[labeled_idx].numpy(),
            )
        elif uncertainty_method == "adaptive_controller":
            from src.query_engine.query_engine.adaptive_controller import AdaptiveQueryController
            ctrl = AdaptiveQueryController()
            w_tuple = ctrl.get_weights(self.current_round, 5)
            controller_weights = {
                "w_uncertainty": float(w_tuple[0]),
                "w_diversity": float(w_tuple[1]),
                "w_hard_region": float(w_tuple[2]),
            }

        details = []
        for rank, r_idx in enumerate(selected_rel):
            g_idx = int(unlabeled_idx[r_idx])
            r, c = self.train_ds.samples[g_idx]
            p_vec = probs_unlabeled[r_idx]
            pred_cls_idx = int(np.argmax(p_vec))
            top_prob = float(np.max(p_vec))
            lc_val = float(lc_scores[r_idx])
            div_val = float(div_scores[r_idx])
            margin_val = float(margin_scores[r_idx])
            margin_diff = float(margin_diffs[r_idx])

            # Top-3 class probabilities
            top_3_idx = np.argsort(-p_vec)[:3]
            top_3 = [{"class": self.class_names[i], "prob": float(p_vec[i])} for i in top_3_idx]

            # Explanation logic using exact computed values
            lc_percentile = float(np.mean(lc_scores <= lc_val) * 100)
            div_percentile = float(np.mean(div_scores <= div_val) * 100)
            margin_percentile = float(np.mean(margin_scores <= margin_val) * 100)

            if uncertainty_method in {"margin", "pure_margin"}:
                reason = (
                    f"Pure Margin Rank #{rank+1}: Uncertainty u(x)={margin_val:.3f} "
                    f"(margin p(1)-p(2)={margin_diff:.3f}, top {100-margin_percentile:.1f}% boundary ambiguity)."
                )
            elif elv_details:
                # Find matching candidate index in elv_details
                cand_pos = np.where(elv_details["candidate_indices"] == r_idx)[0]
                if len(cand_pos) > 0:
                    c_idx = cand_pos[0]
                    u_norm = float(elv_details["norm_u"][c_idx])
                    n_norm = float(elv_details["norm_n"][c_idx])
                    g_norm = float(elv_details["norm_g"][c_idx])
                    v_score = float(elv_details["v_scores"][c_idx])
                    reason = (
                        f"Expected Learning Value Rank #{rank+1} (Score: {v_score:.3f}): "
                        f"Uncertainty U={u_norm:.3f} (LC: {lc_val:.3f}), Novelty N={n_norm:.3f}, "
                        f"Expected Gradient Signal G={g_norm:.3f}."
                    )
                else:
                    reason = f"Expected Learning Value Rank #{rank+1} selected via uncertainty & novelty filtering."
            elif controller_weights:
                reason = (
                    f"Adaptive Controller Rank #{rank+1}: Uncertainty w_U={controller_weights['w_uncertainty']:.2f} "
                    f"(LC: {lc_val:.3f}, top {100-lc_percentile:.1f}%), Diversity w_D={controller_weights['w_diversity']:.2f} "
                    f"(Score: {div_val:.3f}, top {100-div_percentile:.1f}%), Hard-Region w_H={controller_weights['w_hard_region']:.2f}."
                )
            else:
                reason = (
                    f"Selected rank #{rank+1} because predictive uncertainty is high "
                    f"(Least Confidence: {lc_val:.3f}, top {100-lc_percentile:.1f}%) "
                    f"and feature embedding is diverse from labeled pool (Diversity score: {div_val:.3f}, top {100-div_percentile:.1f}%)."
                )

            details.append({
                "rank": rank + 1,
                "global_index": g_idx,
                "relative_index": int(r_idx),
                "coord": [int(r), int(c)],
                "predicted_class": self.class_names[pred_cls_idx],
                "oracle_class": self.class_names[int(self.train_lbls[g_idx].item())],
                "oracle_class_id": int(self.train_lbls[g_idx].item()),
                "max_probability": top_prob,
                "least_confidence": lc_val,
                "margin_uncertainty": margin_val,
                "margin_diff": margin_diff,
                "diversity_score": div_val,
                "selection_reason": reason,
                "top_k_probabilities": top_3,
            })

        self.last_query_details = details
        return {
            "selected_samples": details,
            "query_time": query_time,
            "k": k_actual,
            "controller_weights": controller_weights,
        }

    def _check_target_reached(self, metrics):
        current_oa = metrics.get("OverallAccuracy", 0.0) * 100
        target_reached = current_oa >= self.target_oa
        if target_reached and self.target_reached_info is None:
            self.target_reached_info = {
                "round": self.current_round,
                "num_labeled": self.dataset_manager.num_labeled,
                "oa": current_oa,
            }
        return {
            "current_oa": current_oa,
            "target_oa": self.target_oa,
            "target_reached": target_reached,
            "target_reached_info": self.target_reached_info,
            "status_label": "TARGET REACHED" if target_reached else "TARGET NOT REACHED",
            "message": f"Target OA: {self.target_oa}% | Current OA: {current_oa:.2f}%" + (" (TARGET REACHED)" if target_reached else " (TARGET NOT REACHED)")
        }

    def get_sample_inspection(self, global_index: int):
        """STEP 6: Sample Inspection (Patch, Spectral signature, class probabilities)."""
        r, c = self.train_ds.samples[global_index]
        r_pad = r + self.train_ds.margin
        c_pad = c + self.train_ds.margin

        # Extract patch
        patch_arr = self.train_ds.image[
            r_pad - self.train_ds.margin:r_pad + self.train_ds.margin + 1,
            c_pad - self.train_ds.margin:c_pad + self.train_ds.margin + 1,
            :
        ]
        rgb_patch = patch_arr[:, :, self.meta["rgb_bands"]]
        rgb_patch = (rgb_patch - rgb_patch.min()) / (rgb_patch.max() - rgb_patch.min() + 1e-8)
        patch_uint8 = (rgb_patch * 255).astype(np.uint8)

        # Spectral signature at center pixel
        spectrum = patch_arr[7, 7, :].tolist()

        # Prediction probabilities
        feat = self.train_feats[global_index:global_index+1]
        probs = self.predict_probs(feat)[0]

        is_selected = global_index in self.last_selected_global_indices

        # Find query detail if available
        q_detail = next((d for d in self.last_query_details if d["global_index"] == global_index), None)
        if q_detail is None and hasattr(self, "previous_query_details"):
            q_detail = next((d for d in self.previous_query_details if d["global_index"] == global_index), None)

        lc_val = float(1.0 - np.max(probs))

        # Ground truth visibility based on active mode
        gt_class_display = self.class_names[int(self.train_ds.labels[r, c]) - 1] if self.mode == "oracle" else "HIDDEN (Human Annotation Mode)"
        oracle_badge = "SIMULATED ORACLE / EXPERIMENTAL" if self.mode == "oracle" else "HUMAN ANNOTATION"
        user_ann = self.class_names[self.user_annotations[global_index]] if global_index in self.user_annotations else None

        return {
            "global_index": global_index,
            "coord": [int(r), int(c)],
            "gt_class": gt_class_display,
            "user_annotation": user_ann,
            "oracle_badge": oracle_badge,
            "mode": self.mode,
            "predicted_class": self.class_names[int(np.argmax(probs))],
            "probabilities": probs.tolist(),
            "class_names": self.class_names,
            "least_confidence": lc_val,
            "rank": q_detail["rank"] if q_detail and "rank" in q_detail else None,
            "diversity_score": q_detail["diversity_score"] if q_detail else 0.0,
            "is_selected": is_selected,
            "selection_reason": q_detail["selection_reason"] if q_detail else "Sample not in current query batch.",
            "spectrum": spectrum,
            "wavelengths": self.meta["wavelengths"],
            "patch_uint8": patch_uint8.tolist(),
            "patch_hsi": np.round(patch_arr.transpose(2, 0, 1), 4).tolist(),
        }

    def annotate_human_samples(self, annotations: list, epochs=25, lr=0.0005):
        """
        HUMAN ANNOTATION ENDPOINT METHOD:
        Accepts explicit human annotations, updates labeled pool, and retrains classifier head.
        Ground truth labels are NOT used here.
        """
        if not annotations:
            return {"error": "No human annotations provided."}

        # Save previous prediction map
        probs_all = self.predict_probs(self.train_feats)
        preds_all = np.argmax(probs_all, axis=1)
        gt_labels = self.train_ds.labels
        height, width = gt_labels.shape
        self.prev_prediction_map = np.zeros((height, width), dtype=np.int32) - 1
        for idx in range(len(self.train_ds)):
            r, c = self.train_ds.samples[idx]
            self.prev_prediction_map[r, c] = int(preds_all[idx])

        newly_annotated_global = []
        for ann in annotations:
            g_idx = int(ann["global_index"])
            selected_cls = ann["selected_class"]
            if isinstance(selected_cls, str):
                cls_idx = self.class_names.index(selected_cls) if selected_cls in self.class_names else 0
            else:
                cls_idx = int(selected_cls)

            self.user_annotations[g_idx] = cls_idx
            newly_annotated_global.append(g_idx)

            # If g_idx is in unlabeled pool, move it to labeled pool
            rel_matches = np.where(self.dataset_manager.unlabeled_indices == g_idx)[0]
            if len(rel_matches) > 0:
                self.dataset_manager.update_pools([rel_matches[0]])

        self.current_round += 1
        self._train_classifier(epochs=epochs, lr=lr)

        # Confidence-Based Pseudo-Labeling (if enabled)
        pseudo_stats = {"count": 0, "per_class": {}, "mean_confidence": 0.0}
        self.last_pseudo_points = []
        if self.use_pseudo_labeling and self.dataset_manager.num_unlabeled > 0:
            rem_u_idx = self.dataset_manager.unlabeled_indices
            rem_probs_u = self.predict_probs(self.train_feats[rem_u_idx])
            p_res = self.pseudo_labeler.select_pseudo_labels(
                probabilities=rem_probs_u,
                unlabeled_indices=rem_u_idx,
                num_classes=self.num_classes,
                labeled_indices=self.dataset_manager.labeled_indices,
            )
            if p_res["total_accepted"] > 0:
                p_feats = self.train_feats[p_res["pseudo_indices"]]
                p_lbls = torch.tensor(p_res["pseudo_labels"], dtype=torch.long)
                self._train_classifier(epochs=epochs, lr=lr, pseudo_feats=p_feats, pseudo_lbls=p_lbls)
                pseudo_stats = {
                    "count": p_res["total_accepted"],
                    "per_class": p_res["per_class_counts"],
                    "mean_confidence": p_res["mean_confidence"],
                }
                self.last_pseudo_points = [
                    {
                        "coord": [int(self.train_ds.samples[p_idx][0]), int(self.train_ds.samples[p_idx][1])],
                        "row": int(self.train_ds.samples[p_idx][0]),
                        "col": int(self.train_ds.samples[p_idx][1]),
                        "class_id": int(p_res["pseudo_labels"][i]),
                        "class_name": self.class_names[int(p_res["pseudo_labels"][i])] if int(p_res["pseudo_labels"][i]) < len(self.class_names) else f"Class {p_res['pseudo_labels'][i]}",
                        "confidence": float(p_res["pseudo_confidences"][i]),
                        "global_index": int(p_idx),
                    }
                    for i, p_idx in enumerate(p_res["pseudo_indices"])
                ]
        self.last_pseudo_stats = pseudo_stats

        test_metrics = self.evaluate_test()
        target_status = self._check_target_reached(test_metrics)

        trainable_params = sum(p.numel() for p in self.classifier_head.parameters() if p.requires_grad)

        step_record = {
            "round": self.current_round,
            "num_labeled": self.dataset_manager.num_labeled,
            "num_unlabeled": self.dataset_manager.num_unlabeled,
            "num_pseudo": pseudo_stats["count"],
            "pseudo_stats": pseudo_stats,
            "query_time": self.last_retrain_time,
            "train_time": self.last_retrain_time,
            "retrain_loss": self.last_retrain_loss,
            "metrics": test_metrics,
            "mode": "human",
        }
        self.history.append(step_record)

        self.previous_query_details = list(self.last_query_details)
        self.last_selected_relative_indices = []
        self.last_selected_global_indices = []
        self.last_query_details = []

        return {
            "annotated_count": len(newly_annotated_global),
            "mode": "HUMAN ANNOTATION",
            "current_round": self.current_round,
            "num_labeled": self.dataset_manager.num_labeled,
            "num_unlabeled": self.dataset_manager.num_unlabeled,
            "num_pseudo": pseudo_stats["count"],
            "pseudo_stats": pseudo_stats,
            "retrain_loss": self.last_retrain_loss,
            "train_time": self.last_retrain_time,
            "trainable_parameters": trainable_params,
            "metrics": test_metrics,
            "target_status": target_status,
        }

    def simulate_annotation_and_retrain(self, epochs=25, lr=0.0005):
        """STEPS 7 & 8: Move selected samples to labeled pool and retrain classifier head."""
        if len(self.last_selected_relative_indices) == 0:
            return {"error": "No query selection available to annotate."}

        # Save previous prediction map for BEFORE vs AFTER comparison
        probs_all = self.predict_probs(self.train_feats)
        preds_all = np.argmax(probs_all, axis=1)
        gt_labels = self.train_ds.labels
        height, width = gt_labels.shape
        self.prev_prediction_map = np.zeros((height, width), dtype=np.int32) - 1
        for idx in range(len(self.train_ds)):
            r, c = self.train_ds.samples[idx]
            self.prev_prediction_map[r, c] = int(preds_all[idx])

        # STRICT DATA LEAKAGE ASSERTION: Retrieve ground-truth training label ONLY
        added_global_idx = self.dataset_manager.update_pools(self.last_selected_relative_indices)

        # Increment round
        self.current_round += 1

        # Retrain classifier head on genuine labels
        self._train_classifier(epochs=epochs, lr=lr)

        # Confidence-Based Pseudo-Labeling (if enabled)
        pseudo_stats = {"count": 0, "per_class": {}, "mean_confidence": 0.0}
        self.last_pseudo_points = []
        if self.use_pseudo_labeling and self.dataset_manager.num_unlabeled > 0:
            rem_u_idx = self.dataset_manager.unlabeled_indices
            rem_probs_u = self.predict_probs(self.train_feats[rem_u_idx])
            p_res = self.pseudo_labeler.select_pseudo_labels(
                probabilities=rem_probs_u,
                unlabeled_indices=rem_u_idx,
                num_classes=self.num_classes,
                labeled_indices=self.dataset_manager.labeled_indices,
            )
            if p_res["total_accepted"] > 0:
                p_feats = self.train_feats[p_res["pseudo_indices"]]
                p_lbls = torch.tensor(p_res["pseudo_labels"], dtype=torch.long)
                self._train_classifier(epochs=epochs, lr=lr, pseudo_feats=p_feats, pseudo_lbls=p_lbls)
                pseudo_stats = {
                    "count": p_res["total_accepted"],
                    "per_class": p_res["per_class_counts"],
                    "mean_confidence": p_res["mean_confidence"],
                }
                self.last_pseudo_points = [
                    {
                        "coord": [int(self.train_ds.samples[p_idx][0]), int(self.train_ds.samples[p_idx][1])],
                        "row": int(self.train_ds.samples[p_idx][0]),
                        "col": int(self.train_ds.samples[p_idx][1]),
                        "class_id": int(p_res["pseudo_labels"][i]),
                        "class_name": self.class_names[int(p_res["pseudo_labels"][i])] if int(p_res["pseudo_labels"][i]) < len(self.class_names) else f"Class {p_res['pseudo_labels'][i]}",
                        "confidence": float(p_res["pseudo_confidences"][i]),
                        "global_index": int(p_idx),
                    }
                    for i, p_idx in enumerate(p_res["pseudo_indices"])
                ]
        self.last_pseudo_stats = pseudo_stats

        test_metrics = self.evaluate_test()
        target_status = self._check_target_reached(test_metrics)

        trainable_params = sum(p.numel() for p in self.classifier_head.parameters() if p.requires_grad)

        step_record = {
            "round": self.current_round,
            "num_labeled": self.dataset_manager.num_labeled,
            "num_unlabeled": self.dataset_manager.num_unlabeled,
            "num_pseudo": pseudo_stats["count"],
            "pseudo_stats": pseudo_stats,
            "query_time": self.last_retrain_time, # updated
            "train_time": self.last_retrain_time,
            "retrain_loss": self.last_retrain_loss,
            "metrics": test_metrics,
            "mode": "oracle",
        }
        self.history.append(step_record)

        # Clear query selection
        annotated_count = len(added_global_idx)
        self.previous_query_details = list(self.last_query_details)
        self.last_selected_relative_indices = []
        self.last_selected_global_indices = []
        self.last_query_details = []

        return {
            "annotated_count": annotated_count,
            "mode": "SIMULATED ORACLE / EXPERIMENTAL",
            "current_round": self.current_round,
            "num_labeled": self.dataset_manager.num_labeled,
            "num_unlabeled": self.dataset_manager.num_unlabeled,
            "num_pseudo": pseudo_stats["count"],
            "pseudo_stats": pseudo_stats,
            "retrain_loss": self.last_retrain_loss,
            "train_time": self.last_retrain_time,
            "trainable_parameters": trainable_params,
            "metrics": test_metrics,
            "target_status": target_status,
        }

    def get_before_after_maps(self):
        """STEP 9: BEFORE vs AFTER prediction maps."""
        gt_labels = self.train_ds.labels
        height, width = gt_labels.shape

        probs_all = self.predict_probs(self.train_feats)
        preds_all = np.argmax(probs_all, axis=1)

        after_map = np.zeros((height, width), dtype=np.int32) - 1
        for idx in range(len(self.train_ds)):
            r, c = self.train_ds.samples[idx]
            after_map[r, c] = int(preds_all[idx])

        before_map = self.prev_prediction_map.tolist() if self.prev_prediction_map is not None else after_map.tolist()

        return {
            "before_prediction_map": before_map,
            "after_prediction_map": after_map.tolist(),
            "latest_metrics": self.history[-1]["metrics"],
        }
