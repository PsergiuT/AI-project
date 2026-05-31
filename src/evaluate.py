"""
evaluate.py — Segmentation evaluation metrics for AEA predictions.

Computes per-class and average:
    - Dice Similarity Coefficient (DSC)
    - Intersection over Union (IoU / Jaccard)
    - Hausdorff Distance at 95th percentile (HD95)

All metrics are computed using MONAI's built-in metric classes,
which handle batch processing, device placement, and edge cases
(e.g. empty predictions or ground-truth masks).
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from monai.metrics import (
    DiceMetric,
    MeanIoU,
    HausdorffDistanceMetric,
)
from monai.transforms import AsDiscrete
from monai.utils import set_determinism
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NUM_CLASSES, CLASS_NAMES, EVAL_CONFIG


# ── Metric containers ──────────────────────────────────────────────────────────

class SegmentationMetrics:
    """
    Wrapper around MONAI metric objects for AEA segmentation evaluation.

    Usage:
        metrics = SegmentationMetrics(device="cuda")

        for batch in val_loader:
            pred_logits = model(batch["image"].to(device))
            pred_mask   = metrics.logits_to_onehot(pred_logits)
            gt_mask     = metrics.labels_to_onehot(batch["mask"])
            metrics.update(pred_mask, gt_mask)

        results = metrics.aggregate()
        metrics.reset()
    """

    def __init__(self, device: str = "cpu", voxel_spacing: tuple = (0.4, 0.4, 0.4)):
        """
        Args:
            device:        "cuda" or "cpu".
            voxel_spacing: Physical spacing in mm — used for HD95 computation.
        """
        self.device        = device
        self.voxel_spacing = voxel_spacing
        self.n_classes     = NUM_CLASSES

        # MONAI metric objects — include_background=False excludes class 0
        # (EVAL_CONFIG["include_background"] is False, so we pass False directly)
        self.dice_metric = DiceMetric(
            include_background = EVAL_CONFIG["include_background"],
            reduction          = "mean_batch",
            get_not_nans       = True,
        )
        self.iou_metric = MeanIoU(
            include_background = EVAL_CONFIG["include_background"],
            reduction          = "mean_batch",
            get_not_nans       = True,
        )
        self.hd95_metric = HausdorffDistanceMetric(
            include_background = EVAL_CONFIG["include_background"],
            percentile         = EVAL_CONFIG["percentile"],
            reduction          = "mean_batch",
            get_not_nans       = True,
        )

        # One-hot converters
        # post_pred : logits → argmax class index → one-hot
        # post_label: integer labels (0,1,2) → one-hot directly
        #             NO threshold here — threshold=0.5 would collapse labels 1
        #             and 2 both into channel 1, making AEA Right invisible to metrics
        self.post_pred  = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
        self.post_label = AsDiscrete(to_onehot=NUM_CLASSES)

    # ── Conversion helpers ─────────────────────────────────────────────────────

    def logits_to_onehot(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert raw model output logits to one-hot encoded predictions.

        Args:
            logits: (B, C, H, W, D) tensor of class logits.

        Returns:
            (B, C, H, W, D) one-hot tensor (argmax over C, then one-hot encoded).
        """
        return torch.stack([self.post_pred(p) for p in logits])

    def labels_to_onehot(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Convert integer label mask to one-hot encoding.

        Args:
            labels: (B, 1, H, W, D) tensor with integer class labels.

        Returns:
            (B, C, H, W, D) one-hot tensor.
        """
        return torch.stack([self.post_label(l) for l in labels])

    # ── Accumulation ───────────────────────────────────────────────────────────

    def update(
        self,
        pred_onehot : torch.Tensor,
        gt_onehot   : torch.Tensor,
    ) -> None:
        """
        Add a batch of predictions to the running metric accumulators.

        Args:
            pred_onehot: (B, C, H, W, D) one-hot predictions.
            gt_onehot:   (B, C, H, W, D) one-hot ground truth.
        """
        self.dice_metric(y_pred=pred_onehot, y=gt_onehot)
        self.iou_metric( y_pred=pred_onehot, y=gt_onehot)
        self.hd95_metric(y_pred=pred_onehot, y=gt_onehot)

    def reset(self) -> None:
        """Clear all accumulated metric values (call between epochs)."""
        self.dice_metric.reset()
        self.iou_metric.reset()
        self.hd95_metric.reset()

    # ── Aggregation ────────────────────────────────────────────────────────────

    def aggregate(self) -> dict:
        """
        Compute final metric values from accumulated batches.

        Returns:
            Dict with keys:
                dice_aeal, dice_aear, dice_mean
                iou_aeal,  iou_aear,  iou_mean
                hd95_aeal, hd95_aear, hd95_mean
        """
        # MONAI returns (metric_value, not_nans_count) tuples
        dice_vals, _ = self.dice_metric.aggregate()   # shape: (C-1,) if no bg
        iou_vals,  _ = self.iou_metric.aggregate()
        hd95_vals, _ = self.hd95_metric.aggregate()

        # Convert to Python floats; handle NaN from empty masks
        def safe_float(t, idx):
            val = t[idx].item() if t.numel() > idx else float("nan")
            return round(val, 4) if not np.isnan(val) else float("nan")

        results = {
            # Dice per class (indices 0, 1 → AEAL, AEAR when bg excluded)
            "dice_aeal" : safe_float(dice_vals, 0),
            "dice_aear" : safe_float(dice_vals, 1),
            "dice_mean" : round(dice_vals.nanmean().item(), 4),

            # IoU per class
            "iou_aeal"  : safe_float(iou_vals, 0),
            "iou_aear"  : safe_float(iou_vals, 1),
            "iou_mean"  : round(iou_vals.nanmean().item(), 4),

            # HD95 per class (in mm)
            "hd95_aeal" : safe_float(hd95_vals, 0),
            "hd95_aear" : safe_float(hd95_vals, 1),
            "hd95_mean" : round(hd95_vals.nanmean().item(), 4),
        }

        return results

    def log_results(self, results: dict, prefix: str = "") -> None:
        """Pretty-print metric results to the logger."""
        tag = f"[{prefix}] " if prefix else ""
        logger.info(f"{tag}{'─' * 45}")
        logger.info(f"{tag}{'Metric':<20} {'AEA Left':>10} {'AEA Right':>10} {'Mean':>10}")
        logger.info(f"{tag}{'─' * 45}")
        logger.info(
            f"{tag}{'Dice (DSC)':<20} "
            f"{results['dice_aeal']:>10.4f} "
            f"{results['dice_aear']:>10.4f} "
            f"{results['dice_mean']:>10.4f}"
        )
        logger.info(
            f"{tag}{'IoU (Jaccard)':<20} "
            f"{results['iou_aeal']:>10.4f} "
            f"{results['iou_aear']:>10.4f} "
            f"{results['iou_mean']:>10.4f}"
        )
        logger.info(
            f"{tag}{'HD95 (mm)':<20} "
            f"{results['hd95_aeal']:>10.4f} "
            f"{results['hd95_aear']:>10.4f} "
            f"{results['hd95_mean']:>10.4f}"
        )
        logger.info(f"{tag}{'─' * 45}")


# ── Full test-set evaluation ───────────────────────────────────────────────────

def evaluate_model(
    model      : torch.nn.Module,
    dataloader : torch.utils.data.DataLoader,
    device     : str,
    roi_size   : tuple = (96, 96, 96),
    sw_batch   : int = 4,
) -> dict:
    """
    Run full evaluation on a dataloader (val or test split).

    Uses sliding window inference so the full volume is evaluated,
    not just patches. This gives a more realistic performance estimate.

    Args:
        model:      Trained SwinUNETR model (in eval mode).
        dataloader: Val or test DataLoader.
        device:     "cuda" or "cpu".
        roi_size:   Sliding window patch size.
        sw_batch:   Number of patches to process in parallel.

    Returns:
        Dict of aggregated metrics.
    """
    from monai.inferers import sliding_window_inference

    model.eval()
    metrics = SegmentationMetrics(device=device)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)

            # Sliding window inference — processes the full volume
            pred_logits = sliding_window_inference(
                inputs      = images,
                roi_size    = roi_size,
                sw_batch_size = sw_batch,
                predictor   = model,
                overlap     = 0.5,
                mode        = "gaussian",
            )

            pred_onehot = metrics.logits_to_onehot(pred_logits)
            gt_onehot   = metrics.labels_to_onehot(masks)
            metrics.update(pred_onehot, gt_onehot)

    results = metrics.aggregate()
    metrics.log_results(results, prefix="Evaluation")
    return results
