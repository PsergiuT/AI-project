"""
train.py — SwinUNETR fine-tuning training loop for AEA segmentation.

This script:
    1. Loads SwinUNETR with MONAI pre-trained self-supervised weights.
    2. Sets up DiceCELoss, AdamW optimizer, and cosine annealing scheduler.
    3. Runs a training loop with periodic validation.
    4. Saves the best model checkpoint based on mean validation Dice.
    5. Implements early stopping to prevent wasted compute on Colab.

Run on Google Colab (recommended) or locally with a CUDA GPU:
    python src/train.py

For Colab, use the provided notebook: notebooks/training.ipynb
"""

import gc
import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from loguru import logger

# ── PyTorch 2.6 compatibility patch
# PyTorch 2.6 changed torch.load to default weights_only=True for security.
# MONAI's PersistentDataset calls torch.load internally (no weights_only arg)
# to read its disk cache, which contains MetaTensor + numpy objects that are
# blocked under strict mode. Rather than allowlisting every type one by one,
# we patch torch.load to keep the old default for any call that doesn't
# explicitly set weights_only. Our own explicit calls already pass weights_only=False.
_torch_load_orig = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load_orig(*args, **kwargs)
torch.load = _torch_load_compat

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODEL_CONFIG, TRAIN_CONFIG, INFERENCE_CONFIG,
    SPLITS_DIR, SWINUNETR_DIR, LOGS_DIR, RANDOM_SEED, NUM_CLASSES,
)
from src.dataset import AEADataModule
from src.evaluate import SegmentationMetrics
from src.utils import set_seed, setup_logger, save_json


# ── Pre-trained weights URL

PRETRAINED_WEIGHTS_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/"
    "0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt"
)


# ── Model initialisation

def build_model(device: str, pretrained: bool = True) -> SwinUNETR:
    """
    Build and optionally load pre-trained SwinUNETR weights.

    The pre-trained checkpoint contains encoder weights only
    (from self-supervised masked-volume pre-training). The decoder
    is randomly initialised and learns from our AEA labels.

    Args:
        device:     "cuda" or "cpu".
        pretrained: Whether to load MONAI pre-trained encoder weights.

    Returns:
        SwinUNETR model moved to the specified device.
    """
    # Build SwinUNETR with API compatibility across MONAI versions.
    # MONAI 1.3.x uses 'img_size' as a keyword argument.
    # MONAI 1.4+ changed the signature — img_size must be passed positionally.
    # MONAI 1.4+ removed img_size entirely — in_channels is now the first arg.
    # Try without img_size first (1.4+), fall back to passing it as keyword (1.3.x).
    _kwargs = dict(
        in_channels    = MODEL_CONFIG["in_channels"],
        out_channels   = MODEL_CONFIG["out_channels"],
        feature_size   = MODEL_CONFIG["feature_size"],
        use_checkpoint = MODEL_CONFIG["use_checkpoint"],
    )
    try:
        # MONAI 1.4+: img_size not needed, removed from signature
        model = SwinUNETR(**_kwargs).to(device)
        logger.info("SwinUNETR built without img_size (MONAI 1.4+)")
    except TypeError:
        # MONAI 1.3.x: img_size required as keyword argument
        logger.info("Falling back to img_size keyword (MONAI 1.3.x)")
        model = SwinUNETR(
            img_size = MODEL_CONFIG["img_size"],
            **_kwargs,
        ).to(device)

    if pretrained:
        weights_path = SWINUNETR_DIR / "pretrained_swinunetr.pt"

        # Download weights if not already cached
        if not weights_path.exists():
            logger.info("Downloading SwinUNETR pre-trained weights …")
            try:
                import urllib.request
                urllib.request.urlretrieve(PRETRAINED_WEIGHTS_URL, str(weights_path))
                logger.info(f"Weights downloaded → {weights_path}")
            except Exception as e:
                logger.warning(f"Could not download pre-trained weights: {e}")
                logger.warning("Training from random initialisation instead.")
                return model

        # Load encoder weights only (decoder stays randomly initialised)
        # weights_only=False needed for PyTorch 2.6+ — MONAI checkpoint contains
        # numpy scalars which are not allowed under the default strict mode.
        # Safe to use here: checkpoint sourced from MONAI's official GitHub release.
        checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)

        # The MONAI checkpoint wraps weights under a "state_dict" key
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Filter to encoder-only keys (swinViT prefix)
        encoder_state = {
            k.replace("swinViT.", ""): v
            for k, v in state_dict.items()
            if k.startswith("swinViT.")
        }

        missing, unexpected = model.swinViT.load_state_dict(
            encoder_state, strict=False
        )
        logger.info(
            f"Pre-trained encoder loaded. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable:,} trainable")
    return model


# ── Learning rate scheduler with warmup ───────────────────────────────────────

class WarmupCosineScheduler:
    """
    Linear warmup for the first `warmup_epochs` epochs, then cosine annealing.

    This prevents large gradient updates in the early epochs when the
    randomly initialised decoder interacts with the pre-trained encoder.
    """

    def __init__(
        self,
        optimizer     : torch.optim.Optimizer,
        warmup_epochs : int,
        max_epochs    : int,
        base_lr       : float,
    ):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs    = max_epochs
        self.base_lr       = base_lr
        self.cosine        = CosineAnnealingLR(
            optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-7
        )
        self.current_epoch = 0

    def step(self) -> float:
        """Advance one epoch and return the current learning rate."""
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            self.cosine.step()
            lr = self.optimizer.param_groups[0]["lr"]
        return lr


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    pretrained   : bool = True,
    cache_rate   : float = 1.0,
    num_workers  : int = 4,
    resume_from  : str | None = None,
    persistent   : bool = False,
    finetune     : bool = False,
) -> None:
    """
    Main training function.

    Args:
        pretrained:  Whether to load MONAI pre-trained encoder weights.
                     Ignored when finetune=True (weights come from resume_from).
        cache_rate:  Fraction of data to cache in RAM (ignored when persistent=True).
        num_workers: DataLoader worker processes (set 0 on Windows / Colab).
        resume_from: Path to a checkpoint to resume training from.
        persistent:  Use PersistentDataset (disk cache) instead of RAM cache.
                     Recommended on Colab where RAM < 16GB. First run is slower
                     while the cache is built; all subsequent epochs are fast.
        finetune:    Fine-tune mode — loads only the model weights from resume_from
                     (NOT the optimizer state), resets the optimizer with a lower
                     learning rate (finetune_lr from TRAIN_CONFIG), and disables
                     the warmup phase.  Use this to continue from a trained
                     swinunetr_best.pth with improved loss / augmentations without
                     disrupting the already-learned feature representations.
    """
    set_seed(RANDOM_SEED)
    setup_logger(LOGS_DIR)

    # ── Device setup ───────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # ── Data ───────────────────────────────────────────────────────────────────
    # PersistentDataset writes processed tensors to /tmp/aea_cache (Colab SSD).
    # This avoids loading large CBCT volumes into RAM on every epoch.
    cache_dir = Path("/tmp/aea_cache") if persistent else None
    if persistent:
        logger.info(f"Using PersistentDataset — disk cache at {cache_dir}")
    else:
        logger.info(f"Using CacheDataset — RAM cache rate: {cache_rate}")

    dm = AEADataModule(
        splits_dir  = SPLITS_DIR,
        cache_dir   = cache_dir,
        cache_rate  = cache_rate,
        num_workers = num_workers,
    )
    dm.setup()

    # ── Model ──────────────────────────────────────────────────────────────────
    # In fine-tune mode the model is loaded from resume_from, so there is no
    # point downloading the MONAI self-supervised weights again.
    model = build_model(device, pretrained=(pretrained and not finetune))

    # ── Loss function ──────────────────────────────────────────────────────────
    # Class weights for the cross-entropy component:
    #   background (0) → 0.1  — dominant class, downweight to prevent the network
    #                            from ignoring the rare AEA voxels
    #   AEA Left   (1) → 1.0
    #   AEA Right  (2) → 1.0
    # The Dice component is naturally balanced because it operates per-class.
    # Aggressive downweighting of background forces the network to focus
    # almost entirely on the rare AEA voxels during CE loss computation.
    ce_weights = torch.tensor([0.1, 1.5, 1.5], dtype=torch.float32, device=device)
    loss_fn = DiceCELoss(
        to_onehot_y   = True,    # Convert integer labels to one-hot internally
        softmax       = True,    # Apply softmax to logits before loss
        lambda_dice   = TRAIN_CONFIG["dice_weight"],
        lambda_ce     = TRAIN_CONFIG["ce_weight"],
        weight        = ce_weights,  # CE class weights (MONAI 1.4 uses 'weight')
    )

    # ── Optimiser & scheduler ──────────────────────────────────────────────────
    # Fine-tune mode uses a much smaller LR so we don't undo already-learned
    # representations with large gradient steps.
    base_lr = TRAIN_CONFIG["finetune_lr"] if finetune else TRAIN_CONFIG["learning_rate"]

    optimizer = AdamW(
        model.parameters(),
        lr           = base_lr,
        weight_decay = TRAIN_CONFIG["weight_decay"],
    )

    # Fine-tune mode skips the linear warmup (model is already trained).
    warmup_epochs = 0 if finetune else TRAIN_CONFIG["warmup_epochs"]
    scheduler = WarmupCosineScheduler(
        optimizer     = optimizer,
        warmup_epochs = warmup_epochs,
        max_epochs    = TRAIN_CONFIG["max_epochs"],
        base_lr       = base_lr,
    )

    # ── Load / resume checkpoint ───────────────────────────────────────────────
    start_epoch    = 0
    best_dice      = -1.0
    no_improve_cnt = 0

    if resume_from and Path(resume_from).exists():
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])

        if finetune:
            # Fine-tune: restore model weights only.
            # A fresh optimizer at base_lr gives the network a clean slate to
            # adapt to the improved loss / augmentations without momentum
            # artefacts from the previous training run.
            best_dice = checkpoint.get("best_dice", -1.0)
            logger.info(
                f"Fine-tune mode: loaded model weights from {resume_from} "
                f"(prior best Dice: {best_dice:.4f})"
            )
            logger.info(
                f"Optimizer reset — LR: {base_lr:.1e}, warmup: {warmup_epochs} epochs"
            )
            # Reset best_dice so the fine-tuned model must beat it to save a checkpoint
            # (keeps the original best.pth safe until fine-tuning actually improves it)
            best_dice = best_dice   # keep as floor — fine-tune must beat this
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_dice   = checkpoint.get("best_dice", -1.0)
            logger.info(f"Resumed from {resume_from} (epoch {start_epoch})")

    # ── AMP scaler — halves VRAM by computing in float16 ──────────────────────
    use_amp = (device == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    logger.info(f"Automatic Mixed Precision (AMP): {'enabled' if use_amp else 'disabled'}")

    # ── Metric tracker ─────────────────────────────────────────────────────────
    val_metrics = SegmentationMetrics(device=device)
    history     = {"train_loss": [], "val_dice": [], "val_iou": [], "val_hd95": []}

    logger.info("=" * 60)
    logger.info("Starting training" + (" (FINE-TUNE mode)" if finetune else ""))
    logger.info(f"  Max epochs   : {TRAIN_CONFIG['max_epochs']}")
    logger.info(f"  Batch size   : {TRAIN_CONFIG['batch_size']}")
    logger.info(f"  Learning rate: {base_lr:.1e}")
    logger.info(f"  Warmup epochs: {warmup_epochs}")
    logger.info(f"  Patience     : {TRAIN_CONFIG['patience']} epochs")
    logger.info(f"  CE class wts : [0.1, 1.0, 1.0] (bg downweighted)")
    logger.info("=" * 60)

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, TRAIN_CONFIG["max_epochs"]):
        epoch_start = time.time()
        model.train()
        epoch_loss  = 0.0
        num_batches = 0

        # ── Training step ──────────────────────────────────────────────────────
        for batch in dm.train_loader:
            images = batch["image"].to(device)   # (B, 1, H, W, D)
            masks  = batch["mask"].to(device)    # (B, 1, H, W, D) integer labels
            del batch   # release MetaTensor metadata dict immediately

            # Clamp labels to valid range [0, NUM_CLASSES-1].
            # Some NRRD masks may contain unexpected label values (e.g. 255)
            # which cause a CUDA index-out-of-bounds crash inside CE loss.
            masks = masks.long()
            masks = torch.clamp(masks, 0, NUM_CLASSES - 1)

            optimizer.zero_grad()

            # AMP forward pass — computes in float16 to save VRAM
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)           # (B, C, H, W, D)
                loss   = loss_fn(logits, masks)

            # Scaled backward pass — GradScaler handles float16 → float32 safely
            scaler.scale(loss).backward()

            # Gradient clipping prevents exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        lr       = scheduler.step()
        history["train_loss"].append(avg_loss)

        elapsed = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch + 1:>4}/{TRAIN_CONFIG['max_epochs']} | "
            f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | {elapsed:.0f}s"
        )

        # ── Validation step ────────────────────────────────────────────────────
        if (epoch + 1) % TRAIN_CONFIG["val_every"] == 0:
            model.eval()

            val_metrics.reset()
            with torch.no_grad():
                for batch in dm.val_loader:
                    images = batch["image"].to(device)
                    masks  = batch["mask"].to(device)
                    del batch

                    pred_logits = sliding_window_inference(
                        inputs        = images,
                        roi_size      = INFERENCE_CONFIG["roi_size"],
                        sw_batch_size = INFERENCE_CONFIG["sw_batch_size"],
                        predictor     = model,
                        overlap       = INFERENCE_CONFIG["overlap"],
                        mode          = INFERENCE_CONFIG["mode"],
                    )

                    pred_oh = val_metrics.logits_to_onehot(pred_logits)
                    gt_oh   = val_metrics.labels_to_onehot(masks)
                    val_metrics.update(pred_oh, gt_oh)

            results = val_metrics.aggregate()
            val_metrics.reset()   # free metric buffers immediately after aggregation
            val_metrics.log_results(results, prefix=f"Val Epoch {epoch + 1}")

            history["val_dice"].append(results["dice_mean"])
            history["val_iou"].append(results["iou_mean"])
            history["val_hd95"].append(results["hd95_mean"])

            # ── Checkpoint: save best model ────────────────────────────────────
            current_dice = results["dice_mean"]
            if current_dice > best_dice:
                best_dice      = current_dice
                no_improve_cnt = 0
                checkpoint = {
                    "epoch"     : epoch,
                    "model"     : model.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    "best_dice" : best_dice,
                    "metrics"   : results,
                }
                best_path = SWINUNETR_DIR / TRAIN_CONFIG["best_model_name"]
                torch.save(checkpoint, str(best_path))
                logger.info(
                    f"✓ New best model saved (Dice: {best_dice:.4f}) → {best_path}"
                )

                # ── Auto-backup best checkpoint to GCS (if configured) ─────────
                # Set the env var AEA_GCS_BACKUP to a gs:// path before training.
                # Example: os.environ['AEA_GCS_BACKUP'] = 'gs://my-bucket/aea'
                # Colab Enterprise has gsutil pre-installed and authenticated.
                _gcs = os.environ.get("AEA_GCS_BACKUP", "").strip()
                if _gcs:
                    _dst = f"{_gcs.rstrip('/')}/{TRAIN_CONFIG['best_model_name']}"
                    _ret = os.system(f'gsutil -q cp "{best_path}" "{_dst}"')
                    if _ret == 0:
                        logger.info(f"  ↑ Best checkpoint backed up to GCS → {_dst}")
                    else:
                        logger.warning(f"  ✗ GCS backup failed (gsutil exit {_ret})")
            else:
                no_improve_cnt += TRAIN_CONFIG["val_every"]
                logger.info(
                    f"No improvement for {no_improve_cnt} epochs "
                    f"(best Dice: {best_dice:.4f})"
                )

            # ── Early stopping ─────────────────────────────────────────────────
            if no_improve_cnt >= TRAIN_CONFIG["patience"]:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(no improvement for {no_improve_cnt} epochs)."
                )
                break

        # Force Python GC to collect unreferenced tensors/dicts from this epoch
        gc.collect()

        # ── Save last checkpoint every 10 epochs ──────────────────────────────
        if (epoch + 1) % 10 == 0:
            last_path = SWINUNETR_DIR / TRAIN_CONFIG["last_model_name"]
            torch.save({
                "epoch"     : epoch,
                "model"     : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "best_dice" : best_dice,
            }, str(last_path))
            logger.info(f"Checkpoint saved (epoch {epoch + 1}) → {last_path}")

            # ── Auto-backup last checkpoint to GCS every 10 epochs ────────────
            # A crash can never cost more than 10 epochs of work.
            # Resume by downloading swinunetr_last.pth from GCS.
            _gcs = os.environ.get("AEA_GCS_BACKUP", "").strip()
            if _gcs:
                _dst = f"{_gcs.rstrip('/')}/{TRAIN_CONFIG['last_model_name']}"
                _ret = os.system(f'gsutil -q cp "{last_path}" "{_dst}"')
                if _ret == 0:
                    logger.info(f"  ↑ Last checkpoint backed up to GCS → {_dst}")
                else:
                    logger.warning(f"  ✗ GCS backup failed (gsutil exit {_ret})")

    # ── Save training history ──────────────────────────────────────────────────
    save_json(history, LOGS_DIR / "training_history.json")
    logger.info("Training complete.")
    logger.info(f"Best validation Dice: {best_dice:.4f}")
    logger.info(f"Training history saved → {LOGS_DIR / 'training_history.json'}")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SwinUNETR for AEA segmentation")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Train from random weights (not recommended)")
    parser.add_argument("--cache_rate", type=float, default=1.0,
                        help="Fraction of data to cache in RAM (ignored with --persistent)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes (use 0 on Colab / Windows)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--persistent", action="store_true",
                        help="Cache processed tensors to disk instead of RAM. "
                             "Recommended on Colab (avoids RAM OOM). "
                             "First run builds the cache (~5 min), then fast.")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune mode: load model weights only from --resume, "
                             "reset optimizer to finetune_lr (1e-5), no warmup. "
                             "Use with --resume path/to/swinunetr_best.pth to continue "
                             "from a trained checkpoint with improved loss & augmentations.")
    args = parser.parse_args()

    train(
        pretrained  = not args.no_pretrained,
        cache_rate  = args.cache_rate,
        num_workers = args.num_workers,
        resume_from = args.resume,
        persistent  = args.persistent,
        finetune    = args.finetune,
    )
