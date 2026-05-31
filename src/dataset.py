"""
dataset.py — MONAI Dataset and DataLoader setup for AEA segmentation.

This module defines:
  - get_transforms()    : MONAI transform pipelines for train / val / test
  - get_dataloaders()   : Build DataLoaders from JSON split manifests
  - AEADataModule       : Convenience wrapper that loads all three splits at once
"""

import sys
import torch
from pathlib import Path
from typing import Optional

from monai import transforms as T
from monai.data import (
    CacheDataset,
    DataLoader,
    PersistentDataset,
    pad_list_data_collate,
)
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAIN_CONFIG, MODEL_CONFIG, HU_MIN, HU_MAX, NUM_CLASSES
from src.utils import load_json


# ── Custom transforms ──────────────────────────────────────────────────────────

class ClampMaskLabeld(T.MapTransform):
    """
    Clamp mask label values to the valid range [0, num_classes - 1].

    Some NRRD segmentation files contain unexpected label values (e.g. 255)
    caused by software defaults or file-format quirks.  When such values
    reach RandCropByPosNegLabeld, the sampler treats them as background
    (because they are ≥ num_classes) and reports "Num foregrounds 0" on
    every epoch, completely defeating foreground oversampling.

    Applying this transform immediately before the crop sampler ensures that
    any off-spec label is mapped to the nearest valid class:
      - values < 0   → 0  (background)
      - values > 2   → 2  (AEA Right, the highest valid class)

    This is safe because the valid labels are {0, 1, 2} and any value outside
    that range is a data artefact rather than a meaningful annotation.
    """

    def __init__(self, keys, num_classes: int):
        super().__init__(keys)
        self.num_classes = num_classes

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = d[key].long().clamp(0, self.num_classes - 1)
        return d


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(mode: str = "train") -> T.Compose:
    """
    Build MONAI transform pipelines for each data split.

    The pipeline differs between training (with augmentation) and
    validation/test (no augmentation, full-volume sliding window).

    Args:
        mode: One of "train", "val", or "test".

    Returns:
        A MONAI Compose transform that processes a dict with keys
        "image" and "mask" (for train/val) or just "image" (for inference).
    """
    patch_size   = TRAIN_CONFIG["patch_size"]
    num_samples  = TRAIN_CONFIG["num_samples"]
    pos_ratio    = TRAIN_CONFIG["pos_sample_ratio"]
    neg_ratio    = TRAIN_CONFIG["neg_sample_ratio"]

    # ── Shared base transforms (always applied) ────────────────────────────────
    base = [
        # Load NIfTI image and mask from file paths stored in the dict
        T.LoadImaged(keys=["image", "mask"], image_only=False),

        # Ensure channel dimension exists: (H, W, D) → (1, H, W, D)
        T.EnsureChannelFirstd(keys=["image", "mask"]),

        # Make sure array types are correct
        T.EnsureTyped(keys=["image", "mask"]),

        # Reorient to RAS+ standard orientation (Right-Anterior-Superior)
        T.Orientationd(keys=["image", "mask"], axcodes="RAS"),

        # Resample to consistent 0.4mm isotropic spacing
        # (data is already at 0.4mm, but this guards against any case-level variation)
        T.Spacingd(
            keys       = ["image", "mask"],
            pixdim     = (0.4, 0.4, 0.4),
            mode       = ("bilinear", "nearest"),
        ),

        # Clip and normalise CBCT Hounsfield Unit values to [0, 1]
        T.ScaleIntensityRanged(
            keys    = ["image"],
            a_min   = HU_MIN,
            a_max   = HU_MAX,
            b_min   = 0.0,
            b_max   = 1.0,
            clip    = True,
        ),

        # Clamp mask labels to [0, NUM_CLASSES-1] BEFORE foreground sampling.
        # Some NRRD files contain label values > 2 (e.g. 255), which causes
        # RandCropByPosNegLabeld to report "Num foregrounds 0" and silently
        # fall back to random patch sampling, hurting convergence.
        ClampMaskLabeld(keys=["mask"], num_classes=NUM_CLASSES),
    ]

    if mode == "train":
        # ── Training-specific transforms ───────────────────────────────────────
        augmentation = [
            # Crop patches with foreground oversampling:
            # pos_ratio:neg_ratio = 1:1 means half the patches contain AEA voxels
            T.RandCropByPosNegLabeld(
                keys        = ["image", "mask"],
                label_key   = "mask",
                spatial_size= patch_size,
                pos         = pos_ratio,
                neg         = neg_ratio,
                num_samples = num_samples,
                image_key   = "image",
                image_threshold = 0,
            ),

            # Random flips along each axis (AEA can be on either side)
            T.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            T.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),

            # Random 90-degree rotations
            T.RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),

            # Random intensity shift and scale to simulate scanner variability
            T.RandScaleIntensityd(keys=["image"], factors=0.1,  prob=0.5),
            T.RandShiftIntensityd(keys=["image"], offsets=0.1,  prob=0.5),

            # ── Additional augmentations (improve generalisation) ──────────────

            # Gaussian noise: simulates sensor/electronic noise across scanners
            T.RandGaussianNoised(
                keys  = ["image"],
                prob  = 0.2,
                mean  = 0.0,
                std   = 0.01,
            ),

            # Gaussian blur: simulates slight defocus or reconstruction differences
            T.RandGaussianSmoothd(
                keys    = ["image"],
                prob    = 0.2,
                sigma_x = (0.5, 1.15),
                sigma_y = (0.5, 1.15),
                sigma_z = (0.5, 1.15),
            ),

            # Contrast adjustment: simulates different CBCT acquisition protocols
            T.RandAdjustContrastd(
                keys  = ["image"],
                prob  = 0.2,
                gamma = (0.7, 1.5),
            ),

            # Elastic deformation: simulates anatomical variability between patients.
            # Magnitude kept small — large deformations on a tiny artery (~5 voxels
            # wide) misalign image and label, giving contradictory training signal.
            T.Rand3DElasticd(
                keys            = ["image", "mask"],
                prob            = 0.15,
                sigma_range     = (3, 5),
                magnitude_range = (10, 30),
                mode            = ("bilinear", "nearest"),
                padding_mode    = "border",
            ),

            # Convert to PyTorch tensors
            T.ToTensord(keys=["image", "mask"]),
        ]
        return T.Compose(base + augmentation)

    else:
        # ── Validation / Test transforms (no augmentation) ─────────────────────
        val_transforms = [
            T.ToTensord(keys=["image", "mask"]),
        ]
        return T.Compose(base + val_transforms)


def get_inference_transforms() -> T.Compose:
    """
    Minimal transform pipeline for inference on a new unseen scan.
    Does not expect a mask key.
    """
    return T.Compose([
        T.LoadImaged(keys=["image"], image_only=False),
        T.EnsureChannelFirstd(keys=["image"]),
        T.EnsureTyped(keys=["image"]),
        T.Orientationd(keys=["image"], axcodes="RAS"),
        T.Spacingd(keys=["image"], pixdim=(0.4, 0.4, 0.4), mode="bilinear"),
        T.ScaleIntensityRanged(
            keys  = ["image"],
            a_min = HU_MIN,
            a_max = HU_MAX,
            b_min = 0.0,
            b_max = 1.0,
            clip  = True,
        ),
        T.ToTensord(keys=["image"]),
    ])


# ── DataLoaders ────────────────────────────────────────────────────────────────

def get_dataloader(
    manifest     : list[dict],
    mode         : str,
    cache_dir    : Optional[Path] = None,
    cache_rate   : float = 1.0,
    num_workers  : int = 4,
) -> DataLoader:
    """
    Build a MONAI DataLoader for a given manifest and mode.

    Uses CacheDataset (loads all data into RAM) if cache_rate=1.0, which is
    recommended for small datasets like ours (130 cases fit comfortably in
    ~16GB RAM). On Colab, set cache_rate=0.5 if memory is tight.

    Args:
        manifest:    List of {"case_id", "image", "mask"} dicts.
        mode:        "train", "val", or "test".
        cache_dir:   If set, use PersistentDataset (cache to disk) instead.
        cache_rate:  Fraction of data to cache in memory (CacheDataset only).
        num_workers: Number of DataLoader worker processes.

    Returns:
        MONAI DataLoader ready for the training loop.
    """
    transforms  = get_transforms(mode)
    batch_size  = TRAIN_CONFIG["batch_size"] if mode == "train" else 1
    shuffle     = (mode == "train")

    if cache_dir is not None:
        # PersistentDataset caches preprocessed tensors to disk
        # — useful on Colab where RAM is limited
        cache_dir = Path(cache_dir) / mode
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset = PersistentDataset(
            data        = manifest,
            transform   = transforms,
            cache_dir   = str(cache_dir),
        )
    else:
        dataset = CacheDataset(
            data        = manifest,
            transform   = transforms,
            cache_rate  = cache_rate,
            num_workers = num_workers,
        )

    loader = DataLoader(
        dataset    = dataset,
        batch_size = batch_size,
        shuffle    = shuffle,
        num_workers= num_workers,
        pin_memory = True,
        collate_fn = pad_list_data_collate,  # Handles variable-size patches
    )

    logger.info(
        f"DataLoader [{mode}]: {len(dataset)} cases, "
        f"batch_size={batch_size}, shuffle={shuffle}"
    )
    return loader


# ── Convenience data module ────────────────────────────────────────────────────

class AEADataModule:
    """
    Convenience wrapper that loads all three splits and exposes DataLoaders.

    Usage:
        dm = AEADataModule(splits_dir=SPLITS_DIR)
        dm.setup()
        for batch in dm.train_loader:
            images = batch["image"]   # (B, 1, H, W, D)
            masks  = batch["mask"]    # (B, 1, H, W, D)
    """

    def __init__(
        self,
        splits_dir  : Path,
        cache_dir   : Optional[Path] = None,
        cache_rate  : float = 1.0,
        num_workers : int = 4,
    ):
        self.splits_dir  = Path(splits_dir)
        self.cache_dir   = cache_dir
        self.cache_rate  = cache_rate
        self.num_workers = num_workers

        self.train_loader = None
        self.val_loader   = None
        self.test_loader  = None

    @staticmethod
    def _filter_empty_masks(manifest: list) -> list:
        """
        Remove training cases where the mask contains no foreground voxels.

        Cases with all-zero masks contribute nothing to foreground sampling —
        RandCropByPosNegLabeld falls back to random background patches for them,
        wasting compute and slowing convergence. Filtering them out ensures every
        training case actually contains AEA voxels for the model to learn from.

        This check reads the NIfTI mask header only (not the full volume) so it
        runs in a few seconds even for large datasets.
        """
        import SimpleITK as sitk
        import numpy as np

        filtered, skipped = [], []
        for case in manifest:
            mask = sitk.GetArrayFromImage(sitk.ReadImage(case["mask"]))
            if np.any(mask > 0):
                filtered.append(case)
            else:
                skipped.append(case["case_id"])

        if skipped:
            logger.warning(
                f"Removed {len(skipped)} training cases with empty masks "
                f"(no AEA annotation): {skipped}"
            )
        logger.info(
            f"Training cases after empty-mask filter: "
            f"{len(filtered)}/{len(filtered) + len(skipped)}"
        )
        return filtered

    def setup(self) -> None:
        """Load JSON manifests and create DataLoaders for all splits."""
        train_manifest = load_json(self.splits_dir / "train.json")
        val_manifest   = load_json(self.splits_dir / "val.json")
        test_manifest  = load_json(self.splits_dir / "test.json")

        # Remove training cases with no foreground — they hurt foreground sampling
        train_manifest = self._filter_empty_masks(train_manifest)

        logger.info(
            f"Loaded splits: {len(train_manifest)} train / "
            f"{len(val_manifest)} val / {len(test_manifest)} test"
        )

        self.train_loader = get_dataloader(
            train_manifest, "train",
            cache_dir=self.cache_dir, cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )
        self.val_loader = get_dataloader(
            val_manifest, "val",
            cache_dir=self.cache_dir, cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )
        self.test_loader = get_dataloader(
            test_manifest, "test",
            cache_dir=self.cache_dir, cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )
