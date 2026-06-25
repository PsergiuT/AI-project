"""
postprocess.py — Post-processing for AEA segmentation predictions.

The raw model output can contain small isolated "islands" of predicted voxels
that are anatomically impossible — the AEA is a single continuous structure,
not a scattered collection of disconnected blobs.

Connected Component Analysis (CCA) removes these false positives by:
    1. Labelling every connected group of voxels in the prediction.
    2. Keeping only the largest connected component per class.
    3. Discarding all smaller components below a size threshold.

This is a well-established post-processing step for vascular segmentation
and is documented as one of the improvements in the project plan.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LABEL_AEA_LEFT, LABEL_AEA_RIGHT, LABEL_BACKGROUND


# ── Core CCA function ──

def keep_largest_component(
    binary_mask   : np.ndarray,
    min_size_voxels: int = 10,
) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.

    Args:
        binary_mask:     3D boolean or 0/1 numpy array.
        min_size_voxels: If the largest component is smaller than this,
                         return an empty mask (prediction deemed unreliable).

    Returns:
        Cleaned binary mask with only the largest connected component.
    """
    if binary_mask.sum() == 0:
        return binary_mask  # Nothing to clean

    # Label all connected components (6-connectivity in 3D)
    structure = ndi.generate_binary_structure(3, 1)  # 6-connectivity
    labeled_array, num_features = ndi.label(binary_mask, structure=structure)

    if num_features == 0:
        return binary_mask

    # Find the largest component by voxel count
    component_sizes = ndi.sum(binary_mask, labeled_array, range(1, num_features + 1))
    largest_label   = np.argmax(component_sizes) + 1  # +1 because label 0 = background
    largest_size    = component_sizes[largest_label - 1]

    if largest_size < min_size_voxels:
        logger.warning(
            f"Largest component has only {int(largest_size)} voxels "
            f"(threshold: {min_size_voxels}). Returning empty mask."
        )
        return np.zeros_like(binary_mask)

    cleaned = (labeled_array == largest_label).astype(binary_mask.dtype)
    removed = int(binary_mask.sum() - cleaned.sum())
    if removed > 0:
        logger.debug(f"CCA: removed {removed} voxels ({num_features - 1} extra components)")

    return cleaned


def remove_small_components(
    binary_mask   : np.ndarray,
    min_size_voxels: int = 10,
) -> np.ndarray:
    """
    Remove all connected components smaller than min_size_voxels.

    Unlike keep_largest_component, this preserves multiple components
    as long as each is large enough. Useful if the AEA is occasionally
    predicted as two separate segments due to a small gap.

    Args:
        binary_mask:      3D boolean or 0/1 numpy array.
        min_size_voxels:  Minimum component size to keep.

    Returns:
        Cleaned binary mask.
    """
    if binary_mask.sum() == 0:
        return binary_mask

    structure = ndi.generate_binary_structure(3, 1)
    labeled_array, num_features = ndi.label(binary_mask, structure=structure)
    component_sizes = ndi.sum(binary_mask, labeled_array, range(1, num_features + 1))

    # Create output mask keeping only components above threshold
    cleaned = np.zeros_like(binary_mask)
    for label_idx, size in enumerate(component_sizes):
        if size >= min_size_voxels:
            cleaned[labeled_array == (label_idx + 1)] = 1

    removed = int(binary_mask.sum() - cleaned.sum())
    logger.debug(f"Small component removal: {removed} voxels removed")
    return cleaned


# ── Full mask post-processing ──

def postprocess_mask(
    pred_mask       : np.ndarray | torch.Tensor,
    min_size_voxels : int = 10,
    strategy        : str = "largest",
) -> np.ndarray:
    """
    Apply post-processing to a 3-class segmentation mask.

    Processes each foreground class (AEAL, AEAR) independently,
    preserving the background class unchanged.

    Args:
        pred_mask:        3D array of shape (H, W, D) with integer labels
                          {0: background, 1: AEAL, 2: AEAR}.
                          Can be a numpy array or a torch Tensor.
        min_size_voxels:  Minimum connected component size to retain.
        strategy:         "largest" — keep only the largest component per class.
                          "threshold" — remove all components below min_size.

    Returns:
        Cleaned 3D numpy array with same shape and dtype as input.
    """
    # Convert torch Tensor to numpy if needed
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()

    pred_mask = np.asarray(pred_mask, dtype=np.int16)
    cleaned   = np.zeros_like(pred_mask)

    for label in [LABEL_AEA_LEFT, LABEL_AEA_RIGHT]:
        class_name   = "AEA Left" if label == LABEL_AEA_LEFT else "AEA Right"
        binary       = (pred_mask == label).astype(np.uint8)
        voxel_count  = int(binary.sum())

        if voxel_count == 0:
            logger.debug(f"[Postprocess] {class_name}: no predicted voxels — skipping.")
            continue

        logger.debug(f"[Postprocess] {class_name}: {voxel_count} predicted voxels before CCA.")

        if strategy == "largest":
            cleaned_binary = keep_largest_component(binary, min_size_voxels)
        elif strategy == "threshold":
            cleaned_binary = remove_small_components(binary, min_size_voxels)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. Use 'largest' or 'threshold'.")

        cleaned[cleaned_binary == 1] = label
        logger.debug(
            f"[Postprocess] {class_name}: {int(cleaned_binary.sum())} voxels after CCA."
        )

    return cleaned


# ── Morphological smoothing (optional) ──

def smooth_mask(
    pred_mask    : np.ndarray,
    iterations   : int = 1,
) -> np.ndarray:
    """
    Apply morphological closing to smooth the segmentation boundary.

    Closing = dilation followed by erosion.
    It fills small holes and smooths rough edges without significantly
    changing the overall shape or position of the segmentation.

    Args:
        pred_mask:   3D integer label mask.
        iterations:  Number of closing iterations (1 is usually enough).

    Returns:
        Smoothed mask.
    """
    smoothed = np.zeros_like(pred_mask)
    struct   = ndi.generate_binary_structure(3, 1)

    for label in [LABEL_AEA_LEFT, LABEL_AEA_RIGHT]:
        binary  = (pred_mask == label).astype(bool)
        closed  = ndi.binary_closing(binary, structure=struct, iterations=iterations)
        smoothed[closed] = label

    return smoothed


# ── Pipeline convenience function ──────────────────────────────────────────────

def full_postprocess(
    pred_mask       : np.ndarray | torch.Tensor,
    min_size_voxels : int  = 10,
    apply_smoothing : bool = False,
) -> np.ndarray:
    """
    Run the complete post-processing pipeline on a prediction mask.

    Steps:
        1. Connected component analysis (keep largest per class).
        2. (Optional) Morphological smoothing.

    Args:
        pred_mask:        Raw model prediction (integer labels 0/1/2).
        min_size_voxels:  Minimum connected component size.
        apply_smoothing:  Whether to apply morphological closing.

    Returns:
        Post-processed integer label mask as numpy array.
    """
    mask = postprocess_mask(pred_mask, min_size_voxels=min_size_voxels, strategy="largest")

    if apply_smoothing:
        mask = smooth_mask(mask, iterations=1)

    return mask
