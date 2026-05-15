"""
agent/tools.py — LangChain tool definitions for the AEA segmentation agent.

Each tool is a Python function decorated with @tool from LangChain.
The agent (Llama 3.1 via Ollama) can call these tools in any order
to fulfil a user's natural language instruction.

── How state is shared between tools ──────────────────────────────────────────
Tools in LangChain are stateless by design — they receive arguments as strings
and return strings. To pass large objects (tensors, numpy arrays) between tool
calls without serialising them, we use a module-level SESSION_STORE dictionary.

Each pipeline run is assigned a unique session_id. Tools write their outputs
into SESSION_STORE[session_id] and return the session_id string to the agent.
The agent passes this session_id to subsequent tools as an argument.

── Available tools ─────────────────────────────────────────────────────────────
1. load_and_preprocess   — Load a DICOM folder and convert to a model-ready tensor
2. run_segmentation      — Run SwinUNETR inference on the preprocessed volume
3. postprocess_segmentation — Remove small disconnected components from the mask
4. evaluate_segmentation — Compute Dice, IoU, HD95 against a ground truth mask
5. generate_report       — Produce a structured JSON + text report
"""

import sys
import uuid
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import SimpleITK as sitk
from langchain_core.tools import tool
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    SWINUNETR_DIR, TRAIN_CONFIG, MODEL_CONFIG,
    INFERENCE_CONFIG, HU_MIN, HU_MAX, LOGS_DIR,
)
from src.postprocess import full_postprocess
from src.report import generate_report, report_to_text

# ── Lazy model loader ──────────────────────────────────────────────────────────
# The model is loaded once and reused across all tool calls in the session.
_model_cache: dict = {"model": None, "device": None}

def _get_model():
    """Load the trained SwinUNETR model (cached after first call)."""
    if _model_cache["model"] is not None:
        return _model_cache["model"], _model_cache["device"]

    from monai.networks.nets import SwinUNETR

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    _kwargs = dict(
        in_channels    = MODEL_CONFIG["in_channels"],
        out_channels   = MODEL_CONFIG["out_channels"],
        feature_size   = MODEL_CONFIG["feature_size"],
        use_checkpoint = False,
    )
    try:
        model = SwinUNETR(img_size=MODEL_CONFIG["img_size"], **_kwargs).to(device)
    except TypeError:
        model = SwinUNETR(MODEL_CONFIG["img_size"], **_kwargs).to(device)

    best_ckpt = SWINUNETR_DIR / TRAIN_CONFIG["best_model_name"]
    if not best_ckpt.exists():
        raise FileNotFoundError(
            f"Trained model not found at {best_ckpt}. "
            "Run training first using notebooks/training.ipynb"
        )

    checkpoint = torch.load(str(best_ckpt), map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    logger.info(f"SwinUNETR model loaded from {best_ckpt} on {device}")

    _model_cache["model"]  = model
    _model_cache["device"] = device
    return model, device


# ── Session store ──────────────────────────────────────────────────────────────
# Holds intermediate pipeline results keyed by session_id.
# Structure per session:
#   {
#     "dicom_path"    : str,
#     "volume_tensor" : torch.Tensor,   # (1, 1, H, W, D) normalised
#     "sitk_image"    : sitk.Image,     # for geometry preservation
#     "raw_mask"      : np.ndarray,     # raw model output (H, W, D)
#     "clean_mask"    : np.ndarray,     # post-processed mask
#     "metrics"       : dict | None,
#     "report"        : dict,
#     "report_text"   : str,
#   }
SESSION_STORE: dict[str, dict] = {}


def _new_session(dicom_path: str) -> str:
    """Create a new session and return its ID."""
    session_id = str(uuid.uuid4())[:8]
    SESSION_STORE[session_id] = {"dicom_path": dicom_path}
    return session_id


# ── Tool 1 — Load and preprocess ──────────────────────────────────────────────

@tool
def load_and_preprocess(dicom_path: str) -> str:
    """
    Load a CBCT scan from a DICOM folder and prepare it for AI segmentation.

    This tool reads all .dcm files in the given folder, sorts them by anatomical
    position, reconstructs the 3D volume, applies Hounsfield Unit normalisation,
    and stores the result ready for the segmentation model.

    Args:
        dicom_path: Absolute path to the folder containing .dcm DICOM files.
                    Example: '/data/patient_001/NL001'

    Returns:
        A session_id string to pass to the next tool (run_segmentation).
        Returns an error message string if loading fails.
    """
    dicom_path = dicom_path.strip().strip('"\'')
    logger.info(f"[Tool 1] Loading DICOM from: {dicom_path}")

    try:
        path = Path(dicom_path)
        if not path.exists():
            return f"ERROR: Path does not exist: {dicom_path}"

        dcm_files = list(path.glob("*.dcm"))
        if not dcm_files:
            # Try one level deeper
            dcm_files = list(path.rglob("*.dcm"))
        if not dcm_files:
            return f"ERROR: No .dcm files found in {dicom_path}"

        # Load the DICOM series using SimpleITK (handles geometry automatically)
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(path))
        if not series_ids:
            return f"ERROR: No valid DICOM series found in {dicom_path}"

        file_names = reader.GetGDCMSeriesFileNames(str(path), series_ids[0])
        reader.SetFileNames(file_names)
        sitk_image = reader.Execute()

        logger.info(
            f"[Tool 1] Loaded volume: size={sitk_image.GetSize()}, "
            f"spacing={[round(s,3) for s in sitk_image.GetSpacing()]}"
        )

        # Convert to numpy and normalise HU values to [0, 1]
        volume_np = sitk.GetArrayFromImage(sitk_image).astype(np.float32)  # (Z, Y, X)
        volume_np = np.clip(volume_np, HU_MIN, HU_MAX)
        volume_np = (volume_np - HU_MIN) / (HU_MAX - HU_MIN)

        # Convert to torch tensor: (1, 1, H, W, D) = (batch, channel, Z, Y, X)
        volume_tensor = torch.from_numpy(volume_np).unsqueeze(0).unsqueeze(0)

        # Store in session
        session_id = _new_session(dicom_path)
        SESSION_STORE[session_id]["volume_tensor"] = volume_tensor
        SESSION_STORE[session_id]["sitk_image"]    = sitk_image

        shape = tuple(volume_tensor.shape)
        return (
            f"SUCCESS. Session ID: {session_id}. "
            f"Volume loaded: shape={shape}, "
            f"size={sitk_image.GetSize()}, "
            f"spacing={[round(s,3) for s in sitk_image.GetSpacing()]} mm. "
            f"Pass session_id='{session_id}' to run_segmentation."
        )

    except Exception as e:
        logger.error(f"[Tool 1] Error: {e}")
        return f"ERROR in load_and_preprocess: {str(e)}"


# ── Tool 2 — Run segmentation ─────────────────────────────────────────────────

@tool
def run_segmentation(session_id: str) -> str:
    """
    Run the trained SwinUNETR AI model to segment the anterior ethmoidal artery.

    Uses sliding window inference: the model processes overlapping 96×96×96 mm
    patches across the full volume, then averages overlapping predictions.
    This is the core AI step that produces the AEA segmentation mask.

    Args:
        session_id: The session ID returned by load_and_preprocess.

    Returns:
        Confirmation string with the session_id and prediction statistics.
        Returns an error message string if inference fails.
    """
    session_id = session_id.strip()
    logger.info(f"[Tool 2] Running segmentation for session: {session_id}")

    if session_id not in SESSION_STORE:
        return f"ERROR: Session '{session_id}' not found. Run load_and_preprocess first."

    session = SESSION_STORE[session_id]
    if "volume_tensor" not in session:
        return "ERROR: No volume in this session. Run load_and_preprocess first."

    try:
        from monai.inferers import sliding_window_inference
        from monai.transforms import AsDiscrete

        model, device = _get_model()
        volume_tensor = session["volume_tensor"].to(device)

        logger.info(f"[Tool 2] Running sliding window inference on {device}...")
        with torch.no_grad():
            logits = sliding_window_inference(
                inputs        = volume_tensor,
                roi_size      = INFERENCE_CONFIG["roi_size"],
                sw_batch_size = INFERENCE_CONFIG["sw_batch_size"],
                predictor     = model,
                overlap       = INFERENCE_CONFIG["overlap"],
                mode          = INFERENCE_CONFIG["mode"],
            )

        # Argmax over class dimension → integer label mask
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (H, W, D)

        aeal_voxels = int(np.sum(pred == 1))
        aear_voxels = int(np.sum(pred == 2))

        SESSION_STORE[session_id]["raw_mask"] = pred
        logger.info(
            f"[Tool 2] Inference complete. "
            f"AEAL: {aeal_voxels} voxels, AEAR: {aear_voxels} voxels"
        )

        return (
            f"SUCCESS. Segmentation complete for session '{session_id}'. "
            f"Raw prediction — AEA Left: {aeal_voxels} voxels, "
            f"AEA Right: {aear_voxels} voxels. "
            f"Run postprocess_segmentation('{session_id}') next to clean the mask."
        )

    except Exception as e:
        logger.error(f"[Tool 2] Error: {e}")
        return f"ERROR in run_segmentation: {str(e)}"


# ── Tool 3 — Post-process ─────────────────────────────────────────────────────

@tool
def postprocess_segmentation(session_id: str) -> str:
    """
    Clean the raw AI segmentation mask using connected component analysis.

    Removes small disconnected 'islands' of predicted voxels that are
    anatomically impossible — the AEA is a single continuous structure.
    Only the largest connected component per class is retained.

    Args:
        session_id: The session ID from the previous tool call.

    Returns:
        Confirmation string with cleaned voxel counts.
    """
    session_id = session_id.strip()
    logger.info(f"[Tool 3] Post-processing for session: {session_id}")

    if session_id not in SESSION_STORE:
        return f"ERROR: Session '{session_id}' not found."

    session = SESSION_STORE[session_id]
    if "raw_mask" not in session:
        return "ERROR: No raw mask found. Run run_segmentation first."

    try:
        raw_mask   = session["raw_mask"]
        clean_mask = full_postprocess(raw_mask, min_size_voxels=10)

        aeal_before = int(np.sum(raw_mask   == 1))
        aear_before = int(np.sum(raw_mask   == 2))
        aeal_after  = int(np.sum(clean_mask == 1))
        aear_after  = int(np.sum(clean_mask == 2))

        SESSION_STORE[session_id]["clean_mask"] = clean_mask
        logger.info(
            f"[Tool 3] Post-processing done. "
            f"AEAL: {aeal_before}→{aeal_after}, AEAR: {aear_before}→{aear_after}"
        )

        return (
            f"SUCCESS. Post-processing complete for session '{session_id}'. "
            f"AEA Left: {aeal_before} → {aeal_after} voxels "
            f"({aeal_before - aeal_after} removed). "
            f"AEA Right: {aear_before} → {aear_after} voxels "
            f"({aear_before - aear_after} removed). "
            f"Run generate_report('{session_id}') to create the final report, "
            f"or evaluate_segmentation('{session_id}', gt_nrrd_path) if you have ground truth."
        )

    except Exception as e:
        logger.error(f"[Tool 3] Error: {e}")
        return f"ERROR in postprocess_segmentation: {str(e)}"


# ── Tool 4 — Evaluate (optional, requires ground truth) ───────────────────────

@tool
def evaluate_segmentation(session_id: str, gt_nrrd_path: str) -> str:
    """
    Compute segmentation quality metrics by comparing prediction to a ground truth mask.

    This tool is optional — only use it if a manual segmentation mask (.nrrd file)
    is available for the case. If no ground truth is available, skip this tool
    and go directly to generate_report.

    Computes:
        - Dice Similarity Coefficient (DSC): overlap quality, range 0–1
        - IoU / Jaccard Index: overlap quality, range 0–1
        - HD95: boundary accuracy in mm (lower is better)

    Args:
        session_id:   The session ID from postprocess_segmentation.
        gt_nrrd_path: Absolute path to the ground truth .nrrd segmentation file.

    Returns:
        Formatted string with metric values for both AEA sides.
    """
    session_id   = session_id.strip()
    gt_nrrd_path = gt_nrrd_path.strip().strip('"\'')
    logger.info(f"[Tool 4] Evaluating session '{session_id}' against {gt_nrrd_path}")

    if session_id not in SESSION_STORE:
        return f"ERROR: Session '{session_id}' not found."

    session = SESSION_STORE[session_id]
    if "clean_mask" not in session:
        return "ERROR: No cleaned mask found. Run postprocess_segmentation first."

    try:
        import nrrd
        from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
        from monai.transforms import AsDiscrete
        from config import NUM_CLASSES

        # Load ground truth
        gt_data, _ = nrrd.read(gt_nrrd_path)
        gt_array   = np.transpose(gt_data, (2, 1, 0)).astype(np.int16)  # (Z, Y, X)

        pred = session["clean_mask"]

        # Ensure same shape (resample GT if needed)
        if gt_array.shape != pred.shape:
            logger.warning(
                f"[Tool 4] Shape mismatch: pred={pred.shape}, gt={gt_array.shape}. "
                "Attempting resize..."
            )
            from scipy.ndimage import zoom
            zoom_factors = [p/g for p, g in zip(pred.shape, gt_array.shape)]
            gt_array = zoom(gt_array, zoom_factors, order=0).astype(np.int16)

        # Convert to one-hot tensors for MONAI metrics
        to_onehot = AsDiscrete(to_onehot=NUM_CLASSES)
        pred_oh = to_onehot(
            torch.from_numpy(pred).unsqueeze(0).long()
        ).unsqueeze(0).float()
        gt_oh = to_onehot(
            torch.from_numpy(gt_array).unsqueeze(0).long()
        ).unsqueeze(0).float()

        # Compute metrics
        dice_m = DiceMetric(include_background=False, reduction="mean_batch")
        iou_m  = MeanIoU(include_background=False, reduction="mean_batch")
        hd_m   = HausdorffDistanceMetric(include_background=False, percentile=95,
                                          reduction="mean_batch")

        dice_m(y_pred=pred_oh, y=gt_oh)
        iou_m( y_pred=pred_oh, y=gt_oh)
        hd_m(  y_pred=pred_oh, y=gt_oh)

        dice_vals, _ = dice_m.aggregate()
        iou_vals,  _ = iou_m.aggregate()
        hd_vals,   _ = hd_m.aggregate()

        def s(t, i):
            return round(t[i].item(), 4) if t.numel() > i else float("nan")

        metrics = {
            "dice_aeal": s(dice_vals, 0), "dice_aear": s(dice_vals, 1),
            "dice_mean": round(dice_vals.nanmean().item(), 4),
            "iou_aeal" : s(iou_vals,  0), "iou_aear" : s(iou_vals,  1),
            "iou_mean" : round(iou_vals.nanmean().item(), 4),
            "hd95_aeal": s(hd_vals,   0), "hd95_aear": s(hd_vals,   1),
            "hd95_mean": round(hd_vals.nanmean().item(), 4),
        }

        SESSION_STORE[session_id]["metrics"] = metrics
        logger.info(f"[Tool 4] Metrics: {metrics}")

        return (
            f"SUCCESS. Evaluation complete for session '{session_id}'.\n"
            f"{'Metric':<18} {'AEA Left':>10} {'AEA Right':>10} {'Mean':>10}\n"
            f"{'-'*50}\n"
            f"{'Dice (DSC)':<18} {metrics['dice_aeal']:>10.4f} {metrics['dice_aear']:>10.4f} {metrics['dice_mean']:>10.4f}\n"
            f"{'IoU':<18} {metrics['iou_aeal']:>10.4f} {metrics['iou_aear']:>10.4f} {metrics['iou_mean']:>10.4f}\n"
            f"{'HD95 (mm)':<18} {metrics['hd95_aeal']:>10.4f} {metrics['hd95_aear']:>10.4f} {metrics['hd95_mean']:>10.4f}\n"
            f"Run generate_report('{session_id}') to produce the final report."
        )

    except Exception as e:
        logger.error(f"[Tool 4] Error: {e}")
        return f"ERROR in evaluate_segmentation: {str(e)}"


# ── Tool 5 — Generate report ──────────────────────────────────────────────────

@tool
def generate_final_report(session_id: str, patient_id: str = "unknown") -> str:
    """
    Generate the final structured segmentation report for the patient case.

    Produces a JSON report and a formatted text summary that include:
    anatomical findings (which AEA sides detected), volumetric measurements,
    quantitative metrics (if ground truth was provided), and clinical interpretation.
    The report is saved to the logs directory and returned as formatted text.

    Args:
        session_id: The session ID from postprocess_segmentation (or evaluate_segmentation).
        patient_id: Optional patient identifier to include in the report.
                    Defaults to the DICOM folder name if not provided.

    Returns:
        Formatted text report string ready for display.
    """
    session_id = session_id.strip()
    patient_id = patient_id.strip() if patient_id else "unknown"
    logger.info(f"[Tool 5] Generating report for session '{session_id}', patient '{patient_id}'")

    if session_id not in SESSION_STORE:
        return f"ERROR: Session '{session_id}' not found."

    session = SESSION_STORE[session_id]
    mask_key = "clean_mask" if "clean_mask" in session else "raw_mask"

    if mask_key not in session:
        return "ERROR: No segmentation mask found. Run run_segmentation and postprocess_segmentation first."

    try:
        # Use DICOM folder name as patient_id fallback
        if patient_id == "unknown" and "dicom_path" in session:
            patient_id = Path(session["dicom_path"]).name

        report = generate_report(
            case_id       = patient_id,
            pred_mask     = session[mask_key],
            metrics       = session.get("metrics", None),
            dicom_path    = session.get("dicom_path"),
            output_dir    = LOGS_DIR / "reports",
        )

        report_text = report_to_text(report)

        SESSION_STORE[session_id]["report"]      = report
        SESSION_STORE[session_id]["report_text"] = report_text
        logger.info(f"[Tool 5] Report generated for session '{session_id}'")

        return report_text

    except Exception as e:
        logger.error(f"[Tool 5] Error: {e}")
        return f"ERROR in generate_final_report: {str(e)}"


# ── Tool registry ──────────────────────────────────────────────────────────────

ALL_TOOLS = [
    load_and_preprocess,
    run_segmentation,
    postprocess_segmentation,
    evaluate_segmentation,
    generate_final_report,
]
