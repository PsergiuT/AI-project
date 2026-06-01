"""
report.py — Structured report generation for AEA segmentation results.

Generates a JSON report and a human-readable text summary that are
displayed in the web UI and available for download.

The report includes:
    - Patient / case metadata
    - Per-class segmentation metrics (Dice, IoU, HD95)
    - Clinical interpretation of the results
    - Anatomical findings (which side has AEA, volumes in mm³)
    - Timestamp and model version info
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLASS_NAMES, TARGET_SPACING


# Volume computation

def compute_volumes(mask: np.ndarray, voxel_spacing_mm: tuple = TARGET_SPACING) -> dict:
    """
    Compute the predicted volume of each AEA segment in mm³.

    Args:
        mask:             3D integer label array (0=bg, 1=AEAL, 2=AEAR).
        voxel_spacing_mm: Physical voxel size in mm per axis.

    Returns:
        Dict with keys "aeal_mm3", "aear_mm3", "total_mm3".
    """
    voxel_vol = voxel_spacing_mm[0] * voxel_spacing_mm[1] * voxel_spacing_mm[2]
    aeal_vox  = int(np.sum(mask == 1))
    aear_vox  = int(np.sum(mask == 2))

    return {
        "aeal_voxels"  : aeal_vox,
        "aear_voxels"  : aear_vox,
        "aeal_mm3"     : round(aeal_vox * voxel_vol, 2),
        "aear_mm3"     : round(aear_vox * voxel_vol, 2),
        "total_mm3"    : round((aeal_vox + aear_vox) * voxel_vol, 2),
    }


# Clinical interpretation

def interpret_dice(dice: float, side: str) -> str:
    """
    Return a plain-language interpretation of a Dice score.

    Thresholds based on published AEA segmentation literature:
        ≥ 0.80 → Strong performance, clinically reliable
        ≥ 0.70 → Acceptable, suitable for surgical planning with review
        ≥ 0.50 → Moderate, manual review strongly recommended
        < 0.50 → Poor, do not use without extensive correction
    """
    if dice >= 0.80:
        quality = "strong"
        advice  = "Segmentation is clinically reliable. Suitable for preoperative planning."
    elif dice >= 0.70:
        quality = "acceptable"
        advice  = "Segmentation is acceptable. Radiologist review recommended before use in surgery."
    elif dice >= 0.50:
        quality = "moderate"
        advice  = "Moderate segmentation quality. Manual correction required before clinical use."
    elif dice >= 0.0:
        quality = "poor"
        advice  = "Poor segmentation quality. Do not use for surgical planning without full correction."
    else:
        quality = "unavailable"
        advice  = "Metric not available (no ground truth provided)."

    return f"{side} AEA [{quality}]: {advice}"


def interpret_hd95(hd95: float) -> str:
    """
    Return a plain-language interpretation of HD95 in mm.
    """
    if hd95 < 0:
        return "HD95: Not available (no ground truth provided)."
    elif hd95 <= 1.0:
        return f"HD95 = {hd95:.2f} mm — Excellent boundary accuracy (< 1 mm)."
    elif hd95 <= 2.0:
        return f"HD95 = {hd95:.2f} mm — Good boundary accuracy (within 2 mm, clinically acceptable)."
    elif hd95 <= 5.0:
        return f"HD95 = {hd95:.2f} mm — Moderate boundary error. Review segmentation boundaries."
    else:
        return f"HD95 = {hd95:.2f} mm — Large boundary error. Manual correction strongly recommended."


def detect_anatomical_findings(mask: np.ndarray) -> dict:
    """
    Extract basic anatomical findings from the predicted mask.

    Args:
        mask: 3D integer label array.

    Returns:
        Dict describing what was detected in the scan.
    """
    aeal_present = bool(np.any(mask == 1))
    aear_present = bool(np.any(mask == 2))

    findings = {
        "aeal_detected" : aeal_present,
        "aear_detected" : aear_present,
        "bilateral"     : aeal_present and aear_present,
        "unilateral"    : aeal_present ^ aear_present,
        "none_detected" : not aeal_present and not aear_present,
    }

    if findings["bilateral"]:
        findings["summary"] = "Bilateral AEA detected. Both left and right arteries segmented."
    elif aeal_present:
        findings["summary"] = "Unilateral AEA detected — left side only."
    elif aear_present:
        findings["summary"] = "Unilateral AEA detected — right side only."
    else:
        findings["summary"] = (
            "No AEA detected. Possible causes: artery not visible on this scan, "
            "or model confidence below threshold. Manual review required."
        )

    return findings


# Main report generator

def generate_report(
    case_id       : str,
    pred_mask     : np.ndarray,
    metrics       : Optional[dict] = None,
    dicom_path    : Optional[str]  = None,
    output_dir    : Optional[Path] = None,
    model_version : str = "SwinUNETR-AEA-v1",
) -> dict:
    """
    Generate a structured segmentation report for a single patient case.

    Args:
        case_id:       Patient or case identifier string.
        pred_mask:     Post-processed 3D integer label mask (0/1/2).
        metrics:       Dict of evaluation metrics (Dice, IoU, HD95).
                       Pass None if no ground truth is available.
        dicom_path:    Path to the source DICOM folder (for provenance).
        output_dir:    If set, save the JSON report to this directory.
        model_version: Model identifier string for provenance.

    Returns:
        Structured report dict.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    volumes   = compute_volumes(pred_mask)
    findings  = detect_anatomical_findings(pred_mask)

    # Build metrics section
    if metrics:
        dice_aeal = metrics.get("dice_aeal", -1.0)
        dice_aear = metrics.get("dice_aear", -1.0)
        iou_aeal  = metrics.get("iou_aeal",  -1.0)
        iou_aear  = metrics.get("iou_aear",  -1.0)
        hd95_aeal = metrics.get("hd95_aeal", -1.0)
        hd95_aear = metrics.get("hd95_aear", -1.0)
    else:
        dice_aeal = dice_aear = iou_aeal = iou_aear = hd95_aeal = hd95_aear = -1.0

    report = {
        # ── Metadata ──────────────────────────────────────────────────────────
        "report_id"        : f"AEA-{case_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "case_id"          : case_id,
        "timestamp"        : timestamp,
        "model_version"    : model_version,
        "source_dicom"     : str(dicom_path) if dicom_path else "Not specified",

        # ── Anatomical findings ────────────────────────────────────────────────
        "anatomical_findings": findings,

        # ── Volumetric measurements ────────────────────────────────────────────
        "volumes": {
            "aeal_mm3"   : volumes["aeal_mm3"],
            "aear_mm3"   : volumes["aear_mm3"],
            "total_mm3"  : volumes["total_mm3"],
            "aeal_voxels": volumes["aeal_voxels"],
            "aear_voxels": volumes["aear_voxels"],
            "voxel_size" : f"{TARGET_SPACING[0]} × {TARGET_SPACING[1]} × {TARGET_SPACING[2]} mm",
        },

        # ── Quantitative metrics (only if ground truth available) ──────────────
        "metrics": {
            "dice": {
                "aeal"  : dice_aeal,
                "aear"  : dice_aear,
                "mean"  : round((dice_aeal + dice_aear) / 2, 4)
                          if dice_aeal >= 0 and dice_aear >= 0 else -1.0,
            },
            "iou": {
                "aeal"  : iou_aeal,
                "aear"  : iou_aear,
                "mean"  : round((iou_aeal + iou_aear) / 2, 4)
                          if iou_aeal >= 0 and iou_aear >= 0 else -1.0,
            },
            "hd95_mm": {
                "aeal"  : hd95_aeal,
                "aear"  : hd95_aear,
                "mean"  : round((hd95_aeal + hd95_aear) / 2, 4)
                          if hd95_aeal >= 0 and hd95_aear >= 0 else -1.0,
            },
            "ground_truth_available": metrics is not None,
        },

        # ── Clinical interpretation ────────────────────────────────────────────
        "interpretation": {
            "aeal_quality"   : interpret_dice(dice_aeal, "Left"),
            "aear_quality"   : interpret_dice(dice_aear, "Right"),
            "boundary_quality": interpret_hd95(hd95_aeal),
            "overall_summary": findings["summary"],
            "clinical_note"  : (
                "This segmentation was generated by an AI model and is intended "
                "to assist preoperative planning. It must be reviewed and validated "
                "by a qualified otolaryngologist or radiologist before clinical use."
            ),
        },
    }

    # Save to disk if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{report['report_id']}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved → {report_path}")
        report["_saved_to"] = str(report_path)

    return report


def report_to_text(report: dict) -> str:
    """
    Convert a structured report dict to a formatted human-readable text string.

    Used for display in the Gradio UI.
    """
    m  = report["metrics"]
    v  = report["volumes"]
    f  = report["anatomical_findings"]
    i  = report["interpretation"]

    lines = [
        "=" * 55,
        "   AEA SEGMENTATION REPORT",
        "=" * 55,
        f"  Case ID      : {report['case_id']}",
        f"  Timestamp    : {report['timestamp']}",
        f"  Model        : {report['model_version']}",
        "",
        "── ANATOMICAL FINDINGS " + "─" * 32,
        f"  {f['summary']}",
        f"  Left AEA     : {'Detected ✓' if f['aeal_detected'] else 'Not detected ✗'}",
        f"  Right AEA    : {'Detected ✓' if f['aear_detected'] else 'Not detected ✗'}",
        "",
        "── VOLUMETRIC MEASUREMENTS " + "─" * 28,
        f"  Left  AEA volume : {v['aeal_mm3']:.2f} mm³  ({v['aeal_voxels']} voxels)",
        f"  Right AEA volume : {v['aear_mm3']:.2f} mm³  ({v['aear_voxels']} voxels)",
        f"  Total            : {v['total_mm3']:.2f} mm³",
        f"  Voxel size       : {v['voxel_size']}",
        "",
    ]

    if m["ground_truth_available"]:
        lines += [
            "── QUANTITATIVE METRICS " + "─" * 32,
            f"  {'Metric':<18} {'AEA Left':>10} {'AEA Right':>10} {'Mean':>10}",
            f"  {'-'*50}",
            f"  {'Dice (DSC)':<18} {m['dice']['aeal']:>10.4f} {m['dice']['aear']:>10.4f} {m['dice']['mean']:>10.4f}",
            f"  {'IoU (Jaccard)':<18} {m['iou']['aeal']:>10.4f} {m['iou']['aear']:>10.4f} {m['iou']['mean']:>10.4f}",
            f"  {'HD95 (mm)':<18} {m['hd95_mm']['aeal']:>10.4f} {m['hd95_mm']['aear']:>10.4f} {m['hd95_mm']['mean']:>10.4f}",
            "",
        ]
    else:
        lines += [
            "── QUANTITATIVE METRICS " + "─" * 32,
            "  No ground truth provided — metrics not available.",
            "  Upload the NRRD mask alongside the DICOM to enable metrics.",
            "",
        ]

    lines += [
        "── CLINICAL INTERPRETATION " + "─" * 28,
        f"  {i['aeal_quality']}",
        f"  {i['aear_quality']}",
        f"  {i['boundary_quality']}",
        "",
        "── IMPORTANT NOTICE " + "─" * 35,
        "  " + i["clinical_note"],
        "=" * 55,
    ]

    return "\n".join(lines)
