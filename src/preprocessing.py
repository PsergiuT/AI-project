"""
preprocessing.py — Convert raw DICOM + NRRD data to NIfTI format for MONAI.

Usage (run once before training):
    python src/preprocessing.py --data_root ../dateArteraEtimoidala \
                                 --output_dir data/processed

What this script does for each patient case:
    1. Finds all DICOM slices in the patient's subfolder.
    2. Loads and sorts slices by their 3D position (ImagePositionPatient).
    3. Constructs a SimpleITK image with correct geometry (origin, spacing, direction).
    4. Loads the NRRD segmentation mask (already aligned to the DICOM space).
    5. Resamples the mask to exactly match the DICOM image grid.
    6. Saves both as compressed NIfTI files (image.nii.gz, mask.nii.gz).
    7. Writes a JSON manifest listing all processed cases.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import pydicom
import nrrd
from loguru import logger
from tqdm import tqdm

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CROP_FOLDERS, PROCESSED_DIR, SPLITS_DIR,
    TARGET_SPACING, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO,
)
from src.utils import build_manifest, split_manifest, save_json, setup_logger, set_seed


# DICOM loading

def load_dicom_series(dicom_dir: Path) -> sitk.Image:
    """
    Load a DICOM series from a directory and return a SimpleITK 3D image.

    Slices are sorted by the Z component of ImagePositionPatient to ensure
    correct anatomical ordering regardless of filename ordering.

    Args:
        dicom_dir: Path to the folder containing .dcm files.

    Returns:
        SimpleITK Image with correct geometry (origin, spacing, direction cosines).
    """
    dicom_files = sorted(dicom_dir.glob("*.dcm"))
    if not dicom_files:
        raise FileNotFoundError(f"No .dcm files found in {dicom_dir}")

    # Read all slices and sort by ImagePositionPatient Z coordinate
    slices = []
    for f in dicom_files:
        ds = pydicom.dcmread(str(f), stop_before_pixels=False)
        slices.append(ds)

    # Sort by Z position in patient coordinate system
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except AttributeError:
        # Fallback: sort by InstanceNumber if ImagePositionPatient is missing
        slices.sort(key=lambda s: int(s.InstanceNumber))
        logger.warning(f"ImagePositionPatient missing — sorted by InstanceNumber in {dicom_dir}")

    # Extract pixel arrays and stack into a 3D numpy array (Z, Y, X)
    pixel_arrays = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        # Apply rescale slope/intercept to convert to Hounsfield Units
        slope     = float(getattr(s, "RescaleSlope",     1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        pixel_arrays.append(arr)

    volume = np.stack(pixel_arrays, axis=0)  # (Z, Y, X)

    # Build SimpleITK image with proper geometry
    ref = slices[0]
    row_spacing, col_spacing = [float(x) for x in ref.PixelSpacing]
    slice_thickness = float(getattr(ref, "SliceThickness", TARGET_SPACING[2]))

    # SimpleITK expects (X, Y, Z) order — transpose from numpy's (Z, Y, X)
    sitk_image = sitk.GetImageFromArray(volume)
    sitk_image.SetSpacing((col_spacing, row_spacing, slice_thickness))

    # Set origin from first slice
    origin = [float(x) for x in ref.ImagePositionPatient]
    sitk_image.SetOrigin(origin)

    # Set direction cosines from ImageOrientationPatient
    try:
        iop = [float(x) for x in ref.ImageOrientationPatient]
        row_cosine = iop[:3]
        col_cosine = iop[3:]
        # Compute normal (slice direction) as cross product
        normal = [
            row_cosine[1] * col_cosine[2] - row_cosine[2] * col_cosine[1],
            row_cosine[2] * col_cosine[0] - row_cosine[0] * col_cosine[2],
            row_cosine[0] * col_cosine[1] - row_cosine[1] * col_cosine[0],
        ]
        direction = row_cosine + col_cosine + normal
        sitk_image.SetDirection(direction)
    except AttributeError:
        logger.warning(f"ImageOrientationPatient missing — using identity direction in {dicom_dir}")

    return sitk_image


# NRRD mask loading

def load_nrrd_mask(nrrd_path: Path) -> sitk.Image:
    """
    Load a 3D Slicer NRRD segmentation file and return a SimpleITK image.

    The NRRD contains a 3-class labelmap:
        0 = background, 1 = AEA Left (AEAL), 2 = AEA Right (AEAR)

    Args:
        nrrd_path: Path to the .nrrd file.

    Returns:
        SimpleITK Image with integer labels.
    """
    data, header = nrrd.read(str(nrrd_path))

    # NRRD data shape is (X, Y, Z) — SimpleITK also uses (X, Y, Z) internally
    # but GetImageFromArray expects (Z, Y, X), so we transpose
    data_zyx = np.transpose(data, (2, 1, 0)).astype(np.int16)
    sitk_mask = sitk.GetImageFromArray(data_zyx)

    # Extract geometry from NRRD header
    if "space directions" in header:
        directions = header["space directions"]
        # directions is a 3x3 matrix: each row is a basis vector
        dx = float(directions[0][0])
        dy = float(directions[1][1])
        dz = float(directions[2][2])
        sitk_mask.SetSpacing((abs(dx), abs(dy), abs(dz)))

    if "space origin" in header:
        origin = [float(x) for x in header["space origin"]]
        sitk_mask.SetOrigin(origin)

    return sitk_mask


# Resampling

def resample_mask_to_image(
    mask: sitk.Image,
    reference: sitk.Image,
) -> sitk.Image:
    """
    Resample the mask to match the reference image's grid (size, spacing, origin).

    Uses nearest-neighbour interpolation to preserve integer label values.
    This is necessary because 3D Slicer may export the NRRD on a different
    grid than the original DICOM series.

    Args:
        mask:      The segmentation mask to resample.
        reference: The DICOM image whose grid we want to match.

    Returns:
        Resampled mask aligned to the reference image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkInt16)
    return resampler.Execute(mask)


# Per-case processing

def process_case(
    case_id    : str,
    dicom_dir  : Path,
    nrrd_path  : Path,
    output_dir : Path,
) -> dict | None:
    """
    Process a single patient case: load, align, and save as NIfTI.

    Args:
        case_id:    Unique identifier for this patient (e.g. "case_001").
        dicom_dir:  Directory containing the .dcm slice files.
        nrrd_path:  Path to the .nrrd segmentation mask.
        output_dir: Root output directory (a subdirectory named case_id is created).

    Returns:
        Dict with {"case_id", "image", "mask"} paths on success, None on failure.
    """
    case_output = output_dir / case_id
    image_out   = case_output / "image.nii.gz"
    mask_out    = case_output / "mask.nii.gz"

    # Skip if already processed
    if image_out.exists() and mask_out.exists():
        logger.info(f"[{case_id}] Already processed — skipping.")
        return {"case_id": case_id, "image": str(image_out), "mask": str(mask_out)}

    case_output.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Load DICOM volume
        logger.info(f"[{case_id}] Loading DICOM from {dicom_dir} …")
        image = load_dicom_series(dicom_dir)

        # 2. Load NRRD mask
        logger.info(f"[{case_id}] Loading NRRD mask from {nrrd_path} …")
        mask = load_nrrd_mask(nrrd_path)

        # 3. Resample mask to match image grid
        mask_resampled = resample_mask_to_image(mask, image)

        # 4. Validate label values
        mask_array = sitk.GetArrayFromImage(mask_resampled)
        unique_labels = np.unique(mask_array)
        logger.info(f"[{case_id}] Mask labels: {unique_labels}")
        if not all(l in [0, 1, 2] for l in unique_labels):
            logger.warning(f"[{case_id}] Unexpected label values: {unique_labels}")

        # 5. Save as NIfTI
        sitk.WriteImage(image,           str(image_out))
        sitk.WriteImage(mask_resampled,  str(mask_out))
        logger.info(f"[{case_id}] Saved → {case_output}")

        return {"case_id": case_id, "image": str(image_out), "mask": str(mask_out)}

    except Exception as e:
        logger.error(f"[{case_id}] Failed: {e}")
        return None


# Dataset discovery

def discover_cases(data_root: Path) -> list[tuple[str, Path, Path]]:
    """
    Walk through CROP1–CROP7 folders and collect (case_id, dicom_dir, nrrd_path).

    Expected structure:
        data_root/
            CROP1/
                001. NL001 NUNA LUCA/
                    NL001/          ← DICOM files
                    NL001.nrrd      ← segmentation mask
                002. …
            CROP2/
                …

    Returns:
        List of (case_id, dicom_dir, nrrd_path) tuples, sorted by case_id.
    """
    cases = []
    data_root = Path(data_root)

    for crop_folder in CROP_FOLDERS:
        crop_path = data_root / crop_folder
        if not crop_path.exists():
            logger.warning(f"CROP folder not found: {crop_path}")
            continue

        for patient_dir in sorted(crop_path.iterdir()):
            if not patient_dir.is_dir():
                continue

            # Extract numeric case ID from folder name (e.g. "001. NL001 …" → "001")
            folder_name = patient_dir.name
            case_num = folder_name.split(".")[0].strip().zfill(3)
            case_id  = f"case_{case_num}"

            # Find the DICOM subfolder (named after the patient code, e.g. "NL001")
            dicom_dirs = [
                d for d in patient_dir.iterdir()
                if d.is_dir() and any(d.glob("*.dcm"))
            ]
            nrrd_files = list(patient_dir.glob("*.nrrd"))

            if not dicom_dirs:
                logger.warning(f"[{case_id}] No DICOM subfolder found in {patient_dir}")
                continue
            if not nrrd_files:
                logger.warning(f"[{case_id}] No NRRD file found in {patient_dir}")
                continue

            dicom_dir = dicom_dirs[0]
            nrrd_path = nrrd_files[0]
            cases.append((case_id, dicom_dir, nrrd_path))

    cases.sort(key=lambda x: x[0])
    logger.info(f"Discovered {len(cases)} cases in {data_root}")
    return cases


# Main

def main(data_root: Path, output_dir: Path) -> None:
    set_seed(RANDOM_SEED)
    setup_logger(output_dir.parent / "logs")

    logger.info("=" * 60)
    logger.info("AEA Segmentation — Preprocessing Pipeline")
    logger.info("=" * 60)

    # Discover all cases
    cases = discover_cases(data_root)
    if not cases:
        logger.error("No cases found. Check DATA_ROOT in config.py.")
        return

    # Process each case
    manifest = []
    for case_id, dicom_dir, nrrd_path in tqdm(cases, desc="Processing cases"):
        result = process_case(case_id, dicom_dir, nrrd_path, output_dir)
        if result:
            manifest.append(result)

    logger.info(f"Successfully processed {len(manifest)} / {len(cases)} cases.")

    # Save full manifest
    manifest_path = SPLITS_DIR / "manifest.json"
    save_json(manifest, manifest_path)

    # Create train/val/test splits
    train, val, test = split_manifest(
        manifest,
        train_ratio = TRAIN_RATIO,
        val_ratio   = VAL_RATIO,
        seed        = RANDOM_SEED,
    )
    save_json(train, SPLITS_DIR / "train.json")
    save_json(val,   SPLITS_DIR / "val.json")
    save_json(test,  SPLITS_DIR / "test.json")

    logger.info("Preprocessing complete.")
    logger.info(f"  Manifest → {manifest_path}")
    logger.info(f"  Train    → {SPLITS_DIR / 'train.json'} ({len(train)} cases)")
    logger.info(f"  Val      → {SPLITS_DIR / 'val.json'} ({len(val)} cases)")
    logger.info(f"  Test     → {SPLITS_DIR / 'test.json'} ({len(test)} cases)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess AEA CBCT dataset")
    parser.add_argument(
        "--data_root",
        type    = Path,
        default = Path("../dateArteraEtimoidala"),
        help    = "Path to the folder containing CROP1 … CROP7",
    )
    parser.add_argument(
        "--output_dir",
        type    = Path,
        default = PROCESSED_DIR,
        help    = "Output directory for processed NIfTI files",
    )
    args = parser.parse_args()
    main(args.data_root, args.output_dir)
