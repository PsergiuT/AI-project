"""
ui/app.py — Gradio dashboard for the AEA Segmentation Agent.

Layout (3-column dashboard):
┌────────────────┬─────────────────────────┬──────────────────────┐
│  INPUT PANEL   │    SLICE VIEWER         │   RESULTS PANEL      │
│                │                         │                      │
│  DICOM upload  │  Axial / Coronal /      │  Metrics table       │
│  NRRD upload   │  Sagittal tabs          │  Report text         │
│  Patient ID    │  Slice slider           │  Download report     │
│  Instruction   │  Overlay toggle         │                      │
│  [Run] button  │                         │                      │
├────────────────┴─────────────────────────┴──────────────────────┤
│                  AGENT REASONING LOG                             │
│  (shows Thought → Action → Observation trace)                    │
└──────────────────────────────────────────────────────────────────┘

Run with:
    python ui/app.py
    or
    python run.py
"""

import sys
import os
import json
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import gradio as gr
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LOGS_DIR, CLASS_NAMES, HU_MIN, HU_MAX
from src.agent.tools import SESSION_STORE


# ── Colour palette (matches 3D Slicer annotation colours) ─────────────────────
AEAL_COLOUR = (0.13, 0.55, 0.13, 0.65)   # Green with transparency — AEA Left
AEAR_COLOUR = (0.94, 0.50, 0.13, 0.65)   # Orange with transparency — AEA Right
BG_COLOUR   = "#1a1a2e"                   # Dark background for the UI theme


# ── Slice renderer ─────────────────────────────────────────────────────────────

def render_slice(
    volume      : np.ndarray,
    mask        : np.ndarray,
    slice_idx   : int,
    plane       : str = "axial",
    show_overlay: bool = True,
) -> Image.Image:
    """
    Render a single 2D slice from a 3D volume with optional mask overlay.

    Args:
        volume:       3D float32 array (Z, Y, X) — normalised HU [0,1].
        mask:         3D int array (Z, Y, X) — labels {0, 1, 2}.
        slice_idx:    Index of the slice to render.
        plane:        "axial" (Z), "coronal" (Y), or "sagittal" (X).
        show_overlay: Whether to draw the segmentation overlay.

    Returns:
        PIL Image of the rendered slice.
    """
    # Extract 2D slice based on anatomical plane
    if plane == "axial":
        slice_idx = int(np.clip(slice_idx, 0, volume.shape[0] - 1))
        img_2d  = volume[slice_idx, :, :]
        mask_2d = mask[slice_idx,   :, :]
    elif plane == "coronal":
        slice_idx = int(np.clip(slice_idx, 0, volume.shape[1] - 1))
        img_2d  = volume[:, slice_idx, :]
        mask_2d = mask[:,   slice_idx, :]
    else:  # sagittal
        slice_idx = int(np.clip(slice_idx, 0, volume.shape[2] - 1))
        img_2d  = volume[:, :, slice_idx]
        mask_2d = mask[:,   :, slice_idx]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Base CBCT image (grayscale)
    ax.imshow(img_2d, cmap="gray", vmin=0.05, vmax=0.95, aspect="equal")

    if show_overlay and mask_2d.max() > 0:
        # AEA Left (label 1) — green
        aeal_mask = np.ma.masked_where(mask_2d != 1, np.ones_like(mask_2d))
        ax.imshow(aeal_mask, cmap=None, alpha=0.0)  # placeholder for legend
        aeal_rgba = np.zeros((*mask_2d.shape, 4))
        aeal_rgba[mask_2d == 1] = AEAL_COLOUR
        ax.imshow(aeal_rgba, aspect="equal")

        # AEA Right (label 2) — orange
        aear_rgba = np.zeros((*mask_2d.shape, 4))
        aear_rgba[mask_2d == 2] = AEAR_COLOUR
        ax.imshow(aear_rgba, aspect="equal")

        # Legend
        legend_patches = [
            mpatches.Patch(color=AEAL_COLOUR[:3], label="AEA Left"),
            mpatches.Patch(color=AEAR_COLOUR[:3], label="AEA Right"),
        ]
        ax.legend(
            handles   = legend_patches,
            loc       = "lower right",
            fontsize  = 9,
            facecolor = "black",
            edgecolor = "white",
            labelcolor= "white",
        )

    ax.set_title(
        f"{plane.capitalize()} — Slice {slice_idx}",
        color="white", fontsize=11, pad=6,
    )
    ax.axis("off")
    plt.tight_layout(pad=0.2)

    # Convert matplotlib figure to PIL Image
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf  = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    plt.close(fig)
    return Image.fromarray(buf)


def render_placeholder(message: str = "Upload a CBCT scan\nand click Run Segmentation") -> Image.Image:
    """Return a placeholder dark image with a given message."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("#111111")
    ax.text(
        0.5, 0.5,
        message,
        ha="center", va="center",
        color="#555555", fontsize=13,
        transform=ax.transAxes,
    )
    ax.axis("off")
    plt.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf  = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    plt.close(fig)
    return Image.fromarray(buf)


def render_waiting() -> Image.Image:
    """Return a placeholder image shown while the agent is running."""
    return render_placeholder("⏳ Please wait for the agent\nto finish running the\nsegmentation pipeline...")


# ── File handling ──────────────────────────────────────────────────────────────

def extract_upload(uploaded_file_path: str, tmp_dir: Path) -> tuple[str | None, str | None]:
    """
    Handle uploaded zip file: extract and find DICOM folder and NRRD mask.

    Returns:
        (dicom_dir_path, nrrd_path) — either may be None if not found.
    """
    dicom_dir = None
    nrrd_path = None

    uploaded = Path(uploaded_file_path)

    if uploaded.suffix.lower() == ".zip":
        extract_to = tmp_dir / "extracted"
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(uploaded, "r") as z:
            z.extractall(extract_to)
        search_root = extract_to
    else:
        # Assume it's already a folder path
        search_root = uploaded if uploaded.is_dir() else uploaded.parent

    # Find DICOM folder (contains .dcm files)
    dcm_files = list(search_root.rglob("*.dcm"))
    if dcm_files:
        dicom_dir = str(dcm_files[0].parent)

    # Find NRRD mask
    nrrd_files = list(search_root.rglob("*.nrrd"))
    if nrrd_files:
        nrrd_path = str(nrrd_files[0])

    return dicom_dir, nrrd_path


# ── Agent runner ───────────────────────────────────────────────────────────────

# Global state — holds the last pipeline result for the slice viewer
_pipeline_state: dict = {
    "volume"     : None,
    "mask"       : None,
    "session_id" : None,
    "report"     : None,
    "steps"      : [],
}


def run_pipeline_ui(
    dicom_zip_path  : str,
    patient_id      : str,
    instruction     : str,
    nrrd_path       : str,
    use_gt          : bool,
) -> tuple:
    """
    Called when the user clicks the Run button.

    Returns a tuple of Gradio component updates:
        (status_msg, axial_img, coronal_img, sagittal_img,
         axial_max, coronal_max, sagittal_max,
         metrics_text, report_text, agent_log_text, report_json_path)
    """
    global _pipeline_state

    # ── Validate inputs ────────────────────────────────────────────────────────
    if not dicom_zip_path:
        return _error_return("Please upload a DICOM zip file or provide a path.")

    if not patient_id.strip():
        patient_id = "unknown"

    if not instruction.strip():
        instruction = "Segment the anterior ethmoidal artery and generate a clinical report."

    # ── Setup temp directory ───────────────────────────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="aea_ui_"))

    try:
        # ── Extract uploaded file ──────────────────────────────────────────────
        status = "⏳ Extracting uploaded files..."
        yield _progress_return(status)

        dicom_dir, auto_nrrd = extract_upload(dicom_zip_path, tmp_dir)

        if not dicom_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            yield _error_return("No DICOM (.dcm) files found in the uploaded archive.")
            return

        gt_path = None
        if use_gt:
            gt_path = nrrd_path.strip() if nrrd_path and nrrd_path.strip() else auto_nrrd

        logger.info(f"DICOM dir: {dicom_dir}")
        logger.info(f"GT NRRD:   {gt_path}")

        # ── Run the agent ──────────────────────────────────────────────────────
        status = "🤖 Agent is running the segmentation pipeline..."
        yield _progress_return(status)

        from src.agent.agent import AEAAgent
        agent  = AEAAgent()
        result = agent.run(
            instruction  = instruction,
            dicom_path   = dicom_dir,
            patient_id   = patient_id.strip(),
            gt_nrrd_path = gt_path,
        )

        if not result["success"]:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            yield _error_return(f"Pipeline error: {result['output']}")
            return

        # ── Retrieve results from session store ────────────────────────────────
        session_id = result["session_id"]

        # Detailed diagnostics
        logger.info(f"Agent result success: {result['success']}")
        logger.info(f"Agent output: {result['output'][:300]}")
        logger.info(f"session_id from agent: {session_id}")
        logger.info(f"All session IDs in store: {list(SESSION_STORE.keys())}")
        logger.info(f"Agent steps ({len(result.get('steps', []))}):")
        for s in result.get("steps", []):
            logger.info(f"  tool={s['tool']} | input={s['tool_input'][:80]} | obs={s['observation'][:120]}")

        # Fallback: if regex didn't capture session_id, use the last entry in SESSION_STORE
        if session_id is None and SESSION_STORE:
            session_id = list(SESSION_STORE.keys())[-1]
            logger.warning(f"session_id not captured from agent output — using last session: {session_id}")

        session    = SESSION_STORE.get(session_id, {})
        logger.info(f"Session keys for '{session_id}': {list(session.keys())}")

        mask       = session.get("clean_mask", session.get("raw_mask"))
        sitk_img   = session.get("sitk_image")
        report     = session.get("report", {})
        steps      = result.get("steps", [])

        logger.info(f"mask is None: {mask is None}")
        logger.info(f"sitk_img is None: {sitk_img is None}")

        if mask is None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            yield _error_return("Segmentation did not produce a mask. Check logs.")
            return

        # ── Reconstruct normalised volume for display ──────────────────────────
        import SimpleITK as _sitk
        volume_np = _sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        volume_np = np.clip(volume_np, HU_MIN, HU_MAX)
        volume_np = (volume_np - HU_MIN) / (HU_MAX - HU_MIN)  # Normalise to [0, 1]

        # Store in global state for slider callbacks
        _pipeline_state.update({
            "volume"    : volume_np,
            "mask"      : mask,
            "session_id": session_id,
            "report"    : report,
            "steps"     : steps,
        })

        # ── Render initial slices ──────────────────────────────────────────────
        # Find the best slice (middle of AEA extent) for each plane
        aea_voxels = np.argwhere(mask > 0)
        if len(aea_voxels) > 0:
            ax_mid  = int(np.median(aea_voxels[:, 0]))
            cor_mid = int(np.median(aea_voxels[:, 1]))
            sag_mid = int(np.median(aea_voxels[:, 2]))
        else:
            ax_mid  = volume_np.shape[0] // 2
            cor_mid = volume_np.shape[1] // 2
            sag_mid = volume_np.shape[2] // 2

        axial_img    = render_slice(volume_np, mask, ax_mid,  "axial")
        coronal_img  = render_slice(volume_np, mask, cor_mid, "coronal")
        sagittal_img = render_slice(volume_np, mask, sag_mid, "sagittal")

        # ── Build metrics text ─────────────────────────────────────────────────
        metrics_text = _format_metrics(session.get("metrics"))

        # ── Build agent log ────────────────────────────────────────────────────
        agent_log = _format_agent_log(steps)

        # ── Report text ────────────────────────────────────────────────────────
        from src.report import report_to_text
        report_text = report_to_text(report) if report else result["output"]

        # ── Save report JSON for download ──────────────────────────────────────
        # Save report to LOGS_DIR (not tmp_dir which gets deleted immediately)
        report_json_path = None
        if report:
            report_json_path = str(LOGS_DIR / f"report_{patient_id}.json")
            with open(report_json_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved → {report_json_path}")

        shutil.rmtree(tmp_dir, ignore_errors=True)

        yield (
            "✅ Segmentation complete!",
            axial_img, coronal_img, sagittal_img,
            gr.Slider(maximum=volume_np.shape[0] - 1, value=ax_mid),
            gr.Slider(maximum=volume_np.shape[1] - 1, value=cor_mid),
            gr.Slider(maximum=volume_np.shape[2] - 1, value=sag_mid),
            metrics_text,
            report_text,
            agent_log,
            report_json_path,
            gr.Markdown(visible=False),
        )

    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.error(f"UI pipeline error: {e}")
        yield _error_return(f"Unexpected error: {str(e)}")


def _error_return(msg: str) -> tuple:
    """Return a uniform error state for all Gradio outputs."""
    placeholder = render_placeholder()
    return (
        f"❌ {msg}",
        placeholder, placeholder, placeholder,
        gr.Slider(value=0), gr.Slider(value=0), gr.Slider(value=0),
        "No metrics available.", msg, "No agent log.", None,
        gr.Markdown(visible=False),
    )


def _progress_return(msg: str) -> tuple:
    """Return a progress state (used with yield for streaming updates)."""
    waiting = render_waiting()
    return (
        msg,
        waiting, waiting, waiting,
        gr.Slider(value=0), gr.Slider(value=0), gr.Slider(value=0),
        "", "", "", None,
        gr.Markdown(visible=True),
    )


# ── Slider callbacks ───────────────────────────────────────────────────────────

def update_axial(slice_idx: int) -> Image.Image:
    v, m = _pipeline_state["volume"], _pipeline_state["mask"]
    if v is None:
        return render_placeholder()
    return render_slice(v, m, int(slice_idx), "axial")


def update_coronal(slice_idx: int) -> Image.Image:
    v, m = _pipeline_state["volume"], _pipeline_state["mask"]
    if v is None:
        return render_placeholder()
    return render_slice(v, m, int(slice_idx), "coronal")


def update_sagittal(slice_idx: int) -> Image.Image:
    v, m = _pipeline_state["volume"], _pipeline_state["mask"]
    if v is None:
        return render_placeholder()
    return render_slice(v, m, int(slice_idx), "sagittal")


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _format_metrics(metrics: Optional[dict]) -> str:
    """Format metrics dict into a readable string for the UI panel."""
    if not metrics:
        return (
            "No ground truth provided.\n"
            "Upload a .nrrd mask alongside the DICOM\n"
            "and enable 'Use ground truth for evaluation'\n"
            "to compute Dice, IoU and HD95."
        )
    lines = [
        "SEGMENTATION QUALITY METRICS",
        "─" * 34,
        f"{'Metric':<16} {'Left':>8} {'Right':>8} {'Mean':>8}",
        f"{'─'*34}",
        f"{'Dice (DSC)':<16} {metrics.get('dice_aeal', float('nan')):>8.4f} "
        f"{metrics.get('dice_aear', float('nan')):>8.4f} "
        f"{metrics.get('dice_mean', float('nan')):>8.4f}",
        f"{'IoU':<16} {metrics.get('iou_aeal', float('nan')):>8.4f} "
        f"{metrics.get('iou_aear', float('nan')):>8.4f} "
        f"{metrics.get('iou_mean', float('nan')):>8.4f}",
        f"{'HD95 (mm)':<16} {metrics.get('hd95_aeal', float('nan')):>8.4f} "
        f"{metrics.get('hd95_aear', float('nan')):>8.4f} "
        f"{metrics.get('hd95_mean', float('nan')):>8.4f}",
        "─" * 34,
        "",
        "Thresholds: Dice ≥ 0.80 → excellent",
        "            HD95 ≤ 2 mm → clinically acceptable",
    ]
    return "\n".join(lines)


def _format_agent_log(steps: list) -> str:
    """Format the agent's reasoning trace for display."""
    if not steps:
        return "No agent steps recorded."

    lines = ["AGENT REASONING TRACE", "=" * 50]
    for i, step in enumerate(steps, 1):
        lines.append(f"\nStep {i}: {step['tool']}")
        lines.append(f"  Input  : {step['tool_input'][:120]}...")
        obs = step["observation"]
        obs_preview = obs[:300] + "..." if len(obs) > 300 else obs
        lines.append(f"  Result : {obs_preview}")
        lines.append("  " + "─" * 46)

    return "\n".join(lines)


# ── Gradio UI definition ───────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks dashboard."""

    custom_css = """
    .panel-header  { font-size: 15px; font-weight: 600; color: #e2e8f0; margin-bottom: 8px; }
    .metric-box    { font-family: monospace; font-size: 13px; }
    .status-bar    { font-size: 14px; font-weight: 500; }
    .divider       { border: none; border-top: 3px solid rgba(59,130,246,0.5); margin: 10px 0; }
    .section-title { font-size: 15px; font-weight: 700; text-transform: uppercase;
                     letter-spacing: 0.05em; border-bottom: 3px solid rgba(59,130,246,0.5);
                     padding-bottom: 4px; margin-bottom: 8px; }
    footer         { display: none !important; }
    """

    with gr.Blocks(
        title   = "AEA Segmentation — Preoperative Planning Tool",
        theme   = gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css     = custom_css,
    ) as demo:

        # ── Header ─────────────────────────────────────────────────────────────
        gr.Markdown("# Anterior Ethmoidal Artery Segmentation")
        gr.Markdown(
            "**AI-assisted preoperative localization of the AEA on CBCT scans.** "
            "Upload a CBCT scan, type a natural language instruction, and the AI agent "
            "will automatically segment both the left and right anterior ethmoidal arteries."
        )

        # ── Status bar ─────────────────────────────────────────────────────────
        status_bar = gr.Textbox(
            value       = "Ready. Upload a CBCT scan to begin.",
            label       = "Status",
            interactive = False,
            elem_classes= ["status-bar"],
        )
        gr.HTML("<hr style='border:none;border-top:3px solid rgba(59,130,246,0.5);margin:6px 0;'>")

        # ── Main 3-column layout ───────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── LEFT: Input panel ──────────────────────────────────────────────
            with gr.Column(scale=1, min_width=280):
                gr.HTML("<div class='section-title'>Input</div>")

                dicom_upload = gr.File(
                    label       = "CBCT Scan (ZIP containing DICOM folder)",
                    file_types  = [".zip"],
                    file_count  = "single",
                )

                patient_id_box = gr.Textbox(
                    label       = "Patient ID",
                    placeholder = "e.g. NL001",
                    value       = "",
                )

                instruction_box = gr.Textbox(
                    label       = "Natural language instruction",
                    placeholder = "e.g. Segment the anterior ethmoidal artery and give me a report.",
                    value       = "Segment the anterior ethmoidal artery on both sides and generate a preoperative report.",
                    lines       = 3,
                )

                with gr.Accordion("Ground truth evaluation (optional)", open=False):
                    use_gt_checkbox = gr.Checkbox(
                        label = "Use ground truth NRRD for metric evaluation",
                        value = False,
                    )
                    nrrd_path_box = gr.Textbox(
                        label       = "Path to .nrrd mask file",
                        placeholder = "/path/to/NL001.nrrd",
                        value       = "",
                    )
                    gr.Markdown(
                        "_The NRRD file is used to compute Dice, IoU and HD95. "
                        "Leave empty to auto-detect from the ZIP._"
                    )

                run_btn = gr.Button(
                    "Run Segmentation",
                    variant = "primary",
                    size    = "lg",
                )

                gr.HTML("<hr style='border:none;border-top:3px solid rgba(59,130,246,0.5);margin:10px 0;'>")
                gr.Markdown(
                    "**Model:** SwinUNETR (fine-tuned)  \n"
                    "**Agent:** Llama 3.1 8B via Ollama  \n"
                    "**Dataset:** 130 CBCT cases"
                )

            # ── Vertical divider ───────────────────────────────────────────────
            gr.HTML("<div style='border-left:3px solid rgba(59,130,246,0.5);min-height:600px;margin:0 8px;'></div>")

            # ── CENTRE: Slice viewer ───────────────────────────────────────────
            with gr.Column(scale=3, min_width=500):
                gr.HTML("<div class='section-title'>Slice Viewer</div>")
                gr.Markdown(
                    "_Green = AEA Left · Orange = AEA Right · "
                    "Use sliders to navigate slices_"
                )

                viewer_status = gr.Markdown(
                    value   = "",
                    visible = False,
                )

                with gr.Tabs():
                    with gr.Tab("Axial"):
                        axial_img    = gr.Image(
                            value   = render_placeholder(),
                            label   = "Axial view",
                            type    = "pil",
                            height  = 420,
                        )
                        axial_slider = gr.Slider(
                            minimum = 0, maximum = 300, value = 150, step = 1,
                            label   = "Axial slice (Z)",
                        )

                    with gr.Tab("Coronal"):
                        coronal_img    = gr.Image(
                            value   = render_placeholder(),
                            label   = "Coronal view",
                            type    = "pil",
                            height  = 420,
                        )
                        coronal_slider = gr.Slider(
                            minimum = 0, maximum = 300, value = 150, step = 1,
                            label   = "Coronal slice (Y)",
                        )

                    with gr.Tab("Sagittal"):
                        sagittal_img    = gr.Image(
                            value   = render_placeholder(),
                            label   = "Sagittal view",
                            type    = "pil",
                            height  = 420,
                        )
                        sagittal_slider = gr.Slider(
                            minimum = 0, maximum = 300, value = 150, step = 1,
                            label   = "Sagittal slice (X)",
                        )

            # ── Vertical divider ───────────────────────────────────────────────
            gr.HTML("<div style='border-left:3px solid rgba(59,130,246,0.5);min-height:600px;margin:0 8px;'></div>")

            # ── RIGHT: Results panel ───────────────────────────────────────────
            with gr.Column(scale=2, min_width=300):
                gr.HTML("<div class='section-title'>Results</div>")

                metrics_box = gr.Textbox(
                    label       = "Segmentation Metrics",
                    value       = "Metrics will appear here after segmentation.",
                    lines       = 12,
                    interactive = False,
                    elem_classes= ["metric-box"],
                )

                report_box = gr.Textbox(
                    label       = "Clinical Report",
                    value       = "Report will appear here after segmentation.",
                    lines       = 18,
                    interactive = False,
                    elem_classes= ["metric-box"],
                )

                report_download = gr.File(
                    label       = "Download Report (JSON)",
                    interactive = False,
                )

        # ── Bottom divider + Agent reasoning log ───────────────────────────────
        gr.HTML("<hr style='border:none;border-top:3px solid rgba(59,130,246,0.5);margin:10px 0;'>")
        with gr.Accordion("Agent Reasoning Log", open=False):
            gr.Markdown(
                "This log shows the step-by-step reasoning of the AI agent: "
                "which tools it called, in what order, and what each tool returned."
            )
            agent_log_box = gr.Textbox(
                label       = "Agent Trace (Thought → Action → Observation)",
                value       = "Agent log will appear here after segmentation.",
                lines       = 15,
                interactive = False,
                elem_classes= ["metric-box"],
            )

        # ── Event handlers ─────────────────────────────────────────────────────

        # Show waiting image immediately when Run is clicked (before queue picks it up)
        run_btn.click(
            fn      = lambda: (render_waiting(), render_waiting(), render_waiting()),
            inputs  = [],
            outputs = [axial_img, coronal_img, sagittal_img],
            queue   = False,
        )

        # Run button → full pipeline
        run_btn.click(
            fn      = run_pipeline_ui,
            inputs  = [
                dicom_upload, patient_id_box, instruction_box,
                nrrd_path_box, use_gt_checkbox,
            ],
            outputs = [
                status_bar,
                axial_img, coronal_img, sagittal_img,
                axial_slider, coronal_slider, sagittal_slider,
                metrics_box, report_box, agent_log_box, report_download,
                viewer_status,
            ],
            queue   = True,
        )

        # Slice sliders → update individual views
        axial_slider.change(
            fn      = update_axial,
            inputs  = [axial_slider],
            outputs = [axial_img],
        )
        coronal_slider.change(
            fn      = update_coronal,
            inputs  = [coronal_slider],
            outputs = [coronal_img],
        )
        sagittal_slider.change(
            fn      = update_sagittal,
            inputs  = [sagittal_slider],
            outputs = [sagittal_img],
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

def launch(share: bool = False, port: int = 7860) -> None:
    """Launch the Gradio dashboard."""
    demo = build_ui()
    logger.info(f"Launching AEA Segmentation dashboard on port {port}...")
    demo.queue()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = port,
        share       = share,
        show_error  = True,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AEA Segmentation Web UI")
    parser.add_argument("--port",  type=int,  default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    args = parser.parse_args()
    launch(share=args.share, port=args.port)
