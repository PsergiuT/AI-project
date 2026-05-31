# Anterior Ethmoidal Artery Segmentation on CBCT

> AI-assisted preoperative localization of the AEA using a Vision Transformer agent

---

## Overview

This project automatically segments the **Anterior Ethmoidal Artery (AEA)** on
Cone Beam CT (CBCT) scans using a fine-tuned **SwinUNETR** (Swin Transformer U-Net)
orchestrated by a **LangChain ReAct agent** (Llama 3.1 8B via Ollama).

The system reduces the risk of AEA injury during endoscopic nasal surgery by providing
precise preoperative localization of both the left and right AEA through a simple web UI.

---

## Project Structure

```
aea-segmentation/
├── config.py                  ← All paths and hyperparameters
├── requirements.txt           ← Python dependencies
├── run.py                     ← Entry point — launch the web UI
│
├── src/
│   ├── preprocessing.py       ← DICOM + NRRD → NIfTI conversion
│   ├── dataset.py             ← MONAI Dataset, DataLoaders, transforms
│   ├── train.py               ← SwinUNETR fine-tuning training loop
│   ├── evaluate.py            ← Dice, IoU, HD95 metric computation
│   ├── postprocess.py         ← Connected component cleanup
│   ├── report.py              ← Structured JSON + text report generation
│   └── agent/
│       ├── tools.py           ← LangChain tool definitions (5 tools)
│       └── agent.py           ← ReAct agent setup and orchestration
│
├── ui/
│   └── app.py                 ← Gradio dashboard (3-column layout)
│
├── notebooks/
│   └── training.ipynb         ← Google Colab training notebook
│
├── data/
│   ├── processed/             ← NIfTI files after preprocessing
│   └── splits/                ← train.json / val.json / test.json
│
└── models/
    └── swinunetr/
        └── swinunetr_best.pth ← Trained model checkpoint (after training)
```

---

## Setup

### 1. Install Python dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install PyTorch with CUDA support (check https://pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Install Ollama (for the LLM agent)

1. Download and install Ollama from **https://ollama.com**
2. Start the Ollama server in a terminal:
   ```bash
   ollama serve
   ```
3. Pull the Llama 3.1 model (one-time download, ~5 GB):
   ```bash
   ollama pull llama3.1
   ```

### 3. Preprocess the dataset (run once)

```bash
python src/preprocessing.py --data_root /path/to/dateArteraEtimoidala
```

This converts all 130 CBCT cases from DICOM+NRRD to NIfTI format and
creates the train/val/test splits in `data/splits/`.

### 4. Train the model (Google Colab recommended)

Open `notebooks/training.ipynb` in Google Colab:
- Go to **Runtime → Change runtime type → GPU (T4)**
- Follow the 10 steps in the notebook
- Training takes ~2–4 hours on a free T4 GPU
- Download `models/swinunetr/swinunetr_best.pth` when done

### 5. Launch the web UI

```bash
python run.py
```

Open your browser at **http://localhost:7860**

For a public demo link (e.g. for presentation):
```bash
python run.py --share
```

---

## Usage

1. **Upload** a CBCT scan as a ZIP file containing the DICOM folder
2. **Enter** the patient ID and a natural language instruction
3. **Click** "Run Segmentation"
4. **View** the segmentation overlay in the axial, coronal, and sagittal views
5. **Navigate** slices using the sliders
6. **Download** the structured JSON report

### Example instructions

- *"Segment the anterior ethmoidal artery and generate a clinical report."*
- *"Detect and segment both AEA sides, evaluate against the ground truth, and give me the metrics."*
- *"Run the full segmentation pipeline and tell me if the AEA is present bilaterally."*

---

## AI Architecture

| Component | Technology |
|---|---|
| Segmentation model | SwinUNETR (Swin Transformer + U-Net decoder) |
| Pre-trained weights | MONAI self-supervised pre-training (5000 epochs) |
| Training framework | MONAI 1.3 + PyTorch 2.1 |
| Agent LLM | Llama 3.1 8B (local, via Ollama) |
| Agent framework | LangChain ReAct pattern |
| Web UI | Gradio 4.x (Blocks API) |
| Medical I/O | SimpleITK, pydicom, pynrrd |

---

## Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| Dice (DSC) | Volumetric overlap (0–1) | ≥ 0.70 |
| IoU (Jaccard) | Intersection over union (0–1) | ≥ 0.55 |
| HD95 | 95th percentile boundary distance (mm) | ≤ 2 mm |

---

## SDGs Addressed

- **SDG 3 — Good Health and Well-being:** Reducing surgical complications from AEA injury
  during endoscopic nasal surgery through AI-assisted preoperative planning.
- **SDG 9 — Industry, Innovation and Infrastructure:** Developing an automated
  AI pipeline that removes the manual segmentation bottleneck (15–30 min per case)
  and makes precision surgical planning scalable and accessible.

---

## Bibliography

1. Huang et al. (2020). *An AI algorithm that differentiates anterior ethmoidal artery location on sinus CT scans.* Journal of Laryngology & Otology.
2. Amarnath & Suresh Kumar (2019). *Study of variants of anterior ethmoidal artery on CT of paranasal sinuses.* Int J Otorhinolaryngol Head Neck Surg.
3. Itayem et al. (2019). *Increased accuracy, confidence, and efficiency in AEA identification with segmented image guidance.* Otolaryngology–Head and Neck Surgery.
4. Tang et al. (2022). *Self-supervised pre-training of Swin Transformers for 3D medical image analysis.* CVPR.

