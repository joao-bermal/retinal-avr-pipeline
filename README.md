<h1 align="center">
  👁️ Retinal AVR Cardiovascular Risk Analysis Pipeline
</h1>

<p align="center">
  Deep-learning pipeline for retinal vessel segmentation, artery/vein classification, and
  scientifically correct AVR (Arteriolar-to-Venular Ratio) calculation — originally built
  for an MBA thesis (TCC) on non-invasive cardiovascular risk screening.
</p>

---

## 📌 Introduction

This is a modular Python codebase for computing the AVR (Arteriolar-to-Venular Ratio) from
retinal fundus photographs. It started as a set of scattered Jupyter notebooks (see
`docs/PROJECT_HISTORY.md`) and was consolidated here into a scalable, trackable pipeline.

The pipeline has four stages:
1. **Vessel segmentation** (`EnhancedUNet`): isolates the vascular tree from the fundus
   image. Dice 0.7942 on DRIVE (thesis target: 0.7965).
2. **A/V classification** (`EnhancedMultiDatasetAVNet`, ResNet-50 backbone): classifies
   segmented vessels as arteries or veins. Macro F1 0.9577 (thesis target: 0.78).
3. **Optic disc detection** (`src/pipeline/optic_disc.py`): locates the optic disc, hybrid
   between a trained model and a classical computer-vision fallback. See "Optic disc fix"
   below — this is the part that was scientifically wrong in the original thesis and has
   since been corrected.
4. **Scientific AVR calculation** (`ScientificAVRCalculator`): measures vessel calibers
   inside the peripapillary Zone B (Knudtson protocol) and applies the CRAE/CRVE formulas.

Full metrics, evidence plots, and the exact GPU/ROCm setup used to train these models are
in [`docs/`](docs/) — see [`docs/METRICS.md`](docs/METRICS.md) and
[`docs/GPU_ROCM_RX6800XT.md`](docs/GPU_ROCM_RX6800XT.md).

### Optic disc fix

> The originally submitted thesis assumed the **geometric center of the image** as the
> optic disc position when building the peripapillary Zone B — a simplification explicitly
> documented as a limitation in the thesis's "future work" section. This has been fixed:
> `src/pipeline/optic_disc.py` now does real optic disc detection, hybrid between a trained
> model (falls back gracefully outside its training domain) and a classical CV heuristic —
> never silently assuming the image center. See `docs/METRICS.md` for the before/after
> numbers (center-detection error dropped from ~328px to ~11px median).

---

## 🛠️ Codebase architecture

```text
├── api/                   # FastAPI HTTP layer over the pipeline (POST /analyze, GET /health)
├── data/                  # DRIVE, IOSTAR, RITE, LES-AV datasets (see EXECUTION_GUIDE.md)
├── docs/                  # Metrics, GPU/ROCm notes, project history, thesis (PT + EN)
├── EXECUTION_GUIDE.md     # 🚀 Step-by-step guide to run everything from zero
├── frontend/              # Next.js web UI calling the API
├── main.py                # 🚪 CLI entrypoint (training + inference)
├── notebooks/
│   └── 00_Unified_Master_Pipeline.ipynb  # Final notebook with plots & interactive inference
├── src/
│   ├── config/            # Global settings and hyperparameters (one block per model)
│   ├── data/               # PyTorch Dataset classes and preprocessing
│   ├── models/             # Neural architectures: U-Net, MultiDatasetAVNet
│   ├── pipeline/           # ScientificAVRPipeline (end-to-end), avr_calculator.py
│   │                       # (Knudtson/Zone B) and optic_disc.py (hybrid disc detection)
│   └── training/           # Training loop (epochs, validation metrics, checkpointing)
└── tests/
    ├── sanity_check.py         # Quick architecture smoke test (forward pass, no dataset)
    └── test_avr_calculator.py  # Zone B geometry and Knudtson formula unit tests
```

---

## ⚙️ Installation

```bash
git clone https://github.com/joao-bermal/retinal-avr-pipeline.git
cd retinal-avr-pipeline

# NVIDIA GPUs (CUDA):
pip install -r requirements.txt

# AMD GPUs (ROCm) — see docs/GPU_ROCM_RX6800XT.md for a full setup/troubleshooting log:
pip install -r requirements-rocm.txt

# Optional: HTTP API layer (on top of either of the above)
pip install -r requirements-api.txt
```

---

## 🚀 Usage

Read **[EXECUTION_GUIDE.md](./EXECUTION_GUIDE.md)** for the full step-by-step walkthrough.

**CLI summary (`main.py`)**:

- Train the vessel segmentation U-Net:
  `python main.py --train_seg`
- Train the A/V classification network:
  `python main.py --train_av`
- Train the optic disc detector (peripapillary Zone B):
  `python main.py --train_od`
- Run the full pipeline on one image (segmentation → A/V → optic disc → AVR/risk):
  `python main.py --run_pipeline "data/drive/test/images/my_image.tif"`
- Override training hyperparameters:
  `python main.py --train_seg --epochs 200 --lr 0.001`

**HTTP API**:
```bash
pip install -r requirements-api.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
curl -F "image=@data/DRIVE/test/images/01_test.tif" http://localhost:8000/analyze
```

**Web frontend** (Next.js, calls the API above):
```bash
cd frontend
npm install
npm run dev
```

For in-depth medical visualizations and iterative reports, the master notebook is still
available:
> `notebooks/00_Unified_Master_Pipeline.ipynb`

---

## 📚 Project history and legacy folders

This repository (`retinal-avr-pipeline`) is the **active source of truth**. During thesis
development, GPU environment constraints (ROCm only works on Linux) forced work across
several local folders and notebook copies outside this repository. They still exist as
historical reference but are no longer maintained — any useful logic found there (like the
scientific AVR calculator) has been ported into `src/`. See
[`docs/PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md) for the full timeline.
