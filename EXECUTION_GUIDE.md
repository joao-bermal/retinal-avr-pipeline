# 🚀 End-to-End Execution Guide (From Zero to Pipeline)

This guide covers the full path from **Step 0** (no trained model at all) to generating
*metrics, inference, and integrated predictions*. The whole architecture is modularized in
native Python (`src/`) to allow terminal/MLOps automation, with visualizations centralized
under `notebooks/`.

---

## 📦 Step 0: Initial Setup

### 1. Environment and dependencies
Make sure you're inside an active virtual environment (venv or conda) with the essential
libraries installed, PyTorch in particular.

```bash
# NVIDIA GPU (CUDA):
pip install -r requirements.txt

# AMD GPU (ROCm):
pip install -r requirements-rocm.txt
```

*(PyTorch/torchvision have hardware-specific builds — use the files above to pull the
correct wheels for your GPU. If you're on an AMD card, read
[`docs/GPU_ROCM_RX6800XT.md`](docs/GPU_ROCM_RX6800XT.md) first — it documents a real,
reproducible GPU thermal-runaway issue and its fix, plus the exact driver setup that works.)*

### 2. Dataset layout
Images are >130MB, so they don't live on GitHub. Download the author's `data.zip`:
1. **Get `data.zip` from:** [Google Drive - Retinal AVR Data](https://drive.google.com/file/d/1VUNfJkRd8V9RmR--NlnI_RZg84Ioz-E8/view?usp=sharing)
2. **Extract it** directly into the project root, so `data/` is recreated with the
   subfolders the system expects.

Resulting structure for segmentation training on DRIVE:

```text
data/
└── drive/
    ├── training/
    │   ├── images/
    │   └── 1st_manual/
    └── test/
        ├── images/
        └── 1st_manual/
```

*(Do the same for the A/V classification datasets — IOSTAR, RITE, LES-AV.)*

### 3. Original IOSTAR images (needed for the optic disc model)

The `data.zip` above only includes IOSTAR's `GT/`, `AV_GT/`, `mask/`, and `mask_OD/` —
missing the 30 original fundus images (`image/*.jpg`), which is the only downloaded source
with optic disc mask ground truth (`mask_OD/`). Without them, `--train_od` and optic disc
detector evaluation won't work. Place them in `data/IOSTAR/image/`.

---

## 🏋️‍♂️ Step 1: Model Training

All training goes through the central entrypoint, `main.py`. It drives the training
functions (`src/training/`) through the full split/loading/logging flow, using the
hyperparameters in `src/config/settings.py`.

### A) Train the segmentation model (Enhanced U-Net)
To train the network responsible for extracting the vascular tree (segmentation):

```bash
python main.py --train_seg
```
- **What happens**: the dataloader processes `*.tif` images with geometric/color
  transforms (CLAHE), feeds `EnhancedUNet`, and saves the checkpoint under
  `models/segmentation/run_<timestamp>/model_epoch<NNN>_dice_<score>.pth`.
- **Logs/metrics**: loss and metric evolution (Dice, epochs, etc.) print every round in the
  terminal, and are also written to `logs/segmentation/run_<timestamp>/metrics_*.csv`.

### B) Train the A/V classification model
To train the more complex A/V architecture:

```bash
python main.py --train_av
```
- **What happens**: uses the datasets configured in `settings.py` to feed
  `EnhancedMultiDatasetAVNet`.
- Final weights saved under `models/av_classification/run_<timestamp>/`.

### C) Train the optic disc detector

Needed for correctly computing the peripapillary Zone B (see "Optic disc fix" below).
Small dataset (~30 IOSTAR images), so training is fast even without a top-tier GPU:

```bash
python main.py --train_od
```
- **What happens**: reuses the `EnhancedUNet` architecture (same as vessel segmentation,
  just a different checkpoint) to segment the optic disc from `data/IOSTAR/image/` +
  `data/IOSTAR/mask_OD/`.
- Final weights saved under `models/optic_disc/run_<timestamp>/model_epoch<NNN>_dice_<score>.pth`.
- **Without a trained checkpoint**, the pipeline automatically falls back to a classical
  computer-vision heuristic (brightest region in the red channel + largest connected
  component) — it works, but is meaningfully less accurate, especially on IOSTAR (SLO
  images, quite different from the consumer fundus photos the heuristic was designed for).
- ⚠️ **If you're on an AMD GPU**, read
  [`docs/GPU_ROCM_RX6800XT.md`](docs/GPU_ROCM_RX6800XT.md) before running this specific
  command — this is the training workload that reproduced a real GPU thermal-runaway
  issue during development, and the doc has the pre-flight checklist that avoids it.

### D) Dynamic hyperparameter overrides (optional)
No need to edit `settings.py` by hand for routine changes — use CLI flags:

```bash
python main.py --train_seg --epochs 250 --batch_size 8 --lr 0.0001
```

Optionally, create an `experimento_1.yaml` file to override more complex `settings.py`
variables and run:
```bash
python main.py --train_av --config experimento_1.yaml
```

---

## 📊 Step 2: Sanity Checks

To verify the forward/backward pass ("dry run") with mocked tensors — checking machine/GPU
memory syntax without loading the full dataset:

```bash
python tests/sanity_check.py
python tests/test_avr_calculator.py   # Zone B geometry + Knudtson formulas, no GPU needed
```

This isolated script confirms the PyTorch model architectures (`src/models/*`) compile
correctly against zero/one-filled tensors before committing hours of real training.

---

## 🧠 Step 3: Evaluation and the Integrated Pipeline (Inference & Plots)

Once `models/` holds trained `.pth` checkpoints, the pipeline can run real-world
predictions, overlay them, and plot the results.

### CLI — single image
```bash
python main.py --run_pipeline "data/drive/test/images/01_test.tif"
```
The result includes `avr`, `crae`, `crve`, `risk_level`, and also
`optic_disc_center`/`optic_disc_radius`/`optic_disc_method` (which method localized the
disc: `TRAINED_MODEL`, `CV_BRIGHTEST_REGION`, or, only as a last resort,
`FALLBACK_IMAGE_CENTER`).

### HTTP API
```bash
pip install -r requirements-api.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
curl -F "image=@data/DRIVE/test/images/01_test.tif" http://localhost:8000/analyze
```

### Web frontend
```bash
cd frontend
npm install
npm run dev   # opens on http://localhost:3000, calls the API above
```

### Optic disc fix (peripapillary Zone B)

The originally submitted thesis (section "Região de Interesse (ROI Peripapilar)")
documented a known limitation: without automatic optic disc detection, the disc center was
estimated as the image's geometric center, and its radius as 15% of the smaller image
dimension — a simplification that is not clinically valid (the optic disc is not centered
in a fundus photograph). `src/pipeline/optic_disc.py` replaces this with real detection, in
order of preference:

1. Trained model (`--train_od`) fit on `data/IOSTAR/mask_OD`.
2. Classical CV heuristic (brightest region in the red channel + largest connected
   component + minimum enclosing circle) — always available, no training required.
3. Only if nothing else works: the old image-center heuristic, now always logged
   explicitly (never silent).

See [`docs/METRICS.md`](docs/METRICS.md) for the before/after accuracy numbers.

### Interactive visualizations and reports (Jupyter notebook)
Deep result analysis and unified visual control live in the master notebook:

1. Start Jupyter:
   ```bash
   jupyter notebook notebooks/00_Unified_Master_Pipeline.ipynb
   ```
2. **Inside the notebook:**
   - Run Cell 1 & 2 to spin up the modular backend, `ScientificAVRPipeline()`.
   - Pass batches of test images or direct image paths.
   - The pipeline object returns dicts with inference time (`inference_time_ms`), masks,
     and lets you plot colorized side-by-side medical reports with `matplotlib`. All the
     training "spaghetti code" is isolated away from this iterative view.
3. This is the environment to use for a thesis defense or any presentation requiring
   precise, reproducible model reports generated through the Python engine.
