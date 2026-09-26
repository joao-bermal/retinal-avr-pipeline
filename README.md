# Retinal AVR cardiovascular risk pipeline

Deep learning pipeline that segments retinal vessels, classifies them into arteries and
veins, locates the optic disc and computes the arteriolar-to-venular ratio (AVR) with the
Knudtson protocol, as a non-invasive cardiovascular risk indicator. It is the code of the
MBA thesis (TCC, USP/Esalq) "Modelo Computacional para Análise da Razão Arteríolo-Venular
em Retinografias e Associação com o Risco Cardiovascular".

## Pipeline

Numbers below are from the **beta run** (git tag `beta`); a documented retraining
(run 1) that fixes the limitations listed in [`docs/METRICS.md`](docs/METRICS.md) is next.


1. **Vessel segmentation** (`EnhancedUNet`): Dice 0.7942 on held-out DRIVE images
   (thesis target 0.7965).
2. **Artery/vein classification** (`EnhancedMultiDatasetAVNet`, ResNet-50 encoder):
   Macro F1 0.9577 (see the limitations in [`docs/METRICS.md`](docs/METRICS.md)).
3. **Optic disc detection** (`src/pipeline/optic_disc.py`): a trained U-Net, a classical
   computer vision heuristic as fallback, and the image center only as a logged last
   resort. Median center error 11.1 px on held-out IOSTAR images, against 357.9 px for
   the image center assumption used in the original thesis.
4. **AVR calculation** (`ScientificAVRCalculator`): vessel calibers inside the
   peripapillary Zone B (0.5 to 1.0 disc diameters around the detected disc), CRAE and
   CRVE by iterative Knudtson combination, AVR = CRAE / CRVE, and a risk category.

A FastAPI server (`api/`) exposes the pipeline over HTTP and a Next.js page (`frontend/`)
lets you upload an image and see the result.

## Quick start

With the datasets in `data/` and trained checkpoints in `models/` (the
[reproduction guide](EXECUTION_GUIDE.md) covers both):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-rocm.txt        # or requirements.txt for NVIDIA/CPU
pip install -r requirements-api.txt

python main.py --run_pipeline data/DRIVE/test/images/01_test.tif

uvicorn api.main:app --host 127.0.0.1 --port 8000
cd frontend && npm install && npm run dev   # http://localhost:3000
```

## Documentation

| Document | Content |
|---|---|
| [`EXECUTION_GUIDE.md`](EXECUTION_GUIDE.md) | Step by step reproduction: environment, datasets, training, metric images, evaluation, tests, CLI, API, frontend, thesis figure |
| [`docs/METRICS.md`](docs/METRICS.md) | All results, the optic disc before/after comparison and the known limitations of the numbers |
| [`docs/GPU_ROCM_RX6800XT.md`](docs/GPU_ROCM_RX6800XT.md) | ROCm setup on the RX 6800XT and how the GPU overheating during training was diagnosed and avoided |
| [`docs/PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md) | Why the repository looks the way it does |
| [`docs/thesis/`](docs/thesis/) | Thesis in Portuguese and English (docx and PDF), figures, and how the additions follow the USP/Esalq manual |

## Repository layout

```text
api/                  FastAPI server (GET /health, POST /analyze)
data/                 DRIVE, IOSTAR, RITE, LES-AV (not in git, see the guide)
docs/                 Metrics, ROCm notes, project history, thesis
frontend/             Next.js web page that calls the API
main.py               Command line entry point: --train_seg, --train_av, --train_od, --run_pipeline
models/               Checkpoints written by training (not in git)
notebooks/            Master notebook for interactive exploration
results/              training_results.json and metric images of each run
scripts/              Standalone trainers, optic disc evaluation, thesis figure, GPU watchdog
src/
  config/settings.py  Hyperparameters and paths for every model
  data/               Dataset classes and preprocessing
  models/             Enhanced U-Net and AVNet
  pipeline/           End to end pipeline, AVR calculator, optic disc detection
  training/           Training loops, losses, checkpointing
  utils/              Metric plots
tests/                Architecture smoke test and AVR calculator unit tests
```
