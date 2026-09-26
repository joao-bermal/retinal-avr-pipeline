# Reproduction guide

This guide reproduces every step of the project from a fresh clone: environment,
datasets, training of the three models, the metric images, the optic disc evaluation,
the thesis figure, the command line pipeline, the HTTP API and the web frontend. Every
command below was run on the reference machine and the expected outputs are the ones it
produced.

Reference machine: Ubuntu, kernel 6.17, AMD Radeon RX 6800XT with ROCm 6.4.2,
Python 3.12.7, Node.js 22.14.0. NVIDIA/CUDA and CPU work through `requirements.txt`,
but the published metrics came from the ROCm environment.

All commands run from the repository root unless stated otherwise.

## Contents

1. [Environment](#1-environment)
2. [Datasets](#2-datasets)
3. [Before training on an AMD GPU](#3-before-training-on-an-amd-gpu)
4. [Training](#4-training)
5. [Metric images and evaluation](#5-metric-images-and-evaluation)
6. [Tests](#6-tests)
7. [Running the pipeline on an image](#7-running-the-pipeline-on-an-image)
8. [HTTP API](#8-http-api)
9. [Web frontend](#9-web-frontend)
10. [Thesis figure](#10-thesis-figure)
11. [Output locations](#11-output-locations)
12. [Troubleshooting](#12-troubleshooting)

## 1. Environment

```bash
git clone https://github.com/joao-bermal/retinal-avr-pipeline.git
cd retinal-avr-pipeline
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install exactly one of the two dependency files:

```bash
# AMD GPU with ROCm 6.4 (the environment the metrics came from)
pip install -r requirements-rocm.txt

# NVIDIA GPU (CUDA 12.8 wheels) or CPU only
pip install -r requirements.txt
```

Then the API layer, needed for sections 8 and 9:

```bash
pip install -r requirements-api.txt
```

Check that PyTorch sees the GPU (ROCm builds also report through `torch.cuda`):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected on the reference machine: `2.9.1+rocm6.4 True`. `False` means CPU only; every
step still works, training is just much slower.

The ROCm driver installation itself (Secure Boot, `amdgpu-install --no-dkms`) is
documented in [`docs/GPU_ROCM_RX6800XT.md`](docs/GPU_ROCM_RX6800XT.md).

## 2. Datasets

The images are too large for GitHub. Download `data.zip` from the author's
[Google Drive](https://drive.google.com/file/d/1VUNfJkRd8V9RmR--NlnI_RZg84Ioz-E8/view?usp=sharing)
and extract it in the repository root so that `data/` is created.

`data.zip` does not contain the 30 original IOSTAR color images (`data/IOSTAR/image/`),
which the optic disc model needs because IOSTAR is the only dataset here with optic disc
masks (`mask_OD/`). Obtain them from the IOSTAR distribution (RetinaCheck project,
Zhang et al., 2016) and place them in `data/IOSTAR/image/` with their original names
(`STAR 01_OSC.jpg` and so on).

Expected layout (folder, file count):

```text
data/
├── DRIVE/
│   ├── training/images/       20   vessel segmentation input
│   ├── training/1st_manual/   20   vessel segmentation ground truth
│   ├── training/mask/         20
│   ├── test/images/           20   no public ground truth; used for inference demos
│   └── test/mask/             20
├── IOSTAR/
│   ├── image/                 30   original color images (not in data.zip)
│   ├── GT/                    30   vessel masks (A/V model input)
│   ├── AV_GT/                 30   artery/vein ground truth
│   ├── mask/                  30   field of view masks
│   └── mask_OD/               30   optic disc masks
├── RITE/{training,test}/{images,vessel,av}/   20 each
├── AV_groundTruth/                            not used by the current code
└── LES-AV/                                    present, but see the note in section 4.2
```

Check it:

```bash
for d in DRIVE/training/images DRIVE/training/1st_manual IOSTAR/image IOSTAR/mask_OD IOSTAR/AV_GT RITE/training/av RITE/test/av; do
  printf "%-26s %s\n" "$d" "$(ls "data/$d" | wc -l)"
done
```

Expected: 20, 20, 30, 30, 30, 20, 20.

## 3. Before training on an AMD GPU

On the reference machine, training drove the GPU from idle to its 110 C shutdown
threshold in seconds and powered the computer off twice. The fixes are documented in
[`docs/GPU_ROCM_RX6800XT.md`](docs/GPU_ROCM_RX6800XT.md). Before every training run:

```bash
sudo rocm-smi --setperflevel manual
sudo rocm-smi --setsclk 0              # lowest clock level; confirm with: rocm-smi --showclocks
sudo rocm-smi --setfan 200             # fixed fan speed; 255 is 100%
export MIOPEN_FIND_MODE=FAST
export MIOPEN_DEBUG_CONV_IMMEDIATE_FALLBACK=1
```

and run the training command under the temperature watchdog, which kills it at 85 C:

```bash
scripts/temp_guard.sh ".venv/bin/python main.py --train_od" logs/train_od.log
tail -f logs/train_od.log.temp          # one junction temperature reading per second
```

The clock lock was observed to revert on its own, so check `rocm-smi --showclocks` again
right before starting. On NVIDIA or CPU this section does not apply.

## 4. Training

A fresh clone has no checkpoints (`models/` is gitignored), so all three models have to
be trained before sections 7 to 10 work. Hyperparameters live in
[`src/config/settings.py`](src/config/settings.py); `--epochs`, `--lr`, `--batch_size`
and `--config <file.yaml>` override them from the command line.

Every run gets an id `run_<YYYYMMDD_HHMMSS>` and writes:

- `models/<task>/run_<id>/`: checkpoints. Only epochs that improve the validation metric
  are saved, so the newest file is the best one. The pipeline and the API load the newest
  checkpoint under `models/<task>/`.
- `logs/<task>/run_<id>/metrics_*.csv`: one row per epoch (gitignored).
- `results/<task>/run_<id>/`: `training_results.json` (full history and final metrics)
  and `evidence/` with the metric images listed in section 5 (committed).

### 4.1 Vessel segmentation

```bash
scripts/temp_guard.sh ".venv/bin/python main.py --train_seg" logs/train_seg.log
```

Enhanced U-Net on DRIVE `training/` (the DRIVE test set has no public ground truth):
the first 16 images train, the last 4 validate. Up to 150 epochs, early stopping after
25 epochs without improvement. Reference run: 135 epochs, under 15 minutes, peak GPU
temperature 71 C, validation Dice 0.7942.

### 4.2 Artery/vein classification

```bash
scripts/temp_guard.sh ".venv/bin/python main.py --train_av" logs/train_av.log
```

AVNet (ResNet-50 encoder) on IOSTAR (30) and RITE (40). Up to 250 epochs, early stopping
after 30. Reference run: 132 epochs, under 15 minutes, peak 65 C, Macro F1 0.9577.

Two known limitations of this step, reproduced as is so the numbers match:

- The A/V dataset classes do not split by `phase`, so the training and validation sets
  are the same 70 images and the reported Macro F1 is training-set performance, not a
  held-out estimate.
- LES-AV is enabled in the config but loads 0 images: the loader looks for
  `images/*.jpg`, `artery_masks/` and `vein_masks/`, while the dataset ships
  `images/*.png`, `arteries/` and `veins/`.

### 4.3 Optic disc detection

```bash
scripts/temp_guard.sh ".venv/bin/python main.py --train_od" logs/train_od.log
```

The same Enhanced U-Net architecture with one output channel, trained on IOSTAR
`image/` + `mask_OD/`: the first 24 sorted images train, the last 6 are held out. Up to
80 epochs, early stopping after 15, and a 0.3 s pause after every batch (this tiny
dataset otherwise keeps the GPU at 100% with no idle gap, which is what triggered the
overheating). Reference run: 54 epochs, under 5 minutes, peak 59 C, best epoch 39,
validation Dice 0.8545.

At the end of this command, `scripts/evaluate_optic_disc.py` runs automatically and
writes the center error evidence of section 5.3, including an epoch sweep over every
saved checkpoint.

`scripts/train_segmentation.py`, `scripts/train_classification.py` and
`scripts/train_optic_disc.py` are standalone equivalents of the three commands above;
`main.py` is the entry point this guide uses.

## 5. Metric images and evaluation

### 5.1 Segmentation evidence (written by `--train_seg`)

`results/segmentation/run_<id>/evidence/`:

| File | Content |
|---|---|
| `training_curves_final.png` | Dice, loss, IoU and learning rate per epoch |
| `fig03_training_curves_complete.png` | Six panel training analysis |
| `fig02_architecture_analysis.png` | Architecture diagram, parameters per component, ROC and precision-recall curves on the validation images |
| `sample_predictions.png` | Image, ground truth and prediction for 3 validation images |
| `final_metrics.csv` | Dice, IoU, accuracy, sensitivity, specificity of the best checkpoint |

The trainer does not compute Dice on the training set; the `train_dice` and `train_iou`
series in `training_results.json` are 0.0 placeholders, so the "train" curves in these
plots are flat at zero and should be ignored.

### 5.2 A/V classification evidence (written by `--train_av`)

`results/av_classification/run_<id>/evidence/`: `confusion_matrix.png`,
`pr_curve_av.png`, `sample_predictions.png`, `consolidated_analysis.png`,
`final_metrics.csv` (Macro F1, accuracy, per class F1).

### 5.3 Optic disc evaluation (any time after `--train_od`)

This compares the trained model, the classical heuristic, the original image center
assumption and the hybrid dispatcher against the real disc centroid of every IOSTAR
mask. It does not train anything and runs on CPU in about 90 seconds:

```bash
python scripts/evaluate_optic_disc.py --device cpu
python scripts/evaluate_optic_disc.py --device cpu --sweep     # also scores every saved epoch
```

Expected output with the reference checkpoint:

```text
Median center error (px):
split            all  held_out
method
trained_model   10.1      11.1
cv_heuristic   223.7     203.4
image_center   327.9     357.9
hybrid          10.2      11.1
```

Files written to `results/optic_disc/run_<id>/evidence/`: `center_error_summary.csv`,
`per_image_center_error.csv`, `center_error_by_method.png`, `held_out_detections.png`,
`training_curves.png` and, with `--sweep`, `epoch_sweep.csv`.

The segmentation and A/V images are produced only at the end of training. To regenerate
them without retraining, load the checkpoint and call the corresponding functions in
`src/utils/metrics.py`, the same way `main.py` does after `trainer.train()`.

## 6. Tests

No GPU or trained model needed:

```bash
python tests/sanity_check.py            # forward pass of both architectures
python tests/test_avr_calculator.py     # Zone B geometry and Knudtson formulas
```

Both end with a success line (`All sanity checks passed successfully.` and
`All AVR calculator tests passed.`).

## 7. Running the pipeline on an image

Requires the three checkpoints from section 4.

```bash
python main.py --run_pipeline data/DRIVE/test/images/01_test.tif
```

Expected output with the reference checkpoints (excerpt):

```text
optic_disc_method: CV_BRIGHTEST_REGION
avr: 0.9430
crae: 15.2528
crve: 16.1754
risk_level: NORMAL
risk_description: Low cardiovascular risk
avr_status: SUCCESS
avr_method: SCIENTIFIC_KNUDTSON
```

`optic_disc_method` tells which detector placed Zone B: `TRAINED_MODEL`,
`CV_BRIGHTEST_REGION`, or `FALLBACK_IMAGE_CENTER` as a logged last resort. On DRIVE the
trained detector is outside its IOSTAR training domain and reports low confidence, so
the classical heuristic is used, as above.

To force CPU on a machine with a GPU, prefix any command with
`HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=`.

## 8. HTTP API

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The models load once at startup (a few seconds). Then, from another terminal:

```bash
curl http://localhost:8000/health
curl -F "image=@data/DRIVE/test/images/01_test.tif" http://localhost:8000/analyze
```

`/health` returns `{"status":"ok","device":"cuda"}` (or `"cpu"`). `/analyze` accepts
JPG, PNG, TIFF or BMP in the `image` form field and returns JSON with `avr`, `crae`,
`crve`, `risk_level`, `risk_description`, `avr_confidence`, `vessel_percentage`,
`confidence`, `optic_disc_center`, `optic_disc_radius`, `optic_disc_method`,
`optic_disc_confidence`, `inference_time_ms` and `av_inference_time_ms`. For
`01_test.tif` the values match section 7. Interactive documentation is at
`http://localhost:8000/docs`.

CORS allows only `http://localhost:3000` and `http://127.0.0.1:3000` (the frontend).
There is no authentication; do not expose this server publicly.

## 9. Web frontend

Requires Node.js 20.9 or newer (tested with 22.14.0) and the API from section 8
running.

```bash
cd frontend
npm install
cp .env.local.example .env.local        # only if the API is not at http://localhost:8000
npm run dev
```

Open `http://localhost:3000`, choose a fundus image and click Analyze. The page shows
the AVR with its risk badge, CRAE, CRVE, vessel area, A/V confidence, the optic disc
method, confidence, center and radius, and inference times. A warning appears when the
optic disc fell back to the image center.

Production build: `npm run build && npm run start`. Lint: `npm run lint`.

## 10. Thesis figure

Figure 7 of the thesis (`docs/thesis/`) shows Zone B centered on the detected optic
disc. It is generated by the same detection code the pipeline uses:

```bash
python scripts/make_zone_b_figure.py --lang pt --out docs/thesis/figures/zone_b_pt.png
python scripts/make_zone_b_figure.py --lang en --out docs/thesis/figures/zone_b_en.png
```

Expected console line: `01_test.tif: method=CV_BRIGHTEST_REGION center=(111.0, 240.0)
radius=66.7px`. `--image` selects another fundus image. The figure has no title inside
it and uses Arial (or Liberation Sans), as the USP/Esalq formatting manual requires
(see [`docs/thesis/README.md`](docs/thesis/README.md)).

## 11. Output locations

| Path | Content | In git |
|---|---|---|
| `models/<task>/run_<id>/*.pth` | Checkpoints | no |
| `logs/<task>/run_<id>/` | Per epoch CSV logs | no |
| `logs/*.log`, `logs/*.log.temp` | Watchdog output and temperature readings | no |
| `results/<task>/run_<id>/training_results.json` | History and final metrics | yes |
| `results/<task>/run_<id>/evidence/` | Metric images and CSVs | yes |
| `docs/thesis/figures/` | Thesis figures | yes |

The runs behind the published numbers are `results/segmentation/run_20260925_225243`,
`results/av_classification/run_20260925_234145` and
`results/optic_disc/run_20260926_122349`. See [`docs/METRICS.md`](docs/METRICS.md).

## 12. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Training or inference hangs for minutes on ROCm with `MIOpen ... Searching the best solution` | Export `MIOPEN_FIND_MODE=FAST` and `MIOPEN_DEBUG_CONV_IMMEDIATE_FALLBACK=1` (section 3). |
| Computer powers off during training, `dmesg` shows `GPU over temperature range (SW CTF)` | Section 3: lock the clock, fix the fan speed, use the watchdog. |
| `FileNotFoundError: data/IOSTAR/image does not exist` | The IOSTAR color images are missing (section 2). |
| `No segmentation checkpoint found` from the pipeline or the API | Train the models (section 4). |
| Frontend shows "Could not reach the API" | Start the API (section 8) or set `NEXT_PUBLIC_API_URL` in `frontend/.env.local`. |
| `pip install -r requirements.txt` fails on a torch version | Check the CUDA index line at the top of the file; CPU installs can use the default PyPI wheel. |
