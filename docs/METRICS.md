# Metrics

All numbers below were produced by this exact codebase, trained end-to-end on an AMD RX
6800XT via ROCm (see [`GPU_ROCM_RX6800XT.md`](GPU_ROCM_RX6800XT.md) for the environment
setup and thermal-management notes needed to reproduce this on similar hardware). Raw
training logs, per-epoch CSVs, and evidence plots (training curves, confusion matrix,
PR curve, sample predictions) are committed under `results/<task>/run_<timestamp>/`.

## Summary

| Model | Metric | Value | Thesis target | Run directory |
|---|---|---|---|---|
| Vessel segmentation (Enhanced U-Net) | Dice score | **0.7942** | 0.7965 | `results/segmentation/run_20260925_225243/` |
| Vessel segmentation | Accuracy | 0.9605 | 0.96 | same |
| A/V classification (AVNet, ResNet-50) | Macro F1 | **0.9577** | 0.78 | `results/av_classification/run_20260925_234145/` |
| A/V classification | Accuracy | 0.9910 | 0.96 | same |
| A/V classification | F1 (artery) | 0.9412 | 0.75 | same |
| A/V classification | F1 (vein) | 0.9366 | 0.80 | same |
| Optic disc detection (U-Net, 1 channel) | Dice (held-out validation) | **0.8545** | 0.85 | `results/optic_disc/run_20260926_122349/` |
| Optic disc detection | Median center-to-center error, 6 true held-out images | **11.1px** | — (new metric) | same |

The optic disc row's held-out evaluation used the 6 IOSTAR images never included in
training (a deterministic 80/20 split — the last 6 of 30 sorted images), scored against
each saved epoch checkpoint using the real disc centroid extracted from
`data/IOSTAR/mask_OD/`, not just the internal validation Dice. The epoch with the best
internal validation Dice (epoch 39) was also the epoch with the lowest center-distance
error — confirming Dice-based checkpoint selection already picks the best-generalizing
epoch for this task.

## Optic disc detection: before / after

The whole point of re-implementing this module was fixing a documented thesis limitation:
the peripapillary Zone B (used to measure vessel calibers for CRAE/CRVE) was built around
the image's geometric center, not the real optic disc position.

| Method | Median center error (IOSTAR) | Notes |
|---|---|---|
| Original thesis approach: assume image center | ~328px | The bug being fixed |
| Classical CV heuristic (no training — brightest region + largest connected component) | ~224px | `src/pipeline/optic_disc.py::detect_optic_disc_cv`, always-available fallback |
| **Trained model (this session)** | **~11px** (true held-out) / ~8.5px (all 30, including training images) | `src/pipeline/optic_disc.py::OpticDiscDetector` |

The trained model was fit only on IOSTAR (the only downloaded dataset with disc-mask
ground truth). It generalizes well within that domain but not automatically to other
fundus camera domains (e.g. DRIVE) — the hybrid dispatcher in `optic_disc.py` correctly
detects low confidence out-of-domain and falls back to the CV heuristic rather than
returning a wrong high-confidence answer. Extending ground truth to DRIVE/RITE/LES-AV
(not currently downloaded) is the natural next step for cross-domain generalization.

## Reproducing these numbers

```bash
python main.py --train_seg   # ~150 epochs, early stopping patience 25
python main.py --train_av    # ~250 epochs, early stopping patience 30
python main.py --train_od    # ~80 epochs, early stopping patience 15

python main.py --run_pipeline "data/DRIVE/test/images/01_test.tif"
python tests/sanity_check.py
python tests/test_avr_calculator.py
```

On the RX 6800XT/ROCm setup this session used, **read
[`GPU_ROCM_RX6800XT.md`](GPU_ROCM_RX6800XT.md) first** — there is a real, reproducible GPU
thermal-runaway risk on this hardware/driver combination that the pre-flight checklist
there addresses. Segmentation and A/V classification training took under 15 minutes each
on this GPU; optic disc training under 5 minutes.
