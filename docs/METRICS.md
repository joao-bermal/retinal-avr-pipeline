# Metrics

Every number below comes from this codebase, trained on an AMD RX 6800XT with ROCm (see
[`GPU_ROCM_RX6800XT.md`](GPU_ROCM_RX6800XT.md)), and can be regenerated with the commands
in [`../EXECUTION_GUIDE.md`](../EXECUTION_GUIDE.md). The final metrics and full per epoch
history of each run are in `results/<task>/run_<id>/training_results.json`, and the metric
images are in the `evidence/` folder next to it.

## Summary

| Model | Metric | Value | Thesis target | Evaluated on | Run |
|---|---|---|---|---|---|
| Vessel segmentation (Enhanced U-Net) | Dice | **0.7942** | 0.7965 | 4 DRIVE training images held out from training | `results/segmentation/run_20260925_225243/` |
| Vessel segmentation | IoU | 0.6587 | | same | same |
| Vessel segmentation | Accuracy | 0.9642 | 0.96 | same | same |
| Vessel segmentation | Sensitivity / specificity | 0.8220 / 0.9773 | | same | same |
| A/V classification (AVNet, ResNet-50) | Macro F1 | **0.9577** | 0.78 | the 70 training images (see limitation 2) | `results/av_classification/run_20260925_234145/` |
| A/V classification | Accuracy | 0.9910 | 0.96 | same | same |
| A/V classification | F1 artery / vein | 0.9412 / 0.9366 | 0.75 / 0.80 | same | same |
| Optic disc detection (U-Net, 1 channel) | Dice | **0.8545** | 0.85 | 6 IOSTAR images held out from training | `results/optic_disc/run_20260926_122349/` |
| Optic disc detection | Median center error | **11.1 px** | none (new metric) | same 6 images | same |

## Optic disc detection: before and after

The optic disc module replaced a documented limitation of the original thesis: the
peripapillary Zone B, where vessel calibers for CRAE/CRVE are measured, was built around
the geometric center of the image instead of the real optic disc.

Distance between the detected center and the centroid of the real disc mask
(`data/IOSTAR/mask_OD/`), in pixels at native resolution (1024 x 1024). Produced by
`python scripts/evaluate_optic_disc.py`, saved in
`results/optic_disc/run_20260926_122349/evidence/center_error_summary.csv`:

| Method | Median error, 6 held-out images | Median error, all 30 images |
|---|---|---|
| Original thesis assumption: image center | 357.9 | 327.9 |
| Classical CV heuristic (no training) | 203.4 | 223.7 |
| **Trained model** | **11.1** | **10.1** (one image without detection) |
| Hybrid dispatcher used by the pipeline | 11.1 | 10.2 |

Evidence images: `center_error_by_method.png`, `held_out_detections.png` (the 6 held-out
images with ground truth and each method's center) and `training_curves.png`.

The epoch with the best validation Dice (39) was kept as the checkpoint. The trained
model only saw IOSTAR, the only dataset here with optic disc masks. On other camera
domains such as DRIVE it reports low confidence and the hybrid dispatcher falls back to
the classical heuristic, which is what happens for `data/DRIVE/test/images/01_test.tif`
(method `CV_BRIGHTEST_REGION`, center (111, 240), radius 66.7 px).

## Known limitations of these numbers

1. **Segmentation validation set.** DRIVE's test split has no public ground truth, so the
   last 4 of the 20 DRIVE training images are used as validation. The same 4 images
   select the best checkpoint and produce the reported metrics, so the numbers are
   slightly optimistic.
2. **A/V classification has no held-out set.** The dataset classes in
   `src/data/av_classification_dataset.py` ignore the `phase` argument, so training and
   "validation" are the same 70 images (IOSTAR 30, RITE 40). The Macro F1 of 0.9577 is
   training-set performance. Measuring generalization requires a real split and a new
   training run.
3. **LES-AV is not used.** It is enabled in `src/config/settings.py` but loads 0 images,
   because the loader expects `images/*.jpg`, `artery_masks/` and `vein_masks/` and the
   dataset has `images/*.png`, `arteries/` and `veins/`.
4. **Training Dice is not recorded.** The segmentation trainer stores 0.0 placeholders
   for `train_dice` and `train_iou`, so the "train" curves in the segmentation plots are
   flat at zero.
5. **Correction of an earlier figure.** An earlier version of this document reported
   8.5 px as the trained detector's median error over all 30 IOSTAR images. The committed
   evaluation script gives 10.1 px, which is the value above.

These are documented rather than fixed because fixing 2 and 3 changes the training data
and requires retraining, after which every A/V number here would change.

## Training runs

| Run | Epochs (early stopped) | Time on RX 6800XT | Peak GPU temperature |
|---|---|---|---|
| Segmentation | 135 of 150 | under 15 min | 71 C |
| A/V classification | 132 of 250 | under 15 min | 65 C |
| Optic disc | 54 of 80 | under 5 min | 59 C (with the 0.3 s batch pause) |

## Reproducing

```bash
scripts/temp_guard.sh ".venv/bin/python main.py --train_seg" logs/train_seg.log
scripts/temp_guard.sh ".venv/bin/python main.py --train_av" logs/train_av.log
scripts/temp_guard.sh ".venv/bin/python main.py --train_od" logs/train_od.log
python scripts/evaluate_optic_disc.py --device cpu --sweep
```

On AMD hardware, do the pre-flight steps in section 3 of the reproduction guide first.
