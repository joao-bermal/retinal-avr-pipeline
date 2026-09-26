# Running this project on an AMD RX 6800XT with ROCm

This document is the operational record of getting real GPU training working
on this specific machine (AMD Radeon RX 6800XT, Ubuntu, kernel `6.17.0-19-generic`)
during the 2026-09-25/26 session, including a **reproducible GPU thermal
runaway** that shut the machine down twice. It exists so that a future session
(human or another model instance) re-training these models does not have to
re-discover any of this from scratch.

If you are about to run `python main.py --train_seg`, `--train_av`, or
`--train_od` on this machine (or a similar RDNA2 card on Linux/ROCm), **read
the "Mandatory pre-flight checklist" section before starting anything.**

## TL;DR

- ROCm 6.4.2 (userspace) + the in-tree `amdgpu` kernel driver (kernel ≥ 6.14,
  no DKMS needed) works. Secure Boot must be **disabled** — see "Secure Boot"
  below for why the DKMS/MOK route was abandoned in favor of this.
- Two PyTorch settings caused real GPU thermal events (up to a firmware-forced
  power-off) on this card, independent of thermal paste/pads/PSU (already
  ruled out by the user beforehand). Both are already fixed in this codebase;
  see "Root causes and fixes" if you need to touch `main.py` or the trainers.
- Even after both fixes, this specific GPU/cooler combination still spikes
  from idle to 90-110°C junction temperature within a few seconds of *any*
  sustained GPU compute burst, reproducibly. The mitigation is running with
  the GPU clock locked to its lowest DPM level and a temperature watchdog —
  see "Mandatory pre-flight checklist."

## Environment that is known to work

```
OS:            Ubuntu (kernel 6.17.0-19-generic)
Secure Boot:   disabled
GPU driver:    in-tree amdgpu (no DKMS, no /opt/amdgpu-pro)
ROCm:          6.4.2 (installed via: sudo amdgpu-install --usecase=rocm --no-dkms)
PyTorch:       torch==2.9.1+rocm6.4 / torchvision==0.24.1+rocm6.4
               (see requirements-rocm.txt — pip index url
               https://download.pytorch.org/whl/rocm6.4)
Python env:    dedicated venv at .venv/ (NOT the system/anaconda base env —
               a generic `pip install torchvision` in a shared env pulled a
               CUDA build and had to be replaced)
```

Confirm the driver stack with:
```bash
lsmod | grep amdgpu          # should show the amdgpu kernel module loaded
ls /dev/kfd                  # must exist (ROCm compute interface)
rocm-smi                     # should list the GPU with a sane temperature
dkms status                  # should be EMPTY (see "Secure Boot" below)
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Secure Boot: why DKMS was abandoned

The machine originally had Secure Boot **enabled** with a DKMS-built,
self-signed `amdgpu` kernel module (`amdgpu-install`'s default `--usecase`
includes `dkms`). The signing key was never enrolled in the MOK (Machine
Owner Key) list — `mokutil --list-enrolled` only showed Canonical's own
certificate — so the kernel silently refused to load the module: no
`/dev/kfd`, `lsmod | grep amdgpu` empty, no error in `dmesg` (the load was
never attempted, not rejected). This is the classic Ubuntu/DKMS/Secure-Boot
gotcha, often compounded by BIOS Fast Boot skipping the one-time MOK
enrollment screen (`mokutil --import`, then a blue "MOK management" screen at
the *next* boot, before GRUB).

The MOK enrollment path was attempted (`mokutil --import
/var/lib/shim-signed/mok/MOK.der`, enroll at boot) but the DKMS module then
caused **boot to fail to bring up graphics at all**, requiring an
`amdgpu.blacklist` kernel boot parameter to get back in, followed by `sudo
amdgpu-uninstall` to remove the broken DKMS/amdgpu-pro stack. After that:

- Secure Boot was disabled entirely.
- The kernel in use (6.17) already ships a working in-tree `amdgpu` driver —
  no DKMS module is needed at all. `dkms status` is empty and `/dev/kfd`
  exists purely from the in-tree driver.
- ROCm was then reinstalled **without** the kernel-driver component:
  ```bash
  sudo amdgpu-install --usecase=rocm --no-dkms
  ```
  `--no-dkms` is the important flag — it installs only the ROCm userspace
  libraries (HIP, MIOpen, rocBLAS, etc.) and leaves the kernel driver alone.

**If you're setting this up fresh on a kernel that does NOT have working
in-tree amdgpu/KFD support**, you will need Secure Boot disabled (simplest)
or a properly-enrolled MOK key, before `--no-dkms` becomes an option at all.

## Root causes and fixes (already applied in this codebase)

### 1. `torch.backends.cudnn.benchmark = True` on ROCm → MIOpen exhaustive search hang

`main.py` used to unconditionally set `torch.backends.cudnn.benchmark = True`
whenever a GPU was available. On CUDA this is a cheap, worthwhile per-shape
algorithm search. On this ROCm install (pip wheels, no prebuilt MIOpen
perf-db for `gfx1030`), it degenerated into an **exhaustive per-shape kernel
search** (`MIOpen(HIP): Warning [SearchImpl] Searching the best solution in
the 9 dim space...`) that took **over an hour for a single inference call**.

Fix (`main.py`):
```python
torch.backends.cudnn.benchmark = torch.version.hip is None
```
Only enable benchmark mode on CUDA; never on a ROCm/HIP build.

Additionally, these two environment variables make MIOpen use a fast
heuristic instead of any search, even for genuinely new shapes:
```bash
export MIOPEN_FIND_MODE=FAST
export MIOPEN_DEBUG_CONV_IMMEDIATE_FALLBACK=1
```
**Set these two variables in every shell that runs training or inference on
this machine.** They are not baked into the codebase (no good place to set
process-wide env vars from within Python before the HIP runtime initializes),
so remember to export them yourself, e.g. wrap your command:
```bash
MIOPEN_FIND_MODE=FAST MIOPEN_DEBUG_CONV_IMMEDIATE_FALLBACK=1 python3 main.py --train_seg
```

### 2. GPU thermal runaway during training (real, reproducible, not a paste/pad issue)

The user had already replaced thermal paste, reseated all thermal pads, and
upgraded the PSU before this session — ruling out the obvious hardware
explanations. Confirmed independently during this session:

- Two full **firmware-forced shutdowns** occurred (`dmesg`: `amdgpu: ERROR:
  GPU over temperature range(SW CTF) detected!` / `System is going to
  shutdown due to GPU SW CTF!`) while training was running.
- The GPU fan's automatic curve (`pwm1_enable=2`) does not react fast enough
  to sudden 100%-utilization compute bursts: junction temperature was
  observed jumping from ~50°C to 110°C (the hardware's own critical
  threshold) in as little as 4-11 seconds.
- Forcing the fan to 100% manually (`sudo rocm-smi --setfan 255`) reduced
  peak temperature somewhat but **did not eliminate the fast spike** —
  confirming this is not purely a "fan not spinning" problem.
- Locking the GPU to its lowest DPM clock level (well below default boost)
  reduced the spike further (peaks dropped from 110°C → ~85-95°C) but still
  did not eliminate it for the optic-disc training workload specifically.
- The actual fix for the optic-disc case turned out to be about **duty
  cycle**, not just clock/power: that dataset is tiny (24 training images)
  and loads instantly, so the GPU is fed batches back-to-back with *zero*
  natural idle gap — unlike vessel segmentation or A/V classification, whose
  larger images and heavier augmentation pipeline give the GPU brief
  breathing room between batches "for free." Both of those trainings
  completed (135 and 132 epochs respectively) with a **gradual** temperature
  ramp and a **safe** peak (71°C and 65°C) with no changes beyond the clock
  lock — only the optic-disc training needed an explicit throttle.

Fix (`src/training/segmentation_trainer.py`): a `batch_pause_seconds`
constructor argument on `EnhancedSegmentationTrainer` inserts
`time.sleep(batch_pause_seconds)` after every training batch. It is wired up
for `--train_od` only (`main.py`, `batch_pause_seconds=0.3`), since that is
the one workload that reproduced the fast spike; `--train_seg` and
`--train_av` do not use it and completed safely without it. If you add
another tiny/fast-loading dataset in the future and see the same fast-spike
pattern, pass `batch_pause_seconds=0.2-0.5` for that trainer too.

With both fixes (clock locked low + `batch_pause_seconds=0.3`), the
optic-disc training completed all 54 epochs (early stopping) with a
**gradual** ramp and a peak of only 59°C — no incidents.

## Mandatory pre-flight checklist (do this before any `--train_*` run)

The clock/fan settings below are **not persistent** — they were observed to
silently revert (DPM unlocking back to full boost clock) across the session
even without a reboot, for reasons that were not fully root-caused. **Always
re-check and re-apply immediately before training, not just once per boot
session.**

1. Confirm the GPU is visible and cool:
   ```bash
   rocm-smi --showtemp --showclocks --showperflevel --showfan
   ```
2. Lock the clock to its lowest DPM level (do NOT rely on `--setperflevel
   low` alone — it was observed to report "low" while `sclk` silently sat at
   the boost level; the explicit two-step manual pin held reliably):
   ```bash
   sudo rocm-smi --setperflevel manual
   sudo rocm-smi --setsclk 0
   ```
   Re-run `rocm-smi --showclocks` and confirm `sclk clock level: 0` (≈500MHz
   on this card — check `rocm-smi -s` for the exact level-0 frequency, it can
   differ per card/BIOS). If it still shows the boost level, repeat the two
   commands; something intermittently resets this.
3. Set the fan to a high, fixed speed (manual beats the automatic curve,
   which was proven too slow to react):
   ```bash
   sudo rocm-smi --setfan 200   # ~78%; 255 = 100% if you want max margin
   ```
4. Export the MIOpen env vars in the same shell you'll launch training from:
   ```bash
   export MIOPEN_FIND_MODE=FAST
   export MIOPEN_DEBUG_CONV_IMMEDIATE_FALLBACK=1
   ```
5. **Run training under a temperature watchdog**, not bare. A minimal
   version of what was used this session:
   ```bash
   #!/bin/bash
   # save as e.g. scripts/temp_guard.sh, chmod +x
   set -u
   CMD="$1"; LOGFILE="$2"; LIMIT_C=85; CHECK_INTERVAL=1
   bash -c "$CMD" > "$LOGFILE" 2>&1 &
   PID=$!
   while kill -0 "$PID" 2>/dev/null; do
     TEMP=$(rocm-smi --showtemp --json 2>/dev/null | grep -o '"Temperature (Sensor junction) (C)": "[0-9.]*"' | grep -o '[0-9.]*' | head -1)
     TEMP_INT=${TEMP%.*}
     if [ -n "$TEMP_INT" ] && [ "$TEMP_INT" -ge "$LIMIT_C" ]; then
       echo "ABORTING: junction temp ${TEMP}C >= ${LIMIT_C}C limit"
       kill -TERM "$PID"; sleep 5; kill -KILL "$PID" 2>/dev/null
       exit 1
     fi
     sleep "$CHECK_INTERVAL"
   done
   wait "$PID"
   ```
   85°C leaves ~25°C of margin below the hardware's own critical/emergency
   cutoff (110°C / 115°C on this card — check `sensors` for your card's
   values) while still tolerating the normal gradual ramp seen in
   segmentation/A-V training (peaked at 71°C / 65°C).
6. If a run gets killed by the watchdog (or by a hardware shutdown), **do
   not just restart the exact same command in a loop hoping it clears up** —
   re-check clocks/fan (step 2-3, they may have reverted) before retrying.

## Reference: metrics achieved this session (2026-09-25/26)

All three models were retrained from scratch on this GPU/ROCm setup, using
the consolidated codebase (not the original scattered notebooks). See
`results/<task>/run_*/` for the full evidence (training curves, confusion
matrix, PR curve, sample predictions) behind these numbers.

| Model | Metric | Value | Target (thesis) | Run |
|---|---|---|---|---|
| Vessel segmentation (Enhanced U-Net) | Dice | 0.7942 | 0.7965 | `results/segmentation/run_20260925_225243/` |
| A/V classification (AVNet, ResNet-50) | Macro F1 | 0.9577 | 0.78 | `results/av_classification/run_20260925_234145/` |
| Optic disc detection (U-Net, 1 channel) | Dice (held-out val) | 0.8545 | 0.85 | `results/optic_disc/run_20260926_122349/` |
| Optic disc detection | Median center distance (6 true held-out images, never trained on) | **11.1px** | — (new metric; see below) | same run |

For comparison, the two baselines this replaces:
- Naive "assume the optic disc is the image's geometric center" (the original
  thesis's documented limitation): median center error ≈ 328px.
- Classical CV heuristic (brightest region + largest connected component,
  no training, `src/pipeline/optic_disc.py::detect_optic_disc_cv`): median
  center error ≈ 224px, and notably weaker/less consistent than the trained
  model on this SLO-derived (IOSTAR) image domain.

The optic-disc model was trained and validated on IOSTAR only (the only
downloaded dataset with disc-mask ground truth, `data/IOSTAR/mask_OD/`); it
does not reliably generalize to other fundus camera domains (e.g. DRIVE) and
the hybrid dispatcher in `src/pipeline/optic_disc.py` correctly falls back to
the CV heuristic there (low confidence on out-of-domain images). If you want
better cross-domain generalization, the next step is sourcing OD-mask ground
truth for DRIVE/RITE/LES-AV (not currently downloaded) and adding them to
`src/data/optic_disc_dataset.py`.

## Known un-root-caused issue: `GemmBwdRest` MIOpen fallback solver

During the optic-disc thermal investigation, every fast temperature spike
was preceded by this exact warning:
```
MIOpen(HIP): Warning [IsEnoughWorkspace] [GetSolutionsFallback AI] Solver <GemmBwdRest>, workspace required: ..., provided ptr: 0 size: 0
```
This solver is selected when PyTorch provides **zero workspace** for a
backward convolution and MIOpen falls back to a GEMM-based implementation.
It's plausible (not confirmed) that this specific fallback path draws
disproportionate transient power on this card/driver combination. This was
never independently isolated from the "no natural inter-batch pause" theory
above (both were fixed simultaneously: clock lock + `batch_pause_seconds`).
If you're revisiting this, a controlled experiment (reproduce with
`batch_pause_seconds=0` again now that the clock is reliably locked, and
watch specifically for this warning) would be worth running — and would be a
legitimate bug to report upstream to ROCm/MIOpen if confirmed, since it's
quite reproducible on this hardware.
