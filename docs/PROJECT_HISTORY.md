# Project history

Context for anyone (human or AI) picking this project up without having lived through its
development. This is not needed to run the pipeline — see `EXECUTION_GUIDE.md` for that —
but explains *why* the repository looks the way it does.

## Origin

This project started as the practical part of an MBA thesis (TCC — Trabalho de Conclusão
de Curso), "Modelo Computacional para Análise da Razão Arteríolo-Venular em Retinografias
e Associação com o Risco Cardiovascular" (Computational Model for Arteriolar-to-Venular
Ratio Analysis in Fundus Photographs and Its Association with Cardiovascular Risk), USP
ESALQ MBA in Software Engineering, 2025. The thesis document lives in
`docs/thesis/` (Portuguese original + English translation, both updated to reflect the
optic disc fix — see below).

## Why the code was scattered across so many folders

The author's primary GPU is an AMD RX 6800XT, and ROCm (AMD's CUDA equivalent) only
supports Linux. Development therefore bounced between a Linux machine (for GPU-accelerated
training) and other environments, producing parallel, drifting copies of the same
notebooks across several local folders: `retinal-avr-cardiovascular-risk`,
`retinal-vessel-segmentation`, `RAV-image-segmentation`, and various backups under
`legacy_projects/`. None of those folders were ever a proper git repository with full
history — they were folder-level snapshots and manual "copy 2", "copy 3" duplicates.

`retinal-avr-pipeline` (this repository) was started as a deliberate refactor: pull the
working logic out of the best notebook version found in that sprawl and rebuild it as a
tracked, modular Python codebase instead of a pile of notebooks. That refactor was
initially incomplete — the CLI's `--run_pipeline` was a stub, and the scientific AVR
calculator (Knudtson formulas, Zone B geometry) existed only inside one specific notebook
in `retinal-avr-cardiovascular-risk/notebooks/03_integrated_pipeline.ipynb`, never ported
over.

## The 2026-09-25/26 consolidation session

A single extended session did the following, in order:

1. **Consolidation**: ported the scientific AVR calculator (`ScientificAVRCalculator`,
   Knudtson CRAE/CRVE formulas, Zone B geometry) from the orphaned notebook into
   `src/pipeline/avr_calculator.py`, fixing a real bug in the process — the notebook's
   iterative combination step used a naive left-fold instead of the canonical
   largest+smallest Knudtson pairing algorithm.
2. **The optic disc fix**: implemented real optic disc detection
   (`src/pipeline/optic_disc.py`) to replace the "assume image center" simplification that
   the thesis itself documented as a limitation. Hybrid design: a trained model (preferred)
   with a classical computer-vision fallback (always available, no training required), and
   only as a last resort the old image-center heuristic — now always logged explicitly
   instead of silent.
3. **Wired the pipeline end-to-end**: `main.py --run_pipeline` actually runs
   `ScientificAVRPipeline.process_image()` now (previously a no-op stub hidden behind a
   `hasattr()` check that was always `False`), and checkpoint loading fails loudly instead
   of silently running on randomly-initialized weights if a checkpoint is missing.
4. **Got ROCm actually working** on the author's machine (Secure Boot/DKMS issue, see
   `GPU_ROCM_RX6800XT.md`) and used it to retrain all three models from scratch on real
   GPU hardware, including diagnosing and fixing a reproducible GPU thermal-runaway issue
   along the way.
5. **Built an HTTP API** (`api/`, FastAPI) and a **web frontend** (`frontend/`, Next.js) on
   top of the existing pipeline, so it can be used interactively instead of only via CLI.
6. **Repo cleanup**: archived the legacy notebooks locally (gitignored — kept on disk for
   reference, not pushed) and translated all documentation to English.

See `docs/METRICS.md` for the resulting numbers and `GPU_ROCM_RX6800XT.md` for the detailed
GPU/driver/thermal story — both are meant to give a future session (this project is
actively intended to be picked up and extended by other AI coding sessions) enough context
to retrain these models without repeating the same discovery process.

## What's still genuinely open

- Optic disc detector only generalizes within its training domain (IOSTAR/SLO-style
  images). Extending to DRIVE/RITE/LES-AV needs disc-mask ground truth for those datasets,
  which isn't downloaded yet.
- The `GemmBwdRest` MIOpen thermal correlation (see `GPU_ROCM_RX6800XT.md`) was mitigated,
  not root-caused. Worth an upstream ROCm/MIOpen bug report if revisited.
- No authentication/deployment story for the API or frontend — both are local-development
  tools right now, not meant to be exposed publicly as-is.
