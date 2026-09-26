# Thesis document

- [`TCC_Bermal_Santaniello_PT.docx`](TCC_Bermal_Santaniello_PT.docx) — original Portuguese
  thesis submission, with the peripapillary ROI section, Figure 7, and its source note
  updated to reflect the real optic disc detection fix (see
  [`../PROJECT_HISTORY.md`](../PROJECT_HISTORY.md)) and a "Nota de Atualização (2026)"
  section documenting the retraining results.
- [`TCC_Bermal_Santaniello_EN.docx`](TCC_Bermal_Santaniello_EN.docx) — full English
  translation of the above, built from the same corrected source. The document already
  contained an English abstract/keywords block in the original submission; the redundant
  Portuguese abstract repeat was removed rather than translated a second time.

Both files carry the same updated Figure 7 (real detected optic disc center instead of
the image-center assumption) and the same "2026 Update Note" summarizing the retraining
results on the RX 6800XT. See [`../METRICS.md`](../METRICS.md) and
[`../GPU_ROCM_RX6800XT.md`](../GPU_ROCM_RX6800XT.md) for the full technical detail behind
that note.
