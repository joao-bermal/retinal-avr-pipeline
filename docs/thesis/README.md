# Thesis

These documents correspond to the **beta run** (git tag `beta`). They will be revised
after run 1 to follow the formatting manual in full, with run 1 figures and metrics.

| File | Content |
|---|---|
| `TCC_Bermal_Santaniello_PT.docx` / `.pdf` | Thesis in Portuguese, with the optic disc correction |
| `TCC_Bermal_Santaniello_EN.docx` / `.pdf` | Full English translation of the same document |
| `figures/zone_b_pt.png`, `figures/zone_b_en.png` | Figure 7 (peripapillary Zone B), regenerated with `scripts/make_zone_b_figure.py` |

The PDFs were rendered with LibreOffice, which substitutes Arial with the metric compatible
Liberation Sans; open the .docx in Word for the exact submission layout.

## What changed from the submitted version

The submitted thesis placed the peripapillary Zone B around the geometric center of the
image because there was no optic disc detection. The pipeline now detects the optic disc
(see [`../METRICS.md`](../METRICS.md)), and the thesis was updated accordingly:

- The "Região de Interesse (ROI Peripapilar)" subsection describes the hybrid detection
  (trained U-Net on IOSTAR, classical heuristic as fallback, image center only as a
  logged last resort) and its validation: Dice 0.8545 and median center error 11.1 px on
  the six held-out IOSTAR images, against 203.4 px for the heuristic and 357.9 px for the
  image center.
- Figure 7 was regenerated with Zone B centered on the detected disc.
- One sentence in the Conclusion states that Zone B is positioned from the detected disc.
- The English version has the same content. Its duplicated Portuguese abstract block was
  removed because the original already carried an English abstract.

## Formatting manual

The additions follow the USP/Esalq MBA "Manual de Instruções e Normas TCC" and its
formatting checklist (local copies in `docs/thesis/reference/`, not redistributed in
this repository):

- No new section: the manual allows only Resumo, Introdução, Metodologia (or Material e
  Métodos), Resultados e Discussão, Conclusão, Agradecimentos, Referências and optional
  Apêndices/Anexos, so an earlier "Nota de Atualização (2026)" section was removed and
  its content moved into the methodology subsection above.
- Methodology text in impersonal past tense, numbers from zero to ten spelled out,
  decimal comma in Portuguese, no direct quotations, no new references needed.
- Figure caption below the figure, "Figura 7." followed by the text with no final
  period, single spacing, and the source line "Fonte: Dados originais da pesquisa"
  (the manual's wording for figures in the methodology section).
- No title inside the figure image, Arial style font.
- Page header uses a hyphen before the year, as the manual specifies.

No travessão (em or en dash) is used anywhere in either document.

## Pre-existing points that do not follow the manual

These come from the originally submitted text and were left unchanged:

1. There is no "Resultados e Discussão" section; results are reported inside the
   methodology subsections.
2. Figure numbers repeat: Figures 6, 7 and 8 appear twice (confusion matrix, PR curve and
   AVNet analysis, then AVR pipeline, Zone B and skeletonization), and the text cites
   Figures 10 and 11, which are not in the document.
3. The other captions use "Figura N:" with a colon, 1.5 line spacing and a final period
   in the source line ("Fonte: Resultados da pesquisa."), and the source wording differs
   from the manual's "Resultados originais da pesquisa" / "Dados originais da pesquisa".
4. The section is titled "Agradecimento" (the manual uses "Agradecimentos") and names a
   person, which the manual allows only with documented authorization.
