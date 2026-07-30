# Reference distributions

## `spider_normative_canal.tsv`

Per-level spinal canal AP diameter percentiles, measured with `spineps.metrics.measure_canal` over the
**210 sagittal T2 studies of the SPIDER dataset**, using SPIDER's *human-annotated* canal masks. SPINEPS is
not involved in producing these numbers, so they are a reference for the measurement rather than for the
model.

### Why this file exists

Canal AP diameter is not constant along the spine. Across these 210 studies the median rises from about
10.2 mm at the most caudal lumbar level to about 14.4 mm around L1 (r = +0.671 with caudal-to-cranial
position), then plateaus. **A single absolute threshold applied to every level is therefore wrong**: 6 mm
is unremarkable at the most caudal level and striking three levels higher. Compare a measurement against
its own level's distribution here.

### Columns

| column | meaning |
|---|---|
| `spider_vertebra_index` | SPIDER's vertebra numbering, **1 = most caudal**. SPIDER does not assign anatomical names, so neither does this table. |
| `structure` | `vertebra` (the level's own band) or `disc` (the intervertebral level). |
| `n` | Studies contributing to the row. |
| `p5` … `p95` | Percentiles of `narrow_ap_diameter_mm` in millimetres. |

### Caveats

- These are **sagittal** AP diameters from 3–4 mm slice acquisitions. A narrowing that is worst off-midline
  is under-measured. This is not the axial dural sac cross-sectional area used in axial stenosis grading.
- SPIDER is a low-back-pain cohort, not a healthy population, so these are *typical* values for patients
  being imaged, not normal values.
- Indices are not anatomical names. Mapping index to level requires knowing what the study covered.
- No clinical threshold is implied. Nothing here has been validated against a reader or an outcome.

### Attribution

Derived from the SPIDER dataset, used under CC BY 4.0:

> van der Graaf, J.W., van Hooff, M.L., Buckens, C.F.M. et al. Lumbar spine segmentation in MR images: a
> dataset and a public benchmark. *Scientific Data* 11, 264 (2024). https://doi.org/10.1038/s41597-024-03090-w

Dataset: https://zenodo.org/records/8009680
