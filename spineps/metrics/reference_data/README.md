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

### Scale: SPINEPS output is wider than these numbers

These percentiles come from **human** canal annotations. SPINEPS' canal label is systematically wider.
Measured on 40 SPIDER studies with identical level bands so that only the canal source differed:

| | value |
|---|---|
| Paired levels | 278 |
| Bias (SPINEPS − reference) | **+1.20 mm** |
| Mean absolute difference | 1.32 mm |
| Correlation | r = +0.913 |
| Per-level bias range | +0.64 to +1.59 mm, no caudal-cranial drift |

Comparing a raw SPINEPS measurement against this table would therefore make every level look about a
millimetre roomier than it is — a systematic bias towards *missing* narrowing. `percentile_rank()` and
`is_unusually_narrow()` default to `source="spineps"` and subtract `SPINEPS_CANAL_OFFSET_MM` for you; pass
`source="reference"` if you measured on a human annotation.

**The agreement above is optimistic.** SPINEPS was trained on 179 of SPIDER's 218 subjects, and this
comparison did not exclude them, so r = +0.913 reflects partly-seen data. The 1.20 mm offset should be
re-estimated on a dataset SPINEPS has never seen before it is relied on.

---

## `rsna_l4l5_operating_points.tsv`

Sensitivity, specificity and predictive values for `narrow_ap_diameter_mm` measured at the L4/L5 disc,
scored against radiologist-assigned canal stenosis grade (Moderate or Severe versus Normal/Mild).

**This is the only part of this package validated against a clinical reference.** Everything else —
backend reproducibility, agreement with human canal annotation, coverage invariance — establishes that the
measurement is well behaved, not that it means anything.

### What was measured

| | |
|---|---|
| Dataset | RSNA 2024 Lumbar Spine Degenerative Classification, sagittal T2/STIR |
| Studies | 286 measured at L4/L5 (of 300 segmented, 0 segmentation failures) |
| Level match rate | 99% (281/284 at the time of the level-matching count) |
| **AUC** | **0.917** (0.908 on a prevalence-matched subsample) |
| Sampling | class-balanced 150/150, positives drawn representatively (48.7% Moderate / 51.3% Severe vs 48.2/51.8 in the population) |

Balancing does not inflate AUC — it is a rank statistic over positive-negative pairs and so is invariant
to class mixing; the prevalence-matched subsample confirms this empirically. Balancing *does* inflate
anything conditioned on a positive call, which is why the table reports predictive values at the true
population prevalence of **24.9%** rather than at the sampled 50%.

### How to read it

**This behaves as a rule-out test.** NPV runs 0.94–0.99 across the useful range: a wide canal on sagittal
T2 reliably means a reader will not call significant stenosis. PPV at the Youden-optimal 10.8 mm is 0.67,
so a narrow measurement triages rather than diagnoses. That asymmetry is expected — canal stenosis is
graded on the **axial** dural sac, so a sagittal AP diameter is a good screen and a poor adjudicator.

### Do not over-apply this

- **L4/L5 only.** Prevalence and anatomy differ by level (24.9% at L4/L5 against 3.5% at L5/S1), so these
  thresholds do not transfer. Canal diameter also varies by level; see the normative table above.
- **Diameters here are raw SPINEPS output**, not offset-corrected. Do not combine these thresholds with
  `calibrate()`.
- Predictive values are derived from sensitivity/specificity via Bayes, not directly observed at 24.9%.
- One dataset, one grading scheme, no external replication, and grades carry inter-reader variability.
