# SPINEPS and VERIDAH

This is a segmentation pipeline to automatically, and robustly, segment the whole spine in T2w sagittal images.

> ## About this fork
>
> This is a fork of **[Hendrik-code/spineps](https://github.com/Hendrik-code/spineps)** whose purpose is to
> make SPINEPS run on machines without an NVIDIA GPU: Apple Silicon through Metal (MPS), and CPU-only hosts.
>
> **The method, the trained models, and the research behind them are entirely upstream's work.** This fork
> changes no methodology, no network architecture, and no model weights — it fixes platform support and
> bugs. If you use it, **cite the original papers** ([Citation](#citation)); there is nothing here to cite.
>
> - What is different, and what it costs you: **[Changes in this fork](#changes-in-this-fork)**
> - Numerical agreement between backends: **[How closely does Metal agree with the CPU?](#how-closely-does-metal-agree-with-the-cpu)**
> - For the canonical, CUDA-targeted version, use [upstream](https://github.com/Hendrik-code/spineps).
>
> Bugs in the fork-specific behaviour belong in [this fork's issues](https://github.com/kevinj24fr/spineps/issues).
> Anything about the segmentation itself is an upstream matter.

## Changes in this fork

| Area | Change |
|---|---|
| **Runs without CUDA** | Upstream's nnU-Net phases call `torch.cuda.mem_get_info()` unconditionally, so on a machine with no CUDA they abort with `ValueError: Expected a cuda device` — `-cpu` included, leaving no working fallback. Those queries are now device-aware. |
| **Metal (MPS) support** | Every device decision went through `cuda if torch.cuda.is_available() else cpu`, so Apple GPUs were never used. A single tested policy (`spineps/utils/device.py`) now resolves the backend, exposed as `-device {auto,cpu,cuda,mps}`. |
| **Named backends fail loudly** | Asking for a backend that is unavailable raises instead of quietly running elsewhere, because the backends do not produce identical masks. Only `auto` falls back, and it logs its choice. |
| **`-cpu` actually applies** | The vertebra-labeling classifier ignored `use_cpu` entirely and used the GPU regardless. |
| **Device recorded in output** | Each model entry in `*_ctd.json` now carries the device it ran on, so a mask's provenance includes where it was computed. |
| **`spineps sample --help`** | Crashed with an `AssertionError` from argparse at every terminal width. Fixed. |
| **SciPy 2.0 readiness** | The labeling phase imported from `scipy.ndimage.interpolation`, a namespace scheduled for removal. |
| **Honest dependency metadata** | `torch` was never declared, though Metal depends on it: before 2.3, `mps.is_available()` returns `True` and then `Conv3D` raises. Now pinned `>=2.3`. The Python range claimed up to 3.14 while the pinned `antspyx` caps at 3.12. |
| **Derived measurements** | A `spineps.metrics` package turning the masks into numbers: per-level canal geometry, level-numbering confidence flags, and paraspinal muscle/fat volumes read from the whole-body segmentation the pipeline already computes and discards. Includes a per-level reference distribution from 210 human-annotated studies, and the one endpoint validated against a clinical reference (canal AP diameter at L4/L5 versus radiologist stenosis grade, AUC 0.917 on 286 RSNA studies). See [`spineps/metrics/reference_data/README.md`](spineps/metrics/reference_data/README.md) for what that does and does not establish. |
| **Housekeeping** | Removed 1,479 lines of unreachable vendored nnU-Net code; `ruff check` and all pre-commit hooks now pass; CI tests macOS as well as Linux and Windows. |

Everything else — models, labels, pipeline structure, CLI semantics — is upstream's and unchanged.

## Installation (macOS / Apple Silicon)

> Fork-specific. Upstream does not run without CUDA — see [Changes in this fork](#changes-in-this-fork).

### How closely does Metal agree with the CPU?

Closely, but **not exactly**, and the difference is worth understanding before you use it for anything
that gets published. Measured on a 10-study clinical cohort of sagittal T2w data, each study run on
both backends:

| | Metal vs CPU |
|---|---|
| Voxels identical | 99.7% |
| Mean Dice, semantic mask | 0.966 |
| Mean Dice, instance mask | 0.972 |
| Vertebra level identities | **identical on every study** |

Disagreement concentrates in thin structures, where a single voxel of boundary jitter costs a lot of
Dice: spinous process 0.931 and endplates 0.921, against 0.978 for the spinal canal.

**The level identities are the reassuring part.** Across the cohort, both backends assigned the same
names to the same vertebrae every time — including in the four of five studies where SPINEPS called a
transitional level (T13 or L6). Nothing about the backend choice shifted the numbering.

**Per-structure measurements do move, though.** In a downstream cross-sectional-area measurement built
on these masks, half of the shared levels came out bit-identical and the rest shifted by under 2 mm²,
with two outliers at 13–14 mm². If your endpoint is the absolute value of a small structure, that is not
noise you can ignore. If your endpoint depends on the *ordering* of levels, it was unaffected: two
independent correlation results were unchanged to three decimals across backends.

The cause is ordinary floating-point non-determinism, not a Metal bug. The two backends accumulate
convolutions in a different order, which shifts logits by around 1e-2. Almost everywhere that is
irrelevant, but at a partial-volume boundary where two classes are nearly tied it can flip the argmax.
This is the same class of difference you get between two CUDA GPUs of different generations.

Practical guidance:

- **Run a cohort on one backend.** Don't mix Metal and CPU results within a study.
- The resolved device is recorded per model in the output `*_ctd.json`, so you can always tell after
  the fact where a mask was computed.
- Naming a backend explicitly (`-device mps`) is treated as an instruction: if it isn't available the
  run fails rather than silently using something else. Only `-device auto` will fall back.
- Whether this matters for *your* endpoint depends on the endpoint. A measurement that uses the
  segmentation only as a seed and then grows a region on the source image can be completely unaffected;
  one that integrates a thin structure's volume directly will not be. Check yours rather than assuming.

### Runtime

Secondary to the above, but the practical reason this is usable: on the same 10-study cohort, Metal ran
each study in 47–124 s against 571 s on the CPU, roughly 9–10x. The point is that the pipeline runs at
all on this hardware; the speed is what makes running it repeatedly bearable.

### Setup on a Mac

Two things matter:

1. **Use a native arm64 Python.** An x86_64 interpreter running under Rosetta cannot reach Metal at all and
   silently falls back to the CPU. Check with:
```bash
python -c "import platform; print(platform.machine())"   # must print arm64
```
2. **Confirm Metal is visible to PyTorch:**
```bash
python -c "import torch; print(torch.backends.mps.is_available())"   # must print True
```

Then install the package as described below. Metal is selected automatically, so no flag is needed:
```bash
spineps sample -i path/to/image.nii.gz -ms t2w
```
To pin the device explicitly, pass `-device mps` (or `-device cpu` to compare). See
[Device selection](#device-selection).

> Apple Silicon GPUs have no CUDA, so `nvidia-smi` and `torch.cuda.is_available()` do not apply.

> **Memory:** peak Metal allocation is about **3.2 GB** for a single vertebra cutout, so 16 GB of unified
> memory is comfortable and 8 GB is tight. Cutouts are deliberately processed one at a time; batching them
> is slower on Metal, not faster.


### Device selection

By default SPINEPS picks the fastest backend available: **CUDA**, then **Metal (MPS)** on Apple Silicon,
then the **CPU**. Override it with `-device`:

```bash
spineps sample -i <image> -ms t2w -device auto   # default: cuda > mps > cpu
spineps sample -i <image> -ms t2w -device mps    # force the Apple Silicon GPU
spineps sample -i <image> -ms t2w -device cuda   # force an NVIDIA GPU
spineps sample -i <image> -ms t2w -device cpu    # force the CPU (same as -cpu)
```

**Naming a backend is an instruction, not a preference.** `-device mps` on a machine without Metal, or
`-device cuda` without an NVIDIA GPU, fails with a clear error. It does not quietly run somewhere else,
because the backends do not produce identical masks (see
[How closely does Metal agree with the CPU?](#how-closely-does-metal-agree-with-the-cpu)) and a batch
script that asked for `cuda` and silently got Metal would produce different results with no record of it.
Only `-device auto` is allowed to degrade, and it logs which backend it picked.

`-cpu` is kept as an alias for `-device cpu`. Unlike upstream it now also takes effect for the
vertebra-labeling classifier, which previously ignored it and used the GPU regardless.

### Issues

- import issues: try installing via the requirements again, somethings it doesn't install everything
- pytorch / cuda issues: good luck! :3
- on macOS, `torch.backends.mps.is_available()` returning `False` almost always means the Python
  interpreter is x86_64 under Rosetta rather than native arm64


## SPINEPS Capabilities

The pipeline can process either:
- Single Nifty (.nii.gz) files
- Whole Datasets

#### Example
```bash
#T2w sagittal
spineps sample -ignore_bids_filter -ignore_inference_compatibility -i /path/sub-testsample_T2w.nii.gz -model_semantic t2w -model_instance instance
#T1w sagittal
spineps sample -ignore_bids_filter -ignore_inference_compatibility -i ~/path/sub-testsample_T1w.nii.gz -model_semantic t1w -model_instance instance
```


## Citation

Cite these whether you use upstream or this fork. The fork should not displace the original work.

If you are using SPINEPS, please cite the following:

```
SPINEPS:

Hendrik Möller, Robert Graf, Joachim Schmitt, Benjamin Keinert, Hanna Schön, Matan Atad,
Anjany Sekuboyina, Felix Streckenbach, Florian Kofler, Thomas Kroencke, Stefanie Bette,
Stefan N. Willich, Thomas Keil, Thoralf Niendorf, Tobias Pischon, Beate Endemann, Bjoern Menze,
Daniel Rueckert, Jan S. Kirschke. SPINEPS—automatic whole spine segmentation of
T2-weighted MR images using a two-phase approach to multi-class semantic and instance segmentation.
Eur Radiol (2024). https://doi.org/10.1007/s00330-024-11155-y

Source of the T2w/T1w Segmentation:

Robert Graf, Joachim Schmitt, Sarah Schlaeger, Hendrik Kristian Möller, Vasiliki
Sideri-Lampretsa, Anjany Sekuboyina, Sandro Manuel Krieg, Benedikt Wiestler, Bjoern
Menze, Daniel Rueckert, Jan Stefan Kirschke. Denoising diffusion-based MRI to CT image
translation enables automated spinal segmentation. Eur Radiol Exp 7, 70 (2023).
https://doi.org/10.1186/s41747-023-00385-2
```

## License

Copyright 2023 Hendrik Möller

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
