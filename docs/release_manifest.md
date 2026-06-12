# Release Manifest

This manifest records the scope of the public lightweight release and its
alignment with the manuscript.

## Manuscript-Aligned Model Facts

- Reported main model: one consistently trained 120,000-iteration checkpoint.
- Reported architecture: 76 intermediate H-MoE sites plus one shared
  reconstruction head.
- Reported routing: deterministic known-modality routing; no learned soft
  expert competition and no load-balancing loss in the H-MoE path.
- Reported output path: one shared reconstruction head for MRI, CT, and PET.
- Not used for reported main results: 4,000-step refinement checkpoints,
  low-learning-rate continuation checkpoints, or historical 79-registered-module
  tail-wrapped variants.

## Included in This Repository

- Core model definition for a shared-reconstruction-head Mamba-MoE variant.
- Project-compatible VSSBlock adapter used by the released model code.
- Deterministic H-MoE wrapper with modality-selected residual experts.
- Lightweight paired-file dataset loader for local restoration experiments.
- Minimal training, evaluation, inference, and prediction-export utilities.
- Metric definitions and reproducibility notes.

## Not Included in This Lightweight Release

- Medical image datasets.
- Pretrained checkpoints.
- Full benchmark-specific dataloaders and saved-prediction directories.
- Full case-level statistical analysis artifacts.
- Raw training logs.

These artifacts are large, access-restricted, or benchmark-specific. They are
tracked separately in the project evidence package and will be released when
the manuscript release package is finalized.

## Evidence Package to Preserve Outside Git

For manuscript provenance, preserve the following outside this code repository:

- final checkpoint filename and SHA-256 hash;
- final training configuration;
- strict saved-prediction evaluation CSV files;
- per-case metric CSV files;
- bootstrap confidence-interval scripts and outputs;
- scripts used to generate manuscript figures and tables.

Do not commit checkpoints, raw medical images, or saved prediction volumes to
this repository.

## Artifact Release Checklist

| Artifact | Current status | Planned release path |
| --- | --- | --- |
| Final 120k checkpoint | not included | external release artifact with SHA-256 |
| Checkpoint SHA-256 | pending | release manifest update |
| Final training configuration | summarized only | full artifact package |
| Table 1 summary CSV | not included | full artifact package |
| Per-case metric CSV files | not included | full artifact package |
| Bootstrap CI scripts and outputs | not included | full artifact package |
| PET lesion Dice / SUV bias evaluator | not included | full artifact package |
| CT sanity evaluation script | not included | full artifact package |
| Routing diagnostic scripts | not included | full artifact package |
| Module diagnostic scripts | not included | full artifact package |
| Figure-generation scripts | not included | full artifact package |
