# Changelog

## 2026-05-13 - Space Advisor/PFF SOTA hardening

### Added

- Added a reproducible Docker image-size measurement script:
  `scripts/package/measure-image-sizes.sh`.
- Added a Docker runtime matrix covering CPU-only, NVIDIA/CUDA, integrated/no GPU,
  Linux, Windows/WSL2, macOS and CI in `docs/docker-runtime-matrix.md`.
- Added a prioritized SOTA audit in `docs/sota-repo-audit.md`, ranked by impact,
  effort, risk and urgency.
- Added formal proof material to `docs/deep-research-report-abnt.pdf`: definitions,
  hypotheses, lemmas, theorem, demonstrations, limits, threats to validity and
  intellectual-property potential without claiming patentability.
- Added Advisor ablation evidence to the PDF/report using the real cutoff-25 payload:
  full run, no surrogate, no interactions, no internal importances, no bootstrap and
  no self-audit.
- Added a paired 50-trial benchmark script for TPE, GP-BO and Advisor ablations:
  `scripts/benchmarks/search_space_advisor_paired_benchmark.py`.

### Changed

- Docker build defaults now produce only the CPU image. Runtime, tools, test and all
  targets are explicit in `scripts/package/build-images.sh`.
- Packaging smoke defaults to CPU-only and supports `PFF_SMOKE_BUILD_TARGET=none`
  to reuse existing images or `PFF_SMOKE_BUILD_TARGET=runtime` for CPU+CUDA.
- GPU smoke no longer runs against an arbitrary existing CUDA image in skip-build mode
  unless `PFF_SMOKE_RUN_GPU=1` or `PFF_SMOKE_REQUIRE_GPU=1` is set.
- Packaging smoke now mounts temporary `data`, `logs` and `outputs` directories by
  default so `clean deep` cannot delete local research artifacts.
- Main dependency lock resolves the CPU wheel (`torch==2.7.0+cpu`) while the
  public project requirement stays at `torch==2.7.0`, so CUDA builds can swap
  to `torch==2.7.0+cu128` without package metadata conflicts.
- CUDA image builds explicitly swap to `torch==2.7.0+cu128` and `triton==3.3.0`
  in the CUDA target.
- README now records observed image sizes, image budgets and reproducible size
  measurement commands.
- CI now builds `pff:ci` with Buildx cache `type=gha`, loads the runtime image into
  the Docker daemon, runs `measure-image-sizes.sh --fail-on-budget` and uploads the
  TSV budget report.
- Dashboard bundle verification no longer depends on the removed chart barrel file.
- Pyright target version now matches the project runtime contract: Python 3.12.

### Verified

- `pff:cpu-lock-check`: 2.84 GB in Docker CLI display; 2.64 GiB by image bytes.
- `pff:cuda-lock-check`: 8.45 GiB by image bytes after the CPU-first lock change.
- CPU runtime: `torch==2.7.0+cpu`, `torch.version.cuda is None`, no `triton` or
  `nvidia-*-cu12` packages, no Poetry cache in `/root/.cache/pypoetry`.
- Advisor cutoff-25 audit: 21 recommendations, 25 complete trials, validation
  Wilson-LB 0.8454, mean confidence 0.5847, directional self-audit Wilson-LB 0.3628.
- Advisor real 50-complete audit for study `deep_research_advisor_real50_gpu_20260506`:
  50 complete trials observed in a 60-entry resumed dashboard payload, best objective
  0.469644, 21 recommendations, 37 evaluated prefixes, directional hit-rate 0.7903
  and validation Wilson-LB 0.7733.
- Advisor ablations: removing bootstrap reduced mean confidence from 0.5847 to
  0.4367 while preserving the action distribution in this cutoff.
- Paired synthetic benchmark at 50 trials: Advisor full improved over TPE mean by
  +0.027655 with 4/5 seed wins, but did not support a universal SOTA claim against
  GP-BO (`GPSampler` mean delta vs TPE +0.122074; Advisor delta vs GP-BO -0.094419).
- ABNT PDF regenerated from the final dashboard/audit payload with Pandoc, XeLaTeX
  and the ABNT CSL style in `docs/abnt.csl`; visual render checks covered the cover
  page and the paired benchmark table.
- CPU packaging smoke passed with `PFF_SMOKE_BUILD_TARGET=none` against
  `pff:cpu-lock-check`; GPU smoke was skipped unless explicitly requested.

### Remaining Work

- Run a real CUDA smoke on an NVIDIA host after the CPU-first lock change.
