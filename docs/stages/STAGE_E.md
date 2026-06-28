# Stage E — Stabilization, Evaluation, Ablation, and Export

## Summary

Stabilize full training, add branch/region metrics, implement all required
ablations and render modes, export every model/mesh/cache artifact, and document
reproducible experiments from a clean environment.

## Prerequisites

- Stages A–D are accepted and committed without regression.
- Full D/R/T training, checkpoint resume, mesh cache, and debug decomposition are
  operational.
- `docs/STATUS.md` explicitly advances the project to Stage E.

## Acceptance entry

Begin final acceptance only when every mandated ablation and metric can run, all
render/export modes work, training resumes with configuration metadata, the
README reproduces results from a clean environment, all engineering assumptions
are logged, and a release commit/tag is ready.
