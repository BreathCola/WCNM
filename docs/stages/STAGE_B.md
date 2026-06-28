# Stage B — Differentiable Ray Tracing and Reflection

## Summary

Add an independent Reflection surfel field, a shared differentiable Gaussian ray
tracer, reflection-ray generation, full microfacet reflection shading, and the
specular-mask constraint. Transmittance and mesh-guided second bounce remain out
of scope.

## Prerequisites

- Stage A acceptance is fully complete and committed.
- Diffuse surfel geometry/material output, debug maps, losses, baseline
  compatibility, and checkpoint resume are verified.
- `docs/STATUS.md` explicitly advances the project to Stage B.

## Acceptance entry

Begin Stage B acceptance only after an independent Reflection model/optimizer,
differentiable accelerated tracing, correct reflection rays, reflection debug
maps, full non-split-sum microfacet BRDF, specular-mask behavior, finite outputs,
and Stage A regression tests are all implemented and tested.
