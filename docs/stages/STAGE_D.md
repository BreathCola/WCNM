# Stage D — Transmittance Gaussian and Full RT-GS

## Summary

Add an independent Transmittance surfel field, first-bounce inside tracing,
mesh-guided second-bounce Diffuse tracing, alpha-over transmittance composition,
the delayed depth constraint, and joint D/R/T optimization/final composition.

## Prerequisites

- Stages A–C are accepted and committed without regression.
- Reflection tracing and full microfacet shading are stable.
- The transparent mesh and two-hit cache are fixed, valid, and versioned.
- `docs/STATUS.md` explicitly advances the project to Stage D.

## Acceptance entry

Begin Stage D acceptance only after three independent fields/optimizers,
inside/outside tracing and debug outputs, constrained first-hit depth, correct
alpha-over transmittance, full D/R/T composition, expected branch separation,
finite gradients, and prior-stage regression tests are verified.
