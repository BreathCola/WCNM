# RT-GS Repository Instructions

`RTGS_MASTER_PLAN.md` at the repository root is the sole long-term technical
specification for this project. Do not create or rely on a second copy.

Before changing code, every contributor and agent must read, in order:

1. `RTGS_MASTER_PLAN.md`
2. `docs/STATUS.md`
3. `docs/DECISIONS.md`
4. `docs/stages/STAGE_<CURRENT>.md`

Only implement work allowed by the current stage. A stage may advance only
after its acceptance checklist, tests, debug outputs, checkpoint requirements,
status updates, and rollback commit are complete. Record every choice affecting
rendering formulas, data representation, losses, path definitions, or training
schedule in `docs/DECISIONS.md`.

Preserve the original 3DGS baseline. Never merge Diffuse, Reflection, and
Transmittance fields into one array, crop the external environment from training,
add dummy outputs for unimplemented features, claim unrun tests, or implement a
later stage early.
