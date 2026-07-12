#!/usr/bin/env python3
"""Audit D-015 semantic-renderer zero-update ablation outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage_d_training import FORMAL_RELEASE_SHA256, FORMAL_SOURCE_SHA256, FORMAL_STEMS
from utils.semantic_renderer_ablation import ARM_NAMES, SCHEMA


def load_json(path: Path, errors: list[str], label: str):
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"unreadable {label}: {exc}")
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    errors, warnings = [], []
    operator = load_json(output / "operator_record.json", errors, "operator record")
    if operator.get("schema") != SCHEMA:
        errors.append("operator schema mismatch")
    if operator.get("optimizer_updates") != 0:
        errors.append("operator recorded optimizer updates")
    if operator.get("checkpoints_written") != 0 or operator.get("ply_written") != 0:
        errors.append("operator recorded checkpoint/PLY writes")
    if operator.get("release_aggregate_before") != FORMAL_RELEASE_SHA256:
        errors.append("release aggregate before mismatch")
    if operator.get("release_aggregate_after") != FORMAL_RELEASE_SHA256:
        errors.append("release aggregate after mismatch")
    if operator.get("release_unchanged") is not True:
        errors.append("release tree changed or was not audited")
    forbidden = [
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.name.startswith("chkpnt") or path.suffix.lower() == ".ply"
    ]
    if forbidden:
        errors.append(f"forbidden resumable artifacts exist: {forbidden}")
    aggregate = load_json(
        output / "semantic_renderer_ablation_aggregate.json",
        errors, "aggregate JSON",
    )
    expected_rows = 2 * len(FORMAL_STEMS) * len(ARM_NAMES)
    if len(aggregate if isinstance(aggregate, list) else []) != expected_rows:
        errors.append("aggregate row count mismatch")
    groups = operator.get("group_metadata", {})
    for group in ("fresh_15000", "trained_20000"):
        metadata = load_json(output / group / "group_metadata.json", errors, f"{group} metadata")
        proof = metadata.get("zero_optimizer_update_proof", {})
        if proof.get("backward_called") is not False:
            errors.append(f"{group} backward proof failed")
        if proof.get("optimizer_step_called") is not False:
            errors.append(f"{group} optimizer proof failed")
        if proof.get("scheduler_step_called") is not False:
            errors.append(f"{group} scheduler proof failed")
        if proof.get("state_hash_unchanged") is not True:
            errors.append(f"{group} model hash changed")
        if metadata.get("model_hash_before") != metadata.get("model_hash_after"):
            errors.append(f"{group} before/after hashes differ")
        if group == "fresh_15000" and metadata.get("source_checkpoint_sha256") != FORMAL_SOURCE_SHA256:
            errors.append("fresh_15000 source SHA mismatch")
        snapshots = set((metadata.get("shared_t_snapshot_hashes") or {}).values())
        if len(snapshots) != 1:
            errors.append(f"{group} did not reuse one T snapshot across fixed views")
        for arm in ARM_NAMES:
            for stem in FORMAL_STEMS:
                stats = load_json(output / group / arm / stem / "stats.json", errors, f"{group}/{arm}/{stem} stats")
                if stats.get("schema") != SCHEMA:
                    errors.append(f"{group}/{arm}/{stem} stats schema mismatch")
                    continue
                parity = stats.get("outside_mask_parity") or {}
                if not parity or not all(parity.values()):
                    errors.append(f"{group}/{arm}/{stem} outside-mask parity failed")
                if stats.get("black_attribution", {}).get("near_black_threshold") != 0.10:
                    errors.append(f"{group}/{arm}/{stem} black threshold mismatch")
                tensor_dir = output / group / arm / stem / "tensors"
                if not tensor_dir.is_dir():
                    errors.append(f"{group}/{arm}/{stem} missing tensor directory")
    verdict = "BLOCKED" if errors else "AWAITING_USER_REVIEW"
    result = {
        "schema": "rtgs_stage_d_semantic_renderer_ablation_audit_v1",
        "verdict": verdict,
        "errors": errors,
        "warnings": warnings,
        "semantic_separation_claimed": False,
    }
    (output / "semantic_renderer_ablation_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
