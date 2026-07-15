import json
from argparse import Namespace

import pytest
import torch

import tools.audit_stage_d_internal_object_townership_to_20000 as audit
import tools.materialize_stage_d_internal_object_townership_to_20000 as materializer
from stage_d_training import (
    FORMAL_RELEASE_SHA256,
    FORMAL_SOURCE_SHA256,
    INTERNAL_OBJECT_TO_20000_NODES,
    INTERNAL_OBJECT_TO_20000_SOURCE_SHA256,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _checkpoint(node, t_delta=0.0, d_delta=0.0):
    return {
        "format": "rtgs_stage_d",
        "checkpoint_version": 1,
        "optimizer_step_completed": True,
        "global_iteration": node,
        "reflection_iteration": 12000,
        "transmittance_iteration": node - 15000,
        "geometry_release_id": "stage_c_geometry_release_v1",
        "geometry_release_aggregate_sha256": FORMAL_RELEASE_SHA256,
        "source": {"sha256": FORMAL_SOURCE_SHA256},
        "config": {
            "source_stage_b_checkpoint_sha256": FORMAL_SOURCE_SHA256,
            "stage_d_depth_start_iteration": 40000,
        },
        "diffuse": {"xyz": torch.ones(2, 3) + d_delta, "optimizer": {"step": torch.tensor([0])}},
        "reflection": {"xyz": torch.ones(3, 3), "optimizer": {"step": torch.tensor([0])}},
        "transmittance": {
            "xyz": torch.full((4096, 3), float(t_delta)),
            "optimizer": {"step": torch.tensor([node - 15000])},
        },
        "runtime_state": {},
        "rng_state": {},
    }


def _metrics():
    return {
        "bird_ain_mean": 0.8,
        "bird_ain_p50": 0.8,
        "bird_ain_p95": 0.9,
        "bird_ain_above_alpha_floor_ratio": 0.9,
        "internal_base_ain_mean": 0.7,
        "internal_base_ain_p50": 0.7,
        "internal_base_ain_p95": 0.8,
        "internal_base_ain_above_alpha_floor_ratio": 0.8,
        "union_ain_mean": 0.75,
        "union_ain_p50": 0.75,
        "union_ain_p95": 0.85,
        "union_ain_above_alpha_floor_ratio": 0.85,
        "mneg_ain_mean": 0.01,
        "ain_negative_spill_ge_0_10": 0.02,
        "bird_cin_energy": 0.2,
        "internal_base_cin_energy": 0.3,
        "mneg_cout_energy": 0.4,
    }


def _telemetry_rows():
    rows = []
    for step in range(15501, 20001):
        rows.append({
            "schema": audit.TELEMETRY_SCHEMA,
            "global_iteration": step,
            "transmittance_local_iteration": step - 15000,
            "training_phase": "internal_object_townership_t_only",
            "optimizer_updates_this_step": {"diffuse": 0, "reflection": 0, "transmittance": 1},
            "counts": {"diffuse": 2, "reflection": 3, "transmittance": 4096},
            "topology_event": {
                "transmittance_densify_prune_called": False,
                "diffuse_delta": 0,
                "reflection_delta": 0,
                "transmittance_delta": 0,
            },
            "loss": {
                "lambda_depth_enabled": False,
                "rgb": 0.1 + (20000 - step) * 1e-6,
                "l_object": 0.02,
                "total": 0.2,
            },
            "internal_object_metrics": _metrics(),
            "nonfinite_count": 0,
        })
    return rows


def _make_output(tmp_path):
    output = tmp_path / audit.INTERNAL_OBJECT_TO_20000_OUTPUT_NAME
    output.mkdir()
    source = tmp_path / "chkpnt15500.pth"
    torch.save(_checkpoint(15500, t_delta=0.0), source)
    for node in INTERNAL_OBJECT_TO_20000_NODES:
        torch.save(_checkpoint(node, t_delta=float(node)), output / f"chkpnt{node}.pth")
        for branch in ("diffuse", "reflection", "transmittance"):
            ply = output / "point_cloud" / branch / f"iteration_{node}" / "point_cloud.ply"
            ply.parent.mkdir(parents=True, exist_ok=True)
            ply.write_text("ply\n", encoding="utf-8")
    _write_json(output / "internal_object_townership_to_20000_metadata.json", {
        "schema": audit.METADATA_SCHEMA,
        "global": [15501, 20000],
        "transmittance_local": [501, 5000],
        "diffuse_optimizer_updates": 0,
        "reflection_optimizer_updates": 0,
        "transmittance_optimizer_updates": 4500,
        "expected_t_count": 4096,
        "t_reinitialization": False,
        "transferred_d_selection_rerun": False,
        "random_fill_rerun": False,
        "source": {"git_head": "head"},
    })
    (output / "stage_d_telemetry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in _telemetry_rows()),
        encoding="utf-8",
    )
    posthoc = output / "posthoc_review"
    for name in (
        "overview_final_15500_20000.png",
        "overview_cin_15500_20000.png",
        "overview_ain_15500_20000.png",
    ):
        (posthoc / name).parent.mkdir(parents=True, exist_ok=True)
        (posthoc / name).write_bytes(b"png")
    for node in audit.REVIEW_NODES:
        debug = posthoc / "debug" / f"iteration_{node:06d}"
        (debug / "contact_sheet.png").parent.mkdir(parents=True, exist_ok=True)
        (debug / "contact_sheet.png").write_bytes(b"png")
        (debug / "semantic_repair_contact_sheet.png").write_bytes(b"png")
        for stem in audit.FORMAL_STEMS:
            view = debug / stem
            view.mkdir(parents=True, exist_ok=True)
            for name in audit.REQUIRED_DEBUG:
                if name.endswith(".json"):
                    _write_json(view / name, {
                        "schema": "rtgs_stage_d_internal_object_townership_to_20000_float_metrics_v1",
                        "finite": 1.0,
                    })
                else:
                    (view / name).write_bytes(b"png")
    _write_json(posthoc / "materialization_manifest.json", {
        "schema": audit.POSTHOC_SCHEMA,
        "node_source_contract": {
            "deterministic_fresh_replay": False,
            "t_reinitialization": False,
            "transferred_d_selection_rerun": False,
            "random_fill_rerun": False,
        },
        "review_nodes": list(audit.REVIEW_NODES),
        "posthoc_review_artifacts": {"debug_root": "posthoc_review/debug"},
        "immutable_files_before_after": {},
        "no_optimizer_execution": True,
        "no_backward": True,
        "no_checkpoint_write": True,
    })
    plan = tmp_path / "plan.json"
    _write_json(plan, {
        "schema": audit.PLAN_SCHEMA,
        "output": str(output),
        "source_sha256": INTERNAL_OBJECT_TO_20000_SOURCE_SHA256,
        "release_aggregate_sha256": FORMAL_RELEASE_SHA256,
        "internal_object_role": "stage_d_internal_object_masks_reviewed",
        "internal_object_human_status": "accepted",
        "internal_object_semantics_version": "tihubird_bird_and_internal_base_v3",
        "iterations": {"end_inclusive": 20000},
        "nodes": list(INTERNAL_OBJECT_TO_20000_NODES),
        "command": ["python", "train.py"],
    })
    return output, plan, source


def _args(output, plan):
    return Namespace(
        output=output,
        operator_plan=plan,
        geometry_manifest=output / "geometry.json",
        scene=output / "scene",
        images="images",
        internal_object_manifest=output / "internal.json",
    )


def _patch_identities(monkeypatch, source):
    monkeypatch.setattr(audit, "SOURCE", source)
    monkeypatch.setattr(materializer, "SOURCE", source)
    monkeypatch.setattr(audit, "PILOT_AUDIT", source.parent / "pilot_audit.json")
    _write_json(source.parent / "pilot_audit.json", {
        "verdict": "D016_PILOT_PASS_AWAITING_USER_REVIEW",
        "technical_run_complete": True,
        "errors": [],
    })
    monkeypatch.setattr(audit, "sha256_file", lambda path: INTERNAL_OBJECT_TO_20000_SOURCE_SHA256 if path == source else "hash")
    monkeypatch.setattr(materializer, "sha256_file", lambda path: INTERNAL_OBJECT_TO_20000_SOURCE_SHA256 if path == source else "hash")
    monkeypatch.setattr(
        audit,
        "validate_geometry_release",
        lambda path: {
            "geometry_release_id": "stage_c_geometry_release_v1",
            "aggregate_sha256": FORMAL_RELEASE_SHA256,
        },
    )
    monkeypatch.setattr(
        audit,
        "validate_internal_object_mask_set",
        lambda *args: {
            "role": "stage_d_internal_object_masks_reviewed",
            "human_status": "accepted",
            "internal_object_semantics_version": "tihubird_bird_and_internal_base_v3",
            "aggregate_sha256": audit.EXPECTED_INTERNAL_AGGREGATE,
            "manifest_file_sha256": "manifest",
        },
    )


def test_to_20000_audit_passes_complete_synthetic_run(monkeypatch, tmp_path):
    output, plan, source = _make_output(tmp_path)
    _patch_identities(monkeypatch, source)

    result, verdict = audit._audit(_args(output, plan))

    assert verdict == audit.VERDICT_PASS
    assert result["technical_run_complete"] is True
    assert result["telemetry_rows"] == 4500
    assert result["joint_training_authorized"] is False
    assert result["stage_e_authorized"] is False
    assert result["recommended_review_candidates"]["endpoint_20000"] == 20000


def test_to_20000_audit_blocks_missing_checkpoint(monkeypatch, tmp_path):
    output, plan, source = _make_output(tmp_path)
    _patch_identities(monkeypatch, source)
    (output / "chkpnt18000.pth").unlink()

    result, verdict = audit._audit(_args(output, plan))

    assert verdict == audit.VERDICT_BLOCKED
    assert "missing checkpoint 18000" in result["errors"]


def test_to_20000_audit_blocks_telemetry_gap(monkeypatch, tmp_path):
    output, plan, source = _make_output(tmp_path)
    _patch_identities(monkeypatch, source)
    rows = _telemetry_rows()
    rows.pop(10)
    (output / "stage_d_telemetry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result, verdict = audit._audit(_args(output, plan))

    assert verdict == audit.VERDICT_BLOCKED
    assert any("continuous global 15501--20000" in error for error in result["errors"])


def test_to_20000_audit_blocks_dr_change(monkeypatch, tmp_path):
    output, plan, source = _make_output(tmp_path)
    _patch_identities(monkeypatch, source)
    torch.save(_checkpoint(20000, t_delta=1.0, d_delta=1.0), output / "chkpnt20000.pth")

    result, verdict = audit._audit(_args(output, plan))

    assert verdict == audit.VERDICT_BLOCKED
    assert any("diffuse parameters changed" in error for error in result["errors"])


def test_to_20000_audit_blocks_replay_materialization(monkeypatch, tmp_path):
    output, plan, source = _make_output(tmp_path)
    _patch_identities(monkeypatch, source)
    manifest = json.loads((output / "posthoc_review" / "materialization_manifest.json").read_text())
    manifest["node_source_contract"]["deterministic_fresh_replay"] = True
    _write_json(output / "posthoc_review" / "materialization_manifest.json", manifest)

    result, verdict = audit._audit(_args(output, plan))

    assert verdict == audit.VERDICT_BLOCKED
    assert any("fresh replay" in error for error in result["errors"])


def test_to_20000_materializer_refuses_existing_posthoc(tmp_path):
    output = tmp_path / materializer.INTERNAL_OBJECT_TO_20000_OUTPUT_NAME
    output.mkdir()
    (output / "posthoc_review").mkdir()

    with pytest.raises(FileExistsError, match="refusing existing posthoc output"):
        materializer.validate_preconditions(output, output / "posthoc_review")


def test_to_20000_materializer_validates_plan_contract(tmp_path):
    output = tmp_path / materializer.INTERNAL_OBJECT_TO_20000_OUTPUT_NAME
    output.mkdir()
    plan = {
        "schema": materializer.PLAN_SCHEMA,
        "output": str(output),
        "source_sha256": INTERNAL_OBJECT_TO_20000_SOURCE_SHA256,
        "release_aggregate_sha256": FORMAL_RELEASE_SHA256,
        "internal_object_role": "stage_d_internal_object_masks_reviewed",
        "internal_object_human_status": "accepted",
        "internal_object_semantics_version": "tihubird_bird_and_internal_base_v3",
        "iterations": {"end_inclusive": 20000},
        "nodes": list(INTERNAL_OBJECT_TO_20000_NODES),
    }
    materializer.validate_plan(plan, output)
    plan["internal_object_role"] = "stage_d_internal_object_mask_proposal_111_v3"
    with pytest.raises(ValueError, match="role mismatch"):
        materializer.validate_plan(plan, output)
