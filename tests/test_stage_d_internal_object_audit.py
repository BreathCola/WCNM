import json
from argparse import Namespace

import torch

import tools.audit_stage_d_internal_object_townership as audit
from stage_d_training import FORMAL_RELEASE_SHA256, FORMAL_SOURCE_SHA256, INTERNAL_OBJECT_NODES


FILTER_METADATA = {
    "schema": "rtgs_stage_d_internal_object_transfer_filter_v3",
    "pre_filter_D_indices_sha256": "pre",
    "selected_D_indices_sha256": "selected",
    "rejected_D_indices_sha256": "rejected",
    "random_fill_count": 2864,
    "pre_object_mask_candidate_count": 4096,
    "post_object_mask_candidate_count": 1232,
    "selected_transferred_count": 1232,
    "rejected_min_views_count": 10,
    "rejected_support_ratio_count": 20,
    "valid_projection_views_histogram": {"3": 1},
    "visible_domain_views_histogram": {"4": 2},
    "positive_object_views_histogram": {"5": 3},
    "per_surfel_support_summary": {"min": 0.6, "max": 1.0},
}
INITIAL_T_HASH = "0" * 64


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _checkpoint(node, t_delta=0.0):
    return {
        "format": "rtgs_stage_d",
        "global_iteration": node,
        "reflection_iteration": 12000,
        "transmittance_iteration": node - 15000,
        "geometry_release_aggregate_sha256": FORMAL_RELEASE_SHA256,
        "config": {
            "source_stage_b_checkpoint_sha256": FORMAL_SOURCE_SHA256,
            "stage_d_depth_start_iteration": 40000,
        },
        "diffuse": {"xyz": torch.ones(2, 3), "optimizer": {"step": torch.tensor([0])}},
        "reflection": {"xyz": torch.ones(3, 3), "optimizer": {"step": torch.tensor([0])}},
        "transmittance": {
            "xyz": torch.full((4096, 3), float(t_delta)),
            "optimizer": {"step": torch.tensor([node - 15000])},
        },
    }


def _split_metrics():
    metrics = {}
    for name in ("bird", "internal_base", "union", "mneg"):
        metrics[f"{name}_pixel_count"] = 8
        metrics[f"{name}_ain_mean"] = 0.2
        metrics[f"{name}_ain_median"] = 0.2
        metrics[f"{name}_ain_p05"] = 0.1
        metrics[f"{name}_ain_p50"] = 0.2
        metrics[f"{name}_ain_p95"] = 0.4
        metrics[f"{name}_ain_max"] = 0.5
        metrics[f"{name}_ain_nonzero_ratio"] = 1.0
        metrics[f"{name}_ain_above_alpha_floor_ratio"] = 0.25
    metrics.update({
        "mneg_cout_energy": 0.3,
        "outside_union_cout_energy": 0.4,
    })
    return metrics


def _make_output(tmp_path):
    output = tmp_path / "stage_d_tihubird_c03r8_internal_object_townership_pilot_v1"
    output.mkdir()
    plan = tmp_path / "d016_plan.json"
    source = tmp_path / "source_chkpnt15000.pth"
    torch.save(_checkpoint(15000, t_delta=0.0), source)
    _write_json(plan, {
        "schema": "rtgs_stage_d_internal_object_operator_plan_v1",
        "execute": True,
        "output": str(output),
        "source": str(source),
        "source_sha256": FORMAL_SOURCE_SHA256,
        "release_aggregate_sha256": FORMAL_RELEASE_SHA256,
        "internal_object_manifest_sha256": "manifest",
    })
    _write_json(output / "internal_object_townership_metadata.json", {
        "schema": "rtgs_stage_d_internal_object_townership_pilot_v1",
        "pilot_global": [15001, 15500],
        "diffuse_optimizer_updates": 0,
        "reflection_optimizer_updates": 0,
        "transmittance_optimizer_updates": 500,
        "expected_t_count": 4096,
        "actual_transmittance_initialization": {
            "selection": {
                "internal_object_filter": dict(FILTER_METADATA),
            }
        },
    })
    rows = []
    for step in range(15001, 15501):
        rows.append({
            "schema": "rtgs_stage_d_internal_object_townership_telemetry_v1",
            "global_iteration": step,
            "optimizer_updates_this_step": {"diffuse": 0, "reflection": 0, "transmittance": 1},
            "loss": {"lambda_depth_enabled": False, "total": 1.0},
            "counts": {"diffuse": 2, "reflection": 3, "transmittance": 4096},
            "topology_event": {
                "transmittance_densify_prune_called": False,
                "diffuse_delta": 0,
                "reflection_delta": 0,
                "transmittance_delta": 0,
            },
            "internal_object_metrics": _split_metrics(),
            "nonfinite_count": 0,
        })
    (output / "stage_d_telemetry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    for node in audit.TRAINING_NODES:
        torch.save(_checkpoint(node, t_delta=1.0), output / f"chkpnt{node}.pth")
        for branch in ("diffuse", "reflection", "transmittance"):
            ply = output / "point_cloud" / branch / f"iteration_{node}" / "point_cloud.ply"
            ply.parent.mkdir(parents=True, exist_ok=True)
            ply.write_text("ply\n", encoding="utf-8")
    posthoc = output / "posthoc_review"
    _write_json(posthoc / "initial_state_replay" / "initial_state_replay.json", {
        "schema": "rtgs_stage_d_internal_object_initial_replay_v1",
        "initial_state_kind": "deterministic_zero_update_replay",
        "optimizer_updates": 0,
        "transmittance_count": 4096,
        "first_replay_transmittance_state_sha256": INITIAL_T_HASH,
        "second_replay_transmittance_state_sha256": INITIAL_T_HASH,
        "replay_deterministic": True,
        "internal_object_filter": dict(FILTER_METADATA),
    })
    _write_json(posthoc / "materialization_manifest.json", {
        "schema": audit.POSTHOC_SCHEMA,
        "posthoc_replayed_initial_state": {
            "initial_state_kind": "deterministic_zero_update_replay",
            "optimizer_updates": 0,
            "transmittance_count": 4096,
            "first_replay_transmittance_state_sha256": INITIAL_T_HASH,
            "second_replay_transmittance_state_sha256": INITIAL_T_HASH,
            "replay_deterministic": True,
            "internal_object_filter": dict(FILTER_METADATA),
        },
        "posthoc_review_artifacts": {
            "debug_root": "posthoc_review/debug",
            "nodes": list(INTERNAL_OBJECT_NODES),
        },
        "raw_pilot_output_tree_sha256_before_materialization": "raw-tree",
        "immutable_files_before_after": {},
        "newly_added_derived_files": [],
        "no_optimizer_execution_during_materialization": True,
    })
    for node in INTERNAL_OBJECT_NODES:
        debug_root = posthoc / "debug" / f"iteration_{node:06d}"
        (debug_root / "contact_sheet.png").parent.mkdir(parents=True, exist_ok=True)
        (debug_root / "contact_sheet.png").write_bytes(b"png")
        for stem in audit.FORMAL_STEMS:
            view = debug_root / stem
            view.mkdir(parents=True, exist_ok=True)
            for name in audit.REQUIRED_DEBUG:
                if name.endswith(".json"):
                    _write_json(view / name, {"finite": 1.0})
                else:
                    (view / name).write_bytes(b"png")
    return output, plan


def _args(output, plan):
    return Namespace(
        output=output,
        operator_plan=plan,
        geometry_manifest=output / "geometry.json",
        scene=output / "scene",
        images="images",
        internal_object_manifest=output / "internal.json",
        expected_internal_aggregate="c0e49503e5f5c30b1ab26b7cfd79332ac9c486f35656f1425c9cefea516d4052",
    )


def _patch_identities(monkeypatch):
    monkeypatch.setattr(
        audit,
        "validate_geometry_release",
        lambda path: {
            "geometry_release_id": "stage_c_geometry_release_v1",
            "aggregate_sha256": FORMAL_RELEASE_SHA256,
            "cache_count": 111,
            "coverage": {"valid_fraction_hard_mean": 0.97},
            "runtime_generation_required": False,
        },
    )
    monkeypatch.setattr(
        audit,
        "validate_internal_object_mask_set",
        lambda *args: {
            "role": "stage_d_internal_object_masks_reviewed",
            "human_status": "accepted",
            "internal_object_semantics_version": "tihubird_bird_and_internal_base_v3",
            "aggregate_sha256": "c0e49503e5f5c30b1ab26b7cfd79332ac9c486f35656f1425c9cefea516d4052",
            "manifest_file_sha256": "manifest",
        },
    )


def test_internal_object_cpu_audit_passes_complete_synthetic_pilot(monkeypatch, tmp_path):
    output, plan = _make_output(tmp_path)
    _patch_identities(monkeypatch)

    result, verdict = audit._audit(_args(output, plan))

    assert verdict == audit.VERDICT_PASS
    assert result["technical_run_complete"] is True
    assert result["semantic_separation_claimed"] is False
    assert result["stage_e_authorized"] is False


def test_internal_object_cpu_audit_blocks_missing_split_metrics(monkeypatch, tmp_path):
    output, plan = _make_output(tmp_path)
    _patch_identities(monkeypatch)
    rows = [
        json.loads(line)
        for line in (output / "stage_d_telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["internal_object_metrics"] = {}
    (output / "stage_d_telemetry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result, verdict = audit._audit(_args(output, plan))

    assert verdict == audit.VERDICT_BLOCKED
    assert any("missing split internal-object metrics" in error for error in result["errors"])


def test_internal_object_cpu_audit_rejects_old_filter_metadata_path(monkeypatch, tmp_path):
    output, plan = _make_output(tmp_path)
    _patch_identities(monkeypatch)
    metadata = json.loads((output / "internal_object_townership_metadata.json").read_text(encoding="utf-8"))
    metadata["actual_transmittance_initialization"] = {
        "internal_object_filter": dict(FILTER_METADATA),
    }
    _write_json(output / "internal_object_townership_metadata.json", metadata)

    result, verdict = audit._audit(_args(output, plan))

    assert verdict == audit.VERDICT_BLOCKED
    assert any("actual_transmittance_initialization.selection" in error for error in result["errors"])


def test_internal_object_cpu_audit_does_not_require_root_15000_checkpoint(monkeypatch, tmp_path):
    output, plan = _make_output(tmp_path)
    _patch_identities(monkeypatch)

    result, verdict = audit._audit(_args(output, plan))

    assert verdict == audit.VERDICT_PASS
    assert not (output / "chkpnt15000.pth").exists()
    assert result["actual_training_checkpoints"] == [
        str(output / "chkpnt15100.pth"),
        str(output / "chkpnt15250.pth"),
        str(output / "chkpnt15500.pth"),
    ]


def test_internal_object_cpu_audit_blocks_without_posthoc_manifest(monkeypatch, tmp_path):
    output, plan = _make_output(tmp_path)
    _patch_identities(monkeypatch)
    (output / "posthoc_review" / "materialization_manifest.json").unlink()

    result, verdict = audit._audit(_args(output, plan))

    assert verdict == audit.VERDICT_BLOCKED
    assert "missing posthoc review materialization manifest" in result["errors"]


def test_internal_object_cpu_audit_blocks_missing_real_training_checkpoint(monkeypatch, tmp_path):
    output, plan = _make_output(tmp_path)
    _patch_identities(monkeypatch)
    (output / "chkpnt15250.pth").unlink()

    result, verdict = audit._audit(_args(output, plan))

    assert verdict == audit.VERDICT_BLOCKED
    assert "missing checkpoint 15250" in result["errors"]


def test_internal_object_cpu_audit_blocks_filter_replay_mismatch(monkeypatch, tmp_path):
    output, plan = _make_output(tmp_path)
    _patch_identities(monkeypatch)
    manifest_path = output / "posthoc_review" / "materialization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["posthoc_replayed_initial_state"]["internal_object_filter"]["selected_D_indices_sha256"] = "wrong"
    _write_json(manifest_path, manifest)

    result, verdict = audit._audit(_args(output, plan))

    assert verdict == audit.VERDICT_BLOCKED
    assert any("posthoc replay filter identity mismatch" in error for error in result["errors"])
