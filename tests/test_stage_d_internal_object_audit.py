import json
from argparse import Namespace

import torch

import tools.audit_stage_d_internal_object_townership as audit
from stage_d_training import FORMAL_RELEASE_SHA256, FORMAL_SOURCE_SHA256, INTERNAL_OBJECT_NODES


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
    _write_json(plan, {
        "schema": "rtgs_stage_d_internal_object_operator_plan_v1",
        "execute": True,
        "output": str(output),
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
            "internal_object_filter": {
                "pre_filter_D_indices_sha256": "a",
                "selected_D_indices_sha256": "b",
                "rejected_D_indices_sha256": "c",
                "random_fill_count": 12,
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
    for node in INTERNAL_OBJECT_NODES:
        torch.save(_checkpoint(node, t_delta=0.0 if node == 15000 else 1.0), output / f"chkpnt{node}.pth")
        for branch in ("diffuse", "reflection", "transmittance"):
            ply = output / "point_cloud" / branch / f"iteration_{node}" / "point_cloud.ply"
            ply.parent.mkdir(parents=True, exist_ok=True)
            ply.write_text("ply\n", encoding="utf-8")
        debug_root = output / "debug" / f"iteration_{node:06d}"
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
