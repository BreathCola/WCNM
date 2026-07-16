import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tools.audit_stage_d_internal_object_gated_joint as audit
import tools.run_stage_d_internal_object_gated_joint as operator
from stage_d_training import (
    FORMAL_RELEASE_SHA256,
    FORMAL_SOURCE_SHA256,
    INTERNAL_OBJECT_GATED_JOINT_ENDPOINT,
    INTERNAL_OBJECT_GATED_JOINT_NODES,
    INTERNAL_OBJECT_GATED_JOINT_OUTPUT_NAME,
    _configure_internal_object_gated_joint_optimizer,
    _transmittance_topology_update_allowed,
    _validate_args,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _branch(count, *, color_value=0.0):
    return {
        "xyz": torch.ones(count, 3),
        "rotation": torch.ones(count, 4),
        "scaling_2d": torch.ones(count, 2),
        "opacity_raw": torch.ones(count, 1),
        "base_color_raw": torch.full((count, 3), color_value),
        "roughness_raw": torch.zeros(count, 1),
        "f0_raw": torch.zeros(count, 3),
        "ks_raw": torch.zeros(count, 1),
        "color_raw": torch.full((count, 3), color_value),
        "exposure": torch.eye(3, 4)[None],
        "optimizer": {"state": {}, "param_groups": []},
        "exposure_optimizer": {"state": {}, "param_groups": []},
    }


def _checkpoint(node, *, color_value=0.0):
    return {
        "format": "rtgs_stage_d",
        "checkpoint_version": 1,
        "global_iteration": node,
        "reflection_iteration": 12000 + max(0, node - 16500),
        "transmittance_iteration": 1500 + max(0, node - 16500),
        "geometry_release_id": "stage_c_geometry_release_v1",
        "geometry_release_aggregate_sha256": FORMAL_RELEASE_SHA256,
        "source": {"sha256": FORMAL_SOURCE_SHA256},
        "config": {
            "internal_object_ownership": {
                "schema": "rtgs_stage_d_internal_object_townership_v1",
            },
            "internal_object_to_20000": {
                "schema": "rtgs_stage_d_internal_object_townership_to_20000_v1",
            },
        },
        "diffuse": _branch(2, color_value=color_value),
        "reflection": _branch(3, color_value=color_value),
        "transmittance": _branch(4096, color_value=color_value),
        "optimizer_step_completed": True,
    }


def _patch_operator_inputs(monkeypatch, tmp_path):
    source = tmp_path / "chkpnt16500.pth"
    output = tmp_path / INTERNAL_OBJECT_GATED_JOINT_OUTPUT_NAME
    torch.save(_checkpoint(16500), source)
    audit_path = tmp_path / "to20000.json"
    _write_json(audit_path, {
        "verdict": "D016_TO_20000_PASS_AWAITING_USER_REVIEW",
        "technical_run_complete": True,
        "errors": [],
        "recommended_review_candidates": {
            "best_rgb": 16500,
            "best_leakage_tradeoff": 16500,
        },
    })
    color_audit = tmp_path / "color.json"
    _write_json(color_audit, {"verdict": "D016_COLOR_RECOVERY_BLOCKED"})
    monkeypatch.setattr(operator, "SOURCE", source)
    monkeypatch.setattr(operator, "OUTPUT", output)
    monkeypatch.setattr(operator, "TO_20000_AUDIT", audit_path)
    monkeypatch.setattr(operator, "COLOR_RECOVERY_AUDIT", color_audit)
    monkeypatch.setattr(
        operator,
        "sha256_file",
        lambda path: operator.INTERNAL_OBJECT_GATED_JOINT_SOURCE_SHA256,
    )
    monkeypatch.setattr(operator, "_git_identity", lambda: {
        "head": "abc",
        "branch": "feature/stage-d-transmittance",
        "ahead_behind": "0\t0",
        "worktree_clean": False,
        "dirty_entries": [" M docs/STATUS.md"],
    })
    monkeypatch.setattr(
        operator,
        "validate_geometry_release",
        lambda path: {
            "geometry_release_id": "stage_c_geometry_release_v1",
            "aggregate_sha256": FORMAL_RELEASE_SHA256,
        },
    )
    monkeypatch.setattr(
        operator,
        "validate_specular_mask_set",
        lambda *args: {
            "manifest_file_sha256": "glass-manifest",
            "aggregate_sha256": "glass-aggregate",
        },
    )
    monkeypatch.setattr(
        operator,
        "validate_internal_object_mask_set",
        lambda *args: {
            "manifest_file_sha256": "internal-manifest",
            "aggregate_sha256": operator.EXPECTED_INTERNAL_AGGREGATE,
            "role": "stage_d_internal_object_masks_reviewed",
            "human_status": "accepted",
            "internal_object_semantics_version": "tihubird_bird_and_internal_base_v3",
        },
    )
    return output


def _run_operator(monkeypatch, argv):
    stream = io.StringIO()
    monkeypatch.setattr("sys.argv", ["run_stage_d_internal_object_gated_joint.py", *argv])
    with redirect_stdout(stream):
        code = operator.main()
    return code, json.loads(stream.getvalue())


def test_gated_joint_plan_only_does_not_execute(monkeypatch, tmp_path):
    output = _patch_operator_inputs(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(operator.subprocess, "call", lambda *args, **kwargs: calls.append(args) or 1)

    code, plan = _run_operator(monkeypatch, [])

    assert code == 0
    assert calls == []
    assert not output.exists()
    assert plan["execute"] is False
    assert plan["iterations"]["updates"] == 1000
    assert plan["nodes"] == list(INTERNAL_OBJECT_GATED_JOINT_NODES)
    assert "--stage_d_internal_object_gated_joint" in plan["command"]
    assert "--stage_d_internal_object_color_recovery" not in plan["command"]
    assert plan["renderer"]["transparent_direct_mode"] == "interface_only"
    assert plan["renderer"]["transparent_reflection_mode"] == "support_safe_outside"
    assert plan["parameter_group_contract"]["trainable"]["transmittance"] == [
        "color", "opacity",
    ]


def test_gated_joint_optimizer_contract_freezes_geometry_and_exposure():
    def model(names):
        params = {name: torch.nn.Parameter(torch.ones(2, 1)) for name in names}
        optimizer = torch.optim.Adam([
            {"params": [param], "lr": 0.1, "name": name}
            for name, param in params.items()
        ])
        return SimpleNamespace(optimizer=optimizer, _exposure=torch.nn.Parameter(torch.ones(1)))

    diffuse = model(("xyz", "base_color", "opacity", "scaling", "rotation", "roughness", "f0", "ks"))
    diffuse.exposure_optimizer = torch.optim.Adam([diffuse._exposure], lr=0.1)
    reflection = model(("xyz", "color", "opacity", "scaling", "rotation"))
    transmittance = model(("xyz", "color", "opacity", "scaling", "rotation"))

    report = _configure_internal_object_gated_joint_optimizer(
        diffuse, reflection, transmittance,
    )

    assert report["schema"] == "rtgs_stage_d_internal_object_gated_joint_parameter_groups_v1"
    for branch in (report["diffuse"], report["reflection"], report["transmittance"]):
        for name, requires_grad in branch["requires_grad"].items():
            if name in branch["trainable"]:
                assert requires_grad is True
                assert branch["optimizer_group_lrs"][name] > 0
            else:
                assert requires_grad is False
                assert branch["optimizer_group_lrs"][name] == 0.0
    assert diffuse._exposure.requires_grad is False
    assert report["exposure"]["optimizer_group_lrs"] == [0.0]


def test_gated_joint_training_gate_and_topology():
    dataset = SimpleNamespace(
        stage="stage_d",
        model_type="surfel",
        geometry_release_manifest="geometry_releases/stage_c_geometry_release_v1.json",
        transmittance_init_mode="transferred_d_inside",
        transmittance_init_count=4096,
        transmittance_compose="alpha_over",
        ray_background="scene",
        specular_masks="specular_masks_reviewed_v1/manifest.json",
        internal_object_masks="internal_object_masks_reviewed_v3/manifest.json",
        transparent_path_mode="cuboid_front_v1",
        cout_ownership_mode="support_safe_outside",
        transparent_direct_mode="interface_only",
        transparent_reflection_mode="support_safe_outside",
    )
    opt = SimpleNamespace(
        stage_d_ownership_pilot=False,
        stage_d_ownership_t_long=False,
        stage_d_tscale_recovery_preflight=False,
        stage_d_tscale_recovery_long=False,
        stage_d_cached_twarmup=False,
        stage_d_semantic_repair_pilot=False,
        stage_d_internal_object_pilot=False,
        stage_d_internal_object_to_20000=False,
        stage_d_internal_object_color_recovery=False,
        stage_d_internal_object_gated_joint=True,
        stage_d_smoke=False,
        stage_d_formal_onset=False,
        stage_d_reuse_static_cache=False,
        random_background=False,
        lambda_spec=0.2,
        lambda_depth=0.2,
        stage_d_depth_start_iteration=40000,
        stage_d_smoke_max_steps=1,
        lambda_object_positive=0.0,
        lambda_object_negative=0.05,
        lambda_object_cin_color=0.05,
        stage_d_phase_a_end_iteration=INTERNAL_OBJECT_GATED_JOINT_ENDPOINT,
        iterations=INTERNAL_OBJECT_GATED_JOINT_ENDPOINT,
        stage_d_cache_parity_atol=2e-5,
        stage_d_cache_parity_mean_atol=2e-6,
        transparent_interface_margin_mode="exclude",
        lambda_anti_veil_black=0.02,
        lambda_anti_veil_saturation=0.02,
        anti_veil_ramp_start=0,
        anti_veil_ramp_end=200,
    )

    _validate_args(dataset, opt, "chkpnt16500.pth")
    assert _transmittance_topology_update_allowed(opt, 1501) is False


def test_gated_joint_cpu_audit_accepts_minimal_complete_run(monkeypatch, tmp_path):
    output = tmp_path / INTERNAL_OBJECT_GATED_JOINT_OUTPUT_NAME
    output.mkdir()
    monkeypatch.setattr(audit, "SOURCE", tmp_path / "source16500.pth")
    torch.save(_checkpoint(16500), audit.SOURCE)
    monkeypatch.setattr(
        audit,
        "sha256_file",
        lambda path: audit.INTERNAL_OBJECT_GATED_JOINT_SOURCE_SHA256,
    )
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
            "manifest_file_sha256": "internal-manifest",
            "aggregate_sha256": audit.EXPECTED_INTERNAL_AGGREGATE,
            "role": "stage_d_internal_object_masks_reviewed",
            "human_status": "accepted",
            "internal_object_semantics_version": "tihubird_bird_and_internal_base_v3",
        },
    )
    _write_json(output / "internal_object_gated_joint_metadata.json", {
        "schema": "rtgs_stage_d_internal_object_gated_joint_v1",
        "global": [16501, 17500],
        "transmittance_local": [1501, 2500],
        "diffuse_optimizer_updates": 1000,
        "reflection_optimizer_updates": 1000,
        "transmittance_optimizer_updates": 1000,
        "topology_updates_allowed": False,
        "t_reinitialization": False,
        "transferred_d_selection_rerun": False,
        "random_fill_rerun": False,
    })
    rows = []
    for iteration in range(16501, 17501):
        rows.append(json.dumps({
            "schema": "rtgs_stage_d_internal_object_gated_joint_telemetry_v1",
            "global_iteration": iteration,
            "optimizer_updates_this_step": {
                "diffuse": 1, "reflection": 1, "transmittance": 1,
            },
            "topology_event": {
                "transmittance_densify_prune_called": False,
                "diffuse_delta": 0,
                "reflection_delta": 0,
                "transmittance_delta": 0,
            },
            "t_parameter_group_contract": {
                "schema": "rtgs_stage_d_internal_object_gated_joint_parameter_groups_v1",
            },
        }))
    (output / "stage_d_telemetry.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    required = audit.REQUIRED_DEBUG
    for node in audit.INTERNAL_OBJECT_GATED_JOINT_NODES:
        color_value = 0.2 if node == 17500 else 0.0
        torch.save(_checkpoint(node, color_value=color_value), output / f"chkpnt{node}.pth")
        root = output / "debug" / f"iteration_{node:06d}"
        root.mkdir(parents=True, exist_ok=True)
        (root / "contact_sheet.png").write_bytes(b"png")
        for stem in audit.FORMAL_STEMS:
            view = root / stem
            view.mkdir(parents=True, exist_ok=True)
            for name in required - {"float_metrics.json"}:
                (view / name).write_bytes(b"png")
            _write_json(view / "float_metrics.json", {
                "schema": "rtgs_stage_d_internal_object_gated_joint_float_metrics_v1",
                "value": 1.0,
            })

    result, verdict = audit._audit(SimpleNamespace(
        output=output,
        operator_plan=None,
        geometry_manifest=tmp_path / "release.json",
        scene=tmp_path / "scene",
        images="images",
        internal_object_manifest=tmp_path / "internal.json",
    ))

    assert verdict == "D016_GATED_JOINT_PASS_AWAITING_USER_REVIEW"
    assert result["errors"] == []
