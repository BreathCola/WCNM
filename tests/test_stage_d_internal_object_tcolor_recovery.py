import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tools.run_stage_d_internal_object_tcolor_recovery as operator
import tools.audit_stage_d_internal_object_tcolor_recovery as audit
from stage_d_training import (
    FORMAL_RELEASE_SHA256,
    FORMAL_SOURCE_SHA256,
    INTERNAL_OBJECT_COLOR_RECOVERY_ENDPOINT,
    INTERNAL_OBJECT_COLOR_RECOVERY_NODES,
    _configure_transmittance_color_recovery_optimizer,
    _phase_for_iteration,
    _transmittance_topology_update_allowed,
    _validate_args,
)
from utils.stage_d_static_cache import state_sha256


def _branch(count):
    return {
        "xyz": torch.ones(count, 3),
        "opacity": torch.ones(count, 1),
        "scaling": torch.ones(count, 2),
        "rotation": torch.ones(count, 4),
        "color": torch.zeros(count, 3),
        "optimizer": {"state": {0: {"step": torch.tensor(1)}}},
    }


def _t_optimizer(groups=("xyz", "color", "opacity", "scaling", "rotation")):
    return {
        "state": {},
        "param_groups": [{"name": name, "lr": 0.01, "params": [index]} for index, name in enumerate(groups)],
    }


def _checkpoint(groups=("xyz", "color", "opacity", "scaling", "rotation"), cache_path=None):
    diffuse = _branch(2)
    reflection = _branch(3)
    transmittance = _branch(4096)
    transmittance["optimizer"] = _t_optimizer(groups)
    frozen = {
        "diffuse": state_sha256(diffuse),
        "reflection": state_sha256(reflection),
    }
    return {
        "format": "rtgs_stage_d",
        "checkpoint_version": 1,
        "global_iteration": 16500,
        "reflection_iteration": 12000,
        "transmittance_iteration": 1500,
        "geometry_release_id": "stage_c_geometry_release_v1",
        "geometry_release_aggregate_sha256": FORMAL_RELEASE_SHA256,
        "source": {"sha256": FORMAL_SOURCE_SHA256},
        "config": {
            "cached_t_warmup": {
                "phase_a_frozen_hash_after": frozen,
                "cache_path": str(cache_path or operator.CACHE),
                "cache_aggregate_sha256": "cache-aggregate",
            },
            "internal_object_to_20000": {
                "schema": "rtgs_stage_d_internal_object_townership_to_20000_v1",
            },
        },
        "diffuse": diffuse,
        "reflection": reflection,
        "transmittance": transmittance,
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _checkpoint_for_audit(node, *, color_value=0.0):
    checkpoint = _checkpoint(cache_path=operator.CACHE)
    checkpoint["global_iteration"] = node
    checkpoint["reflection_iteration"] = 12000
    checkpoint["transmittance_iteration"] = node - 15000
    checkpoint["transmittance"]["color"] = torch.full((4096, 3), color_value)
    return checkpoint


def _patch_valid_inputs(monkeypatch, tmp_path, checkpoint=None):
    source = tmp_path / "chkpnt16500.pth"
    output = tmp_path / "stage_d_tihubird_c03r8_internal_object_color_recovery_16500_17000_v1"
    cache = tmp_path / "pilot" / "static_dr_cache"
    torch.save(checkpoint or _checkpoint(cache_path=cache), source)
    pilot_source = tmp_path / "pilot" / "chkpnt15500.pth"
    pilot_source.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint(cache_path=cache), pilot_source)
    _write_json(cache / "manifest.json", {"aggregate_sha256": "cache-aggregate"})
    audit = tmp_path / "to20000" / "internal_object_townership_to_20000_cpu_audit.json"
    _write_json(audit, {
        "verdict": "D016_TO_20000_PASS_AWAITING_USER_REVIEW",
        "technical_run_complete": True,
        "errors": [],
        "recommended_review_candidates": {
            "best_rgb": 16500,
            "best_leakage_tradeoff": 16500,
            "best_object_ownership": 20000,
        },
    })
    monkeypatch.setattr(operator, "SOURCE", source)
    monkeypatch.setattr(operator, "PILOT_SOURCE", pilot_source)
    monkeypatch.setattr(operator, "OUTPUT", output)
    monkeypatch.setattr(operator, "CACHE", cache)
    monkeypatch.setattr(operator, "TO_20000_AUDIT", audit)
    monkeypatch.setattr(operator, "sha256_file", lambda path: operator.INTERNAL_OBJECT_COLOR_RECOVERY_SOURCE_SHA256)
    monkeypatch.setattr(operator, "_git_identity", lambda: {
        "head": "abc",
        "branch": "feature/stage-d-transmittance",
        "ahead_behind": "0\t0",
        "worktree_clean": True,
        "dirty_entries": [],
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
            "aggregate_sha256": "c0e49503e5f5c30b1ab26b7cfd79332ac9c486f35656f1425c9cefea516d4052",
            "role": "stage_d_internal_object_masks_reviewed",
            "human_status": "accepted",
            "internal_object_semantics_version": "tihubird_bird_and_internal_base_v3",
        },
    )
    return output


def _run_main(monkeypatch, argv):
    stream = io.StringIO()
    monkeypatch.setattr("sys.argv", ["run_stage_d_internal_object_tcolor_recovery.py", *argv])
    with redirect_stdout(stream):
        code = operator.main()
    return code, json.loads(stream.getvalue())


def test_color_recovery_plan_only_does_not_execute_or_create_output(monkeypatch, tmp_path):
    output = _patch_valid_inputs(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(operator.subprocess, "call", lambda *args, **kwargs: calls.append(args) or 1)

    code, plan = _run_main(monkeypatch, [])

    assert code == 0
    assert calls == []
    assert not output.exists()
    assert plan["execute"] is False
    assert plan["iterations"] == {
        "start": 16500,
        "first_update": 16501,
        "end_inclusive": 17000,
        "updates": 500,
        "transmittance_local": [1501, 2000],
    }
    assert plan["nodes"] == list(INTERNAL_OBJECT_COLOR_RECOVERY_NODES)
    assert plan["source_selection_reason"]["best_rgb"] == 16500
    assert plan["source_selection_reason"]["endpoint_20000_is_not_auto_selected"] is True
    assert plan["parameter_group_contract"]["trainable_t_parameter_groups"] == ["color"]
    assert set(plan["parameter_group_contract"]["frozen_t_parameter_groups"]) == {
        "xyz", "opacity", "scaling", "rotation",
    }
    assert plan["losses"]["lambda_object_positive"] == 0.0
    assert "--stage_d_internal_object_color_recovery" in plan["command"]
    assert "--stage_d_internal_object_to_20000" not in plan["command"]
    assert "--lambda_object_cin_color" in plan["command"]


def test_color_recovery_execute_requires_explicit_flag(monkeypatch, tmp_path):
    _patch_valid_inputs(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(operator, "_validate_git_for_execute", lambda: None)
    monkeypatch.setattr(operator, "_validate_no_conflicting_training_process", lambda: None)
    monkeypatch.setattr(operator.subprocess, "call", lambda command, cwd: calls.append((command, cwd)) or 7)

    code, plan = _run_main(monkeypatch, ["--execute"])

    assert code == 7
    assert plan["execute"] is True
    assert len(calls) == 1
    assert Path(calls[0][0][1]).name == "train.py"


def test_color_recovery_refuses_nonseparable_t_groups(monkeypatch, tmp_path):
    cache = tmp_path / "pilot" / "static_dr_cache"
    bad = _checkpoint(groups=("xyz", "opacity", "scaling", "rotation"), cache_path=cache)
    _patch_valid_inputs(monkeypatch, tmp_path, bad)

    with pytest.raises(RuntimeError, match="BLOCKED_BY_PARAMETER_GROUP_CONTRACT"):
        _run_main(monkeypatch, [])


def test_color_recovery_refuses_existing_output(monkeypatch, tmp_path):
    output = _patch_valid_inputs(monkeypatch, tmp_path)
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing existing"):
        _run_main(monkeypatch, [])


def test_color_recovery_optimizer_contract_freezes_geometry_and_opacity():
    params = {
        name: torch.nn.Parameter(torch.ones(2, 1))
        for name in ("xyz", "color", "opacity", "scaling", "rotation")
    }
    optimizer = torch.optim.Adam(
        [{"params": [param], "lr": 0.1, "name": name} for name, param in params.items()]
    )
    model = SimpleNamespace(optimizer=optimizer)

    report = _configure_transmittance_color_recovery_optimizer(model)

    assert report["trainable_t_parameter_groups"] == ["color"]
    for name, parameter in params.items():
        assert parameter.requires_grad is (name == "color")
    for group in optimizer.param_groups:
        if group["name"] == "color":
            assert group["lr"] == pytest.approx(0.1)
        else:
            assert group["lr"] == 0.0


def test_color_recovery_cpu_audit_accepts_minimal_complete_run(monkeypatch, tmp_path):
    output = tmp_path / "stage_d_tihubird_c03r8_internal_object_color_recovery_16500_17000_v1"
    output.mkdir()
    monkeypatch.setattr(audit, "SOURCE", tmp_path / "source16500.pth")
    torch.save(_checkpoint_for_audit(16500), audit.SOURCE)
    monkeypatch.setattr(audit, "sha256_file", lambda path: audit.INTERNAL_OBJECT_COLOR_RECOVERY_SOURCE_SHA256)
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
            "aggregate_sha256": "c0e49503e5f5c30b1ab26b7cfd79332ac9c486f35656f1425c9cefea516d4052",
            "role": "stage_d_internal_object_masks_reviewed",
            "human_status": "accepted",
            "internal_object_semantics_version": "tihubird_bird_and_internal_base_v3",
        },
    )
    _write_json(output / "internal_object_tcolor_recovery_metadata.json", {
        "schema": "rtgs_stage_d_internal_object_tcolor_recovery_v1",
        "global": [16501, 17000],
        "transmittance_local": [1501, 2000],
        "diffuse_optimizer_updates": 0,
        "reflection_optimizer_updates": 0,
        "transmittance_color_optimizer_updates": 500,
        "transmittance_geometry_opacity_optimizer_updates": 0,
        "expected_t_count": 4096,
        "t_reinitialization": False,
        "transferred_d_selection_rerun": False,
        "random_fill_rerun": False,
    })
    rows = []
    for iteration in range(16501, 17001):
        rows.append(json.dumps({
            "schema": "rtgs_stage_d_internal_object_tcolor_recovery_telemetry_v1",
            "global_iteration": iteration,
            "optimizer_updates_this_step": {
                "diffuse": 0, "reflection": 0, "transmittance": 1,
            },
            "t_parameter_group_contract": {
                "trainable_t_parameter_groups": ["color"],
            },
        }))
    (output / "stage_d_telemetry.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    required = audit.REQUIRED_DEBUG
    for node in audit.INTERNAL_OBJECT_COLOR_RECOVERY_NODES:
        color_value = 0.2 if node == 17000 else 0.0
        torch.save(_checkpoint_for_audit(node, color_value=color_value), output / f"chkpnt{node}.pth")
        root = output / "debug" / f"iteration_{node:06d}"
        (root / "contact_sheet.png").parent.mkdir(parents=True, exist_ok=True)
        (root / "contact_sheet.png").write_bytes(b"png")
        for stem in audit.FORMAL_STEMS:
            view = root / stem
            view.mkdir(parents=True, exist_ok=True)
            for name in required - {"float_metrics.json"}:
                (view / name).write_bytes(b"png")
            _write_json(view / "float_metrics.json", {
                "schema": "rtgs_stage_d_internal_object_tcolor_recovery_float_metrics_v1",
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

    assert verdict == "D016_COLOR_RECOVERY_PASS_AWAITING_USER_REVIEW"
    assert result["errors"] == []


def test_color_recovery_training_gate_phase_and_topology():
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
        transparent_direct_mode="off",
        transparent_reflection_mode="off",
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
        stage_d_internal_object_color_recovery=True,
        stage_d_smoke=False,
        stage_d_formal_onset=False,
        stage_d_reuse_static_cache=True,
        random_background=False,
        lambda_spec=0.2,
        lambda_depth=0.2,
        stage_d_depth_start_iteration=40000,
        stage_d_smoke_max_steps=1,
        lambda_object_positive=0.0,
        lambda_object_negative=0.05,
        lambda_object_cin_color=0.10,
        stage_d_phase_a_end_iteration=INTERNAL_OBJECT_COLOR_RECOVERY_ENDPOINT,
        iterations=INTERNAL_OBJECT_COLOR_RECOVERY_ENDPOINT,
        stage_d_cache_parity_atol=2e-5,
        stage_d_cache_parity_mean_atol=2e-6,
        transparent_interface_margin_mode="exclude",
        lambda_anti_veil_black=0.02,
        lambda_anti_veil_saturation=0.02,
        anti_veil_ramp_start=0,
        anti_veil_ramp_end=200,
    )

    _validate_args(dataset, opt, "chkpnt16500.pth")
    assert _phase_for_iteration(
        16501,
        internal_object=True,
        internal_object_color_recovery=True,
    ) == "internal_object_tcolor_recovery"
    assert _transmittance_topology_update_allowed(opt, 1501) is False
