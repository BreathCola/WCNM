import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tools.run_stage_d_internal_object_townership_to_20000 as operator
from stage_d_training import (
    FORMAL_RELEASE_SHA256,
    FORMAL_SOURCE_SHA256,
    INTERNAL_OBJECT_TO_20000_ENDPOINT,
    INTERNAL_OBJECT_TO_20000_NODES,
    _phase_for_iteration,
    _transmittance_topology_update_allowed,
    _validate_args,
)
from utils.stage_d_static_cache import state_sha256


def _branch(count):
    return {
        "xyz": torch.ones(count, 3),
        "optimizer": {"state": {0: {"step": torch.tensor(1)}}},
    }


def _checkpoint(include_t_optimizer=True, sha_cache=True, cache_path=None):
    diffuse = _branch(2)
    reflection = _branch(3)
    transmittance = _branch(4096)
    if not include_t_optimizer:
        transmittance.pop("optimizer")
    frozen = {
        "diffuse": state_sha256(diffuse),
        "reflection": state_sha256(reflection),
    }
    if not sha_cache:
        frozen["diffuse"] = "wrong"
    return {
        "format": "rtgs_stage_d",
        "checkpoint_version": 1,
        "global_iteration": 15500,
        "reflection_iteration": 12000,
        "transmittance_iteration": 500,
        "geometry_release_id": "stage_c_geometry_release_v1",
        "geometry_release_aggregate_sha256": FORMAL_RELEASE_SHA256,
        "source": {"sha256": FORMAL_SOURCE_SHA256},
        "config": {
            "cached_t_warmup": {
                "phase_a_frozen_hash_after": frozen,
                "cache_path": str(cache_path or operator.CACHE),
                "cache_aggregate_sha256": "cache-aggregate",
            },
        },
        "diffuse": diffuse,
        "reflection": reflection,
        "transmittance": transmittance,
        "optimizer_step_completed": True,
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _patch_valid_inputs(monkeypatch, tmp_path, checkpoint=None):
    source = tmp_path / "chkpnt15500.pth"
    output = tmp_path / "stage_d_tihubird_c03r8_internal_object_townership_15500_20000_v1"
    cache = tmp_path / "pilot" / "static_dr_cache"
    torch.save(checkpoint or _checkpoint(cache_path=cache), source)
    _write_json(cache / "manifest.json", {"aggregate_sha256": "cache-aggregate"})
    audit = tmp_path / "pilot" / "internal_object_townership_cpu_audit.json"
    _write_json(audit, {
        "verdict": "D016_PILOT_PASS_AWAITING_USER_REVIEW",
        "technical_run_complete": True,
        "errors": [],
    })
    monkeypatch.setattr(operator, "SOURCE", source)
    monkeypatch.setattr(operator, "OUTPUT", output)
    monkeypatch.setattr(operator, "CACHE", cache)
    monkeypatch.setattr(operator, "PILOT_AUDIT", audit)
    monkeypatch.setattr(operator, "sha256_file", lambda path: operator.INTERNAL_OBJECT_TO_20000_SOURCE_SHA256)
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
    monkeypatch.setattr("sys.argv", ["run_stage_d_internal_object_townership_to_20000.py", *argv])
    with redirect_stdout(stream):
        code = operator.main()
    return code, json.loads(stream.getvalue())


def test_to_20000_plan_only_does_not_execute_or_create_output(monkeypatch, tmp_path):
    output = _patch_valid_inputs(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(operator.subprocess, "call", lambda *args, **kwargs: calls.append(args) or 1)

    code, plan = _run_main(monkeypatch, [])

    assert code == 0
    assert calls == []
    assert not output.exists()
    assert plan["execute"] is False
    assert plan["iterations"] == {
        "start": 15500,
        "first_update": 15501,
        "end_inclusive": 20000,
        "updates": 4500,
        "transmittance_local": [501, 5000],
    }
    assert plan["nodes"] == list(INTERNAL_OBJECT_TO_20000_NODES)
    assert plan["transmittance"]["reinitialization"] is False
    assert plan["transmittance"]["optimizer_resume"] is True
    assert plan["frozen_branches"] == {"diffuse": True, "reflection": True, "transmittance": False}
    assert "--stage_d_internal_object_to_20000" in plan["command"]
    assert "--stage_d_internal_object_pilot" not in plan["command"]
    assert "--stage_d_reuse_static_cache" in plan["command"]
    assert "--checkpoint_iterations" in plan["command"]


def test_to_20000_execute_requires_explicit_flag(monkeypatch, tmp_path):
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


def test_to_20000_refuses_missing_t_optimizer(monkeypatch, tmp_path):
    _patch_valid_inputs(monkeypatch, tmp_path, _checkpoint(include_t_optimizer=False))

    with pytest.raises(RuntimeError, match="BLOCKED_BY_RESUME_CONTRACT"):
        _run_main(monkeypatch, [])


def test_to_20000_refuses_old_or_wrong_source_hash(monkeypatch, tmp_path):
    _patch_valid_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(operator, "sha256_file", lambda path: "old-d015")

    with pytest.raises(RuntimeError, match="source checkpoint SHA"):
        _run_main(monkeypatch, [])


def test_to_20000_refuses_existing_output(monkeypatch, tmp_path):
    output = _patch_valid_inputs(monkeypatch, tmp_path)
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing existing"):
        _run_main(monkeypatch, [])


def test_to_20000_training_gate_and_phase_are_t_only():
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
        stage_d_internal_object_to_20000=True,
        stage_d_smoke=False,
        stage_d_formal_onset=False,
        stage_d_reuse_static_cache=True,
        random_background=False,
        lambda_spec=0.2,
        lambda_depth=0.2,
        stage_d_depth_start_iteration=40000,
        stage_d_smoke_max_steps=1,
        lambda_object_positive=0.05,
        lambda_object_negative=0.05,
        stage_d_phase_a_end_iteration=INTERNAL_OBJECT_TO_20000_ENDPOINT,
        iterations=INTERNAL_OBJECT_TO_20000_ENDPOINT,
        stage_d_cache_parity_atol=2e-5,
        stage_d_cache_parity_mean_atol=2e-6,
        transparent_interface_margin_mode="exclude",
        lambda_anti_veil_black=0.02,
        lambda_anti_veil_saturation=0.02,
        anti_veil_ramp_start=0,
        anti_veil_ramp_end=200,
    )

    _validate_args(dataset, opt, "chkpnt15500.pth")
    assert _phase_for_iteration(15501, internal_object=True) == "internal_object_townership_t_only"
    assert _transmittance_topology_update_allowed(opt, 501) is False
