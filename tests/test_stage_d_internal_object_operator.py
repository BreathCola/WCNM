import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import tools.run_stage_d_internal_object_townership as operator
from stage_d_training import FORMAL_RELEASE_SHA256, FORMAL_SOURCE_SHA256


def _run_main(monkeypatch, argv):
    stream = io.StringIO()
    monkeypatch.setattr("sys.argv", ["run_stage_d_internal_object_townership.py", *argv])
    with redirect_stdout(stream):
        code = operator.main()
    return code, json.loads(stream.getvalue())


def _patch_valid_inputs(monkeypatch, tmp_path):
    source = tmp_path / "source.pth"
    source.write_bytes(b"checkpoint")
    output = tmp_path / "stage_d_tihubird_c03r8_internal_object_townership_pilot_v1"
    monkeypatch.setattr(operator, "SOURCE", source)
    monkeypatch.setattr(operator, "OUTPUT", output)
    monkeypatch.setattr(operator, "sha256_file", lambda path: FORMAL_SOURCE_SHA256)
    monkeypatch.setattr(
        operator,
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


def test_internal_object_operator_plan_only_is_json_and_does_not_execute(monkeypatch, tmp_path):
    calls = []
    output = _patch_valid_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(operator.subprocess, "call", lambda *args, **kwargs: calls.append(args) or 0)

    code, plan = _run_main(monkeypatch, [])

    assert code == 0
    assert plan["execute"] is False
    assert plan["source_sha256"] == FORMAL_SOURCE_SHA256
    assert plan["release_aggregate_sha256"] == FORMAL_RELEASE_SHA256
    assert plan["glass_mask_manifest_sha256"] == "glass-manifest"
    assert plan["internal_object_manifest_sha256"] == "internal-manifest"
    assert plan["internal_object_aggregate_sha256"].startswith("c0e495")
    assert plan["internal_object_semantics_version"] == "tihubird_bird_and_internal_base_v3"
    assert plan["iterations"] == {"start_exclusive": 15000, "end_inclusive": 15500, "updates": 500}
    assert plan["nodes"] == [15000, 15100, 15250, 15500]
    assert plan["transmittance"] == {"count": 4096, "init_seed": 20260703, "init_mode": "transferred_d_inside"}
    assert plan["frozen_branches"] == {"diffuse": True, "reflection": True, "transmittance": False}
    assert "--execute" not in plan["command"]
    assert "--internal_object_masks" in plan["command"]
    assert "internal_object_masks_reviewed_v3/manifest.json" in plan["command"]
    assert "proposal" not in " ".join(plan["command"])
    assert calls == []
    assert not output.exists()


@pytest.mark.parametrize(
    "patcher,match",
    [
        (lambda monkeypatch: monkeypatch.setattr(operator, "sha256_file", lambda path: "bad"), "source hash"),
        (
            lambda monkeypatch: monkeypatch.setattr(
                operator,
                "validate_geometry_release",
                lambda path: {"geometry_release_id": "stage_c_geometry_release_v1", "aggregate_sha256": "bad"},
            ),
            "release hash",
        ),
        (
            lambda monkeypatch: monkeypatch.setattr(
                operator,
                "validate_internal_object_mask_set",
                lambda *args: (_ for _ in ()).throw(ValueError("proposal internal-object masks are not formal training supervision")),
            ),
            "proposal",
        ),
    ],
)
def test_internal_object_operator_plan_fail_closed(monkeypatch, tmp_path, patcher, match):
    _patch_valid_inputs(monkeypatch, tmp_path)
    patcher(monkeypatch)
    with pytest.raises(Exception, match=match):
        _run_main(monkeypatch, [])


def test_internal_object_operator_refuses_existing_output(monkeypatch, tmp_path):
    output = _patch_valid_inputs(monkeypatch, tmp_path)
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing existing"):
        _run_main(monkeypatch, [])


def test_internal_object_operator_execute_requires_explicit_flag_and_mocked_subprocess(monkeypatch, tmp_path):
    calls = []
    _patch_valid_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(operator.subprocess, "call", lambda command, cwd: calls.append((command, cwd)) or 7)

    code, plan = _run_main(monkeypatch, ["--execute"])

    assert code == 7
    assert plan["execute"] is True
    assert len(calls) == 1
    command, cwd = calls[0]
    assert Path(command[1]).name == "train.py"
    assert cwd == operator.ROOT
