#!/usr/bin/env python3
"""Fail-closed 2x500 Stage D cuboid-front ownership A/B operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_release import sha256_file, validate_geometry_release
from stage_d_training import (
    FORMAL_RELEASE_SHA256, FORMAL_SOURCE_SHA256, OWNERSHIP_ARM_NAMES,
    OWNERSHIP_NODES, OWNERSHIP_OUTPUT_NAME,
)


SOURCE = ROOT / "output/tier2_c03_r8_oneshot_v4_rstart_g03000_to_g15000/chkpnt15000.pth"
MANIFEST = ROOT / "geometry_releases/stage_c_geometry_release_v1.json"
OUTPUT = ROOT / "output" / OWNERSHIP_OUTPUT_NAME
CACHE = OUTPUT / "cuboid_front_cache_v4"
LOGS = OUTPUT / "logs"
ALLOWED_DISPLAY_COMPUTE = {"/usr/libexec/gnome-remote-desktop-daemon": 512}
RETRYABLE_PREFLIGHT_COMMIT = "b697c2507ffdd236fac6603ea7ff1d20ff530ac8"
RETRYABLE_PREFLIGHT_ARCHIVE_SUFFIX = "_failed_preflight_b697c25"
RETRYABLE_SCALE_COMMIT = "17d2663c1bae82407d8b12b3d29adf60b6ff771d"
RETRYABLE_SCALE_ARCHIVE_SUFFIX = "_failed_scale_17d2663"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def classify_compute_processes(text):
    observed, conflicts = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError(f"unexpected nvidia-smi compute-process row: {line!r}")
        row = {
            "pid": int(fields[0]), "process_name": fields[1],
            "used_gpu_memory_mib": int(fields[2]),
        }
        observed.append(row)
        limit = ALLOWED_DISPLAY_COMPUTE.get(row["process_name"])
        if limit is None or row["used_gpu_memory_mib"] > limit:
            conflicts.append(row)
    return observed, conflicts


def independent_bird_roi():
    candidates = sorted((ROOT / "data/TiHuBird").glob("bird_roi*/manifest.json"))
    valid = []
    for path in candidates:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        provenance = value.get("provenance", {})
        if (
            value.get("schema") == "rtgs_independent_bird_roi_v1"
            and provenance.get("model_prediction") is False
            and provenance.get("source") in ("manual", "human_annotation")
        ):
            valid.append(str(path.resolve()))
    if len(valid) > 1:
        raise RuntimeError("multiple independent bird ROI manifests are ambiguous")
    return {
        "status": "AVAILABLE" if valid else "NO_INDEPENDENT_BIRD_ROI",
        "manifest": valid[0] if valid else None,
        "searched_candidates": [str(path.resolve()) for path in candidates],
    }


def retryable_ownership_archive_path(output):
    """Validate an exact known failure and return its unused evidence archive path."""
    output = Path(output)
    if not output.exists():
        return None
    if not output.is_dir():
        raise FileExistsError(f"ownership output exists and is not a directory: {output}")
    record_path = output / "ownership_operator_record.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FileExistsError(
            f"refusing to replace unrecognized ownership output: {output}: {exc}"
        ) from exc
    arm_a = record.get("arms", {}).get("random_strict_inside", {})
    arm_b = record.get("arms", {}).get("transferred_d_inside", {})
    source_release_unchanged = (
        record.get("source_sha256_before") == FORMAL_SOURCE_SHA256
        and record.get("source_sha256_after") == FORMAL_SOURCE_SHA256
        and record.get("release_aggregate_before") == FORMAL_RELEASE_SHA256
        and record.get("release_aggregate_after") == FORMAL_RELEASE_SHA256
    )
    preflight_log = output / "logs/random_strict_inside.log"
    exact_preflight_failure = (
        record.get("status") == "CUBOID_PATH_OWNERSHIP_PILOT_BLOCKED"
        and record.get("git_commit") == RETRYABLE_PREFLIGHT_COMMIT
        and record.get("operator_error")
        == "RuntimeError: ownership arm failed: random_strict_inside exit=1"
        and arm_a.get("exit_code") == 1
        and preflight_log.is_file()
        and "support-safe T initialization requires cuboid space"
        in preflight_log.read_text(encoding="utf-8")
        and source_release_unchanged
        and not (output / "cuboid_front_cache_v4/manifest.json").exists()
        and not any(output.rglob("chkpnt*.pth"))
        and not any(output.rglob("stage_d_telemetry.jsonl"))
    )
    scale_log = output / "logs/transferred_d_inside.log"
    exact_scale_failure = (
        record.get("status") == "CUBOID_PATH_OWNERSHIP_PILOT_BLOCKED"
        and record.get("git_commit") == RETRYABLE_SCALE_COMMIT
        and record.get("operator_error")
        == "RuntimeError: ownership arm failed: transferred_d_inside exit=1"
        and arm_a.get("exit_code") == 0 and arm_b.get("exit_code") == 1
        and scale_log.is_file()
        and "surfel support is too large for the strict cuboid interior"
        in scale_log.read_text(encoding="utf-8")
        and source_release_unchanged
        and (output / "cuboid_front_cache_v4/manifest.json").is_file()
        and (output / "arm_a_random_strict_inside/chkpnt15500.pth").is_file()
        and (output / "arm_a_random_strict_inside/stage_d_telemetry.jsonl").is_file()
        and not (output / "arm_b_transferred_d_inside/chkpnt15500.pth").exists()
    )
    if exact_preflight_failure:
        suffix = RETRYABLE_PREFLIGHT_ARCHIVE_SUFFIX
    elif exact_scale_failure:
        suffix = RETRYABLE_SCALE_ARCHIVE_SUFFIX
    else:
        raise FileExistsError(
            f"refusing to overwrite non-retryable ownership output: {output}"
        )
    archive = output.with_name(output.name + suffix)
    if archive.exists():
        raise FileExistsError(f"retry evidence archive already exists: {archive}")
    return archive


def archive_retryable_ownership_failure(output):
    """Atomically preserve an exact known failure before a fresh matched retry."""
    output = Path(output)
    archive = retryable_ownership_archive_path(output)
    if archive is None:
        return None
    os.replace(output, archive)
    return archive


def training_command(arm):
    if arm not in OWNERSHIP_ARM_NAMES:
        raise ValueError(f"unknown ownership arm: {arm}")
    command = [
        sys.executable, str(ROOT / "train.py"),
        "-s", str(ROOT / "data/TiHuBird"),
        "-m", str(OUTPUT / OWNERSHIP_ARM_NAMES[arm]),
        "--images", "images", "--model_type", "surfel", "--stage", "stage_d",
        "--experiment", "TiHuBird C03-r8 Stage D cuboid-front ownership A/B v4",
        "--resolution", "8",
        "--normal_priors", "diffrender_priors_candidates/axis_smoke/C03/normal",
        "--normal_prior_space", "camera",
        "--specular_masks", "specular_masks_reviewed_v1/manifest.json",
        "--geometry_release_manifest", str(MANIFEST),
        "--transmittance_init_mode", arm,
        "--transmittance_init_count", "4096",
        "--transmittance_init_seed", "20260703",
        "--transmittance_compose", "alpha_over",
        "--transparent_path_mode", "cuboid_front_v1",
        "--transparent_direct_mode", "off",
        "--transparent_reflection_mode", "off",
        "--cout_ownership_mode", "support_safe_outside",
        "--stage_d_static_cache_path", str(CACHE),
        "--ray_background", "scene", "--ray_chunk_size", "2048",
        "--iterations", "15500", "--start_checkpoint", str(SOURCE),
        "--lambda_spec", "0.2", "--specular_k0", "0.9",
        "--lambda_depth", "0.2", "--stage_d_depth_start_iteration", "40000",
        "--stage_d_ownership_pilot", "--stage_d_ownership_arm", arm,
        "--stage_d_phase_a_end_iteration", "15500",
        "--transparent_interface_margin", "0.05",
        "--transparent_interface_margin_mode", "exclude",
        "--transfer_min_views", "3",
        "--transfer_min_total_weight", "0.01",
        "--transfer_depth_margin", "0.05",
        "--transfer_opacity_scale", "0.25",
        "--transfer_opacity_min", "0.005",
        "--transfer_opacity_max", "0.05",
        "--stage_d_cache_parity_atol", "2e-5",
        "--stage_d_cache_parity_mean_atol", "2e-6",
        "--disable_viewer", "--quiet",
        "--save_iterations", *[str(node) for node in OWNERSHIP_NODES],
        "--checkpoint_iterations", *[str(node) for node in OWNERSHIP_NODES],
    ]
    if arm == "transferred_d_inside":
        command.append("--stage_d_reuse_static_cache")
    return command


def common_training_contract(command):
    ignored_flags = {
        "-m", "--transmittance_init_mode", "--stage_d_ownership_arm",
        "--stage_d_reuse_static_cache",
    }
    result, index = [], 0
    while index < len(command):
        value = command[index]
        if value == "--stage_d_reuse_static_cache":
            index += 1; continue
        if value in ignored_flags:
            index += 2; continue
        result.append(value); index += 1
    return result


def run_arm(arm, environment, record):
    command = training_command(arm)
    log = LOGS / f"{arm}.log"
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line); sys.stdout.flush(); handle.write(line); handle.flush()
        code = process.wait()
    record["arms"][arm] = {
        "command": command, "exit_code": code,
        "output": str(OUTPUT / OWNERSHIP_ARM_NAMES[arm]), "log": str(log),
    }
    atomic_json(OUTPUT / "ownership_operator_record.json", record)
    if code != 0:
        raise RuntimeError(f"ownership arm failed: {arm} exit={code}")


def run_audit(environment, record):
    command = [
        sys.executable, str(ROOT / "tools/audit_stage_d_ownership_ab.py"),
        "--output", str(OUTPUT), "--geometry-manifest", str(MANIFEST),
    ]
    if record["bird_roi"]["manifest"]:
        command.extend(("--bird-roi-manifest", record["bird_roi"]["manifest"]))
    cpu_environment = environment.copy(); cpu_environment["CUDA_VISIBLE_DEVICES"] = ""
    code = subprocess.run(command, cwd=ROOT, env=cpu_environment).returncode
    record["audit_command"] = command; record["audit_exit_code"] = code
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="required explicit execution flag")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing to run without --execute")
    prior_failed_output = archive_retryable_ownership_failure(OUTPUT)
    OUTPUT.mkdir(parents=True); LOGS.mkdir()
    record = {
        "schema": "rtgs_stage_d_cuboid_path_ownership_operator_v4",
        "status": "PREFLIGHT", "output": str(OUTPUT),
        "source": str(SOURCE), "geometry_manifest": str(MANIFEST),
        "cache": str(CACHE), "arms": {},
        "commands": {arm: training_command(arm) for arm in OWNERSHIP_ARM_NAMES},
        "bird_roi": independent_bird_roi(),
        "prior_failed_output_archive": (
            str(prior_failed_output.resolve()) if prior_failed_output else None
        ),
    }
    common = {
        arm: common_training_contract(command)
        for arm, command in record["commands"].items()
    }
    record["ab_common_contract_sha256"] = hashlib.sha256(
        json.dumps(common["random_strict_inside"], separators=(",", ":")).encode()
    ).hexdigest()
    if common["random_strict_inside"] != common["transferred_d_inside"]:
        raise RuntimeError("A/B training commands differ beyond initialization/output/cache reuse")
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.8",
        "RTGS_BVH_JIT_ROOT": "/tmp/rtgs-stage-d-ownership-v4-jit",
    })
    exit_code = 2
    try:
        if git("status", "--porcelain"):
            raise RuntimeError("ownership pilot requires a clean committed worktree")
        record["git_commit"] = git("rev-parse", "HEAD")
        record["git_branch"] = git("branch", "--show-current")
        if sha256_file(SOURCE) != FORMAL_SOURCE_SHA256:
            raise ValueError("source checkpoint SHA-256 mismatch")
        header = torch.load(SOURCE, map_location="cpu")
        if header.get("format") != "rtgs_stage_b" or header.get("global_iteration") != 15000 \
                or header.get("reflection_iteration") != 12000:
            raise ValueError("source checkpoint header mismatch")
        if len(header.get("rng_state", {}).get("cuda") or []) != 1:
            raise ValueError("source checkpoint must contain exactly one CUDA RNG state")
        del header
        release = validate_geometry_release(MANIFEST)
        if release["aggregate_sha256"] != FORMAL_RELEASE_SHA256:
            raise ValueError("geometry release aggregate mismatch")
        if shutil.disk_usage(ROOT).free < 30 * 2**30:
            raise RuntimeError("ownership A/B requires at least 30 GiB free storage")
        query = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ], text=True)
        observed, conflicts = classify_compute_processes(query)
        if conflicts:
            raise RuntimeError(f"conflicting GPU compute processes are active: {conflicts}")
        record.update({
            "source_sha256_before": FORMAL_SOURCE_SHA256,
            "release_aggregate_before": FORMAL_RELEASE_SHA256,
            "preexisting_display_compute_processes": observed,
            "status": "RUNNING_ARM_A",
        })
        atomic_json(OUTPUT / "ownership_operator_record.json", record)
        run_arm("random_strict_inside", environment, record)
        if not (CACHE / "manifest.json").is_file():
            raise RuntimeError("Arm A did not create the dedicated cuboid-front cache")
        record["status"] = "RUNNING_ARM_B"
        atomic_json(OUTPUT / "ownership_operator_record.json", record)
        run_arm("transferred_d_inside", environment, record)
        record["source_sha256_after"] = sha256_file(SOURCE)
        record["release_aggregate_after"] = validate_geometry_release(MANIFEST)["aggregate_sha256"]
        if record["source_sha256_after"] != FORMAL_SOURCE_SHA256 \
                or record["release_aggregate_after"] != FORMAL_RELEASE_SHA256:
            raise RuntimeError("immutable source/release changed during ownership A/B")
        record["status"] = "AUDIT_PENDING"
        atomic_json(OUTPUT / "ownership_operator_record.json", record)
        exit_code = run_audit(environment, record)
        audit = json.loads((OUTPUT / "ownership_ab_cpu_audit.json").read_text(encoding="utf-8"))
        record["verdict"] = audit.get("verdict")
        record["status"] = record["verdict"]
    except Exception as exc:
        record["status"] = "CUBOID_PATH_OWNERSHIP_PILOT_BLOCKED"
        record["operator_error"] = f"{type(exc).__name__}: {exc}"
        record.setdefault("source_sha256_after", sha256_file(SOURCE) if SOURCE.is_file() else None)
        try:
            record.setdefault(
                "release_aggregate_after", validate_geometry_release(MANIFEST)["aggregate_sha256"]
            )
        except Exception:
            record.setdefault("release_aggregate_after", None)
        atomic_json(OUTPUT / "ownership_operator_record.json", record)
        try:
            run_audit(environment, record)
        except Exception as audit_exc:
            record["audit_launch_error"] = f"{type(audit_exc).__name__}: {audit_exc}"
        exit_code = 2
    atomic_json(OUTPUT / "ownership_operator_record.json", record)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
