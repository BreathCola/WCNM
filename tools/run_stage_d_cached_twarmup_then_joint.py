#!/usr/bin/env python3
"""Single-command operator for cached T warm-up followed by exact joint Stage D."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_release import sha256_file, validate_geometry_release
from stage_d_training import (
    CACHED_NODES, CACHED_OUTPUT_NAME, FORMAL_RELEASE_SHA256, FORMAL_SOURCE_SHA256,
)


SOURCE = ROOT / "output/tier2_c03_r8_oneshot_v4_rstart_g03000_to_g15000/chkpnt15000.pth"
MANIFEST = ROOT / "geometry_releases/stage_c_geometry_release_v1.json"
OUTPUT = ROOT / "output" / CACHED_OUTPUT_NAME
LOG = OUTPUT.with_suffix(".log")
ALLOWED_DISPLAY_COMPUTE = {"/usr/libexec/gnome-remote-desktop-daemon": 512}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
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
        record = {
            "pid": int(fields[0]), "process_name": fields[1],
            "used_gpu_memory_mib": int(fields[2]),
        }
        observed.append(record)
        limit = ALLOWED_DISPLAY_COMPUTE.get(record["process_name"])
        if limit is None or record["used_gpu_memory_mib"] > limit:
            conflicts.append(record)
    return observed, conflicts


def training_command():
    return [
        sys.executable, str(ROOT / "train.py"),
        "-s", str(ROOT / "data/TiHuBird"), "-m", str(OUTPUT),
        "--images", "images", "--model_type", "surfel", "--stage", "stage_d",
        "--experiment", "TiHuBird C03-r8 Stage D cached T warm-up then exact joint v1",
        "--resolution", "8",
        "--normal_priors", "diffrender_priors_candidates/axis_smoke/C03/normal",
        "--normal_prior_space", "camera",
        "--specular_masks", "specular_masks_reviewed_v1/manifest.json",
        "--geometry_release_manifest", str(MANIFEST),
        "--transmittance_init_mode", "random_bbox",
        "--transmittance_init_count", "4096",
        "--transmittance_init_seed", "20260703",
        "--transmittance_compose", "alpha_over",
        "--ray_background", "scene", "--ray_chunk_size", "2048",
        "--iterations", "20000", "--start_checkpoint", str(SOURCE),
        "--lambda_spec", "0.2", "--specular_k0", "0.9",
        "--lambda_depth", "0.2", "--stage_d_depth_start_iteration", "40000",
        "--stage_d_cached_twarmup", "--stage_d_phase_a_end_iteration", "18000",
        "--stage_d_cache_parity_atol", "2e-5",
        "--stage_d_cache_parity_mean_atol", "2e-6",
        "--disable_viewer", "--quiet",
        "--save_iterations", *[str(node) for node in CACHED_NODES],
        "--checkpoint_iterations", *[str(node) for node in CACHED_NODES],
    ]


def run_audit(environment, record):
    command = [
        sys.executable, str(ROOT / "tools/audit_stage_d_cached_twarmup.py"),
        "--output", str(OUTPUT), "--geometry-manifest", str(MANIFEST),
    ]
    audit_environment = environment.copy()
    audit_environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(command, cwd=ROOT, env=audit_environment)
    record["audit_command"] = command
    record["audit_exit_code"] = result.returncode
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="required explicit execution flag")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing to run without --execute")
    if OUTPUT.exists() or LOG.exists():
        raise FileExistsError(f"refusing to overwrite cached Stage D output/log: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    record_path = OUTPUT / "cached_twarmup_operator_record.json"
    record = {
        "schema": "rtgs_stage_d_cached_twarmup_operator_v1",
        "output": str(OUTPUT), "status": "PREFLIGHT",
        "source": str(SOURCE), "geometry_manifest": str(MANIFEST),
        "training_command": training_command(),
    }
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.8",
        "RTGS_BVH_JIT_ROOT": "/tmp/rtgs-stage-d-cached-twarmup-v1-jit",
    })
    exit_code = 2
    try:
        dirty = git("status", "--porcelain")
        if dirty:
            raise RuntimeError("formal cached Stage D requires a clean committed worktree")
        record["git_commit"] = git("rev-parse", "HEAD")
        record["git_branch"] = git("branch", "--show-current")
        source_sha = sha256_file(SOURCE)
        if source_sha != FORMAL_SOURCE_SHA256:
            raise ValueError("source checkpoint SHA-256 mismatch")
        source_header = torch.load(SOURCE, map_location="cpu")
        if source_header.get("global_iteration") != 15000:
            raise ValueError("source checkpoint global iteration mismatch")
        if len(source_header.get("rng_state", {}).get("cuda") or []) != 1:
            raise ValueError("source checkpoint must contain exactly one CUDA RNG state")
        del source_header
        release = validate_geometry_release(MANIFEST)
        if release["aggregate_sha256"] != FORMAL_RELEASE_SHA256:
            raise ValueError("geometry release aggregate mismatch")
        if shutil.disk_usage(ROOT).free < 30 * 2**30:
            raise RuntimeError("cached Stage D run requires at least 30 GiB free storage")
        gpu_query = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ], text=True)
        observed, conflicts = classify_compute_processes(gpu_query)
        if conflicts:
            raise RuntimeError(f"conflicting GPU compute processes are active: {conflicts}")
        record.update({
            "source_sha256_before": source_sha,
            "release_aggregate_before": release["aggregate_sha256"],
            "preexisting_display_compute_processes": observed,
            "status": "RUNNING",
        })
        atomic_json(record_path, record)
        with LOG.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                record["training_command"], cwd=ROOT, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in process.stdout:
                sys.stdout.write(line); sys.stdout.flush()
                log.write(line); log.flush()
            record["training_exit_code"] = process.wait()
        record["source_sha256_after"] = sha256_file(SOURCE)
        record["release_aggregate_after"] = validate_geometry_release(MANIFEST)["aggregate_sha256"]
        record["status"] = (
            "TRAINING_COMPLETED_AUDIT_PENDING"
            if record["training_exit_code"] == 0 else "CACHED_T_WARMUP_BLOCKED"
        )
        atomic_json(record_path, record)
        exit_code = run_audit(environment, record)
        record["status"] = (
            "AUDIT_COMPLETED" if exit_code == 0 else "CACHED_T_WARMUP_BLOCKED"
        )
    except Exception as exc:
        record["status"] = "CACHED_T_WARMUP_BLOCKED"
        record["operator_error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(record_path, record)
        try:
            run_audit(environment, record)
        except Exception as audit_exc:
            record["audit_launch_error"] = f"{type(audit_exc).__name__}: {audit_exc}"
        exit_code = 2
    atomic_json(record_path, record)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
