#!/usr/bin/env python3
"""One-command operator for the authorized TiHuBird Stage D formal onset run."""

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
    FORMAL_NODES, FORMAL_OUTPUT_NAME, FORMAL_RELEASE_SHA256, FORMAL_SOURCE_SHA256,
)

SOURCE = ROOT / "output/tier2_c03_r8_oneshot_v4_rstart_g03000_to_g15000/chkpnt15000.pth"
MANIFEST = ROOT / "geometry_releases/stage_c_geometry_release_v1.json"
OUTPUT = ROOT / "output" / FORMAL_OUTPUT_NAME
ALLOWED_DISPLAY_COMPUTE = {"/usr/libexec/gnome-remote-desktop-daemon": 512}
ZERO_STEP_ATTEMPT_COMMIT = "405cfca5a311c0013e5eab2f360efa26b9c890eb"
ZERO_STEP_ARCHIVE = "preflight_attempt_cuda_rng_cardinality"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def classify_compute_processes(text):
    observed, conflicts = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError(f"unexpected nvidia-smi compute-process row: {line!r}")
        pid, process_name, memory_text = fields
        record = {"pid": int(pid), "process_name": process_name, "used_gpu_memory_mib": int(memory_text)}
        observed.append(record)
        allowed_limit = ALLOWED_DISPLAY_COMPUTE.get(process_name)
        if allowed_limit is None or record["used_gpu_memory_mib"] > allowed_limit:
            conflicts.append(record)
    return observed, conflicts


def preserve_known_zero_step_attempt(output, log_path):
    """Preserve, never delete, the one audited pre-update v1 launch footprint."""
    if not output.exists() and not log_path.exists():
        output.mkdir(parents=True)
        return None
    if not output.is_dir() or not log_path.is_file():
        raise FileExistsError("formal output/log collision is not the known zero-step attempt")
    archive = output / ZERO_STEP_ARCHIVE
    if archive.exists():
        raise FileExistsError(f"zero-step archive already exists: {archive}")
    entries = list(output.iterdir())
    event_files = [path for path in entries if path.name.startswith("events.out.tfevents.")]
    expected_names = {"cameras.json", "cfg_args", "formal_operator_record.json", "input.ply"}
    if len(event_files) != 1 or {path.name for path in entries if path not in event_files} != expected_names:
        raise FileExistsError("formal partial output has unexpected files and cannot be reused")
    record = json.loads((output / "formal_operator_record.json").read_text(encoding="utf-8"))
    if not (
        record.get("schema") == "rtgs_stage_d_formal_operator_v1"
        and record.get("git_commit") == ZERO_STEP_ATTEMPT_COMMIT
        and record.get("status") == "BLOCKED"
        and record.get("training_exit_code") == 1
        and record.get("source_sha256_before") == FORMAL_SOURCE_SHA256
        and record.get("source_sha256_after") == FORMAL_SOURCE_SHA256
        and record.get("release_aggregate_before") == FORMAL_RELEASE_SHA256
        and record.get("release_aggregate_after") == FORMAL_RELEASE_SHA256
    ):
        raise ValueError("formal partial output record is not the audited zero-step attempt")
    log_text = log_path.read_text(encoding="utf-8")
    if (
        "checkpoint CUDA RNG cardinality does not match visible CUDA devices" not in log_text
        or "Stage D training progress" in log_text
    ):
        raise ValueError("formal partial log does not prove a pre-update RNG-cardinality stop")
    archive.mkdir()
    for path in entries:
        os.replace(path, archive / path.name)
    os.replace(log_path, archive / "operator.log")
    return str(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="required explicit execution flag")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing to run without --execute")
    if git("status", "--porcelain"):
        raise RuntimeError("formal training requires a clean committed worktree")
    if sha256_file(SOURCE) != FORMAL_SOURCE_SHA256:
        raise ValueError("formal source checkpoint SHA-256 mismatch")
    source_header = torch.load(SOURCE, map_location="cpu")
    source_cuda_rng_count = len(source_header.get("rng_state", {}).get("cuda") or [])
    del source_header
    if source_cuda_rng_count != 1:
        raise ValueError(f"formal source must contain exactly one CUDA RNG state, found {source_cuda_rng_count}")
    release = validate_geometry_release(MANIFEST)
    if release["aggregate_sha256"] != FORMAL_RELEASE_SHA256:
        raise ValueError("formal geometry release aggregate mismatch")
    if shutil.disk_usage(ROOT).free < 20 * 2**30:
        raise RuntimeError("formal run requires at least 20 GiB free workspace storage")
    gpu_query = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    observed_compute, conflicting_compute = classify_compute_processes(gpu_query)
    if conflicting_compute:
        raise RuntimeError(f"conflicting GPU compute processes are active: {conflicting_compute}")

    command = [
        sys.executable, str(ROOT / "train.py"),
        "-s", str(ROOT / "data/TiHuBird"), "-m", str(OUTPUT),
        "--images", "images", "--model_type", "surfel", "--stage", "stage_d",
        "--experiment", "TiHuBird C03-r8 Stage D formal T-onset v1",
        "--resolution", "8",
        "--normal_priors", "diffrender_priors_candidates/axis_smoke/C03/normal",
        "--normal_prior_space", "camera",
        "--specular_masks", "specular_masks_reviewed_v1/manifest.json",
        "--geometry_release_manifest", str(MANIFEST),
        "--transmittance_init_mode", "random_bbox",
        "--transmittance_init_count", "4096",
        "--transmittance_init_seed", "20260703",
        "--transmittance_compose", "alpha_over",
        "--ray_background", "scene", "--ray_chunk_size", "512",
        "--iterations", "20000", "--start_checkpoint", str(SOURCE),
        "--lambda_spec", "0.2", "--specular_k0", "0.9",
        "--lambda_depth", "0.2", "--stage_d_depth_start_iteration", "40000",
        "--stage_d_formal_onset", "--disable_viewer", "--quiet",
        "--save_iterations", *[str(node) for node in FORMAL_NODES],
        "--checkpoint_iterations", *[str(node) for node in FORMAL_NODES],
    ]
    log_path = OUTPUT.with_suffix(".log")
    preserved_attempt = preserve_known_zero_step_attempt(OUTPUT, log_path)
    record_path = OUTPUT / "formal_operator_record.json"
    record = {
        "schema": "rtgs_stage_d_formal_operator_v1", "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "source": str(SOURCE), "source_sha256_before": FORMAL_SOURCE_SHA256,
        "geometry_manifest": str(MANIFEST), "release_aggregate_before": FORMAL_RELEASE_SHA256,
        "training_command": command, "output": str(OUTPUT), "status": "RUNNING",
        "preexisting_display_compute_processes": observed_compute,
        "source_cuda_rng_state_count": source_cuda_rng_count,
        "preserved_zero_step_attempt": preserved_attempt,
    }
    atomic_json(record_path, record)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.8",
        "RTGS_BVH_JIT_ROOT": "/tmp/rtgs-stage-d-formal-v1-jit",
    })
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line); sys.stdout.flush(); log.write(line); log.flush()
        return_code = process.wait()
    record["training_exit_code"] = return_code
    record["source_sha256_after"] = sha256_file(SOURCE)
    record["release_aggregate_after"] = validate_geometry_release(MANIFEST)["aggregate_sha256"]
    if return_code != 0:
        record["status"] = "BLOCKED"
        atomic_json(record_path, record)
        raise RuntimeError(f"formal training exited with status {return_code}")
    record["status"] = "TRAINING_COMPLETED_AUDIT_PENDING"
    atomic_json(record_path, record)

    audit_command = [
        sys.executable, str(ROOT / "tools/audit_stage_d_formal.py"),
        "--output", str(OUTPUT), "--geometry-manifest", str(MANIFEST),
    ]
    audit_environment = environment.copy()
    audit_environment["CUDA_VISIBLE_DEVICES"] = ""
    audit = subprocess.run(audit_command, cwd=ROOT, env=audit_environment)
    record["audit_command"] = audit_command
    record["audit_exit_code"] = audit.returncode
    record["status"] = "COMPLETED" if audit.returncode == 0 else "BLOCKED"
    atomic_json(record_path, record)
    return audit.returncode


if __name__ == "__main__":
    raise SystemExit(main())
