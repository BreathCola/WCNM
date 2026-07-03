#!/usr/bin/env python3
"""One-command operator for the 1,000-step Stage D semantic-repair pilot."""

from __future__ import annotations

import argparse
import hashlib
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
    FORMAL_RELEASE_SHA256, FORMAL_SOURCE_SHA256, SEMANTIC_NODES,
    SEMANTIC_OUTPUT_NAME,
)


SOURCE = ROOT / "output/tier2_c03_r8_oneshot_v4_rstart_g03000_to_g15000/chkpnt15000.pth"
MANIFEST = ROOT / "geometry_releases/stage_c_geometry_release_v1.json"
OUTPUT = ROOT / "output" / SEMANTIC_OUTPUT_NAME
LOG = OUTPUT.with_suffix(".log")
V2 = ROOT / "output/stage_d_tihubird_c03r8_cached_twarmup_then_joint_g15000_g20000_v2"
V2_AUDIT = ROOT / "output/stage_d_tihubird_c03r8_semantic_repair_v3_audit"
ALLOWED_DISPLAY_COMPUTE = {"/usr/libexec/gnome-remote-desktop-daemon": 512}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def tree_identity(directory):
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            stat = path.stat()
            rows.append((path.relative_to(directory).as_posix(), stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {"file_count": len(rows), "stat_manifest_sha256": digest}


def classify_compute_processes(text):
    observed, conflicts = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError(f"unexpected nvidia-smi compute-process row: {line!r}")
        record = {"pid": int(fields[0]), "process_name": fields[1], "used_gpu_memory_mib": int(fields[2])}
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
        "--experiment", "TiHuBird C03-r8 Stage D semantic-repair v3 pilot",
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
        "--iterations", "16000", "--start_checkpoint", str(SOURCE),
        "--lambda_spec", "0.2", "--specular_k0", "0.9",
        "--lambda_depth", "0.2", "--stage_d_depth_start_iteration", "40000",
        "--stage_d_semantic_repair_pilot",
        "--stage_d_phase_a_end_iteration", "16000",
        "--transparent_interface_margin", "0.05",
        "--transparent_interface_margin_mode", "exclude",
        "--lambda_anti_veil_black", "0.05",
        "--lambda_anti_veil_saturation", "0.02",
        "--anti_veil_gt_luminance_threshold", "0.15",
        "--anti_veil_high_alpha_threshold", "0.80",
        "--anti_veil_black_luminance_threshold", "0.08",
        "--anti_veil_saturation_alpha_threshold", "0.95",
        "--anti_veil_target_saturation_coverage", "0.35",
        "--anti_veil_gate_temperature", "0.05",
        "--anti_veil_black_temperature", "0.02",
        "--anti_veil_coverage_temperature", "0.02",
        "--anti_veil_ramp_start", "0", "--anti_veil_ramp_end", "200",
        "--stage_d_cache_parity_atol", "2e-5",
        "--stage_d_cache_parity_mean_atol", "2e-6",
        "--disable_viewer", "--quiet",
        "--save_iterations", *[str(node) for node in SEMANTIC_NODES],
        "--checkpoint_iterations", *[str(node) for node in SEMANTIC_NODES],
    ]


def run_cpu(command, environment):
    cpu_environment = environment.copy(); cpu_environment["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(command, cwd=ROOT, env=cpu_environment).returncode


def run_final_audit(environment, record):
    command = [
        sys.executable, str(ROOT / "tools/audit_stage_d_semantic_repair.py"),
        "--output", str(OUTPUT), "--geometry-manifest", str(MANIFEST),
    ]
    code = run_cpu(command, environment)
    record["audit_command"] = command; record["audit_exit_code"] = code
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="required explicit execution flag")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing to run without --execute")
    if OUTPUT.exists() or LOG.exists():
        raise FileExistsError(f"refusing to overwrite semantic pilot output/log: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    record_path = OUTPUT / "semantic_repair_operator_record.json"
    record = {
        "schema": "rtgs_stage_d_semantic_repair_operator_v3",
        "output": str(OUTPUT), "status": "PREFLIGHT",
        "source": str(SOURCE), "geometry_manifest": str(MANIFEST),
        "v2_read_only_source": str(V2), "v2_causal_audit": str(V2_AUDIT),
        "training_command": training_command(),
    }
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.8",
        "RTGS_BVH_JIT_ROOT": "/tmp/rtgs-stage-d-semantic-repair-v3-jit",
    })
    exit_code = 2
    try:
        if git("status", "--porcelain"):
            raise RuntimeError("semantic pilot requires a clean committed worktree")
        record["git_commit"] = git("rev-parse", "HEAD")
        record["git_branch"] = git("branch", "--show-current")
        if not V2.is_dir():
            raise FileNotFoundError("completed v2 output is unavailable for read-only audit")
        record["v2_tree_before"] = tree_identity(V2)
        source_sha = sha256_file(SOURCE)
        if source_sha != FORMAL_SOURCE_SHA256:
            raise ValueError("source checkpoint SHA-256 mismatch")
        header = torch.load(SOURCE, map_location="cpu")
        if header.get("format") != "rtgs_stage_b" or header.get("global_iteration") != 15000:
            raise ValueError("source checkpoint header mismatch")
        if len(header.get("rng_state", {}).get("cuda") or []) != 1:
            raise ValueError("source checkpoint must contain exactly one CUDA RNG state")
        del header
        release = validate_geometry_release(MANIFEST)
        if release["aggregate_sha256"] != FORMAL_RELEASE_SHA256:
            raise ValueError("geometry release aggregate mismatch")
        if shutil.disk_usage(ROOT).free < 20 * 2**30:
            raise RuntimeError("semantic pilot requires at least 20 GiB free storage")
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
            "status": "V2_CAUSAL_AUDIT",
        })
        atomic_json(record_path, record)
        causal_command = [
            sys.executable, str(ROOT / "tools/audit_stage_d_v2_semantic_failure.py"),
            "--v2", str(V2), "--output", str(V2_AUDIT),
            "--interface-margin", "0.05",
        ]
        record["v2_causal_audit_command"] = causal_command
        record["v2_causal_audit_exit_code"] = run_cpu(causal_command, environment)
        if record["v2_causal_audit_exit_code"] != 0:
            raise RuntimeError("v2 causal audit failed")
        record["v2_tree_after_audit"] = tree_identity(V2)
        if record["v2_tree_after_audit"] != record["v2_tree_before"]:
            raise RuntimeError("v2 changed during read-only causal audit")
        record["status"] = "RUNNING"
        atomic_json(record_path, record)
        with LOG.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                record["training_command"], cwd=ROOT, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in process.stdout:
                sys.stdout.write(line); sys.stdout.flush(); log.write(line); log.flush()
            record["training_exit_code"] = process.wait()
        record["source_sha256_after"] = sha256_file(SOURCE)
        record["release_aggregate_after"] = validate_geometry_release(MANIFEST)["aggregate_sha256"]
        record["v2_tree_after"] = tree_identity(V2)
        if record["source_sha256_after"] != FORMAL_SOURCE_SHA256:
            raise RuntimeError("source checkpoint mutated")
        if record["release_aggregate_after"] != FORMAL_RELEASE_SHA256:
            raise RuntimeError("geometry release mutated")
        if record["v2_tree_after"] != record["v2_tree_before"]:
            raise RuntimeError("v2 output mutated")
        record["status"] = (
            "TRAINING_COMPLETED_AUDIT_PENDING"
            if record["training_exit_code"] == 0 else "SEMANTIC_REPAIR_PILOT_BLOCKED"
        )
        atomic_json(record_path, record)
        exit_code = run_final_audit(environment, record)
        audit = json.loads((OUTPUT / "semantic_repair_cpu_audit.json").read_text(encoding="utf-8"))
        record["verdict"] = audit.get("verdict")
        record["status"] = audit.get("verdict", "SEMANTIC_REPAIR_PILOT_BLOCKED")
    except Exception as exc:
        record["status"] = "SEMANTIC_REPAIR_PILOT_BLOCKED"
        record["operator_error"] = f"{type(exc).__name__}: {exc}"
        record.setdefault("source_sha256_after", sha256_file(SOURCE) if SOURCE.is_file() else None)
        try:
            record.setdefault("release_aggregate_after", validate_geometry_release(MANIFEST)["aggregate_sha256"])
        except Exception:
            record.setdefault("release_aggregate_after", None)
        atomic_json(record_path, record)
        try: run_final_audit(environment, record)
        except Exception as audit_exc:
            record["audit_launch_error"] = f"{type(audit_exc).__name__}: {audit_exc}"
        exit_code = 2
    atomic_json(record_path, record)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
