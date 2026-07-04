#!/usr/bin/env python3
"""Run the bounded transferred-T ownership continuation through global 20,000."""

from __future__ import annotations

import argparse
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
    FORMAL_RELEASE_SHA256, FORMAL_SOURCE_SHA256, OWNERSHIP_T_LONG_ENDPOINT,
    OWNERSHIP_T_LONG_NODES, OWNERSHIP_T_LONG_OUTPUT_NAME,
    OWNERSHIP_T_LONG_SOURCE_SHA256,
)
from tools.run_stage_d_ownership_ab import classify_compute_processes


PILOT = ROOT / "output/stage_d_tihubird_c03r8_cuboid_path_ownership_ab500_v4"
SOURCE = PILOT / "arm_b_transferred_d_inside/chkpnt15500.pth"
CACHE = PILOT / "cuboid_front_cache_v4"
PILOT_AUDIT = PILOT / "ownership_ab_cpu_audit.json"
MANIFEST = ROOT / "geometry_releases/stage_c_geometry_release_v1.json"
OUTPUT = ROOT / "output" / OWNERSHIP_T_LONG_OUTPUT_NAME
LOG = OUTPUT / "ownership_t_long.log"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def atomic_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def training_command():
    return [
        sys.executable, str(ROOT / "train.py"),
        "-s", str(ROOT / "data/TiHuBird"), "-m", str(OUTPUT),
        "--images", "images", "--model_type", "surfel", "--stage", "stage_d",
        "--experiment", "TiHuBird C03-r8 Stage D ownership transferred-T long v1",
        "--resolution", "8",
        "--normal_priors", "diffrender_priors_candidates/axis_smoke/C03/normal",
        "--normal_prior_space", "camera",
        "--specular_masks", "specular_masks_reviewed_v1/manifest.json",
        "--geometry_release_manifest", str(MANIFEST),
        "--transmittance_init_mode", "transferred_d_inside",
        "--transmittance_init_count", "4096", "--transmittance_init_seed", "20260703",
        "--transmittance_compose", "alpha_over",
        "--transparent_path_mode", "cuboid_front_v1",
        "--transparent_direct_mode", "off",
        "--transparent_reflection_mode", "off",
        "--cout_ownership_mode", "support_safe_outside",
        "--stage_d_static_cache_path", str(CACHE),
        "--stage_d_reuse_static_cache", "--stage_d_ownership_t_long",
        "--stage_d_ownership_arm", "transferred_d_inside",
        "--ray_background", "scene", "--ray_chunk_size", "2048",
        "--iterations", str(OWNERSHIP_T_LONG_ENDPOINT),
        "--start_checkpoint", str(SOURCE),
        "--lambda_spec", "0.2", "--specular_k0", "0.9",
        "--lambda_depth", "0.2", "--stage_d_depth_start_iteration", "40000",
        "--stage_d_phase_a_end_iteration", str(OWNERSHIP_T_LONG_ENDPOINT),
        "--transparent_interface_margin", "0.05",
        "--transparent_interface_margin_mode", "exclude",
        "--transfer_min_views", "3", "--transfer_min_total_weight", "0.01",
        "--transfer_depth_margin", "0.05", "--transfer_opacity_scale", "0.25",
        "--transfer_opacity_min", "0.005", "--transfer_opacity_max", "0.05",
        "--stage_d_cache_parity_atol", "2e-5",
        "--stage_d_cache_parity_mean_atol", "2e-6",
        "--disable_viewer", "--quiet",
        "--save_iterations", *[str(node) for node in OWNERSHIP_T_LONG_NODES],
        "--checkpoint_iterations", *[str(node) for node in OWNERSHIP_T_LONG_NODES],
    ]


def validate_pilot_source():
    audit = json.loads(PILOT_AUDIT.read_text(encoding="utf-8"))
    if audit.get("technical_run_complete") is not True or audit.get("errors"):
        raise ValueError("ownership A/B source audit is not technically complete")
    if audit.get("verdict") not in (
        "CUBOID_PATH_OWNERSHIP_PILOT_HOLD", "CUBOID_PATH_OWNERSHIP_PILOT_PASS",
    ):
        raise ValueError("ownership A/B source verdict is not continuable")
    endpoint = audit.get("arms", {}).get("transferred_d_inside", {}).get(
        "checkpoints", {}
    ).get("15500", {})
    if endpoint.get("sha256") != OWNERSHIP_T_LONG_SOURCE_SHA256:
        raise ValueError("ownership A/B audit endpoint hash mismatch")
    metadata = json.loads(
        (PILOT / "arm_b_transferred_d_inside/ownership_arm_metadata.json")
        .read_text(encoding="utf-8")
    )
    if metadata.get("phase_a_frozen_hash_before") \
            != metadata.get("phase_a_frozen_hash_after"):
        raise ValueError("ownership A/B source D/R state is not frozen")
    cache = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    if cache.get("identity", {}).get("cache_schema_version") \
            != "rtgs_stage_d_cuboid_front_cache_v4" \
            or len(cache.get("entries", [])) != 111:
        raise ValueError("ownership v4 cache identity/count mismatch")
    return audit, metadata, cache


def run_audit(environment):
    command = [
        sys.executable, str(ROOT / "tools/audit_stage_d_ownership_t_long.py"),
        "--output", str(OUTPUT), "--source", str(SOURCE),
        "--pilot-audit", str(PILOT_AUDIT), "--cache", str(CACHE),
        "--geometry-manifest", str(MANIFEST),
    ]
    cpu = environment.copy(); cpu["CUDA_VISIBLE_DEVICES"] = ""
    return command, subprocess.run(command, cwd=ROOT, env=cpu).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing to run without --execute")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite ownership T-long output: {OUTPUT}")
    if git("status", "--porcelain"):
        raise RuntimeError("ownership T-long requires a clean committed worktree")
    if sha256_file(SOURCE) != OWNERSHIP_T_LONG_SOURCE_SHA256:
        raise ValueError("ownership T-long source checkpoint hash mismatch")
    header = torch.load(SOURCE, map_location="cpu")
    if header.get("format") != "rtgs_stage_d" \
            or header.get("global_iteration") != 15500 \
            or header.get("reflection_iteration") != 12000 \
            or header.get("transmittance_iteration") != 500:
        raise ValueError("ownership T-long source checkpoint header mismatch")
    del header
    release = validate_geometry_release(MANIFEST)
    if release["aggregate_sha256"] != FORMAL_RELEASE_SHA256:
        raise ValueError("Stage C release aggregate mismatch")
    pilot_audit, pilot_metadata, cache = validate_pilot_source()
    if shutil.disk_usage(ROOT).free < 20 * 2**30:
        raise RuntimeError("ownership T-long requires at least 20 GiB free storage")
    query = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    observed, conflicts = classify_compute_processes(query)
    if conflicts:
        raise RuntimeError(f"conflicting GPU compute processes are active: {conflicts}")
    OUTPUT.mkdir(parents=True)
    record = {
        "schema": "rtgs_stage_d_ownership_t_long_operator_v1",
        "status": "RUNNING", "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "source": str(SOURCE), "source_sha256_before": OWNERSHIP_T_LONG_SOURCE_SHA256,
        "geometry_manifest": str(MANIFEST),
        "release_aggregate_before": FORMAL_RELEASE_SHA256,
        "cache": str(CACHE), "cache_identity": cache["identity"]["identity_sha256"],
        "pilot_verdict": pilot_audit["verdict"],
        "pilot_frozen_hash": pilot_metadata["phase_a_frozen_hash_before"],
        "command": training_command(), "preexisting_display_compute_processes": observed,
    }
    atomic_json(OUTPUT / "ownership_t_long_operator_record.json", record)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.8",
        "RTGS_BVH_JIT_ROOT": "/tmp/rtgs-stage-d-ownership-t-long-v1-jit",
    })
    exit_code = 2
    try:
        with LOG.open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(
                training_command(), cwd=ROOT, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in process.stdout:
                sys.stdout.write(line); sys.stdout.flush(); handle.write(line); handle.flush()
            train_code = process.wait()
        record["training_exit_code"] = train_code
        if train_code != 0:
            raise RuntimeError(f"ownership T-long training failed: exit={train_code}")
        record["source_sha256_after"] = sha256_file(SOURCE)
        record["release_aggregate_after"] = validate_geometry_release(MANIFEST)["aggregate_sha256"]
        if record["source_sha256_after"] != OWNERSHIP_T_LONG_SOURCE_SHA256 \
                or record["release_aggregate_after"] != FORMAL_RELEASE_SHA256:
            raise RuntimeError("immutable ownership source/release changed")
        record["status"] = "AUDIT_PENDING"
        atomic_json(OUTPUT / "ownership_t_long_operator_record.json", record)
        audit_command, audit_code = run_audit(environment)
        record["audit_command"] = audit_command; record["audit_exit_code"] = audit_code
        audit = json.loads((OUTPUT / "ownership_t_long_cpu_audit.json").read_text())
        record["verdict"] = audit["verdict"]; record["status"] = audit["verdict"]
        exit_code = audit_code
    except Exception as exc:
        record["status"] = "OWNERSHIP_T_LONG_BLOCKED"
        record["operator_error"] = f"{type(exc).__name__}: {exc}"
        record.setdefault("source_sha256_after", sha256_file(SOURCE))
        record.setdefault(
            "release_aggregate_after", validate_geometry_release(MANIFEST)["aggregate_sha256"]
        )
        atomic_json(OUTPUT / "ownership_t_long_operator_record.json", record)
        try:
            audit_command, audit_code = run_audit(environment)
            record["audit_command"] = audit_command; record["audit_exit_code"] = audit_code
        except Exception as audit_exc:
            record["audit_launch_error"] = f"{type(audit_exc).__name__}: {audit_exc}"
    atomic_json(OUTPUT / "ownership_t_long_operator_record.json", record)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
