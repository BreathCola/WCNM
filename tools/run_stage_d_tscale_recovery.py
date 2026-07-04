#!/usr/bin/env python3
"""Run the committed Stage D projected-scale recovery preflight or continuation."""

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
    FORMAL_RELEASE_SHA256, TSCALE_RECOVERY_LONG_ENDPOINT,
    TSCALE_RECOVERY_LONG_NODES, TSCALE_RECOVERY_LONG_OUTPUT_NAME,
    TSCALE_RECOVERY_PREFLIGHT_ENDPOINT, TSCALE_RECOVERY_PREFLIGHT_NODES,
    TSCALE_RECOVERY_PREFLIGHT_OUTPUT_NAME, TSCALE_RECOVERY_SOURCE_SHA256,
)
from tools.run_stage_d_ownership_ab import classify_compute_processes
from utils.stage_d_static_cache import state_sha256


EVIDENCE = ROOT / "output/stage_d_tihubird_c03r8_cuboid_path_ownership_tlong_g15500_g20000_v1"
PILOT = ROOT / "output/stage_d_tihubird_c03r8_cuboid_path_ownership_ab500_v4"
CACHE = PILOT / "cuboid_front_cache_v4"
MANIFEST = ROOT / "geometry_releases/stage_c_geometry_release_v1.json"
PREFLIGHT_OUTPUT = ROOT / "output" / TSCALE_RECOVERY_PREFLIGHT_OUTPUT_NAME
LONG_OUTPUT = ROOT / "output" / TSCALE_RECOVERY_LONG_OUTPUT_NAME
LEGACY_SOURCE = EVIDENCE / "chkpnt16000.pth"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def atomic_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def profile_contract(profile):
    if profile == "preflight":
        return {
            "source": LEGACY_SOURCE, "output": PREFLIGHT_OUTPUT,
            "endpoint": TSCALE_RECOVERY_PREFLIGHT_ENDPOINT,
            "nodes": TSCALE_RECOVERY_PREFLIGHT_NODES,
            "flag": "--stage_d_tscale_recovery_preflight",
            "experiment": "TiHuBird C03-r8 Stage D T-scale recovery preflight50 v1",
        }
    return {
        "source": PREFLIGHT_OUTPUT / "chkpnt16050.pth", "output": LONG_OUTPUT,
        "endpoint": TSCALE_RECOVERY_LONG_ENDPOINT, "nodes": TSCALE_RECOVERY_LONG_NODES,
        "flag": "--stage_d_tscale_recovery_long",
        "experiment": "TiHuBird C03-r8 Stage D projected T-scale recovery long v1",
    }


def training_command(profile):
    contract = profile_contract(profile)
    return [
        sys.executable, str(ROOT / "train.py"),
        "-s", str(ROOT / "data/TiHuBird"), "-m", str(contract["output"]),
        "--images", "images", "--model_type", "surfel", "--stage", "stage_d",
        "--experiment", contract["experiment"], "--resolution", "8",
        "--normal_priors", "diffrender_priors_candidates/axis_smoke/C03/normal",
        "--normal_prior_space", "camera",
        "--specular_masks", "specular_masks_reviewed_v1/manifest.json",
        "--geometry_release_manifest", str(MANIFEST),
        "--transmittance_init_mode", "transferred_d_inside",
        "--transmittance_init_count", "4096", "--transmittance_init_seed", "20260703",
        "--transmittance_compose", "alpha_over",
        "--transparent_path_mode", "cuboid_front_v1",
        "--transparent_direct_mode", "off", "--transparent_reflection_mode", "off",
        "--cout_ownership_mode", "support_safe_outside",
        "--stage_d_static_cache_path", str(CACHE), "--stage_d_reuse_static_cache",
        contract["flag"], "--stage_d_ownership_arm", "transferred_d_inside",
        "--ray_background", "scene", "--ray_chunk_size", "2048",
        "--iterations", str(contract["endpoint"]),
        "--start_checkpoint", str(contract["source"]),
        "--lambda_spec", "0.2", "--specular_k0", "0.9",
        "--lambda_depth", "0.2", "--stage_d_depth_start_iteration", "40000",
        "--stage_d_phase_a_end_iteration", str(contract["endpoint"]),
        "--transparent_interface_margin", "0.05",
        "--transparent_interface_margin_mode", "exclude",
        "--transfer_min_views", "3", "--transfer_min_total_weight", "0.01",
        "--transfer_depth_margin", "0.05", "--transfer_opacity_scale", "0.25",
        "--transfer_opacity_min", "0.005", "--transfer_opacity_max", "0.05",
        "--stage_d_cache_parity_atol", "2e-5",
        "--stage_d_cache_parity_mean_atol", "2e-6",
        "--disable_viewer", "--quiet",
        "--save_iterations", *[str(node) for node in contract["nodes"]],
        "--checkpoint_iterations", *[str(node) for node in contract["nodes"]],
    ]


def audit_command(profile):
    contract = profile_contract(profile)
    return [
        sys.executable, str(ROOT / "tools/audit_stage_d_tscale_recovery.py"),
        "--profile", profile, "--output", str(contract["output"]),
        "--source", str(contract["source"]), "--cache", str(CACHE),
        "--geometry-manifest", str(MANIFEST),
    ]


def validate_source(profile):
    contract = profile_contract(profile)
    source = contract["source"]
    if not source.is_file():
        raise FileNotFoundError(f"T-scale recovery source is missing: {source}")
    source_hash = sha256_file(source)
    if profile == "preflight" and source_hash != TSCALE_RECOVERY_SOURCE_SHA256:
        raise ValueError("global-16000 legacy checkpoint hash mismatch")
    if profile == "long":
        audit = json.loads(
            (PREFLIGHT_OUTPUT / "tscale_recovery_cpu_audit.json").read_text(encoding="utf-8")
        )
        if audit.get("verdict") != "TSCALE_RECOVERY_PREFLIGHT_PASS" \
                or audit.get("errors"):
            raise ValueError("T-scale recovery preflight is not continuable")
        expected = audit.get("checkpoints", {}).get("16050", {}).get("sha256")
        if source_hash != expected:
            raise ValueError("preflight global-16050 checkpoint hash mismatch")
    checkpoint = torch.load(source, map_location="cpu")
    expected_global = 16000 if profile == "preflight" else 16050
    if checkpoint.get("format") != "rtgs_stage_d" \
            or checkpoint.get("global_iteration") != expected_global \
            or checkpoint.get("reflection_iteration") != 12000 \
            or checkpoint.get("transmittance_iteration") != expected_global - 15000:
        raise ValueError("T-scale recovery source checkpoint header mismatch")
    if profile == "long" and checkpoint["transmittance"].get(
        "scaling_parameterization"
    ) != "cuboid_support_projected_cap_v2":
        raise ValueError("long source lacks projected T-scale parameterization")
    return source_hash


def write_operator_abort_hashes(output, source, error):
    candidates = sorted(output.glob("chkpnt*.pth"))
    checkpoint_path = candidates[-1] if candidates else source
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    atomic_json(output / "operator_abort_hashes.json", {
        "schema": "rtgs_stage_d_tscale_operator_abort_hashes_v1",
        "error": error, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "branch_hashes": {
            branch: state_sha256(checkpoint[branch])
            for branch in ("diffuse", "reflection", "transmittance")
        },
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute-preflight", action="store_true")
    group.add_argument("--execute-long", action="store_true")
    args = parser.parse_args()
    profile = "preflight" if args.execute_preflight else "long"
    contract = profile_contract(profile)
    output, source = contract["output"], contract["source"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T-scale recovery output: {output}")
    if git("status", "--porcelain"):
        raise RuntimeError("T-scale recovery requires a clean committed worktree")
    source_hash = validate_source(profile)
    release = validate_geometry_release(MANIFEST)
    if release["aggregate_sha256"] != FORMAL_RELEASE_SHA256:
        raise ValueError("Stage C release aggregate mismatch")
    cache_manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    if cache_manifest.get("identity", {}).get("cache_schema_version") \
            != "rtgs_stage_d_cuboid_front_cache_v4" \
            or len(cache_manifest.get("entries", [])) != 111:
        raise ValueError("ownership v4 cache identity/count mismatch")
    evidence_before = tree_sha256(EVIDENCE)
    if shutil.disk_usage(ROOT).free < 20 * 2**30:
        raise RuntimeError("T-scale recovery requires at least 20 GiB free storage")
    query = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    observed, conflicts = classify_compute_processes(query)
    if conflicts:
        raise RuntimeError(f"conflicting GPU compute processes are active: {conflicts}")

    output.mkdir(parents=True)
    record = {
        "schema": "rtgs_stage_d_tscale_recovery_operator_v1",
        "profile": profile, "status": "RUNNING",
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "source": str(source), "source_sha256_before": source_hash,
        "evidence": str(EVIDENCE), "evidence_tree_sha256_before": evidence_before,
        "geometry_manifest": str(MANIFEST),
        "release_aggregate_before": FORMAL_RELEASE_SHA256,
        "cache": str(CACHE),
        "cache_identity": cache_manifest["identity"]["identity_sha256"],
        "command": training_command(profile),
        "preexisting_display_compute_processes": observed,
    }
    atomic_json(output / "tscale_recovery_operator_record.json", record)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.8",
        "RTGS_BVH_JIT_ROOT": f"/tmp/rtgs-stage-d-tscale-{profile}-v1-jit",
    })
    log = output / "tscale_recovery.log"
    exit_code = 2
    try:
        with log.open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(
                training_command(profile), cwd=ROOT, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in process.stdout:
                sys.stdout.write(line); sys.stdout.flush()
                handle.write(line); handle.flush()
            train_code = process.wait()
        record["training_exit_code"] = train_code
        if train_code != 0:
            raise RuntimeError(f"T-scale recovery training failed: exit={train_code}")
        record["source_sha256_after"] = sha256_file(source)
        record["release_aggregate_after"] = validate_geometry_release(MANIFEST)[
            "aggregate_sha256"
        ]
        record["evidence_tree_sha256_after"] = tree_sha256(EVIDENCE)
        if record["source_sha256_after"] != source_hash \
                or record["release_aggregate_after"] != FORMAL_RELEASE_SHA256 \
                or record["evidence_tree_sha256_after"] != evidence_before:
            raise RuntimeError("immutable source/release/evidence changed")
        record["status"] = "AUDIT_PENDING"
        atomic_json(output / "tscale_recovery_operator_record.json", record)
        command = audit_command(profile)
        cpu = environment.copy(); cpu["CUDA_VISIBLE_DEVICES"] = ""
        audit_code = subprocess.run(command, cwd=ROOT, env=cpu).returncode
        record["audit_command"] = command; record["audit_exit_code"] = audit_code
        audit = json.loads((output / "tscale_recovery_cpu_audit.json").read_text())
        record["verdict"] = audit["verdict"]; record["status"] = audit["verdict"]
        exit_code = audit_code
    except Exception as exc:
        verdict = (
            "TSCALE_RECOVERY_PREFLIGHT_BLOCKED"
            if profile == "preflight" else "TSCALE_RECOVERY_LONG_BLOCKED"
        )
        record["status"] = verdict
        record["operator_error"] = f"{type(exc).__name__}: {exc}"
        record.setdefault("source_sha256_after", sha256_file(source))
        record.setdefault(
            "release_aggregate_after", validate_geometry_release(MANIFEST)["aggregate_sha256"]
        )
        record.setdefault("evidence_tree_sha256_after", tree_sha256(EVIDENCE))
        write_operator_abort_hashes(output, source, record["operator_error"])
        atomic_json(output / "tscale_recovery_operator_record.json", record)
        try:
            command = audit_command(profile)
            cpu = environment.copy(); cpu["CUDA_VISIBLE_DEVICES"] = ""
            record["audit_command"] = command
            record["audit_exit_code"] = subprocess.run(
                command, cwd=ROOT, env=cpu
            ).returncode
        except Exception as audit_exc:
            record["audit_launch_error"] = f"{type(audit_exc).__name__}: {audit_exc}"
    atomic_json(output / "tscale_recovery_operator_record.json", record)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
