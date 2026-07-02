#!/usr/bin/env python3
"""One-shot coordinator and CPU-only final audit for the C03-r8 Tier-2 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_tier2_gate import ANOMALY_RE, _checkpoint_counts, _finite_scan, build_gate_packet


OUTPUT = ROOT / "output"
EXPERIMENT = "C03-r8 Tier 2 onset study"
RESOLUTION = 8
RETRY_IDENTITY = "oneshot_v4_memory_bounded_retry"
ALLOCATOR_POLICY = "adaptive_pressure_cache_and_ray_retry_v1"
REFERENCE_PEAK_ALLOCATED_BYTES = 0
MINIMUM_PROJECTED_HEADROOM_BYTES = 268435456
BOOTSTRAP = OUTPUT / "tier2_c03_r8_oneshot_shared_d_bootstrap_g00000_07000_v1"
BRANCH_A = OUTPUT / "tier2_c03_r8_oneshot_v4_rstart_g03000_to_g15000"
BRANCH_B = OUTPUT / "tier2_c03_r8_oneshot_v4_rstart_g07000_to_g15000"
STATE_PATH = OUTPUT / "tier2_c03_r8_oneshot_v4_state.json"
REPORT_PATH = OUTPUT / "tier2_c03_r8_oneshot_v4_final_audit.json"
COORDINATOR_LOG = OUTPUT / "tier2_c03_r8_oneshot_v4_coordinator.log"
PACKET_3000 = OUTPUT / "tier2_c03_r8_oneshot_v4_source_g03000_audit.json"
PACKET_7000 = OUTPUT / "tier2_c03_r8_oneshot_v4_source_g07000_audit.json"
PACKET_A100 = OUTPUT / "tier2_c03_r8_oneshot_v4_a_r0100_audit.json"
PACKET_B100 = OUTPUT / "tier2_c03_r8_oneshot_v4_b_r0100_audit.json"
V1_FINAL_AUDIT = OUTPUT / "tier2_c03_r8_oneshot_final_audit_v1.json"
EXPECTED_V1_FINAL_AUDIT_SHA256 = "72887af7b8d93ee64f59747c745030970e96aa379e7dc09f980bd749af8df88a"
V2_FINAL_AUDIT = OUTPUT / "tier2_c03_r8_oneshot_v2_final_audit.json"
EXPECTED_V2_FINAL_AUDIT_SHA256 = "492464757a0776d6b463a80d0b89fe2e668549586ab5a94bb01394142c0bcb7a"
V3_FINAL_AUDIT = OUTPUT / "tier2_c03_r8_oneshot_v3_final_audit.json"
EXPECTED_V3_FINAL_AUDIT_SHA256 = "776ab1102ad74e90ea35985e45ef29b9e73fecb99c20d86b71d53845858c9a4f"
EXPECTED_SOURCE_HASHES = {
    3000: "c8f83b17d53f49a3f283d25078e69cb4c8073b2b09354cc901ecc23eae772e6c",
    7000: "59461b60ac721f4e724b48ced9ee319f98ede49490f38bce91651590bb760d89",
}
SCRIPT = ROOT / "tools" / "tier2_operator.sh"
MASK = ROOT / "data" / "TiHuBird" / "specular_masks_reviewed_v1" / "manifest.json"
PRIORS = ROOT / "data" / "TiHuBird" / "diffrender_priors_candidates" / "axis_smoke" / "C03" / "normal"
POLL_SECONDS = 5.0
GPU_POLL_SECONDS = 30.0

TASK_SPECS = {
    "bootstrap": {
        "action": None, "gpu": None,
        "log": OUTPUT / "tier2_c03_r8_oneshot_shared_d_bootstrap_g00000_07000_v1.log",
        "telemetry": BOOTSTRAP / "telemetry" / "bootstrap_g00001_07000.jsonl",
        "global_start": 1, "local_start": None,
    },
    "branch_a_warmup": {
        "action": "a-phase-a", "gpu": 0,
        "log": OUTPUT / "tier2_c03_r8_oneshot_v4_a_warmup_g03001_03100.log",
        "telemetry": BRANCH_A / "telemetry" / "warmup_g03001_03100.jsonl",
        "global_start": 3001, "local_start": 1,
    },
    "branch_a_formal": {
        "action": "a-long", "gpu": 0,
        "log": OUTPUT / "tier2_c03_r8_oneshot_v4_a_formal_g03101_15000.log",
        "telemetry": BRANCH_A / "telemetry" / "formal_g03101_15000.jsonl",
        "global_start": 3101, "local_start": 101,
    },
    "branch_b_warmup": {
        "action": "b-phase-a", "gpu": 1,
        "log": OUTPUT / "tier2_c03_r8_oneshot_v4_b_warmup_g07001_07100.log",
        "telemetry": BRANCH_B / "telemetry" / "warmup_g07001_07100.jsonl",
        "global_start": 7001, "local_start": 1,
    },
    "branch_b_formal": {
        "action": "b-long", "gpu": 1,
        "log": OUTPUT / "tier2_c03_r8_oneshot_v4_b_formal_g07101_15000.log",
        "telemetry": BRANCH_B / "telemetry" / "formal_g07101_15000.jsonl",
        "global_start": 7101, "local_start": 101,
    },
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl_live(path: Path) -> tuple[list[dict], bool]:
    if not path.is_file():
        return [], False
    records = []
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    malformed_tail = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                malformed_tail = True
                break
            raise
    return records, malformed_tail


def validate_live_telemetry(path: Path, global_start: int, local_start: int | None) -> dict:
    records, malformed_tail = read_jsonl_live(path)
    globals_seen = [int(record["global_iteration"]) for record in records]
    if globals_seen != list(range(global_start, global_start + len(records))):
        raise RuntimeError(f"telemetry global discontinuity: {path}")
    locals_seen = None
    if local_start is not None:
        locals_seen = [int(record["reflection_local_iteration"]) for record in records]
        if locals_seen != list(range(local_start, local_start + len(records))):
            raise RuntimeError(f"telemetry R-local discontinuity: {path}")
    for record in records:
        if int(record["nonfinite_count"]) != 0 or not math.isfinite(float(record["total_loss"])):
            raise FloatingPointError(f"non-finite telemetry record at global {record['global_iteration']}")
        phase = str(record.get("phase_tag", ""))
        if f"experiment={EXPERIMENT}" not in phase or f"resolution={RESOLUTION}" not in phase:
            raise RuntimeError(f"telemetry experiment identity mismatch: {path}")
        if local_start is not None and f"retry={RETRY_IDENTITY}" not in phase:
            raise RuntimeError(f"telemetry retry identity mismatch: {path}")
    return {
        "rows": len(records),
        "last_global": globals_seen[-1] if globals_seen else None,
        "last_r_local": locals_seen[-1] if locals_seen else None,
        "malformed_trailing_live_line": malformed_tail,
    }


def branch_checkpoint_iterations(onset: int) -> list[int]:
    nodes = {onset + 100, onset + 200, onset + 500, onset + 1000, 15000}
    nodes.update(range(((onset + 1000) // 1000) * 1000, 15001, 1000))
    return sorted(value for value in nodes if value > onset)


def _run_checked(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=True,
    ).stdout


def _gpu_inventory() -> tuple[dict[int, dict], list[dict]]:
    gpu_output = _run_checked([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ])
    gpus = {}
    uuid_to_index = {}
    for line in gpu_output.splitlines():
        index, uuid, name, total, used = [value.strip() for value in line.split(",", 4)]
        record = {"uuid": uuid, "name": name, "memory_total_mib": int(total), "memory_used_mib": int(used)}
        gpus[int(index)] = record
        uuid_to_index[uuid] = int(index)
    apps_output = _run_checked([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    apps = []
    for line in apps_output.splitlines():
        if not line.strip():
            continue
        uuid, pid, name, used = [value.strip() for value in line.split(",", 3)]
        apps.append({"gpu": uuid_to_index.get(uuid), "pid": int(pid), "name": name, "used_memory_mib": int(used)})
    return gpus, apps


def preflight() -> dict:
    if os.environ.get("RTGS_TIER2_ACK_RESOLUTION8") != "YES":
        raise RuntimeError("RTGS_TIER2_ACK_RESOLUTION8=YES is required")
    dirty = _run_checked(["git", "status", "--porcelain"]).strip()
    if dirty:
        raise RuntimeError(f"working tree is not clean:\n{dirty}")
    reserved = [
        BRANCH_A, BRANCH_B, STATE_PATH, REPORT_PATH, COORDINATOR_LOG,
        PACKET_3000, PACKET_7000, PACKET_A100, PACKET_B100,
        *(spec["log"] for key, spec in TASK_SPECS.items() if key != "bootstrap"),
    ]
    conflicts = [str(path) for path in reserved if path.exists()]
    if conflicts:
        raise FileExistsError(f"one-shot outputs already exist; refusing overwrite: {conflicts}")
    if not MASK.is_file() or not PRIORS.is_dir():
        raise FileNotFoundError("formal mask or C03 normal priors are missing")
    if not V1_FINAL_AUDIT.is_file() or sha256_file(V1_FINAL_AUDIT) != EXPECTED_V1_FINAL_AUDIT_SHA256:
        raise RuntimeError("oneshot_v1 final audit is missing or changed")
    if not V2_FINAL_AUDIT.is_file() or sha256_file(V2_FINAL_AUDIT) != EXPECTED_V2_FINAL_AUDIT_SHA256:
        raise RuntimeError("oneshot_v2 final audit is missing or changed")
    if not V3_FINAL_AUDIT.is_file() or sha256_file(V3_FINAL_AUDIT) != EXPECTED_V3_FINAL_AUDIT_SHA256:
        raise RuntimeError("oneshot_v3 final audit is missing or changed")
    if not BOOTSTRAP.is_dir() or bool(BOOTSTRAP.stat().st_mode & 0o222):
        raise RuntimeError("protected oneshot_v1 shared bootstrap is missing or writable")
    for iteration, expected_hash in EXPECTED_SOURCE_HASHES.items():
        source = BOOTSTRAP / f"chkpnt{iteration}.pth"
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise RuntimeError(f"protected bootstrap checkpoint {iteration} is missing or changed")
        if bool(source.stat().st_mode & 0o222):
            raise RuntimeError(f"protected bootstrap checkpoint {iteration} is writable")
    gpus, apps = _gpu_inventory()
    if set(gpus) != {0, 1} or any("RTX 3090" not in gpus[index]["name"] for index in (0, 1)):
        raise RuntimeError(f"preflight requires exactly GPU 0/1 RTX 3090 devices: {gpus}")
    conflicting_apps = [
        app for app in apps
        if app["gpu"] in (0, 1) and re.search(r"python|train\.py|conda", app["name"], re.IGNORECASE)
    ]
    if conflicting_apps:
        raise RuntimeError(f"conflicting GPU training processes are present: {conflicting_apps}")
    source = SCRIPT.read_text(encoding="utf-8")
    required_literals = (
        'RESOLUTION=8', 'EXPERIMENT="C03-r8 Tier 2 onset study"',
        'RETRY_IDENTITY="oneshot_v4_memory_bounded_retry"',
        'ALLOCATOR_POLICY="adaptive_pressure_cache_and_ray_retry_v1"',
        'REFERENCE_PEAK_ALLOCATED_BYTES=0',
        'MINIMUM_PROJECTED_HEADROOM_BYTES=268435456',
        'PRESSURE_RELEASE_FREE_BYTES=2147483648',
        'MEMORY_RETRY_MIN_CHUNK_SIZE=512',
        "tier2_c03_r8_oneshot_v4_rstart", "--resolution \"$RESOLUTION\"",
        'PYTORCH_ALLOCATOR_CONFIG="max_split_size_mb:128,garbage_collection_threshold:0.8"',
    )
    if any(value not in source for value in required_literals):
        raise RuntimeError("operator script failed the C03-r8 static identity preflight")
    return {
        "git_commit": _run_checked(["git", "rev-parse", "HEAD"]).strip(),
        "git_clean": True,
        "experiment": EXPERIMENT,
        "retry_identity": RETRY_IDENTITY,
        "resolution": RESOLUTION,
        "gpus": gpus,
        "preexisting_compute_apps": apps,
        "legacy_r2_used": False,
        "shared_bootstrap_reused_read_only": True,
        "v1_final_audit_sha256": EXPECTED_V1_FINAL_AUDIT_SHA256,
        "v2_final_audit_sha256": EXPECTED_V2_FINAL_AUDIT_SHA256,
        "v3_final_audit_sha256": EXPECTED_V3_FINAL_AUDIT_SHA256,
    }


def _pgid(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None


def verify_gpu_isolation(active: dict[str, subprocess.Popen]) -> None:
    gpus, apps = _gpu_inventory()
    del gpus
    groups = {label: process.pid for label, process in active.items() if process.poll() is None}
    expected = {label: int(TASK_SPECS[label]["gpu"]) for label in groups}
    known_pids = set()
    for app in apps:
        app_group = _pgid(app["pid"])
        for label, group in groups.items():
            if app_group == group:
                known_pids.add(app["pid"])
                if app["gpu"] != expected[label]:
                    raise RuntimeError(
                        f"GPU isolation failure: {label} pid {app['pid']} is on GPU {app['gpu']}"
                    )
    foreign_training = [
        app for app in apps if app["pid"] not in known_pids and app["gpu"] in (0, 1)
        and re.search(r"python|train\.py|conda", app["name"], re.IGNORECASE)
    ]
    if foreign_training:
        raise RuntimeError(f"conflicting external GPU training process appeared: {foreign_training}")


def checkpoint_packet(
    checkpoint: Path,
    telemetry: Path,
    run_dir: Path,
    log: Path,
    global_start: int,
    global_end: int,
    packet_path: Path,
    local_start: int | None = None,
    local_end: int | None = None,
    allow_running: bool = False,
    require_ply: bool = False,
    expected_source_hash: str | None = None,
) -> dict:
    args = SimpleNamespace(
        checkpoint=str(checkpoint), telemetry=str(telemetry), run_dir=str(run_dir), log=str(log),
        expected_global_start=global_start, expected_global_end=global_end,
        expected_r_local_start=local_start, expected_r_local_end=local_end,
        allow_running_log=allow_running, require_ply=require_ply,
    )
    packet = build_gate_packet(args)
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    config = checkpoint_data.get("config", {})
    if config.get("experiment") != EXPERIMENT or int(config.get("resolution", -1)) != RESOLUTION:
        packet["missing_or_mismatched"].append("checkpoint experiment/resolution mismatch")
        packet["packet_ready_for_codex_review"] = False
    if checkpoint_data.get("format") == "rtgs_stage_b" and (
        config.get("tier2_retry_identity") != RETRY_IDENTITY
        or config.get("allocator_policy", {}).get("name") != ALLOCATOR_POLICY
    ):
        packet["missing_or_mismatched"].append("checkpoint retry/allocator policy mismatch")
        packet["packet_ready_for_codex_review"] = False
    if expected_source_hash is not None and checkpoint_data.get("provenance", {}).get("sha256") != expected_source_hash:
        packet["missing_or_mismatched"].append("checkpoint shared-source provenance mismatch")
        packet["packet_ready_for_codex_review"] = False
    atomic_json(packet_path, packet)
    if not packet["packet_ready_for_codex_review"]:
        raise RuntimeError(f"checkpoint gate audit failed: {packet_path}")
    return packet


def _log_anomaly(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines():
        if ANOMALY_RE.search(line):
            return line.strip()
    return None


class Coordinator:
    def __init__(self, preflight_data: dict):
        self.state = {
            "schema": "rtgs_tier2_oneshot_v4_state",
            "experiment": EXPERIMENT,
            "retry_identity": RETRY_IDENTITY,
            "resolution": RESOLUTION,
            "coordinator_pid": os.getpid(),
            "started_at": utc_now(),
            "ended_at": None,
            "status": "RUNNING",
            "preflight": preflight_data,
            "tasks": {},
            "events": [],
            "failures": [],
            "source_checkpoint_hashes": {},
            "final_audit": str(REPORT_PATH),
        }
        self.processes: dict[str, subprocess.Popen] = {}
        self.handled: set[str] = set()
        self.stop_requested = False
        self.last_gpu_poll = 0.0
        self._save()

    def _save(self) -> None:
        atomic_json(STATE_PATH, self.state)

    def event(self, message: str) -> None:
        record = {"time": utc_now(), "message": message}
        self.state["events"].append(record)
        self._save()
        line = f"[{record['time']}] {message}"
        print(line, flush=True)
        with COORDINATOR_LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def fail(self, label: str, reason: str) -> None:
        self.state["failures"].append({"time": utc_now(), "task": label, "reason": reason})
        self.event(f"HARD FAILURE [{label}]: {reason}")

    def start(self, label: str) -> None:
        spec = TASK_SPECS[label]
        if spec["action"] is None:
            raise ValueError(f"task {label} is a read-only source, not an executable action")
        log_handle = COORDINATOR_LOG.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [str(SCRIPT), spec["action"], "--execute"],
            cwd=ROOT,
            env=dict(os.environ),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        self.processes[label] = process
        self.state["tasks"][label] = {
            "action": spec["action"], "gpu": spec["gpu"], "launcher_pid": process.pid,
            "pid_kind": "operator process-group leader", "started_at": utc_now(), "ended_at": None,
            "exit_code": None, "status": "RUNNING", "log": str(spec["log"]),
            "telemetry": str(spec["telemetry"]), "last_checkpoint": None,
            "last_telemetry_global": None, "last_telemetry_r_local": None,
        }
        self.event(f"started {label} on GPU {spec['gpu']} as process group {process.pid}")

    def stop(self, label: str, reason: str) -> None:
        process = self.processes.get(label)
        if process is None or process.poll() is not None:
            return
        self.fail(label, reason)
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        except ProcessLookupError:
            pass

    def refresh_task(self, label: str) -> None:
        process = self.processes[label]
        task = self.state["tasks"][label]
        spec = TASK_SPECS[label]
        telemetry = {"last_global": None, "last_r_local": None}
        try:
            telemetry = validate_live_telemetry(
                spec["telemetry"], spec["global_start"], spec["local_start"]
            )
            task["last_telemetry_global"] = telemetry["last_global"]
            task["last_telemetry_r_local"] = telemetry["last_r_local"]
        except Exception as exc:
            self.stop(label, str(exc))
        run_dir = BRANCH_A if "branch_a" in label else BRANCH_B
        completed_global = telemetry["last_global"]
        checkpoints = sorted(
            (
                path for path in run_dir.glob("chkpnt*.pth")
                if completed_global is not None
                and int(path.stem.removeprefix("chkpnt")) <= completed_global
            ),
            key=lambda path: int(path.stem.removeprefix("chkpnt")),
        ) if run_dir.is_dir() else []
        task["last_checkpoint"] = str(checkpoints[-1]) if checkpoints else None
        anomaly = _log_anomaly(spec["log"])
        if anomaly and process.poll() is None:
            self.stop(label, f"training log anomaly: {anomaly}")
        return_code = process.poll()
        if return_code is not None and task["exit_code"] is None:
            task["exit_code"] = int(return_code)
            task["ended_at"] = utc_now()
            task["status"] = "COMPLETED" if return_code == 0 else "HARD_FAILED"
            if return_code != 0 and not any(item["task"] == label for item in self.state["failures"]):
                self.fail(label, f"training process exited with code {return_code}")
            else:
                self.event(f"{label} exited with code {return_code}")
        self._save()

    def completed_ok(self, label: str) -> bool:
        task = self.state["tasks"].get(label)
        return bool(task and task["exit_code"] == 0)

    def audit_sources_and_start_warmups(self) -> None:
        checkpoint_packet(
            BOOTSTRAP / "chkpnt3000.pth", TASK_SPECS["bootstrap"]["telemetry"], BOOTSTRAP,
            TASK_SPECS["bootstrap"]["log"], 1, 3000, PACKET_3000,
        )
        checkpoint_packet(
            BOOTSTRAP / "chkpnt7000.pth", TASK_SPECS["bootstrap"]["telemetry"], BOOTSTRAP,
            TASK_SPECS["bootstrap"]["log"], 1, 7000, PACKET_7000,
            require_ply=True,
        )
        hashes = {
            str(iteration): sha256_file(BOOTSTRAP / f"chkpnt{iteration}.pth")
            for iteration in range(1000, 7001, 1000)
        }
        self.state["source_checkpoint_hashes"]["bootstrap_1k_7k"] = hashes
        self.state["source_checkpoint_hashes"]["global_3000"] = hashes["3000"]
        self.state["source_checkpoint_hashes"]["global_7000"] = hashes["7000"]
        self.event("protected oneshot_v1 bootstrap 3000/7000 audited read-only for v2 reuse")
        self.start("branch_a_warmup")
        self.start("branch_b_warmup")

    def audit_warmup_and_start_formal(self, branch: str) -> None:
        if branch == "a":
            run_dir, global_start, global_end = BRANCH_A, 3001, 3100
            warmup, formal, packet = "branch_a_warmup", "branch_a_formal", PACKET_A100
        else:
            run_dir, global_start, global_end = BRANCH_B, 7001, 7100
            warmup, formal, packet = "branch_b_warmup", "branch_b_formal", PACKET_B100
        checkpoint_packet(
            run_dir / f"chkpnt{global_end}.pth", TASK_SPECS[warmup]["telemetry"], run_dir,
            TASK_SPECS[warmup]["log"], global_start, global_end, packet,
            local_start=1, local_end=100, require_ply=True,
            expected_source_hash=self.state["source_checkpoint_hashes"][f"global_{global_start - 1}"],
        )
        self.event(f"Branch {branch.upper()} warmup audited at R-local 100; starting formal-mask phase")
        self.start(formal)

    def run(self) -> int:
        def request_stop(signum, _frame):
            self.stop_requested = True
            self.event(f"coordinator received signal {signum}; stopping active tasks without cleanup")

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        try:
            self.audit_sources_and_start_warmups()
        except Exception as exc:
            self.fail("shared_bootstrap_source_audit", str(exc))
        while True:
            for label in list(self.processes):
                self.refresh_task(label)
            if self.stop_requested:
                for label in list(self.processes):
                    self.stop(label, "coordinator interruption")
                break
            if time.monotonic() - self.last_gpu_poll >= GPU_POLL_SECONDS:
                try:
                    verify_gpu_isolation(self.processes)
                except Exception as exc:
                    for label, process in list(self.processes.items()):
                        if process.poll() is None:
                            self.stop(label, str(exc))
                self.last_gpu_poll = time.monotonic()

            if self.completed_ok("branch_a_warmup") and "branch_a_warmup" not in self.handled:
                self.handled.add("branch_a_warmup")
                try:
                    self.audit_warmup_and_start_formal("a")
                except Exception as exc:
                    self.fail("branch_a_warmup_audit", str(exc))

            if self.completed_ok("branch_b_warmup") and "branch_b_warmup" not in self.handled:
                self.handled.add("branch_b_warmup")
                try:
                    self.audit_warmup_and_start_formal("b")
                except Exception as exc:
                    self.fail("branch_b_warmup_audit", str(exc))

            source_failed = any(
                item["task"] == "shared_bootstrap_source_audit"
                for item in self.state["failures"]
            )
            a_terminal = (
                self.state["tasks"].get("branch_a_formal", {}).get("exit_code") is not None
                or any(
                    item["task"].startswith("branch_a")
                    for item in self.state["failures"]
                )
                and not any(process.poll() is None for label, process in self.processes.items() if "branch_a" in label)
                or source_failed
            )
            b_terminal = (
                self.state["tasks"].get("branch_b_formal", {}).get("exit_code") is not None
                or (any(
                        item["task"].startswith("branch_b")
                        for item in self.state["failures"]
                    )
                    and not any(process.poll() is None for label, process in self.processes.items() if "branch_b" in label))
                or source_failed
            )
            if a_terminal and b_terminal:
                break
            time.sleep(POLL_SECONDS)

        for label in list(self.processes):
            self.refresh_task(label)
        self.state["ended_at"] = utc_now()
        report = safe_build_final_audit(self.state)
        fully_completed = bool(
            self.completed_ok("branch_a_formal")
            and self.completed_ok("branch_b_formal")
            and report["healthy"]
        )
        completed_major = sum(
            self.completed_ok(label) for label in ("branch_a_formal", "branch_b_formal")
        )
        self.state["status"] = (
            "FULLY_COMPLETED" if fully_completed
            else "HARD_FAILED" if self.state["failures"]
            else "PARTIALLY_COMPLETED" if completed_major else "HARD_FAILED"
        )
        self.state["final_audit_healthy"] = report["healthy"]
        self._save()
        self.event(f"one-shot terminal status: {self.state['status']}; report={REPORT_PATH}")
        report["operator_state"] = self.state
        atomic_json(REPORT_PATH, report)
        return 0 if fully_completed else 1


def _distribution(values: list[float | int | None]) -> dict | None:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not clean:
        return None
    def percentile(q: float) -> float:
        position = (len(clean) - 1) * q
        low = int(position)
        high = min(low + 1, len(clean) - 1)
        if low == high:
            return clean[low]
        return clean[low] * (high - position) + clean[high] * (position - low)
    return {"min": clean[0], "mean": sum(clean) / len(clean), "p50": percentile(0.5),
            "p95": percentile(0.95), "max": clean[-1]}


def scan_checkpoint(
    path: Path,
    expected_global: int,
    expected_local: int | None,
    expected_source_hash: str | None = None,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    stage_b = checkpoint.get("format") == "rtgs_stage_b"
    global_iteration = int(checkpoint["global_iteration"] if stage_b else checkpoint["iteration"])
    local_iteration = int(checkpoint["reflection_iteration"]) if stage_b else None
    d_count, r_count, topology = _checkpoint_counts(checkpoint)
    config = checkpoint.get("config", {})
    result = {
        "path": str(path), "sha256": sha256_file(path), "format": checkpoint.get("format"),
        "version": checkpoint.get("checkpoint_version"), "global": global_iteration,
        "r_local": local_iteration, "d_count": d_count, "r_count": r_count,
        "r_topology_version": topology, "optimizer_step_completed": checkpoint.get("optimizer_step_completed"),
        "has_rng_state": "rng_state" in checkpoint, "has_runtime_state": "runtime_state" in checkpoint,
        "experiment": config.get("experiment"), "resolution": config.get("resolution"),
        "source_path": checkpoint.get("provenance", {}).get("path"),
        "source_sha256": checkpoint.get("provenance", {}).get("sha256"),
        "finite_scan": _finite_scan(checkpoint),
    }
    errors = []
    if global_iteration != expected_global or local_iteration != expected_local:
        errors.append("checkpoint iteration mismatch")
    if config.get("experiment") != EXPERIMENT or int(config.get("resolution", -1)) != RESOLUTION:
        errors.append("checkpoint experiment/resolution mismatch")
    if stage_b and (
        config.get("tier2_retry_identity") != RETRY_IDENTITY
        or config.get("allocator_policy", {}).get("name") != ALLOCATOR_POLICY
    ):
        errors.append("checkpoint retry/allocator policy mismatch")
    if checkpoint.get("checkpoint_version") != 2 or checkpoint.get("optimizer_step_completed") is not True:
        errors.append("checkpoint is not complete version-2 continuation state")
    if result["finite_scan"]["nonfinite_count"]:
        errors.append("checkpoint contains non-finite tensors")
    if expected_source_hash is not None and result["source_sha256"] != expected_source_hash:
        errors.append("checkpoint shared-source provenance mismatch")
    result["errors"] = errors
    return result


def audit_telemetry(paths: list[Path], global_start: int, global_end: int, local_start: int | None) -> dict:
    records = []
    errors = []
    for path in paths:
        try:
            values, malformed = read_jsonl_live(path)
            if malformed:
                errors.append(f"malformed trailing line: {path}")
            records.extend(values)
        except Exception as exc:
            errors.append(f"cannot read telemetry {path}: {exc}")
    globals_seen = [int(record["global_iteration"]) for record in records]
    if globals_seen != list(range(global_start, global_end + 1)):
        errors.append("global telemetry is not complete and contiguous")
    locals_seen = None
    if local_start is not None:
        locals_seen = [int(record["reflection_local_iteration"]) for record in records]
        if locals_seen != list(range(local_start, local_start + len(records))):
            errors.append("R-local telemetry is not complete and contiguous")
    nonfinite = sum(int(record.get("nonfinite_count", 0)) for record in records)
    if nonfinite or any(not math.isfinite(float(record.get("total_loss", math.nan))) for record in records):
        errors.append("telemetry contains non-finite values")
    d_events = [
        {"global": int(record["global_iteration"]), "delta": int(record["d_count_delta"]),
         "count": int(record["d_count"])} for record in records if record.get("d_topology_event")
    ]
    r_events = [
        {"global": int(record["global_iteration"]), "local": int(record["reflection_local_iteration"]),
         "delta": int(record["r_count_delta"]), "count": int(record["r_count"]),
         "version": int(record["r_topology_version"])}
        for record in records if record.get("r_topology_event")
    ]
    allocator = {
        key: _distribution([record.get(key) for record in records])
        for key in (
            "cuda_memory_allocated_bytes", "cuda_memory_reserved_bytes",
            "cuda_max_memory_allocated_bytes", "cuda_max_memory_reserved_bytes",
        )
    }
    result = {
        "paths": [str(path) for path in paths], "rows": len(records),
        "global": [globals_seen[0], globals_seen[-1]] if globals_seen else None,
        "r_local": [locals_seen[0], locals_seen[-1]] if locals_seen else None,
        "loss": _distribution([record.get("total_loss") for record in records]),
        "l_spec": _distribution([record.get("l_spec") for record in records]),
        "wall_ms": _distribution([record.get("whole_step_wall_ms") for record in records]),
        "allocator": allocator, "d_topology_events": d_events, "r_topology_events": r_events,
        "nonfinite_count_sum": nonfinite, "errors": errors,
    }
    if local_start is not None and records:
        warmup = [record for record in records if int(record["reflection_local_iteration"]) <= 100]
        formal = [record for record in records if int(record["reflection_local_iteration"]) >= 101]
        if any(record.get("l_spec") is not None or record.get("mask_support_fraction") is not None for record in warmup):
            errors.append("warmup telemetry unexpectedly contains mask/L_spec")
        if any(record.get("l_spec") is None or record.get("mask_support_fraction") is None for record in formal):
            errors.append("formal telemetry is missing mask/L_spec")
    return result


def audit_debug(run_dir: Path, nodes: list[int], onset: int) -> dict:
    errors = []
    entries = []
    required = {
        "reflection_metadata.json", "final.png", "ks.png", "reflection_contribution.png",
        "reflection_contribution_vis.png", "ray_candidate_count.png", "ray_exact_intersection_count.png",
    }
    for global_iteration in nodes:
        directory = run_dir / "debug" / f"iteration_{global_iteration:06d}"
        local = global_iteration - onset
        node_required = set(required)
        if local >= 101:
            node_required |= {"transparent_mask.png", "overlay.png"}
        missing = sorted(name for name in node_required if not (directory / name).is_file())
        metadata = None
        metadata_path = directory / "reflection_metadata.json"
        if metadata_path.is_file():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            raw = payload["raw_stats"]
            metadata = {
                "candidate_p99": raw["ray_candidate_count"]["p99"],
                "exact_p99": raw["ray_exact_intersection_count"]["p99"],
                "raytrace_timing_ms": payload["raytrace"]["timing_ms"],
                "reflection_contribution_mean": raw["reflection_contribution"]["mean"],
                "reflection_contribution_p99": raw["reflection_contribution"]["p99"],
            }
        if missing:
            errors.append(f"debug global {global_iteration} missing {missing}")
        entries.append({"global": global_iteration, "r_local": local, "path": str(directory),
                        "missing": missing, "metadata": metadata})
    candidates = [entry["metadata"]["candidate_p99"] for entry in entries if entry["metadata"]]
    exact = [entry["metadata"]["exact_p99"] for entry in entries if entry["metadata"]]
    ray_wall = [
        entry["metadata"]["raytrace_timing_ms"].get("raytrace_wall")
        for entry in entries if entry["metadata"]
    ]
    return {
        "nodes": entries, "candidate_p99": _distribution(candidates),
        "exact_p99": _distribution(exact), "raytrace_wall_ms": _distribution(ray_wall),
        "errors": errors,
        "warmup_mask_note": "R-local 100 intentionally has no mask/overlay because Phase A forbids loading the formal mask",
    }


def audit_logs(paths: list[Path]) -> dict:
    entries = []
    errors = []
    memory_retries = []
    for path in paths:
        if not path.is_file():
            entries.append({"path": str(path), "exists": False})
            errors.append(f"missing log: {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.replace("\r", "\n").splitlines():
            marker = "STAGE_B_MEMORY_RETRY "
            if marker in line:
                try:
                    payload = line.split(marker, 1)[1].strip()
                    record, end = json.JSONDecoder().raw_decode(payload)
                    suffix = payload[end:].strip()
                    if suffix and re.fullmatch(
                        r"\[\d{2}/\d{2} \d{2}:\d{2}:\d{2}\]", suffix
                    ) is None:
                        raise json.JSONDecodeError(
                            "unexpected text after memory retry JSON", payload, end
                        )
                    memory_retries.append(record)
                except (json.JSONDecodeError, TypeError):
                    errors.append(f"malformed memory retry record: {path}")
        anomalies = [line.strip() for line in text.replace("\r", "\n").splitlines() if ANOMALY_RE.search(line)]
        complete = "Training complete." in text
        entries.append({"path": str(path), "exists": True, "training_complete": complete,
                        "anomaly_hits": anomalies[:20]})
        if not complete or anomalies:
            errors.append(f"unhealthy log: {path}")
    return {"entries": entries, "memory_retries": memory_retries, "errors": errors}


def audit_headroom(run_dir: Path) -> dict:
    path = run_dir / "allocator_headroom_gate.json"
    errors = []
    warnings = []
    record = None
    if not path.is_file():
        errors.append(f"missing allocator headroom gate: {path}")
    else:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("schema") != "rtgs_stage_b_headroom_gate_v1"
            or record.get("retry_identity") != RETRY_IDENTITY
            or record.get("allocator_policy") != ALLOCATOR_POLICY
            or record.get("reflection_local_iteration") != 101
            or record.get("reference_peak_allocated_bytes") != REFERENCE_PEAK_ALLOCATED_BYTES
            or record.get("minimum_projected_headroom_bytes") != MINIMUM_PROJECTED_HEADROOM_BYTES
            or not isinstance(record.get("passed"), bool)
        ):
            errors.append(f"invalid allocator headroom record: {path}")
        elif record.get("passed") is False:
            warnings.append(f"projected headroom is below advisory margin: {path}")
    return {"path": str(path), "record": record, "warnings": warnings, "errors": errors}


def build_final_audit(state: dict | None = None) -> dict:
    if state is None:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.is_file() else {}
    errors = []
    bootstrap_scans = {}
    for iteration in range(1000, 7001, 1000):
        path = BOOTSTRAP / f"chkpnt{iteration}.pth"
        if not path.is_file():
            errors.append(f"missing bootstrap checkpoint {iteration}")
            continue
        scan = scan_checkpoint(path, iteration, None)
        bootstrap_scans[str(iteration)] = scan
        errors.extend(f"bootstrap {iteration}: {value}" for value in scan["errors"])

    branches = {}
    for name, run_dir, onset in (("A", BRANCH_A, 3000), ("B", BRANCH_B, 7000)):
        scans = {}
        nodes = branch_checkpoint_iterations(onset)
        for iteration in nodes:
            path = run_dir / f"chkpnt{iteration}.pth"
            if not path.is_file():
                errors.append(f"Branch {name} missing checkpoint {iteration}")
                continue
            source_hash = state.get("source_checkpoint_hashes", {}).get(f"global_{onset}")
            scan = scan_checkpoint(path, iteration, iteration - onset, source_hash)
            scans[str(iteration)] = scan
            errors.extend(f"Branch {name} {iteration}: {value}" for value in scan["errors"])
        prefix = "branch_a" if name == "A" else "branch_b"
        telemetry = audit_telemetry(
            [TASK_SPECS[f"{prefix}_warmup"]["telemetry"], TASK_SPECS[f"{prefix}_formal"]["telemetry"]],
            onset + 1, 15000, 1,
        )
        debug = audit_debug(run_dir, nodes, onset)
        logs = audit_logs([TASK_SPECS[f"{prefix}_warmup"]["log"], TASK_SPECS[f"{prefix}_formal"]["log"]])
        headroom = audit_headroom(run_dir)
        errors.extend(
            f"Branch {name}: {value}"
            for value in telemetry["errors"] + debug["errors"] + logs["errors"] + headroom["errors"]
        )
        branches[name] = {
            "onset": onset, "output": str(run_dir), "checkpoints": scans,
            "final_checkpoint": scans.get("15000"), "telemetry": telemetry,
            "debug": debug, "logs": logs, "headroom_gate": headroom,
        }

    bootstrap_telemetry = audit_telemetry(
        [TASK_SPECS["bootstrap"]["telemetry"]], 1, 7000, None,
    )
    bootstrap_logs = audit_logs([TASK_SPECS["bootstrap"]["log"]])
    errors.extend(f"bootstrap: {value}" for value in bootstrap_telemetry["errors"] + bootstrap_logs["errors"])
    protected_hashes = state.get("source_checkpoint_hashes", {})
    for key, iteration in (("global_3000", 3000), ("global_7000", 7000)):
        path = BOOTSTRAP / f"chkpnt{iteration}.pth"
        expected = protected_hashes.get(key)
        if expected and path.is_file() and sha256_file(path) != expected:
            errors.append(f"shared bootstrap checkpoint {iteration} changed after protection")
        if path.is_file() and sha256_file(path) != EXPECTED_SOURCE_HASHES[iteration]:
            errors.append(f"shared bootstrap checkpoint {iteration} differs from approved v1 source")
    if not V1_FINAL_AUDIT.is_file() or sha256_file(V1_FINAL_AUDIT) != EXPECTED_V1_FINAL_AUDIT_SHA256:
        errors.append("oneshot_v1 final audit changed during v4")
    if not V2_FINAL_AUDIT.is_file() or sha256_file(V2_FINAL_AUDIT) != EXPECTED_V2_FINAL_AUDIT_SHA256:
        errors.append("oneshot_v2 final audit changed during v4")
    if not V3_FINAL_AUDIT.is_file() or sha256_file(V3_FINAL_AUDIT) != EXPECTED_V3_FINAL_AUDIT_SHA256:
        errors.append("oneshot_v3 final audit changed during v4")
    report = {
        "schema": "rtgs_tier2_oneshot_v4_final_audit", "generated_at": utc_now(),
        "experiment": EXPERIMENT, "retry_identity": RETRY_IDENTITY,
        "resolution": RESOLUTION, "allocator_policy": ALLOCATOR_POLICY, "cpu_only": True,
        "state_path": str(STATE_PATH), "operator_state": state,
        "v1_evidence": {
            "final_audit": str(V1_FINAL_AUDIT),
            "expected_sha256": EXPECTED_V1_FINAL_AUDIT_SHA256,
            "actual_sha256": sha256_file(V1_FINAL_AUDIT) if V1_FINAL_AUDIT.is_file() else None,
            "used_as_branch_resume": False,
        },
        "v2_evidence": {
            "final_audit": str(V2_FINAL_AUDIT),
            "expected_sha256": EXPECTED_V2_FINAL_AUDIT_SHA256,
            "actual_sha256": sha256_file(V2_FINAL_AUDIT) if V2_FINAL_AUDIT.is_file() else None,
            "used_as_branch_resume": False,
        },
        "v3_evidence": {
            "final_audit": str(V3_FINAL_AUDIT),
            "expected_sha256": EXPECTED_V3_FINAL_AUDIT_SHA256,
            "actual_sha256": sha256_file(V3_FINAL_AUDIT) if V3_FINAL_AUDIT.is_file() else None,
            "used_as_branch_resume": False,
        },
        "bootstrap": {
            "output": str(BOOTSTRAP), "checkpoints": bootstrap_scans,
            "telemetry": bootstrap_telemetry, "logs": bootstrap_logs,
            "reused_read_only": True,
            "read_only": BOOTSTRAP.is_dir() and not bool(BOOTSTRAP.stat().st_mode & 0o222),
        },
        "branches": branches,
        "errors": errors,
    }
    report["healthy"] = not errors
    return report


def safe_build_final_audit(state: dict | None = None) -> dict:
    """Always return a CPU-only terminal artifact, even if an audit reader fails."""
    try:
        return build_final_audit(state)
    except Exception as exc:
        return {
            "schema": "rtgs_tier2_oneshot_v4_final_audit",
            "generated_at": utc_now(),
            "experiment": EXPERIMENT,
            "retry_identity": RETRY_IDENTITY,
            "resolution": RESOLUTION,
            "allocator_policy": ALLOCATOR_POLICY,
            "cpu_only": True,
            "state_path": str(STATE_PATH),
            "operator_state": state or {},
            "healthy": False,
            "errors": [f"CPU-only final-audit reader failed closed: {type(exc).__name__}: {exc}"],
        }


def print_status() -> int:
    if not STATE_PATH.is_file():
        print(json.dumps({"state": "NOT_STARTED", "state_path": str(STATE_PATH)}, indent=2))
        return 0
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    summary = {
        "status": state.get("status"), "started_at": state.get("started_at"),
        "ended_at": state.get("ended_at"), "coordinator_pid": state.get("coordinator_pid"),
        "coordinator_alive": _pgid(int(state.get("coordinator_pid", -1))) is not None,
        "tasks": state.get("tasks", {}), "failures": state.get("failures", []),
        "state_path": str(STATE_PATH), "final_audit": str(REPORT_PATH),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run-all", "status", "final-audit"))
    args = parser.parse_args()
    if args.action == "status":
        return print_status()
    if args.action == "final-audit":
        report = safe_build_final_audit()
        atomic_json(REPORT_PATH, report)
        print(json.dumps({"healthy": report["healthy"], "report": str(REPORT_PATH),
                          "errors": report["errors"]}, indent=2, allow_nan=False))
        return 0 if report["healthy"] else 1
    data = preflight()
    coordinator = Coordinator(data)
    return coordinator.run()


if __name__ == "__main__":
    sys.exit(main())
