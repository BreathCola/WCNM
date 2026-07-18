#!/usr/bin/env python3
"""Plan the Tao global-0 layered early-joint run; training requires --execute."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tao_early_joint_plan import OUTPUT_NAME, build_tao_early_joint_plan


FORBIDDEN_CHECKPOINT_OPTIONS = (
    "--start_checkpoint", "--start-checkpoint", "--checkpoint-source",
    "--resume", "--resume-from", "--source-checkpoint",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "data/Tao")
    parser.add_argument(
        "--reviewed-mask-manifest", type=Path,
        default=ROOT / "data/Tao/specular_masks_reviewed_v1/manifest.json",
    )
    parser.add_argument(
        "--geometry-release", type=Path,
        default=ROOT / "output/stage_c_tao_geometry_release_v1/release_manifest.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output" / OUTPUT_NAME,
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="dispatch the audited plan to train.py; absent means read-only plan output",
    )
    return parser


def _reject_checkpoint_options(argv: list[str]) -> None:
    for token in argv:
        option = token.split("=", 1)[0]
        if option in FORBIDDEN_CHECKPOINT_OPTIONS or "checkpoint_source" in option:
            raise SystemExit(f"CHECKPOINT_SOURCE_FORBIDDEN: {option}")


def _launch_training(plan: dict) -> None:
    """Dispatch only from the explicit execute branch.

    The current task does not call this function.  ``train.py`` is responsible
    for consuming the versioned plan and creating the fresh output atomically.
    """
    output = Path(str(plan["output"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".plan.json", dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2, sort_keys=True)
            handle.write("\n")
        subprocess.run(
            [sys.executable, str(ROOT / "train.py"), "--tao_layered_plan", str(temporary)],
            cwd=ROOT, check=True,
        )
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    actual = list(sys.argv[1:] if argv is None else argv)
    _reject_checkpoint_options(actual)
    args = build_parser().parse_args(actual)
    plan = build_tao_early_joint_plan(
        scene=args.scene, reviewed_mask_manifest=args.reviewed_mask_manifest,
        geometry_release=args.geometry_release, output=args.output,
        execute=args.execute,
    )
    print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False))
    if args.execute:
        _launch_training(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
