#!/usr/bin/env python3
"""Generate a scene-agnostic, review-only DR glass-mask proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.glass_mask_proposal import generate_glass_mask_proposal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--erode-pixels", type=int, default=5)
    parser.add_argument(
        "--resume-build", type=Path,
        help="resume a hidden proposal build after a review-only candidate failure",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = generate_glass_mask_proposal(
        scene=args.scene, images=args.images, raw_root=args.raw_root,
        output=args.output, erode_pixels=args.erode_pixels,
        resume_build=args.resume_build,
    )
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "count": manifest["count"],
        "human_status": manifest["human_status"],
        "training_eligible": manifest["training_eligible"],
        "verdict": "TAO_GLASS_MASK_REVIEW_REQUIRED",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
