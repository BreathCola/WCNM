#!/usr/bin/env python3
"""Audit DR artifacts and generate nine automatic glass-mask review proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.dr_mask_proposal import (
    ProposalAuditError,
    _atomic_json,
    audit_dr_artifacts,
    generate_proposals,
)


DEFAULT_STEMS = (
    "000000", "000014", "000028", "000042", "000055",
    "000069", "000083", "000097", "000110",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=111)
    parser.add_argument("--stems", nargs="+", default=list(DEFAULT_STEMS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    try:
        if output.exists() and any(output.iterdir()):
            raise ProposalAuditError(f"proposal output must be new or empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        audit = audit_dr_artifacts(args.scene, args.raw_root, args.expected_count)
        _atomic_json(output / "audit_manifest.json", audit)
        proposals = generate_proposals(audit, output, args.stems)
    except (OSError, ValueError, ProposalAuditError) as error:
        print(f"DR_MASK_PROPOSAL=FAIL\n{error}")
        return 1
    print(
        json.dumps(
            {
                "audit": audit["status"],
                "real_frames": audit["real_frame_count"],
                "padding_excluded": audit["padding_frame_count"],
                "proposals": len(proposals),
                "output": str(output),
            },
            indent=2,
        )
    )
    print("DR_MASK_PROPOSAL=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
