#!/usr/bin/env python3
"""Generate and strictly package 111 frozen DR glass proposals for review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.dr_mask_proposal import ProposalAuditError, _atomic_json, audit_dr_artifacts
from utils.dr_mask_review import build_review_package, generate_all_real_proposals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "run_state.json"
    try:
        if any(path for path in output.iterdir() if path != state_path):
            raise ProposalAuditError(f"111-view output must be new or empty: {output}")
        _atomic_json(
            state_path,
            {
                "artifact": "111-view automatic proposal generation state",
                "status": "IN_PROGRESS",
                "training_role": None,
            },
        )
        audit = audit_dr_artifacts(args.scene, args.raw_root, expected_count=111)
        _atomic_json(output / "audit_manifest.json", audit)

        def report(index: int, stem: str) -> None:
            print(f"PROPOSAL_PROGRESS={index}/111 stem={stem}", flush=True)

        generate_all_real_proposals(audit, output, progress=report)
        manifest = build_review_package(audit, output)
        _atomic_json(
            state_path,
            {
                "artifact": "111-view automatic proposal generation state",
                "status": "PASS",
                "validated_real_frames": manifest["completeness"]["validated_real_frames"],
                "training_role": None,
                "reviewed_soft_created": False,
                "formal_training_manifest_created": False,
            },
        )
    except (OSError, ValueError, ProposalAuditError) as error:
        _atomic_json(
            state_path,
            {
                "artifact": "111-view automatic proposal generation state",
                "status": "FAIL",
                "error": str(error),
                "training_role": None,
            },
        )
        print(f"DR_MASK_REVIEW_111=FAIL\n{error}")
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "real_frames": manifest["completeness"]["validated_real_frames"],
                "padding_proposals": manifest["padding_exclusion_proof"]["padding_proposal_count"],
                "review_queue": manifest["review_queue_count"],
                "output": str(output),
            },
            indent=2,
        )
    )
    print("DR_MASK_REVIEW_111=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
