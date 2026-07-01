#!/usr/bin/env python3
"""Build high-resolution, read-only review packs for fixed high-risk proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.dr_mask_high_risk_review import generate_high_risk_review_pack
from utils.dr_mask_proposal import ProposalAuditError, _atomic_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "run_state.json"
    try:
        if any(path for path in output.iterdir() if path != state_path):
            raise ProposalAuditError(f"high-risk review output must be new or empty: {output}")
        _atomic_json(
            state_path,
            {
                "artifact": "high-risk proposal review-pack state",
                "status": "IN_PROGRESS",
                "training_role": None,
            },
        )
        manifest = generate_high_risk_review_pack(args.proposal_root, output)
        _atomic_json(
            state_path,
            {
                "artifact": "high-risk proposal review-pack state",
                "status": "PASS",
                "view_count": manifest["view_count"],
                "proposal_files_modified": False,
                "reviewed_soft_created": False,
                "training_role": None,
            },
        )
    except (OSError, ValueError, ProposalAuditError) as error:
        _atomic_json(
            state_path,
            {
                "artifact": "high-risk proposal review-pack state",
                "status": "FAIL",
                "error": str(error),
                "training_role": None,
            },
        )
        print(f"DR_MASK_HIGH_RISK_REVIEW=FAIL\n{error}")
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "views": manifest["view_count"],
                "proposal_files_modified": manifest["proposal_files_modified"],
                "contact_sheet": manifest["contact_sheet"]["path"],
                "output": str(output),
            },
            indent=2,
        )
    )
    print("DR_MASK_HIGH_RISK_REVIEW=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
