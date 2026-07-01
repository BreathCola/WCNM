#!/usr/bin/env python3
"""Generate localized repair candidates for failed proposals 000039--000041."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.dr_mask_proposal import ProposalAuditError, _atomic_json
from utils.dr_mask_repair import generate_repair_candidates


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
            raise ProposalAuditError(f"repair output must be new or empty: {output}")
        _atomic_json(
            state_path,
            {
                "artifact": "three-frame repair-candidate state",
                "status": "IN_PROGRESS",
                "training_role": None,
            },
        )
        report = generate_repair_candidates(args.proposal_root, output)
        _atomic_json(
            state_path,
            {
                "artifact": "three-frame repair-candidate state",
                "status": "PASS",
                "repair_stems": report["repair_stems"],
                "source_proposal_files_modified": False,
                "reviewed_soft_created": False,
                "training_role": None,
            },
        )
    except (OSError, ValueError, ProposalAuditError) as error:
        _atomic_json(
            state_path,
            {
                "artifact": "three-frame repair-candidate state",
                "status": "FAIL",
                "error": str(error),
                "training_role": None,
            },
        )
        print(f"DR_MASK_REPAIR=FAIL\n{error}")
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "repair_stems": report["repair_stems"],
                "source_proposal_files_modified": False,
                "comparison": report["three_frame_comparison"]["path"],
                "output": str(output),
            },
            indent=2,
        )
    )
    print("DR_MASK_REPAIR=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
