#!/usr/bin/env python3
"""Generate full 111-view TiHuBird glass-mask v2 review proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.dr_mask_full_review_v2 import (  # noqa: E402
    DEFAULT_SOURCE_SIZE,
    ProposalAuditError,
    generate_full_review_v2_proposal,
)
from utils.dr_mask_proposal import _atomic_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--reviewed-v1-manifest", type=Path, required=True)
    parser.add_argument("--proposal-v1-root", type=Path, required=True)
    parser.add_argument("--high-risk-root", type=Path, required=True)
    parser.add_argument("--repair-v1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "run_state.json"
    try:
        if any(path for path in output.iterdir() if path != state_path):
            raise ProposalAuditError(f"v2 proposal output must be new or empty: {output}")

        def report(index: int, stem: str) -> None:
            print(f"GLASS_MASK_V2_PROGRESS={index}/111 stem={stem}", flush=True)

        manifest = generate_full_review_v2_proposal(
            scene=args.scene,
            raw_root=args.raw_root,
            reviewed_v1_manifest=args.reviewed_v1_manifest,
            proposal_v1_root=args.proposal_v1_root,
            high_risk_root=args.high_risk_root,
            repair_v1_root=args.repair_v1_root,
            output_root=output,
            expected_count=111,
            source_size=DEFAULT_SOURCE_SIZE,
            require_formal_loader=True,
            progress=report,
        )
    except (OSError, ValueError, ProposalAuditError) as error:
        _atomic_json(
            state_path,
            {
                "artifact": "full glass-mask v2 proposal state",
                "status": "FAIL",
                "error": str(error),
                "training_role": None,
            },
        )
        print(f"GLASS_MASK_V2_PROPOSAL=FAIL\n{error}")
        return 1
    summary = manifest["outputs"]
    print(
        json.dumps(
            {
                "status": "PASS",
                "verdict": summary["verdict"],
                "candidate_png_count": summary["candidate_png_count"],
                "review_page_count": summary["review_page_count"],
                "highest_risk_stem": summary["highest_risk"]["stem"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
