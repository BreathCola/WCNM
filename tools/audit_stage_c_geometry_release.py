#!/usr/bin/env python3
"""Strict CPU-only audit for a frozen Stage C geometry release."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_release import validate_geometry_release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = validate_geometry_release(args.manifest)
    if args.json_output:
        output = args.json_output.resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite release audit: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, output)
    print(result["verdict"])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
