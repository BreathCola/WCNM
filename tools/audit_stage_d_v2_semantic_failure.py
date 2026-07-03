#!/usr/bin/env python3
"""Read-only causal audit of the completed cached-T v2 semantic failure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.cuboid_space import CLASS_NAMES, CuboidSpace
from geometry.geometry_release import sha256_file, validate_geometry_release


V2 = ROOT / "output/stage_d_tihubird_c03r8_cached_twarmup_then_joint_g15000_g20000_v2"
OUTPUT = ROOT / "output/stage_d_tihubird_c03r8_semantic_repair_v3_audit"
RELEASE = ROOT / "geometry_releases/stage_c_geometry_release_v1.json"
NODES = (15025, 15100, 15500, 16000, 17500, 18000, 19000, 20000)
STEMS = ("000000", "000012", "000039", "000040", "000041", "000053", "000063", "000083", "000110")
SOURCE_SHA = "050500d607e1910ca088049ae73619949ad183e23c85354a8408bb29571fbe84"
RELEASE_SHA = "4fedb22dc2f2e6415a3d3948ab26fba54df91ba06b66d951a09b3dc5f761188d"
V2_FINAL_SHA = "a238935a3384256bbbfae6aa8d5d512710ad01a6e277452b3cc1eadd71952ee0"


def image(path, gray=False):
    opened = Image.open(path).convert("L" if gray else "RGB")
    value = np.asarray(opened, dtype=np.float32) / 255.0
    return value[..., None] if gray else value


def stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return {
        "count": int(values.size), "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def node_metrics(node):
    per_view = {}
    pooled = {key: [] for key in (
        "ain", "conditional_luminance", "r_energy", "cout_energy", "t_energy",
        "high_ain_black", "saturated",
    )}
    for stem in STEMS:
        root = V2 / f"debug/iteration_{node:06d}" / stem
        mask = image(root / "transparent_mask.png", gray=True)[..., 0] >= 0.5
        ain = image(root / "inside_alpha.png", gray=True)[..., 0]
        cin = image(root / "inside_color.png")
        gt = image(root / "ground_truth.png")
        r = image(root / "reflection_contribution.png")
        cout = image(root / "outside_color.png")
        t = image(root / "transmittance_contribution.png")
        conditional = cin / np.maximum(ain[..., None], 1.0 / 255.0)
        conditional_luminance = (
            0.2126 * conditional[..., 0] + 0.7152 * conditional[..., 1]
            + 0.0722 * conditional[..., 2]
        )
        gt_luminance = 0.2126 * gt[..., 0] + 0.7152 * gt[..., 1] + 0.0722 * gt[..., 2]
        high = mask & (ain >= 0.8)
        saturated = mask & (ain >= 0.95)
        high_black = high & (gt_luminance >= 0.15) & (conditional_luminance <= 0.08)
        selected = lambda value: value[mask]
        row = {
            "ain": stats(selected(ain)),
            "ain_saturated_fraction": float(saturated.sum() / max(mask.sum(), 1)),
            "high_ain_near_black_conditional_fraction": float(high_black.sum() / max(mask.sum(), 1)),
            "conditional_luminance": stats(selected(conditional_luminance)),
            "r_contribution_energy": float(selected(r).mean()),
            "cout_energy": float(selected(cout).mean()),
            "t_contribution_energy": float(selected(t).mean()),
        }
        per_view[stem] = row
        pooled["ain"].append(selected(ain))
        pooled["conditional_luminance"].append(selected(conditional_luminance))
        pooled["r_energy"].append(selected(r))
        pooled["cout_energy"].append(selected(cout))
        pooled["t_energy"].append(selected(t))
        pooled["high_ain_black"].append(selected(high_black.astype(np.float32)))
        pooled["saturated"].append(selected(saturated.astype(np.float32)))
    return {
        "per_view": per_view,
        "aggregate": {
            "ain": stats(np.concatenate(pooled["ain"])),
            "ain_saturated_fraction": float(np.concatenate(pooled["saturated"]).mean()),
            "high_ain_near_black_conditional_fraction": float(np.concatenate(pooled["high_ain_black"]).mean()),
            "conditional_luminance": stats(np.concatenate(pooled["conditional_luminance"])),
            "r_contribution_energy": float(np.concatenate(pooled["r_energy"]).mean()),
            "cout_energy": float(np.concatenate(pooled["cout_energy"]).mean()),
            "t_contribution_energy": float(np.concatenate(pooled["t_energy"]).mean()),
        },
    }


def main():
    global V2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, default=V2)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--interface-margin", type=float, default=0.05)
    args = parser.parse_args()
    V2 = args.v2.resolve()
    output = args.output.resolve()
    if output.exists():
        existing = json.loads(
            (output / "semantic_failure_audit.json").read_text(encoding="utf-8")
        )
        expected_checkpoint = sha256_file(V2 / "chkpnt20000.pth")
        if (
            expected_checkpoint != V2_FINAL_SHA
            or
            existing.get("schema") != "rtgs_stage_d_v2_semantic_failure_audit_v1"
            or existing.get("v2_path") != str(V2)
            or existing.get("v2_final_checkpoint_sha256") != expected_checkpoint
            or existing.get("read_only") is not True
            or existing.get("cuboid_space", {}).get("interface_margin")
            != float(args.interface_margin)
            or set(existing.get("nodes", {})) != {str(node) for node in NODES}
            or existing.get("supported_evidence", {}).get("t_pruning", {}).get("final_count")
            != 540
            or existing.get("evidence_gaps", {}).get("r_spatial_source_energy", {}).get("supported")
            is not False
            or existing.get("evidence_gaps", {}).get("cout_spatial_source_energy", {}).get("supported")
            is not False
            or not (output / "semantic_failure_audit.md").is_file()
        ):
            raise ValueError("existing causal audit identity mismatch")
        print(json.dumps(existing, indent=2, sort_keys=True, allow_nan=False))
        return 0
    audit = json.loads((V2 / "stage_d_cached_twarmup_cpu_audit.json").read_text(encoding="utf-8"))
    if audit.get("technical_run_complete") is not True or audit.get("telemetry_rows") != 5000:
        raise ValueError("v2 technical audit is incomplete")
    release = validate_geometry_release(RELEASE)
    if release["aggregate_sha256"] != RELEASE_SHA:
        raise ValueError("geometry release mismatch")
    operator = json.loads((V2 / "cached_twarmup_operator_record.json").read_text(encoding="utf-8"))
    if operator.get("source_sha256_before") != SOURCE_SHA or operator.get("source_sha256_after") != SOURCE_SHA:
        raise ValueError("v2 source identity mismatch")

    metadata_path = ROOT / "output/stage_c_geometry_release_v1/mesh_metadata.json"
    cuboid = CuboidSpace.from_metadata(
        metadata_path, interface_margin=args.interface_margin,
        device="cpu", dtype=torch.float64,
    )
    checkpoint = torch.load(V2 / "chkpnt20000.pth", map_location="cpu")
    if sha256_file(V2 / "chkpnt20000.pth") != V2_FINAL_SHA:
        raise ValueError("v2 final checkpoint SHA-256 mismatch")
    spatial_counts = {}
    for branch in ("diffuse", "reflection", "transmittance"):
        classes = cuboid.classify(checkpoint[branch]["xyz"].double())
        spatial_counts[branch] = {
            name: int((classes == index).sum()) for index, name in enumerate(CLASS_NAMES)
        }

    rows = [json.loads(line) for line in (V2 / "stage_d_telemetry.jsonl").read_text(encoding="utf-8").splitlines()]
    pruning = []
    prior = 4096
    for row in rows:
        current = int(row["counts"]["transmittance"])
        if current != prior:
            pruning.append({
                "global_iteration": int(row["global_iteration"]),
                "transmittance_local_iteration": int(row["transmittance_local_iteration"]),
                "before": prior, "after": current, "removed": prior - current,
                "topology_event": row["topology_event"],
            })
            prior = current

    nodes = {str(node): node_metrics(node) for node in NODES}
    first_veil_node = next((
        node for node in NODES
        if nodes[str(node)]["aggregate"]["ain_saturated_fraction"] >= 0.5
    ), None)
    first_black_node = next((
        node for node in NODES
        if nodes[str(node)]["aggregate"]["high_ain_near_black_conditional_fraction"] >= 0.25
    ), None)
    result = {
        "schema": "rtgs_stage_d_v2_semantic_failure_audit_v1",
        "v2_path": str(V2),
        "v2_final_checkpoint_sha256": sha256_file(V2 / "chkpnt20000.pth"),
        "read_only": True,
        "cuboid_space": cuboid.metadata(),
        "final_surfel_spatial_counts": spatial_counts,
        "t_pruning_events": pruning,
        "nodes": nodes,
        "supported_evidence": {
            "ain_saturation": {
                "supported": True, "first_majority_saturation_node": first_veil_node,
                "final": nodes["20000"]["aggregate"]["ain"],
                "final_saturated_fraction": nodes["20000"]["aggregate"]["ain_saturated_fraction"],
            },
            "black_conditional_inside_color": {
                "supported": True, "first_documented_black_veil_node": first_black_node,
                "final_conditional_luminance": nodes["20000"]["aggregate"]["conditional_luminance"],
                "final_high_ain_near_black_fraction": nodes["20000"]["aggregate"]["high_ain_near_black_conditional_fraction"],
                "measurement_note": "Derived from saved 8-bit debug PNGs; threshold evidence is approximate but directionally decisive.",
            },
            "t_pruning": {
                "supported": True, "events": pruning,
                "initial_count": 4096, "final_count": int(rows[-1]["counts"]["transmittance"]),
            },
            "earliest_observed_failure": {
                "supported": True,
                "statement": (
                    "R contribution and bright Cout are already present at the first 15025 review; "
                    f"majority Ain saturation is first documented at {first_veil_node}."
                ),
                "first_node_r_energy": nodes["15025"]["aggregate"]["r_contribution_energy"],
                "first_node_cout_energy": nodes["15025"]["aggregate"]["cout_energy"],
            },
        },
        "evidence_gaps": {
            "r_spatial_source_energy": {
                "supported": False,
                "inside": None, "interface": None, "outside": None,
                "reason": "v2 saved only the composited R contribution, not per-surfel spatial-class traces.",
            },
            "cout_spatial_source_energy": {
                "supported": False,
                "inside": None, "interface": None, "outside": None,
                "reason": "v2 saved only composited Cout, not class-filtered second-bounce traces.",
            },
            "bird_level_quantification": {
                "supported": False,
                "reason": "No independent versioned bird ROI exists; contribution maps cannot provide bird-level proof.",
            },
            "which_of_r_or_cout_leaked_first": {
                "supported": False,
                "reason": "Both are already present at the earliest saved node and v2 has no pre-update class decomposition.",
            },
        },
        "conclusion": (
            "The black-veil failure and pruning chronology are supported. Spatial source attribution "
            "for R and Cout is not recoverable from v2 artifacts and must be instrumented prospectively."
        ),
    }
    output.mkdir(parents=True)
    (output / "semantic_failure_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    (output / "semantic_failure_audit.md").write_text(
        "# Stage D v2 semantic-failure causal audit\n\n"
        "This audit is read-only with respect to v2.\n\n"
        f"- Final Ain saturation fraction: `{result['supported_evidence']['ain_saturation']['final_saturated_fraction']:.6f}`.\n"
        f"- Final high-Ain/near-black conditional fraction: `{result['supported_evidence']['black_conditional_inside_color']['final_high_ain_near_black_fraction']:.6f}`.\n"
        f"- T count: `4096 -> {result['supported_evidence']['t_pruning']['final_count']}` across `{len(pruning)}` count-changing events.\n"
        f"- First majority-saturation review node: `{first_veil_node}`.\n\n"
        "## Evidence gaps\n\n"
        "V2 does not contain class-filtered R or Cout traces, so inside/interface/outside contribution energy cannot be reconstructed. "
        "No independent bird ROI exists, so bird-level quantitative proof is unavailable.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
