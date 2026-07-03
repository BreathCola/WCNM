#!/usr/bin/env python3
"""CPU-only audit and review summary for the Stage D formal T-onset run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_release import sha256_file, validate_geometry_release
from stage_d_training import (
    FORMAL_NODES, FORMAL_RELEASE_ID, FORMAL_RELEASE_SHA256,
    FORMAL_SOURCE_SHA256, FORMAL_STEMS,
)

REQUIRED_DEBUG = {
    "ground_truth.png", "final.png", "diffuse_contribution.png",
    "reflection_contribution.png", "transmittance_contribution.png",
    "inside_color.png", "inside_alpha.png", "inside_depth.png",
    "outside_color.png", "outside_alpha.png", "outside_depth.png",
    "transmittance_color.png", "transmittance_alpha.png",
    "depth_violation.png", "din_vs_far_violation.png", "near_depth.png",
    "far_depth.png", "two_hit_valid.png", "ks.png", "transparent_mask.png",
    "transmittance_metadata.json",
}


def scan_finite(value, path="checkpoint"):
    tensors = elements = 0
    if torch.is_tensor(value):
        tensors, elements = 1, value.numel()
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite tensor at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            a, b = scan_finite(child, f"{path}.{key}")
            tensors += a; elements += b
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            a, b = scan_finite(child, f"{path}[{index}]")
            tensors += a; elements += b
    return tensors, elements


def finite_scalars(value):
    if isinstance(value, dict):
        return all(finite_scalars(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_scalars(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def plot_curves(rows, node_metrics, target):
    def sheet(series_groups, path, width=1400, height=900):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        cols, rows_count = 2, (len(series_groups) + 1) // 2
        panel_w, panel_h = width // cols, height // rows_count
        colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                  "#17becf", "#8c564b", "#e377c2", "#7f7f7f")
        for index, (title, lines) in enumerate(series_groups):
            ox, oy = (index % cols) * panel_w, (index // cols) * panel_h
            left, top, right, bottom = ox + 65, oy + 35, ox + panel_w - 25, oy + panel_h - 45
            draw.rectangle((left, top, right, bottom), outline="#888888")
            draw.text((left, oy + 8), title, fill="black")
            all_points = [point for _, points in lines for point in points]
            xs = [point[0] for point in all_points]; ys = [point[1] for point in all_points]
            xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
            if xmax == xmin: xmax += 1
            if ymax == ymin: ymax += 1
            for line_index, (label, points) in enumerate(lines):
                mapped = [
                    (left + (x - xmin) / (xmax - xmin) * (right - left),
                     bottom - (y - ymin) / (ymax - ymin) * (bottom - top))
                    for x, y in points
                ]
                if len(mapped) > 1:
                    draw.line(mapped, fill=colors[line_index % len(colors)], width=2)
                elif mapped:
                    x, y = mapped[0]; draw.ellipse((x-2, y-2, x+2, y+2), fill=colors[line_index % len(colors)])
                draw.text((left + 6 + 120 * (line_index % 5), bottom + 10 + 15 * (line_index // 5)),
                          label, fill=colors[line_index % len(colors)])
            draw.text((left, bottom + 28), f"x {xmin:g}..{xmax:g}", fill="#555555")
            draw.text((right - 150, bottom + 28), f"y {ymin:.4g}..{ymax:.4g}", fill="#555555")
        canvas.save(path)

    x = [row["global_iteration"] for row in rows]
    groups = (
        ("training losses", [("total", list(zip(x, [r["loss"]["total"] for r in rows]))),
                              ("RGB", list(zip(x, [r["loss"]["rgb"] for r in rows])))]),
        ("constraint diagnostics", [("L_spec", list(zip(x, [r["loss"]["l_spec"] for r in rows]))),
                                     ("L_depth measured", list(zip(x, [r["loss"]["l_depth"] for r in rows])))]),
        ("sampled Din <= t_far", [("fraction", list(zip(x, [r["din_le_t_far_fraction"] for r in rows])))]),
        ("field counts", [(b, list(zip(x, [r["counts"][b] for r in rows]))) for b in ("diffuse", "reflection", "transmittance")]),
        ("step wall time", [("seconds", list(zip(x, [r["whole_step_wall_ms"] / 1000 for r in rows])))]),
        ("scoped CUDA peaks GiB", [("allocated", list(zip(x, [r["cuda_max_memory_allocated_bytes"] / 2**30 for r in rows]))),
                                    ("reserved", list(zip(x, [r["cuda_max_memory_reserved_bytes"] / 2**30 for r in rows])))])
    )
    sheet(groups, target)

    node_target = target.with_name("node_semantic_curves.png")
    nx = sorted(node_metrics)
    metrics = (
        ("din_le_t_far_fraction", "Din <= t_far"),
        ("inside_alpha_mean", "T inside alpha mean"),
        ("inside_energy", "Cin energy"),
        ("outside_energy", "Cout energy"),
    )
    node_groups = []
    for key, title in metrics:
        node_groups.append((title, [
            (stem, list(zip(nx, [node_metrics[n][stem][key] for n in nx])))
            for stem in FORMAL_STEMS
        ]))
    sheet(node_groups, node_target, height=800)
    return node_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    errors, warnings = [], []
    release = validate_geometry_release(args.geometry_manifest)
    if release["geometry_release_id"] != FORMAL_RELEASE_ID:
        errors.append("geometry release ID mismatch")
    if release["aggregate_sha256"] != FORMAL_RELEASE_SHA256:
        errors.append("geometry release aggregate mismatch")

    operator = json.loads((output / "formal_operator_record.json").read_text(encoding="utf-8"))
    if operator.get("schema") != "rtgs_stage_d_formal_operator_v2":
        errors.append("formal operator schema mismatch")
    if operator.get("source_sha256_before") != FORMAL_SOURCE_SHA256:
        errors.append("operator source preflight hash mismatch")
    if operator.get("source_sha256_after") != FORMAL_SOURCE_SHA256:
        errors.append("source checkpoint mutated during run")
    if operator.get("release_aggregate_before") != FORMAL_RELEASE_SHA256:
        errors.append("operator release preflight mismatch")
    if operator.get("release_aggregate_after") != FORMAL_RELEASE_SHA256:
        errors.append("release/cache mutation detected")

    checkpoints, final_checkpoint = {}, None
    for node in FORMAL_NODES:
        path = output / f"chkpnt{node}.pth"
        if not path.is_file():
            errors.append(f"missing checkpoint {node}"); continue
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("format") != "rtgs_stage_d":
            errors.append(f"checkpoint {node} format mismatch"); continue
        expected = (node, node - 3000, node - 15000)
        actual = (
            checkpoint.get("global_iteration"), checkpoint.get("reflection_iteration"),
            checkpoint.get("transmittance_iteration"),
        )
        if actual != expected:
            errors.append(f"checkpoint {node} iteration tuple {actual} != {expected}")
        if checkpoint.get("optimizer_step_completed") is not True:
            errors.append(f"checkpoint {node} lacks completed endpoint update")
        if checkpoint.get("geometry_release_aggregate_sha256") != FORMAL_RELEASE_SHA256:
            errors.append(f"checkpoint {node} release mismatch")
        config = checkpoint.get("config", {})
        if config.get("source_stage_b_checkpoint_sha256") != FORMAL_SOURCE_SHA256:
            errors.append(f"checkpoint {node} source mismatch")
        if config.get("stage_d_depth_start_iteration") != 40000 or config.get("lambda_depth") != 0.2:
            errors.append(f"checkpoint {node} L_depth schedule mismatch")
        policy = config.get("formal_memory_policy", {})
        if config.get("ray_chunk_size") != 2048 or policy.get("attempt_chunk_sizes") != [2048, 1024, 512]:
            errors.append(f"checkpoint {node} formal memory policy mismatch")
        try:
            tensors, elements = scan_finite(checkpoint)
        except Exception as exc:
            errors.append(str(exc)); tensors = elements = 0
        states = {name: checkpoint.get(name, {}) for name in ("diffuse", "reflection", "transmittance")}
        if any(state.get("optimizer") is None for state in states.values()):
            errors.append(f"checkpoint {node} has incomplete optimizer state")
        try:
            storage = [states[name]["xyz"].untyped_storage().data_ptr() for name in states]
            if len(set(storage)) != 3:
                errors.append(f"checkpoint {node} D/R/T storage is not independent")
        except Exception as exc:
            errors.append(f"checkpoint {node} branch state failure: {exc}")
        counts = {name: int(states[name]["xyz"].shape[0]) for name in states if "xyz" in states[name]}
        for branch, count in counts.items():
            ply = output / "point_cloud" / branch / f"iteration_{node}" / "point_cloud.ply"
            if not ply.is_file() or len(PlyData.read(ply)["vertex"].data) != count:
                errors.append(f"{node} {branch} PLY missing/count mismatch")
        checkpoints[node] = {
            "path": str(path), "sha256": sha256_file(path), "counts": counts,
            "finite_tensor_count": tensors, "finite_element_count": elements,
        }
        if node == FORMAL_NODES[-1]:
            final_checkpoint = checkpoint

    telemetry_path = output / "stage_d_telemetry.jsonl"
    rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    expected_steps = list(range(15001, 20001))
    if [row.get("global_iteration") for row in rows] != expected_steps:
        errors.append("telemetry is not exactly continuous from 15001 through 20000")
    for row in rows:
        if row.get("schema") != "rtgs_stage_d_formal_telemetry_v2":
            errors.append("telemetry schema mismatch"); break
        if row.get("nonfinite_count") != 0 or not finite_scalars(row):
            errors.append(f"non-finite telemetry at {row.get('global_iteration')}"); break
        if row["loss"].get("lambda_depth_enabled") is not False:
            errors.append(f"L_depth enabled early at {row.get('global_iteration')}"); break
        if row.get("geometry_release_aggregate_sha256") != FORMAL_RELEASE_SHA256:
            errors.append(f"telemetry release mismatch at {row.get('global_iteration')}"); break
        memory = row.get("ray_memory_policy", {})
        if memory.get("used_chunk_size") not in (2048, 1024, 512):
            errors.append(f"invalid ray chunk at {row.get('global_iteration')}"); break
        if memory.get("checkpoint_chunks") is not True:
            errors.append(f"non-checkpointed formal ray path at {row.get('global_iteration')}"); break

    node_metrics = {}
    for node in FORMAL_NODES:
        root = output / "debug" / f"iteration_{node:06d}"
        if not (root / "contact_sheet.png").is_file():
            errors.append(f"missing contact sheet at {node}")
        else:
            Image.open(root / "contact_sheet.png").verify()
        node_metrics[node] = {}
        for stem in FORMAL_STEMS:
            view = root / stem
            missing = sorted(name for name in REQUIRED_DEBUG if not (view / name).is_file())
            if missing:
                errors.append(f"debug {node}/{stem} missing {missing}"); continue
            metadata = json.loads((view / "transmittance_metadata.json").read_text(encoding="utf-8"))
            if not finite_scalars(metadata):
                errors.append(f"debug metadata {node}/{stem} is non-finite")
            node_metrics[node][stem] = {
                "din_le_t_far_fraction": metadata["din_le_t_far_fraction"],
                "inside_alpha_mean": metadata["raw_stats"]["inside_alpha"]["mean"],
                "inside_energy": metadata["energy"]["inside_color_valid_mean"],
                "outside_energy": metadata["energy"]["outside_color_valid_mean"],
                "full_rgb_l1": metadata["rgb_l1"]["full_image"],
                "transparent_rgb_l1": metadata["rgb_l1"]["transparent_hard"],
                "valid_two_hit_fraction": metadata["valid_two_hit_fraction"],
            }

    t_health = {}
    if final_checkpoint is not None:
        state = final_checkpoint["transmittance"]
        alpha = torch.sigmoid(state["opacity_raw"].float())
        color = torch.sigmoid(state["color_raw"].float())
        t_health = {
            "count": int(alpha.shape[0]), "alpha_min": float(alpha.min()),
            "alpha_mean": float(alpha.mean()), "alpha_max": float(alpha.max()),
            "alpha_std": float(alpha.std()), "color_mean": float(color.mean()),
            "color_std": float(color.std()),
        }
        if not (t_health["count"] > 0 and 1e-8 < t_health["alpha_mean"] < 1 - 1e-8
                and t_health["color_std"] > 0):
            errors.append("T field is zero, saturated, or collapsed")

    evolution = {}
    if all(len(node_metrics.get(node, {})) == len(FORMAL_STEMS) for node in FORMAL_NODES):
        for key in (
            "din_le_t_far_fraction", "inside_alpha_mean", "inside_energy",
            "outside_energy", "full_rgb_l1",
        ):
            values = [
                sum(node_metrics[node][stem][key] for stem in FORMAL_STEMS) / len(FORMAL_STEMS)
                for node in FORMAL_NODES
            ]
            evolution[key] = {"node_means": values, "range": max(values) - min(values)}
            if evolution[key]["range"] <= 1e-12:
                errors.append(f"no visible cross-node evolution for {key}")

    curves = output / "key_curves.png"
    if rows and all(len(node_metrics.get(node, {})) == len(FORMAL_STEMS) for node in FORMAL_NODES):
        node_curve = plot_curves(rows, node_metrics, curves)
    else:
        node_curve = output / "node_semantic_curves.png"
        errors.append("insufficient complete data for key curves")

    verdict = "BLOCKED" if errors else "HOLD_FOR_SEMANTIC_REVIEW"
    memory_summary = {
        "oom_retry_count": sum(
            row.get("ray_memory_policy", {}).get("oom_retry_count", 0) for row in rows
        ),
        "fallback_step_count": sum(
            row.get("ray_memory_policy", {}).get("used_chunk_size") != 2048 for row in rows
        ),
        "cache_release_step_count": sum(bool(row.get("allocator_cache_released")) for row in rows),
        "max_peak_allocated_bytes": max(
            (row.get("cuda_max_memory_allocated_bytes", 0) for row in rows), default=0
        ),
        "max_peak_reserved_bytes": max(
            (row.get("cuda_max_memory_reserved_bytes", 0) for row in rows), default=0
        ),
    }
    result = {
        "verdict": verdict, "technical_run_complete": not errors,
        "semantic_separation_claimed": False,
        "semantic_note": (
            "The audit establishes a finite, non-collapsed T-onset trajectory only; "
            "loss reduction is not evidence that bird and background are separated."
        ),
        "errors": errors, "warnings": warnings, "checkpoints": checkpoints,
        "final_checkpoint_sha256": checkpoints.get(20000, {}).get("sha256"),
        "telemetry_rows": len(rows), "telemetry_range": [15001, 20000],
        "release": release, "t_health": t_health, "node_metrics": node_metrics,
        "cross_node_evolution": evolution,
        "formal_memory_policy": memory_summary,
        "contact_sheet": str(output / "debug" / "iteration_020000" / "contact_sheet.png"),
        "key_curves": str(curves), "node_semantic_curves": str(node_curve),
    }
    audit_path = output / "stage_d_formal_cpu_audit.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    report = output / "formal_t_onset_report.md"
    report.write_text(
        "# Stage D formal T-onset report\n\n"
        f"Verdict: `{verdict}`\n\n"
        f"Updates: global 15001–20000; R-local 12001–17000; T-local 1–5000.\n\n"
        "`L_depth` remained disabled for this trajectory; its recorded future activation is global 40000 with `lambda_depth=0.2`.\n\n"
        f"Final checkpoint SHA-256: `{result['final_checkpoint_sha256']}`.\n\n"
        f"CPU audit: `{audit_path}`. Final nine-view sheet: `{result['contact_sheet']}`. Curves: `{curves}` and `{node_curve}`.\n\n"
        "This verdict does not infer bird/background separation from falling loss. Review Cin, Cout, final brightness, and Din-vs-far across all five nodes and nine fixed views before any continuation.\n"
        + ("\nErrors:\n" + "".join(f"- {error}\n" for error in errors) if errors else ""),
        encoding="utf-8",
    )
    print(verdict)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
