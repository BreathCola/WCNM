#!/usr/bin/env python3
"""Enumerate and visualize all signed-axis mappings without selecting one."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


AUDIT_FRAMES = (0, 14, 28, 42, 55, 69, 83, 97, 110)
PANEL_WH = (320, 180)


@dataclass(frozen=True)
class AxisCandidate:
    candidate_id: str
    permutation: tuple[int, int, int]
    signs: tuple[int, int, int]
    matrix: tuple[tuple[int, int, int], ...]
    determinant: int
    right_handed: bool
    expression: str


def generate_axis_candidates() -> list[AxisCandidate]:
    names = ("x", "y", "z")
    candidates: list[AxisCandidate] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.int8)
            for output_axis, (input_axis, sign) in enumerate(zip(permutation, signs)):
                matrix[output_axis, input_axis] = sign
            determinant = int(round(np.linalg.det(matrix)))
            terms = [
                ("+" if sign > 0 else "-") + names[input_axis]
                for input_axis, sign in zip(permutation, signs)
            ]
            candidates.append(
                AxisCandidate(
                    candidate_id=f"C{len(candidates):02d}",
                    permutation=permutation,
                    signs=signs,
                    matrix=tuple(tuple(int(value) for value in row) for row in matrix),
                    determinant=determinant,
                    right_handed=determinant > 0,
                    expression=f"[x', y', z'] = [{', '.join(terms)}]",
                )
            )
    return candidates


def apply_axis_candidate(normals: np.ndarray, candidate: AxisCandidate) -> np.ndarray:
    values = np.asarray(normals, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"normals must be HWC with 3 components, got {values.shape}")
    matrix = np.asarray(candidate.matrix, dtype=np.float32)
    return np.ascontiguousarray(values @ matrix.T, dtype=np.float32)


def score_candidate(
    candidate_views: Sequence[np.ndarray], stable_views: Sequence[np.ndarray]
) -> dict[str, float]:
    if len(candidate_views) != len(stable_views) or not candidate_views:
        raise ValueError("candidate and StableNormal view lists must have equal nonzero length")
    full: list[float] = []
    center: list[float] = []
    face_forward: list[float] = []
    continuity: list[float] = []
    for candidate, stable in zip(candidate_views, stable_views):
        if candidate.shape != stable.shape:
            raise ValueError("candidate and StableNormal shapes differ")
        cosine = np.sum(candidate * stable, axis=-1)
        full.append(float(cosine.mean()))
        height, width = cosine.shape
        roi = cosine[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
        center.append(float(roi.mean()))
        # In the local COLMAP camera convention, visible geometry usually has +z;
        # this remains a diagnostic only and is not included in rank_score.
        face_forward.append(float((candidate[..., 2] < 0.0).mean()))
        dx = np.sum(candidate[:, 1:] * candidate[:, :-1], axis=-1).mean()
        dy = np.sum(candidate[1:] * candidate[:-1], axis=-1).mean()
        continuity.append(float(0.5 * (dx + dy)))
    stable_cosine = float(np.mean(full))
    stable_center_cosine = float(np.mean(center))
    return {
        "rank_score": 0.7 * stable_cosine + 0.3 * stable_center_cosine,
        "stable_cosine": stable_cosine,
        "stable_center_cosine": stable_center_cosine,
        "negative_z_fraction": float(np.mean(face_forward)),
        "neighbor_continuity": float(np.mean(continuity)),
    }


def _resize(values: np.ndarray, width: int = PANEL_WH[0], height: int = PANEL_WH[1]) -> np.ndarray:
    return cv2.resize(values, (width, height), interpolation=cv2.INTER_AREA)


def _normal_rgb(values: np.ndarray) -> np.ndarray:
    return np.clip((values + 1.0) * 127.5, 0, 255).astype(np.uint8)


def _source_edge_agreement(source_rgb: np.ndarray, normals: np.ndarray) -> tuple[float, np.ndarray]:
    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    source_edge = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    normal_dx = np.linalg.norm(normals[:, 1:] - normals[:, :-1], axis=-1)
    normal_dy = np.linalg.norm(normals[1:] - normals[:-1], axis=-1)
    normal_edge = np.zeros(normals.shape[:2], dtype=np.float32)
    normal_edge[:, 1:] += normal_dx
    normal_edge[1:] += normal_dy
    source_edge /= max(float(source_edge.max()), 1e-6)
    normal_edge /= max(float(normal_edge.max()), 1e-6)
    score = float(np.sum(source_edge * normal_edge) / np.sqrt(
        max(float(np.sum(source_edge**2) * np.sum(normal_edge**2)), 1e-12)
    ))
    mask = normal_edge >= np.quantile(normal_edge, 0.9)
    overlay = source_rgb.copy()
    overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.array([255, 30, 30])).astype(np.uint8)
    return score, overlay


def _label_panel(rgb: np.ndarray, label: str) -> Image.Image:
    panel = Image.fromarray(rgb)
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, 20), fill=(0, 0, 0))
    draw.text((4, 4), label, fill=(255, 255, 255), font=ImageFont.load_default())
    return panel


def _write_grid(rows: Sequence[Sequence[tuple[np.ndarray, str]]], path: Path) -> None:
    canvas = Image.new("RGB", (PANEL_WH[0] * len(rows[0]), PANEL_WH[1] * len(rows)))
    for row_index, row in enumerate(rows):
        for column_index, (rgb, label) in enumerate(row):
            canvas.paste(_label_panel(rgb, label), (column_index * PANEL_WH[0], row_index * PANEL_WH[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB"), dtype=np.uint8)


def _load_view_data(scene: Path, candidates: Path, stable: Path, d_only: Path) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for index in AUDIT_FRAMES:
        stem = f"{index:06d}"
        source = _resize(_load_rgb(scene / "images" / f"{stem}.jpg"))
        raw = np.load(candidates / "raw_identity" / "normal" / f"{stem}.npy", allow_pickle=False)
        stable_normal = np.load(stable / f"{stem}.npy", allow_pickle=False)
        raw = _resize(raw).astype(np.float32)
        stable_normal = _resize(stable_normal).astype(np.float32)
        raw /= np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-6)
        stable_normal /= np.maximum(np.linalg.norm(stable_normal, axis=-1, keepdims=True), 1e-6)
        d_path = d_only / f"{index:05d}" / "normal.png"
        d_rgb = _resize(_load_rgb(d_path)) if d_path.is_file() else np.zeros_like(source)
        views.append({"index": index, "source": source, "raw": raw, "stable": stable_normal, "d": d_rgb, "d_exists": d_path.is_file()})
    return views


def run_audit(scene: Path, candidates: Path, stable: Path, d_only: Path, output: Path) -> dict[str, Any]:
    scene, candidates, stable, d_only, output = map(lambda path: Path(path).resolve(), (scene, candidates, stable, d_only, output))
    views = _load_view_data(scene, candidates, stable, d_only)
    catalog = generate_axis_candidates()
    ranked: list[dict[str, Any]] = []
    sheets = output / "axis_candidates_contact_sheets"
    representative_rows: list[list[tuple[np.ndarray, str]]] = []
    for view in views:
        _, overlay = _source_edge_agreement(view["source"], view["raw"])
        representative_rows.append([
            (view["source"], f'{view["index"]:06d} source'),
            (_normal_rgb(view["raw"]), "raw identity (axis unconfirmed)"),
            (overlay, "raw normal edge overlay"),
            (_normal_rgb(view["stable"]), "StableNormal reference"),
            (view["d"], "D-only world normal reference"),
        ])
    _write_grid(representative_rows, output / "representative_9_source_normal.png")

    invariant_edges = []
    for view in views:
        edge_score, _ = _source_edge_agreement(view["source"], view["raw"])
        invariant_edges.append(edge_score)
    for candidate in catalog:
        transformed = [apply_axis_candidate(view["raw"], candidate) for view in views]
        metrics = score_candidate(transformed, [view["stable"] for view in views])
        metrics["source_normal_edge_agreement"] = float(np.mean(invariant_edges))
        entry = {
            "candidate_id": candidate.candidate_id,
            "expression": candidate.expression,
            "permutation": list(candidate.permutation),
            "signs": list(candidate.signs),
            "matrix": [list(row) for row in candidate.matrix],
            "determinant": candidate.determinant,
            "right_handed": candidate.right_handed,
            "metrics": metrics,
        }
        ranked.append(entry)
        rows = []
        for view, transformed_normal in zip(views, transformed):
            _, overlay = _source_edge_agreement(view["source"], transformed_normal)
            rows.append([
                (view["source"], f'{view["index"]:06d} source'),
                (_normal_rgb(transformed_normal), f'{candidate.candidate_id} {candidate.expression}'),
                (overlay, "source + candidate edges"),
                (_normal_rgb(view["stable"]), "StableNormal reference"),
                (view["d"], "D-only world normal reference"),
            ])
        _write_grid(rows, sheets / f"{candidate.candidate_id}.png")

    ranked.sort(key=lambda item: item["metrics"]["rank_score"], reverse=True)
    for rank, item in enumerate(ranked, 1):
        item["automatic_rank"] = rank
        item["diagnostic_sheet"] = str((sheets / f'{item["candidate_id"]}.png').resolve())
    report = {
        "selection_status": "awaiting_human_review",
        "selected_candidate": None,
        "warning": "Automatic ranking is diagnostic only and must not select the final axis mapping.",
        "rank_basis": "0.7 * mean cosine to StableNormal + 0.3 * center-region cosine to StableNormal",
        "notes": [
            "Source-edge agreement and neighbor continuity are invariant under orthogonal signed-axis mappings.",
            "D-only normals are world-space encoded PNGs, so they are visual references and are not used in numeric ranking.",
            "negative_z_fraction is diagnostic only and is not part of rank_score.",
        ],
        "audit_frames": list(AUDIT_FRAMES),
        "candidate_count": len(ranked),
        "right_handed_count": sum(item["right_handed"] for item in ranked),
        "candidates_by_rank": ranked,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "axis_rank_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    catalog_root = candidates / "axis_audit"
    catalog_root.mkdir(parents=True, exist_ok=True)
    (catalog_root / "candidate_catalog.json").write_text(json.dumps({"selected_candidate": None, "candidates": [item for item in sorted(ranked, key=lambda x: x["candidate_id"])]}, indent=2) + "\n", encoding="utf-8")
    top = ranked[:5]
    lines = ["# DiffusionRenderer normal axis audit", "", "Status: awaiting human review; no axis mapping has been selected.", "", "Automatic top five (diagnostic only):", ""]
    lines.extend(f"- {item['candidate_id']}: `{item['expression']}`, score={item['metrics']['rank_score']:.6f}, right_handed={item['right_handed']}" for item in top)
    lines.extend(["", "Review all candidate sheets before choosing; the automatic score uses StableNormal only as a reference, not ground truth.", ""])
    (output / "axis_audit.md").write_text("\n".join(lines), encoding="utf-8")
    contract = """# RT-GS normal contract audit

- Loader: `utils/camera_utils.py` accepts HWC or CHW `.npy`, converts to float32, bilinearly resizes if needed, rejects non-finite/near-zero vectors through its validity mask, normalizes, and stores CHW.
- Loss path: `train.py` converts the prior back to HWC; camera-space priors pass through `camera_normals_to_world`, then face-forwarding, before cosine `L_mono`.
- Renderer: `gaussian_renderer/surfel_renderer.py` exposes normalized, face-forwarded HWC world-space normals; its debug PNG is therefore visual context, not directly comparable component-wise to a camera-space prior.
- StableNormal manifest identifies the frozen baseline as HWC float32 camera-space unit normals.
- This DR-1C-A output is HWC float32 and unit-normalized, but remains `raw_identity` with an unconfirmed axis convention. No final RT-GS camera-space mapping is selected.
"""
    (output / "normal_contract_report.md").write_text(contract, encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--stable", type=Path, required=True)
    parser.add_argument("--d-only", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_audit(args.scene, args.candidates, args.stable, args.d_only, args.output)
    print(json.dumps({"candidate_count": report["candidate_count"], "right_handed_count": report["right_handed_count"], "selection_status": report["selection_status"]}, indent=2))
    print("DIFFRENDER_NORMAL_AXIS_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
