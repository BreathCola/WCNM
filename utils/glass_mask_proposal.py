"""Scene-agnostic DiffusionRenderer glass-mask proposal packaging.

Every artifact produced here is explicitly review-only.  The formal mask
loader does not accept this schema or directory layout.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from utils.dr_mask_proposal import (
    ProposalAuditError,
    audit_dr_artifacts,
    generate_view_proposal,
    sha256_file,
)


PROPOSAL_SCHEMA = "rtgs_glass_mask_proposal_v1"
GENERIC_RISKS = (
    "subject_texture_may_be_included",
    "background_may_be_included",
    "glass_edge_may_be_missing",
    "reflection_may_be_misclassified",
    "low_confidence_region",
)
RISK_WEIGHTS = {
    "background_may_be_included": 5.0,
    "glass_edge_may_be_missing": 5.0,
    "low_confidence_region": 4.0,
    "reflection_may_be_misclassified": 2.0,
    "subject_texture_may_be_included": 1.0,
}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="PNG")
        with Image.open(temporary) as checked:
            checked.load()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def tree_digest(root: Path, exclude: Iterable[str] = ()) -> dict[str, Any]:
    root = Path(root)
    excluded = set(exclude)
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    )
    digest = hashlib.sha256()
    records = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        value = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": value})
    return {"count": len(records), "aggregate_sha256": digest.hexdigest(), "files": records}


def _generic_risks(raw: dict[str, Any]) -> dict[str, Any]:
    converted = dict(raw)
    legacy_subject = converted.pop("bird_may_be_included", None)
    if legacy_subject is not None and "subject_texture_may_be_included" not in converted:
        converted["subject_texture_may_be_included"] = legacy_subject
    if set(converted) != set(GENERIC_RISKS):
        raise ProposalAuditError(f"proposal emitted unexpected risk roles: {sorted(converted)}")
    return {name: converted[name] for name in GENERIC_RISKS}


def _risk_score(metrics: dict[str, Any]) -> float:
    return float(sum(
        weight for name, weight in RISK_WEIGHTS.items()
        if metrics["risks"][name]["flag"]
    ) + 10.0 * metrics["uncertainty_fraction_gt_0_55"])


def _review_page(source: Path, soft: Path, hard: Path, eroded: Path, overlay: Path,
                 metrics: dict[str, Any], output: Path) -> None:
    thumb = (520, 455)
    panels = []
    for label, path, mode in (
        ("source RGB", source, "RGB"), ("mask_soft (proposal)", soft, "L"),
        ("mask_hard (proposal)", hard, "L"), ("mask_eroded (proposal)", eroded, "L"),
        ("overlay", overlay, "RGB"),
    ):
        with Image.open(path) as opened:
            image = opened.convert(mode)
            if mode == "L":
                image = ImageOps.colorize(image, black=(0, 0, 0), white=(0, 220, 255))
            image = ImageOps.contain(image.convert("RGB"), thumb)
        panel = Image.new("RGB", (thumb[0], thumb[1] + 28), "white")
        panel.paste(image, ((thumb[0] - image.width) // 2, 28))
        ImageDraw.Draw(panel).text((8, 7), label, fill="black")
        panels.append(panel)
    page = Image.new("RGB", (thumb[0] * 3, (thumb[1] + 28) * 2 + 95), "white")
    for index, panel in enumerate(panels):
        page.paste(panel, ((index % 3) * thumb[0], (index // 3) * (thumb[1] + 28)))
    draw = ImageDraw.Draw(page)
    y = (thumb[1] + 28) * 2 + 8
    draw.text((10, y), f"{metrics['stem']}  REVIEW REQUIRED  risk={metrics['risk_score']:.4f}", fill="black")
    draw.text((10, y + 24), f"area={metrics['hard_area_ratio']:.6f} eroded={metrics['eroded_area_ratio']:.6f} uncertainty={metrics['uncertainty_fraction_gt_0_55']:.6f}", fill="black")
    flags = [name for name in GENERIC_RISKS if metrics["risks"][name]["flag"]]
    draw.text((10, y + 48), "flags=" + (", ".join(flags) if flags else "none"), fill="black")
    _save_png(output, page)


def _contact_pages(rows: Sequence[dict[str, Any]], output_root: Path, prefix: str,
                   per_page: int = 12) -> list[str]:
    paths: list[str] = []
    cell = (430, 410)
    for page_index, start in enumerate(range(0, len(rows), per_page), start=1):
        batch = rows[start:start + per_page]
        sheet = Image.new("RGB", (cell[0] * 4, cell[1] * 3), "white")
        for slot, row in enumerate(batch):
            with Image.open(output_root / "overlays" / f"{row['stem']}.png") as opened:
                image = ImageOps.contain(opened.convert("RGB"), (cell[0] - 12, cell[1] - 48))
            x, y = (slot % 4) * cell[0], (slot // 4) * cell[1]
            sheet.paste(image, (x + (cell[0] - image.width) // 2, y + 34))
            draw = ImageDraw.Draw(sheet)
            draw.text((x + 8, y + 7), f"{row['stem']} area={row['hard_area_ratio']:.3f} risk={row['risk_score']:.2f}", fill="black")
        relative = Path("contact_sheets") / f"{prefix}_{page_index:02d}.png"
        _save_png(output_root / relative, sheet)
        paths.append(relative.as_posix())
    return paths


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "stem", "risk_score", "hard_area_ratio", "eroded_area_ratio",
        "uncertainty_fraction_gt_0_55", *GENERIC_RISKS, "review_status", "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({
                "rank": rank, "stem": row["stem"], "risk_score": row["risk_score"],
                "hard_area_ratio": row["hard_area_ratio"],
                "eroded_area_ratio": row["eroded_area_ratio"],
                "uncertainty_fraction_gt_0_55": row["uncertainty_fraction_gt_0_55"],
                **{name: int(row["risks"][name]["flag"]) for name in GENERIC_RISKS},
                "review_status": "", "notes": "",
            })


def generate_glass_mask_proposal(
    *, scene: Path, raw_root: Path, output: Path, images: str = "images",
    erode_pixels: int = 5,
    resume_build: Path | None = None,
) -> dict[str, Any]:
    scene = Path(scene).expanduser().resolve()
    raw_root = Path(raw_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing glass proposal output: {output}")
    if erode_pixels < 0:
        raise ValueError("erode_pixels must be nonnegative")
    audit = audit_dr_artifacts(scene, raw_root, expected_count=None, images=images)
    build = (
        Path(resume_build).expanduser().resolve()
        if resume_build is not None else output.parent / f".{output.name}.building-{os.getpid()}"
    )
    if build.exists() and resume_build is None:
        raise FileExistsError(f"refusing stale glass proposal build: {build}")
    if resume_build is not None and not build.is_dir():
        raise FileNotFoundError(f"resume proposal build is missing: {build}")
    build.mkdir(parents=True, exist_ok=resume_build is not None)
    rows: list[dict[str, Any]] = []
    try:
        for entry in audit["entries"]:
            stem = entry["stem"]
            view_root = build / "views" / stem
            soft_path = view_root / "proposal_soft.png"
            hard_path = view_root / "proposal_hard_preview.png"
            overlay_path = view_root / "proposal_overlay.png"
            resumable = resume_build is not None and all(path.is_file() for path in (
                view_root / "proposal_metadata.json", soft_path, hard_path, overlay_path,
                build / "mask_soft" / f"{stem}.png",
                build / "mask_hard" / f"{stem}.png",
                build / "mask_eroded" / f"{stem}.png",
                build / "overlays" / f"{stem}.png",
                build / "review_pages" / f"{stem}.png",
            ))
            if resumable:
                metadata = json.loads(
                    (view_root / "proposal_metadata.json").read_text(encoding="utf-8")
                )
            else:
                metadata = generate_view_proposal(
                    entry, view_root, allow_review_fallback=True,
                    legacy_risk_names=False,
                )
            with Image.open(soft_path) as opened:
                soft = opened.convert("L")
                soft_values = np.asarray(soft, dtype=np.uint8)
            with Image.open(hard_path) as opened:
                hard_values = np.asarray(opened.convert("L"), dtype=np.uint8) >= 128
            if resumable:
                with Image.open(build / "mask_eroded" / f"{stem}.png") as opened:
                    eroded_values = np.asarray(opened.convert("L"), dtype=np.uint8) >= 128
            else:
                if erode_pixels:
                    diameter = 2 * erode_pixels + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
                    eroded_values = cv2.erode(hard_values.astype(np.uint8), kernel) > 0
                else:
                    eroded_values = hard_values.copy()
            if not hard_values.any() or not eroded_values.any():
                raise ProposalAuditError(f"{stem}: hard/eroded proposal is empty")
            if not resumable:
                _save_png(build / "mask_soft" / f"{stem}.png", soft)
                _save_png(build / "mask_hard" / f"{stem}.png", Image.fromarray(hard_values.astype(np.uint8) * 255))
                _save_png(build / "mask_eroded" / f"{stem}.png", Image.fromarray(eroded_values.astype(np.uint8) * 255))
                (build / "overlays").mkdir(parents=True, exist_ok=True)
                shutil.copy2(overlay_path, build / "overlays" / f"{stem}.png")
            risks = _generic_risks(metadata["risks"])
            row = {
                "stem": stem,
                "source_rgb_sha256": entry["rgb"]["sha256"],
                "dr_input_sha256": metadata["input_sha256"],
                "mask_soft_sha256": sha256_file(build / "mask_soft" / f"{stem}.png"),
                "mask_hard_sha256": sha256_file(build / "mask_hard" / f"{stem}.png"),
                "mask_eroded_sha256": sha256_file(build / "mask_eroded" / f"{stem}.png"),
                "overlay_sha256": sha256_file(build / "overlays" / f"{stem}.png"),
                "source_size": metadata["source_size"],
                "hard_area_ratio": float(hard_values.mean()),
                "soft_area_ratio": float(soft_values.mean() / 255.0),
                "eroded_area_ratio": float(eroded_values.mean()),
                "bbox_xyxy": metadata["bbox_xyxy"],
                "uncertainty_fraction_gt_0_55": metadata["uncertainty_fraction_gt_0_55"],
                "risks": risks,
            }
            row["risk_score"] = _risk_score(row)
            rows.append(row)
            if not resumable:
                _review_page(
                    Path(entry["rgb"]["path"]), build / "mask_soft" / f"{stem}.png",
                    build / "mask_hard" / f"{stem}.png", build / "mask_eroded" / f"{stem}.png",
                    build / "overlays" / f"{stem}.png", row,
                    build / "review_pages" / f"{stem}.png",
                )
        chronological = _contact_pages(rows, build, "chronological")
        risk_ranked = sorted(rows, key=lambda row: (-row["risk_score"], row["stem"]))
        risk_pages = _contact_pages(risk_ranked, build, "risk_ranked")
        _write_csv(build / "review_queue.csv", risk_ranked)
        _write_csv(build / "per_frame_metrics.csv", rows)
        _atomic_json(build / "per_frame_metrics.json", rows)
        _atomic_json(build / "dr_input_audit.json", audit)
        _atomic_json(build / "review_template.json", {
            "schema": "rtgs_glass_mask_human_review_template_v1",
            "accepted": [], "rejected": [], "manual_edit_required": [],
            "note": "All 112 stems require explicit human disposition before promotion.",
        })
        tree_before_manifest = tree_digest(build)
        manifest = {
            "schema": PROPOSAL_SCHEMA,
            "artifact_role": "glass_mask_proposal_for_human_review",
            "human_status": "proposal_requires_review",
            "training_eligible": False,
            "promotion_performed": False,
            "scene": scene.name,
            "scene_path": str(scene),
            "images": images,
            "raw_diffusion_renderer_root": str(raw_root),
            "raw_manifest_sha256": audit["raw_manifest_sha256"],
            "generation_validation_sha256": audit["generation_validation_sha256"],
            "ordered_stems": [row["stem"] for row in rows],
            "count": len(rows),
            "mask_roles": ["mask_soft", "mask_hard", "mask_eroded"],
            "mask_semantics": "complete projected glass enclosure; visible contents are not holes",
            "erode_pixels": erode_pixels,
            "chronological_contact_sheets": chronological,
            "risk_ranked_contact_sheets": risk_pages,
            "per_frame_metrics": rows,
            "tree_before_manifest": tree_before_manifest,
            "verdict": "GLASS_MASK_PROPOSAL_READY_FOR_HUMAN_REVIEW",
        }
        _atomic_json(build / "proposal_manifest.json", manifest)
        os.replace(build, output)
    except Exception:
        raise
    return manifest
