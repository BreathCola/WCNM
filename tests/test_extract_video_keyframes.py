import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import tools.extract_video_keyframes as keyframes
from tools.extract_video_keyframes import (
    Candidate,
    KeyframeError,
    VideoInfo,
    build_manifest,
    build_selection_report,
    select_keyframes,
    stable_output_name,
    validate_parameters,
    validate_resume_state,
    write_metadata_artifacts,
)


def _args(**overrides):
    values = dict(
        target_count=2,
        candidate_fps=4.0,
        min_time_gap_sec=0.25,
        jpeg_quality=98,
        seed=0,
        duplicate_threshold=0.985,
        max_underexposed_ratio=0.45,
        max_overexposed_ratio=0.45,
        dark_threshold=8,
        bright_threshold=247,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _candidate(tmp_path, index, feature, sharpness=100.0):
    path = tmp_path / f"candidate_{index:08d}.jpg"
    Image.new("RGB", (16, 12), color=(index * 20, 64, 128)).save(path)
    return Candidate(
        index=index,
        timestamp_sec=index * 0.5,
        path=path,
        width=16,
        height=12,
        sharpness=sharpness,
        underexposed_ratio=0.0,
        overexposed_ratio=0.0,
        feature=np.asarray(feature, dtype=np.float32),
    )


def _video(tmp_path):
    path = tmp_path / "source.mp4"
    path.write_bytes(b"test-video-metadata-only")
    stat = path.stat()
    return VideoInfo(path, 2.0, 16, 12, "test", 30.0, stat.st_size, stat.st_mtime_ns)


def test_stable_output_naming():
    assert stable_output_name(0) == "000000.jpg"
    assert stable_output_name(249) == "000249.jpg"
    with pytest.raises(ValueError):
        stable_output_name(-1)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"target_count": 0}, "target-count"),
        ({"candidate_fps": 0.0}, "candidate-fps"),
        ({"min_time_gap_sec": -1.0}, "min-time-gap"),
        ({"jpeg_quality": 101}, "jpeg-quality"),
        ({"duplicate_threshold": 1.1}, "duplicate-threshold"),
    ],
)
def test_parameter_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_parameters(_args(**overrides))


def test_missing_ffmpeg_dependency_has_clear_error(monkeypatch):
    monkeypatch.setattr(keyframes.shutil, "which", lambda _name: None)
    with pytest.raises(KeyframeError, match="required executable 'ffprobe'"):
        keyframes.require_dependencies()


def test_duplicate_candidates_are_not_selected_together(tmp_path):
    identical = np.zeros(16, dtype=np.float32)
    different = np.ones(16, dtype=np.float32)
    candidates = [
        _candidate(tmp_path, 0, identical, sharpness=100.0),
        _candidate(tmp_path, 1, identical, sharpness=90.0),
        _candidate(tmp_path, 2, different, sharpness=80.0),
    ]
    selected = select_keyframes(candidates, 2, 0.0, 0.99, 0.45, 0.45, seed=0)
    selected_indices = {candidate.index for candidate in selected}
    assert len(selected) == 2
    assert not {0, 1}.issubset(selected_indices)
    rejected = next(candidate for candidate in candidates if not candidate.selected)
    assert rejected.rejection_reason == "visual_duplicate"


def test_manifest_records_parameters_and_contiguous_names(tmp_path):
    candidates = [
        _candidate(tmp_path, 0, np.zeros(16)),
        _candidate(tmp_path, 1, np.ones(16)),
    ]
    selected = select_keyframes(candidates, 2, 0.0, 1.0, 0.45, 0.45, seed=7)
    parameters = {"target_count": 2, "seed": 7}
    manifest = build_manifest(
        _video(tmp_path), parameters, candidates, selected, True, "complete"
    )
    assert manifest["parameters"] == parameters
    assert [frame["output_name"] for frame in manifest["keyframes"]] == [
        "000000.jpg",
        "000001.jpg",
    ]
    assert all("timestamp_sec" in frame and "selection_reason" in frame for frame in manifest["keyframes"])


def test_dry_run_artifacts_do_not_write_images(tmp_path):
    candidates = [
        _candidate(tmp_path, 0, np.zeros(16)),
        _candidate(tmp_path, 1, np.ones(16)),
    ]
    selected = select_keyframes(candidates, 2, 0.0, 1.0, 0.45, 0.45, seed=0)
    video = _video(tmp_path)
    manifest = build_manifest(video, {"target_count": 2}, candidates, selected, True, "complete")
    report = build_selection_report(video, candidates, selected, 2)
    metadata = tmp_path / "scene"
    output = metadata / "images"

    write_metadata_artifacts(metadata, manifest, report, selected, jpeg_quality=98)

    assert not output.exists()
    assert (metadata / "keyframes.json").is_file()
    assert (metadata / "contact_sheet.jpg").is_file()
    assert (metadata / "selection_report.txt").is_file()
    loaded = json.loads((metadata / "keyframes.json").read_text())
    assert loaded["dry_run"] is True


def test_run_dry_run_never_creates_output_images(tmp_path, monkeypatch):
    candidates = [
        _candidate(tmp_path, 0, np.zeros(16)),
        _candidate(tmp_path, 1, np.ones(16)),
    ]
    video = _video(tmp_path)
    output = tmp_path / "scene" / "images"
    args = _args(video=video.path, output=output, dry_run=True)

    monkeypatch.setattr(keyframes, "require_dependencies", lambda: {"ffprobe": "ffprobe", "ffmpeg": "ffmpeg"})
    monkeypatch.setattr(keyframes, "probe_video", lambda *_args: video)
    monkeypatch.setattr(keyframes, "extract_uniform_candidates", lambda *_args: [candidate.path for candidate in candidates])
    monkeypatch.setattr(keyframes, "analyze_candidates", lambda *_args: candidates)

    manifest = keyframes.run(args)

    assert manifest["dry_run"] is True
    assert not output.exists()
    assert (output.parent / "keyframes.json").is_file()


def test_resume_refuses_existing_images_with_mismatched_manifest(tmp_path):
    candidate = _candidate(tmp_path, 0, np.zeros(16))
    selected = select_keyframes([candidate], 1, 0.0, 1.0, 0.45, 0.45, seed=0)
    video = _video(tmp_path)
    output = tmp_path / "scene" / "images"
    output.mkdir(parents=True)
    with Image.open(candidate.path) as image:
        image.save(output / "000000.jpg", quality=98)
    manifest = build_manifest(
        video,
        {"target_count": 1, "seed": 0},
        [candidate],
        selected,
        False,
        "in_progress",
        written_outputs=["000000.jpg"],
    )

    with pytest.raises(KeyframeError, match="parameters differ"):
        validate_resume_state(
            output,
            manifest,
            video,
            {"target_count": 2, "seed": 0},
            selected,
        )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg integration dependencies are unavailable",
)
def test_ffmpeg_dry_run_and_formal_extraction_on_synthetic_video(tmp_path):
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=2:size=64x48:rate=8",
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            str(video),
        ],
        check=True,
    )
    output = tmp_path / "scene" / "images"
    args = _args(
        video=video,
        output=output,
        target_count=3,
        min_time_gap_sec=0.0,
        duplicate_threshold=1.0,
        max_underexposed_ratio=1.0,
        max_overexposed_ratio=1.0,
        dry_run=True,
    )

    dry_manifest = keyframes.run(args)
    assert dry_manifest["selected_count"] == 3
    assert not output.exists()

    args.dry_run = False
    final_manifest = keyframes.run(args)
    assert final_manifest["status"] == "complete"
    assert [path.name for path in sorted(output.glob("*.jpg"))] == [
        "000000.jpg",
        "000001.jpg",
        "000002.jpg",
    ]
    with Image.open(output / "000000.jpg") as image:
        assert image.size == (64, 48)
