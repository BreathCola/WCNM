from PIL import Image
import pytest

from utils.transmittance_debug import make_semantic_repair_contact_sheet


CORE_SEMANTIC_FILES = (
    "ground_truth.png",
    "final.png",
    "diffuse_contribution.png",
    "transmittance_contribution.png",
    "inside_color.png",
    "inside_alpha.png",
    "transparent_mask.png",
)


def _png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), "black").save(path)


def test_semantic_repair_contact_sheet_allows_missing_optional_reflection_columns(tmp_path):
    root = tmp_path / "iteration_015000"
    for stem in ("000000", "000001"):
        for name in CORE_SEMANTIC_FILES:
            _png(root / stem / name)

    target = make_semantic_repair_contact_sheet(str(root), ("000000", "000001"))

    assert target == str(root / "semantic_repair_contact_sheet.png")
    assert (root / "semantic_repair_contact_sheet.png").is_file()
    assert (root / "contact_sheet.png").is_file()


def test_semantic_repair_contact_sheet_still_requires_core_columns(tmp_path):
    root = tmp_path / "iteration_015000"
    for name in CORE_SEMANTIC_FILES:
        if name != "inside_alpha.png":
            _png(root / "000000" / name)

    with pytest.raises(FileNotFoundError, match="inside_alpha.png"):
        make_semantic_repair_contact_sheet(str(root), ("000000",))

