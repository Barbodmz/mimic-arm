"""Layout checks for the fit-test block. Run with plain pytest.

Fusion's ``adsk`` modules are not imported here.
"""

from __future__ import annotations

import ast
import re
import struct
from pathlib import Path

import pytest

from fit_layout import (
    BODY_DEPTH_MM,
    BODY_LENGTH_MM,
    BODY_WIDTH_MM,
    HORN_HOLE_DIAMETER_MM,
    HORN_HOLE_OFFSET_MM,
    MOUNT_HOLE_DIAMETER_MM,
    OUTPUT_AXIS_X_MM,
    OVERALL_HEIGHT_MM,
    TOP_MOUNT_HOLE_X_MM,
    Settings,
    column_x,
    compute_layout,
    parameter_definitions,
)


HERE = Path(__file__).resolve().parent
REFERENCE_STL = HERE / "fit_test_block_reference.stl"


def test_default_pocket_is_the_servo_plus_clearance():
    layout = compute_layout()
    assert layout.pocket_length == pytest.approx(BODY_LENGTH_MM + 0.2)
    assert layout.pocket_width == pytest.approx(BODY_WIDTH_MM + 0.2)
    assert layout.block_height == pytest.approx(1.6 + 12.0)
    assert layout.outer_length == pytest.approx(layout.pocket_length + 4.0)
    assert layout.block_width == pytest.approx(layout.outer_width + layout.settings.feature_strip_width)


def test_clearance_changes_holes_slots_and_pegs_in_opposite_directions():
    layout = compute_layout()
    m2 = [c for c in layout.circles if c.name.startswith("m2_")]
    m3 = [c for c in layout.circles if c.name.startswith("m3_")]
    pegs = [c for c in layout.circles if c.kind == "peg"]
    assert [c.diameter for c in m2] == pytest.approx([2.1, 2.2, 2.3])
    assert [c.diameter for c in m3] == pytest.approx([3.1, 3.2, 3.3])
    assert [s.width for s in layout.slots] == pytest.approx([3.1, 3.2, 3.3])
    assert [p.diameter for p in pegs] == pytest.approx([3.9, 3.8, 3.7])
    assert [label.text for label in layout.labels if label.where == "pad"] == [
        "0.10",
        "0.20",
        "0.30",
    ]


def test_column_centres_are_symmetric():
    length = 49.6
    assert column_x(0, 3, length) == pytest.approx(-length / 3)
    assert column_x(1, 3, length) == pytest.approx(0.0)
    assert column_x(2, 3, length) == pytest.approx(length / 3)


def test_a_tighter_pocket_clearance_shrinks_only_the_opening():
    loose = compute_layout(Settings(pocket_clearance=0.3))
    tight = compute_layout(Settings(pocket_clearance=0.1))
    assert loose.pocket_length - tight.pocket_length == pytest.approx(0.2)
    assert loose.pocket_width - tight.pocket_width == pytest.approx(0.2)
    assert loose.block_height == tight.block_height


def test_recorded_servo_measurements():
    # Locked to STEP/SO100/STS3215_03a.step. See fit_layout.py for the source.
    assert BODY_LENGTH_MM == 45.4
    assert BODY_WIDTH_MM == 24.8
    assert BODY_DEPTH_MM == 28.8
    assert OVERALL_HEIGHT_MM == 39.6
    assert OUTPUT_AXIS_X_MM == 12.5
    assert HORN_HOLE_DIAMETER_MM == 2.5
    assert HORN_HOLE_OFFSET_MM == 4.95
    assert MOUNT_HOLE_DIAMETER_MM == 1.5
    assert TOP_MOUNT_HOLE_X_MM[1] - TOP_MOUNT_HOLE_X_MM[0] == pytest.approx(20.7)


def test_fusion_settings_match_the_layout_defaults():
    source = (HERE / "fit_test_block.py").read_text()
    module = ast.parse(source)
    wanted = {
        "BODY_LENGTH",
        "BODY_WIDTH",
        "BODY_DEPTH",
        "POCKET_CLEARANCE",
        "TEST_CLEARANCES",
        "WALL_THICKNESS",
        "FLOOR_THICKNESS",
        "POCKET_DEPTH",
        "FEATURE_PAD_THICKNESS",
    }
    found = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            found[target.id] = ast.literal_eval(node.value)
    assert found["BODY_LENGTH"] == BODY_LENGTH_MM
    assert found["BODY_WIDTH"] == BODY_WIDTH_MM
    assert found["BODY_DEPTH"] == BODY_DEPTH_MM
    assert found["POCKET_CLEARANCE"] == 0.2
    assert found["TEST_CLEARANCES"] == (0.1, 0.2, 0.3)
    assert found["WALL_THICKNESS"] == 2.0
    assert found["FLOOR_THICKNESS"] == 1.6
    assert found["POCKET_DEPTH"] == 12.0
    assert found["FEATURE_PAD_THICKNESS"] == 3.2


def test_parameter_expressions_match_the_placed_features():
    layout = compute_layout()
    env = {}
    for name, expression, _comment in parameter_definitions(layout.settings):
        env[name] = _eval_expr(expression, env)
    for circle in layout.circles:
        assert _eval_expr(circle.x_abs_expr, env) == pytest.approx(abs(circle.x))
        assert _eval_expr(circle.y_expr, env) == pytest.approx(circle.y)
        assert _eval_expr(circle.diameter_expr, env) == pytest.approx(circle.diameter)
    for slot in layout.slots:
        assert _eval_expr(slot.x_abs_expr, env) == pytest.approx(abs(slot.x))
        assert _eval_expr(slot.y_expr, env) == pytest.approx(slot.y)
        assert _eval_expr(slot.width_expr, env) == pytest.approx(slot.width)


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        compute_layout(Settings(test_clearances=()))
    with pytest.raises(ValueError):
        compute_layout(Settings(peg_nominal=0.2, test_clearances=(0.3,)))
    with pytest.raises(ValueError):
        compute_layout(Settings(pocket_depth=40.0))


def test_reference_stl_matches_the_layout_box():
    layout = compute_layout()
    triangles, bbox = _read_stl(REFERENCE_STL)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    assert xmax - xmin == pytest.approx(layout.block_length, abs=0.05)
    assert ymax - ymin == pytest.approx(layout.block_width, abs=0.05)
    assert zmax - zmin == pytest.approx(layout.block_height, abs=0.05)
    assert zmin == pytest.approx(0.0, abs=0.05)
    volume = _mesh_volume(triangles)
    solid_box = layout.block_length * layout.block_width * layout.block_height
    pocket = layout.pocket_length * layout.pocket_width * layout.settings.pocket_depth
    # Thin walls and an open pocket: much smaller than a solid brick, and
    # smaller than that brick with only the pocket removed (the pad is short).
    assert volume < solid_box - pocket * 0.5
    assert volume > pocket * 0.3


def _read_stl(path: Path):
    data = path.read_bytes()
    if data[:5].lower() == b"solid" and b"facet" in data[:200]:
        raise AssertionError(f"{path.name} is ASCII STL; expected binary")
    count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + count * 50
    assert len(data) == expected, f"{path.name} is {len(data)} bytes, expected {expected}"
    triangles = []
    mins = [1e9, 1e9, 1e9]
    maxs = [-1e9, -1e9, -1e9]
    offset = 84
    for _ in range(count):
        nums = struct.unpack_from("<12fH", data, offset)
        offset += 50
        verts = tuple(nums[3 + i * 3 : 6 + i * 3] for i in range(3))
        triangles.append(verts)
        for vert in verts:
            for axis, value in enumerate(vert):
                mins[axis] = min(mins[axis], value)
                maxs[axis] = max(maxs[axis], value)
    bbox = (mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2])
    return triangles, bbox


def _eval_expr(expression: str, env: dict[str, float]) -> float:
    """Evaluate a Fusion parameter expression with the names we defined."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(0)
        if name in env:
            return repr(env[name])
        return name

    python = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", replace, expression)
    python = python.replace("mm", "")
    return float(eval(python, {"__builtins__": {}}, {"abs": abs}))  # noqa: S307


def _mesh_volume(triangles) -> float:
    volume = 0.0
    for (ax, ay, az), (bx, by, bz), (cx, cy, cz) in triangles:
        volume += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
    return abs(volume) / 6.0
