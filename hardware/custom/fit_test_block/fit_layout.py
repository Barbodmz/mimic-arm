"""Size and place the STS3215 fit-test block.

This module does not import Fusion (``adsk``). The Fusion script and the
reference-STL builder both call :func:`compute_layout`, and ``pytest`` checks
the numbers here.

Clearance is extra size on the whole feature, not extra per side. A 0.2 mm
clearance makes a pocket 0.2 mm longer and 0.2 mm wider than the servo, and
makes an M3 hole 3.2 mm across (0.1 mm of air on each side of a 3.0 mm screw).

Servo numbers
-------------
Measured from the SO-ARM100 solid, not from the catalog line on the datasheet.

Source file (millimetres):
https://github.com/TheRobotStudio/SO-ARM100/blob/main/STEP/SO100/STS3215_03a.step

``hardware/stock/`` has the 15 official SO-101 arm parts and no servo solid,
so the servo itself comes from that SO-ARM100 STEP. The same shape is meshed
at ``Simulation/SO101/assets/sts3215_03a_v1.stl`` (units metres).

The model's origin is the centre of the rectangular case. +X runs from the
cable end toward the output end. +Z is the output shaft.

Flat case (the part that slides into a pocket), from the constant cross-section
between the top face z = 14.40 and the bottom face z = -14.40:

- length (X) 45.40 mm, from -22.70 to 22.70
- width  (Y) 24.80 mm, from -12.40 to 12.40
- depth  (Z) 28.80 mm

Plan-view corner radii: 2.0 mm at the cable end (min X), 3.0 mm at the output
end (max X). Overall bounding box, including the output spline and the bottom
pins, is 45.40 x 24.80 x 39.60 mm (z from -19.40 to 20.20).

Feetech's datasheet lists the catalog outline as 45.2 x 24.7 x 35 mm. The
pocket uses the STEP case above, because that is the solid the SO-101 parts
were drawn around.

Output shaft axis: (12.50, 0.00) mm.
Horn screw holes: diameter 2.50 mm, four holes at (±4.95, ±4.95) mm from the
shaft axis, so the centres are (7.55, ±4.95) and (17.45, ±4.95).
Case mounting holes: diameter 1.50 mm, vertical, at y = ±10.25.
Top flange: x = -16.50 and x = 4.20. Bottom face: x = -20.30 and x = 4.20.
"""

from __future__ import annotations

from dataclasses import dataclass


# Measured case. Edit only if you re-measure the STEP.
BODY_LENGTH_MM = 45.4
BODY_WIDTH_MM = 24.8
BODY_DEPTH_MM = 28.8
OVERALL_HEIGHT_MM = 39.6
CABLE_END_CORNER_RADIUS_MM = 2.0
OUTPUT_END_CORNER_RADIUS_MM = 3.0
OUTPUT_AXIS_X_MM = 12.5
OUTPUT_AXIS_Y_MM = 0.0
HORN_HOLE_DIAMETER_MM = 2.5
HORN_HOLE_OFFSET_MM = 4.95
MOUNT_HOLE_DIAMETER_MM = 1.5
MOUNT_HOLE_Y_MM = 10.25
TOP_MOUNT_HOLE_X_MM = (-16.5, 4.2)
BOTTOM_MOUNT_HOLE_X_MM = (-20.3, 4.2)

SERVO_STEP_URL = (
    "https://github.com/TheRobotStudio/SO-ARM100/blob/main/STEP/SO100/STS3215_03a.step"
)

# Distances from the pocket's outer +Y wall to each row of test features.
M2_M3_OFFSET_Y_MM = 6.0
HOLE_PAIR_HALF_SPACING_MM = 3.25
SLOT_OFFSET_Y_MM = 12.5
PEG_OFFSET_Y_MM = 19.0
LABEL_OFFSET_Y_MM = 24.0
FEATURE_STRIP_WIDTH_MM = 28.0
LABEL_TEXT_HEIGHT_MM = 3.2
POCKET_LABEL_HEIGHT_MM = 3.5
MIN_FEATURE_GAP_MM = 1.2


@dataclass(frozen=True)
class Settings:
    """Every size the block is built from. Lengths are millimetres."""

    body_length: float = BODY_LENGTH_MM
    body_width: float = BODY_WIDTH_MM
    body_depth: float = BODY_DEPTH_MM
    pocket_clearance: float = 0.2
    test_clearances: tuple[float, ...] = (0.1, 0.2, 0.3)
    wall_thickness: float = 2.0
    floor_thickness: float = 1.6
    pocket_depth: float = 12.0
    feature_pad_thickness: float = 3.2
    feature_strip_width: float = FEATURE_STRIP_WIDTH_MM
    pocket_corner_radius: float = CABLE_END_CORNER_RADIUS_MM
    m2_nominal: float = 2.0
    m3_nominal: float = 3.0
    slot_nominal_width: float = 3.0
    slot_length: float = 8.0
    peg_nominal: float = 4.0
    peg_height: float = 3.0
    label_depth: float = 0.4
    label_text_height: float = LABEL_TEXT_HEIGHT_MM
    m2_m3_offset_y: float = M2_M3_OFFSET_Y_MM
    hole_pair_half_spacing: float = HOLE_PAIR_HALF_SPACING_MM
    slot_offset_y: float = SLOT_OFFSET_Y_MM
    peg_offset_y: float = PEG_OFFSET_Y_MM
    label_offset_y: float = LABEL_OFFSET_Y_MM


@dataclass(frozen=True)
class CircleFeature:
    """A round hole (cut down through the pad) or a peg (stands on the pad)."""

    name: str
    kind: str
    clearance: float
    x: float
    y: float
    diameter: float
    # Fusion parameter expression for the diameter, such as "m2_nominal + clearance_1".
    diameter_expr: str
    # Positive distance from the origin to this centre, axis by axis.
    x_abs_expr: str
    y_expr: str
    x_sign: int


@dataclass(frozen=True)
class SlotFeature:
    """Obround slot. ``length`` is the overall X size, including the round ends."""

    name: str
    clearance: float
    x: float
    y: float
    width: float
    length: float
    width_expr: str
    x_abs_expr: str
    y_expr: str
    x_sign: int


@dataclass(frozen=True)
class Label:
    text: str
    x: float
    y: float
    height: float
    depth: float
    where: str  # "pad" or "pocket_floor"


@dataclass(frozen=True)
class Layout:
    settings: Settings
    pocket_length: float
    pocket_width: float
    outer_length: float
    outer_width: float
    block_length: float
    block_width: float
    block_height: float
    # Footprint: x from -block_length/2, y from -outer_width/2.
    origin_x: float
    origin_y: float
    circles: tuple[CircleFeature, ...]
    slots: tuple[SlotFeature, ...]
    labels: tuple[Label, ...]

    @property
    def xmin(self) -> float:
        return -self.block_length / 2

    @property
    def xmax(self) -> float:
        return self.block_length / 2

    @property
    def ymin(self) -> float:
        return -self.outer_width / 2

    @property
    def ymax(self) -> float:
        return self.ymin + self.block_width


def column_x(index: int, count: int, outer_length: float) -> float:
    """X of clearance column ``index`` (0-based). Middle column is 0 when count is odd."""

    return outer_length * ((index + 0.5) / count - 0.5)


def clearance_name(index: int) -> str:
    """Fusion parameter name for a test clearance. ``index`` is 0-based."""

    return f"clearance_{index + 1}"


def _axis_expr(numeric: float, positive_expr: str) -> tuple[str, int]:
    if abs(numeric) < 1e-9:
        return "0 mm", 0
    sign = 1 if numeric > 0 else -1
    return positive_expr, sign


def _check_settings(settings: Settings) -> None:
    if settings.body_length <= 0 or settings.body_width <= 0 or settings.body_depth <= 0:
        raise ValueError("servo body dimensions must be positive")
    if settings.pocket_clearance < 0:
        raise ValueError("pocket_clearance cannot be negative")
    if len(settings.test_clearances) < 1:
        raise ValueError("need at least one test clearance")
    if any(c < 0 for c in settings.test_clearances):
        raise ValueError("test clearances cannot be negative")
    if settings.wall_thickness <= 0 or settings.floor_thickness <= 0:
        raise ValueError("wall and floor thickness must be positive")
    if settings.pocket_depth <= 0:
        raise ValueError("pocket_depth must be positive")
    if settings.pocket_depth >= settings.body_depth:
        raise ValueError("pocket_depth should stay shallower than the servo case")
    if settings.feature_pad_thickness < settings.floor_thickness:
        raise ValueError("feature pad must be at least as thick as the floor")
    if settings.label_depth <= 0 or settings.label_depth >= settings.floor_thickness:
        raise ValueError("label_depth must be positive and shallower than the floor")
    if settings.label_depth >= settings.feature_pad_thickness:
        raise ValueError("label_depth must be shallower than the feature pad")
    if settings.pocket_corner_radius <= 0:
        raise ValueError("pocket corner radius must be positive")
    if settings.pocket_corner_radius * 2 >= min(settings.body_length, settings.body_width):
        raise ValueError("pocket corner radius is too large for the servo")
    if settings.slot_length <= settings.slot_nominal_width + max(settings.test_clearances):
        raise ValueError("slots need slot_length greater than the widest slot")
    for clearance in settings.test_clearances:
        if settings.peg_nominal - clearance <= 0:
            raise ValueError("a peg clearance is larger than peg_nominal")
    if settings.feature_strip_width <= settings.label_offset_y + settings.label_text_height / 2:
        raise ValueError("feature strip is too narrow for the labels")


def parameter_definitions(settings: Settings) -> list[tuple[str, str, str]]:
    """Fusion user parameters, in an order that defines each name before use.

    Expressions are millimetre quantities. A bare number is written with ``mm``.
    """

    defs: list[tuple[str, str, str]] = [
        ("body_length", _mm(settings.body_length), "STS3215 case length, X, from the STEP model"),
        ("body_width", _mm(settings.body_width), "STS3215 case width, Y, from the STEP model"),
        ("body_depth", _mm(settings.body_depth), "STS3215 flat case height. The pocket is shallower."),
        ("pocket_clearance", _mm(settings.pocket_clearance), "Extra length and width of the servo pocket"),
        ("wall_thickness", _mm(settings.wall_thickness), "Plastic around the pocket"),
        ("floor_thickness", _mm(settings.floor_thickness), "Plastic under the pocket"),
        ("pocket_depth", _mm(settings.pocket_depth), "How far the servo slides in"),
        ("feature_pad_thickness", _mm(settings.feature_pad_thickness), "Thickness of the screw-test pad"),
        ("feature_strip_width", _mm(settings.feature_strip_width), "How far the test pad sticks out from the pocket"),
        ("pocket_corner_radius", _mm(settings.pocket_corner_radius), "Inner corner radius of the pocket"),
        ("m2_nominal", _mm(settings.m2_nominal), "M2 screw diameter"),
        ("m3_nominal", _mm(settings.m3_nominal), "M3 screw diameter"),
        ("slot_nominal_width", _mm(settings.slot_nominal_width), "Slot width before clearance"),
        ("slot_length", _mm(settings.slot_length), "Overall slot length"),
        ("peg_nominal", _mm(settings.peg_nominal), "Peg diameter before clearance is subtracted"),
        ("peg_height", _mm(settings.peg_height), "How far each peg stands up"),
        ("label_depth", _mm(settings.label_depth), "How deep the engraved labels are"),
        ("label_text_height", _mm(settings.label_text_height), "Label letter height"),
        ("m2_m3_offset_y", _mm(settings.m2_m3_offset_y), "M2/M3 row, out from the pocket wall"),
        ("hole_pair_half_spacing", _mm(settings.hole_pair_half_spacing), "M2 left of the column, M3 right"),
        ("slot_offset_y", _mm(settings.slot_offset_y), "Slot row, out from the pocket wall"),
        ("peg_offset_y", _mm(settings.peg_offset_y), "Peg row, out from the pocket wall"),
        ("label_offset_y", _mm(settings.label_offset_y), "Clearance label, out from the pocket wall"),
    ]
    for index, clearance in enumerate(settings.test_clearances):
        defs.append(
            (
                clearance_name(index),
                _mm(clearance),
                "Test clearance. Holes and slots grow by this. Pegs shrink by this.",
            )
        )
    defs.extend(
        [
            ("pocket_length", "body_length + pocket_clearance", "Pocket opening, X"),
            ("pocket_width", "body_width + pocket_clearance", "Pocket opening, Y"),
            ("outer_length", "pocket_length + wall_thickness * 2", "Pocket box length"),
            ("outer_width", "pocket_width + wall_thickness * 2", "Pocket box width"),
            ("block_height", "floor_thickness + pocket_depth", "Tallest point, the pocket walls"),
        ]
    )
    return defs


def _mm(value: float) -> str:
    return f"{value:.4f} mm"


def compute_layout(settings: Settings | None = None) -> Layout:
    """Return pocket, holes, slots, pegs, and labels for ``settings``."""

    settings = settings or Settings()
    _check_settings(settings)

    pocket_length = settings.body_length + settings.pocket_clearance
    pocket_width = settings.body_width + settings.pocket_clearance
    outer_length = pocket_length + 2 * settings.wall_thickness
    outer_width = pocket_width + 2 * settings.wall_thickness
    block_length = outer_length
    block_width = outer_width + settings.feature_strip_width
    block_height = settings.floor_thickness + settings.pocket_depth
    pad_top = settings.feature_pad_thickness
    if pad_top + settings.peg_height > block_height:
        block_height = pad_top + settings.peg_height

    count = len(settings.test_clearances)
    wall_y = outer_width / 2
    circles: list[CircleFeature] = []
    slots: list[SlotFeature] = []
    labels: list[Label] = []

    for index, clearance in enumerate(settings.test_clearances):
        cname = clearance_name(index)
        cx = column_x(index, count, outer_length)
        x_abs, x_sign = _column_axis(index, count, cx)
        y_holes = wall_y + settings.m2_m3_offset_y
        y_slot = wall_y + settings.slot_offset_y
        y_peg = wall_y + settings.peg_offset_y
        y_label = wall_y + settings.label_offset_y
        y_holes_expr = "outer_width / 2 + m2_m3_offset_y"
        y_slot_expr = "outer_width / 2 + slot_offset_y"
        y_peg_expr = "outer_width / 2 + peg_offset_y"

        for side, nominal_name, nominal in (
            (-1, "m2_nominal", settings.m2_nominal),
            (1, "m3_nominal", settings.m3_nominal),
        ):
            hx = cx + side * settings.hole_pair_half_spacing
            if x_sign == 0:
                hole_abs = "hole_pair_half_spacing"
            elif side == x_sign:
                hole_abs = f"({x_abs}) + hole_pair_half_spacing"
            else:
                hole_abs = f"({x_abs}) - hole_pair_half_spacing"
            circles.append(
                CircleFeature(
                    name=f"{'m2' if side < 0 else 'm3'}_{cname}",
                    kind="hole",
                    clearance=clearance,
                    x=hx,
                    y=y_holes,
                    diameter=nominal + clearance,
                    diameter_expr=f"{nominal_name} + {cname}",
                    x_abs_expr=hole_abs,
                    y_expr=y_holes_expr,
                    x_sign=0 if abs(hx) < 1e-9 else (1 if hx > 0 else -1),
                )
            )

        slots.append(
            SlotFeature(
                name=f"slot_{cname}",
                clearance=clearance,
                x=cx,
                y=y_slot,
                width=settings.slot_nominal_width + clearance,
                length=settings.slot_length,
                width_expr=f"slot_nominal_width + {cname}",
                x_abs_expr=x_abs,
                y_expr=y_slot_expr,
                x_sign=x_sign,
            )
        )
        circles.append(
            CircleFeature(
                name=f"peg_{cname}",
                kind="peg",
                clearance=clearance,
                x=cx,
                y=y_peg,
                diameter=settings.peg_nominal - clearance,
                diameter_expr=f"peg_nominal - {cname}",
                x_abs_expr=x_abs,
                y_expr=y_peg_expr,
                x_sign=x_sign,
            )
        )
        labels.append(
            Label(
                text=f"{clearance:.2f}",
                x=cx,
                y=y_label,
                height=settings.label_text_height,
                depth=settings.label_depth,
                where="pad",
            )
        )

    labels.append(
        Label(
            text=f"servo +{settings.pocket_clearance:.2f}",
            x=0.0,
            y=0.0,
            height=POCKET_LABEL_HEIGHT_MM,
            depth=settings.label_depth,
            where="pocket_floor",
        )
    )

    layout = Layout(
        settings=settings,
        pocket_length=pocket_length,
        pocket_width=pocket_width,
        outer_length=outer_length,
        outer_width=outer_width,
        block_length=block_length,
        block_width=block_width,
        block_height=block_height,
        origin_x=0.0,
        origin_y=0.0,
        circles=tuple(circles),
        slots=tuple(slots),
        labels=tuple(labels),
    )
    _check_fit(layout)
    return layout


def _column_axis(index: int, count: int, numeric_x: float) -> tuple[str, int]:
    """Positive Fusion expression for a column centre, and its sign."""

    expr = f"abs(outer_length * (({index} + 0.5) / {count} - 0.5))"
    return _axis_expr(numeric_x, expr)


def _check_fit(layout: Layout) -> None:
    """Raise if a feature would break out of the pad or collide with another."""

    settings = layout.settings
    pad_bottom = layout.outer_width / 2
    pad = _Rect(layout.xmin, pad_bottom, layout.xmax, layout.ymax)
    pocket = _Rect(
        -layout.pocket_length / 2,
        -layout.pocket_width / 2,
        layout.pocket_length / 2,
        layout.pocket_width / 2,
    )
    margin = 1.0

    circles = list(layout.circles)
    for circle in circles:
        radius = circle.diameter / 2
        if not pad.contains_circle(circle.x, circle.y, radius, margin):
            raise ValueError(f"{circle.name} does not sit inside the test pad")
        if pocket.intersects_circle(circle.x, circle.y, radius):
            raise ValueError(f"{circle.name} breaks into the servo pocket")

    for slot in layout.slots:
        half_len = slot.length / 2
        half_w = slot.width / 2
        if not pad.contains_circle(slot.x - half_len, slot.y, half_w, margin):
            raise ValueError(f"{slot.name} does not sit inside the test pad")
        if not pad.contains_circle(slot.x + half_len, slot.y, half_w, margin):
            raise ValueError(f"{slot.name} does not sit inside the test pad")

    for left_index, left in enumerate(circles):
        for right in circles[left_index + 1 :]:
            _require_gap(
                _circle_distance(left, right),
                (left.diameter + right.diameter) / 2,
                left.name,
                right.name,
            )
        for slot in layout.slots:
            _require_gap(
                _circle_capsule_distance(left, slot),
                left.diameter / 2 + slot.width / 2,
                left.name,
                slot.name,
            )

    for left_index, left in enumerate(layout.slots):
        for right in layout.slots[left_index + 1 :]:
            _require_gap(
                _capsule_distance(left, right),
                left.width / 2 + right.width / 2,
                left.name,
                right.name,
            )

    for label in layout.labels:
        if label.where != "pad":
            continue
        half_w = 0.62 * label.height * len(label.text) / 2
        half_h = label.height / 2
        if not pad.contains_box(label.x - half_w, label.y - half_h, label.x + half_w, label.y + half_h, 0.4):
            raise ValueError(f"label {label.text!r} does not sit on the test pad")


def _require_gap(distance: float, radii: float, left: str, right: str) -> None:
    if distance < radii + MIN_FEATURE_GAP_MM:
        raise ValueError(f"{left} is too close to {right}")


def _circle_distance(left: CircleFeature, right: CircleFeature) -> float:
    return _hypot(left.x - right.x, left.y - right.y)


def _circle_capsule_distance(circle: CircleFeature, slot: SlotFeature) -> float:
    return _point_segment_distance(circle.x, circle.y, *_slot_segment(slot))


def _capsule_distance(left: SlotFeature, right: SlotFeature) -> float:
    ax0, ay0, ax1, ay1 = _slot_segment(left)
    bx0, by0, bx1, by1 = _slot_segment(right)
    return min(
        _point_segment_distance(ax0, ay0, bx0, by0, bx1, by1),
        _point_segment_distance(ax1, ay1, bx0, by0, bx1, by1),
        _point_segment_distance(bx0, by0, ax0, ay0, ax1, ay1),
        _point_segment_distance(bx1, by1, ax0, ay0, ax1, ay1),
    )


def _slot_segment(slot: SlotFeature) -> tuple[float, float, float, float]:
    straight = max(slot.length - slot.width, 0.0) / 2
    return slot.x - straight, slot.y, slot.x + straight, slot.y


def _point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return _hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return _hypot(px - (ax + t * dx), py - (ay + t * dy))


def _hypot(dx: float, dy: float) -> float:
    return (dx * dx + dy * dy) ** 0.5


@dataclass(frozen=True)
class _Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains_circle(self, x: float, y: float, radius: float, margin: float) -> bool:
        return (
            x - radius >= self.xmin + margin
            and x + radius <= self.xmax - margin
            and y - radius >= self.ymin + margin
            and y + radius <= self.ymax - margin
        )

    def contains_box(
        self, xmin: float, ymin: float, xmax: float, ymax: float, margin: float
    ) -> bool:
        return (
            xmin >= self.xmin + margin
            and xmax <= self.xmax - margin
            and ymin >= self.ymin + margin
            and ymax <= self.ymax - margin
        )

    def intersects_circle(self, x: float, y: float, radius: float) -> bool:
        nearest_x = min(max(x, self.xmin), self.xmax)
        nearest_y = min(max(y, self.ymin), self.ymax)
        return _hypot(x - nearest_x, y - nearest_y) < radius
