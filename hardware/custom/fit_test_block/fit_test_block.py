"""STS3215 fit-test block for Fusion.

Run this from Utilities > Scripts and Add-Ins. It opens a new design, builds
the block, and (unless you turn it off below) writes an STL next to this file.

Lengths are millimetres. Clearance is extra size on the whole feature, not
extra on each side. 0.20 means the pocket is 0.20 mm longer and 0.20 mm wider
than the servo, and an M3 hole is 3.20 mm across.

The servo measurements and the feature positions live in fit_layout.py so a
unit test can check them without Fusion. This file is the Fusion-only part.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Fusion puts this folder on the path when you Run the script. Keep the
# import working if you start it some other way too.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import adsk.core
import adsk.fusion

import fit_layout


# =============================================================================
# Settings. Change these, save, and Run the script again.
# =============================================================================

# Case size of the Feetech STS3215, measured from
# SO-ARM100 STEP/SO100/STS3215_03a.step. Leave these unless you re-measure.
BODY_LENGTH = 45.4
BODY_WIDTH = 24.8
BODY_DEPTH = 28.8

# The one servo pocket. Start at 0.20. Raise it if the servo will not slide
# in; lower it if the servo rattles.
POCKET_CLEARANCE = 0.2

# Screw holes, slots, and pegs are printed at each of these clearances.
TEST_CLEARANCES = (0.1, 0.2, 0.3)

WALL_THICKNESS = 2.0
FLOOR_THICKNESS = 1.6
POCKET_DEPTH = 12.0
FEATURE_PAD_THICKNESS = 3.2

# STL written when the script finishes. Blank means "next to this script".
EXPORT_STL = True
STL_PATH = ""


def run(_context: str) -> None:
    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        path = build_fit_test_block()
    except Exception:
        ui.messageBox("Fit-test block failed:\n\n{}".format(traceback.format_exc()))
        return
    note = (
        "Fit-test block is ready.\n\n"
        "Pocket clearance is {pocket:.2f} mm. "
        "The test columns are labeled with their own clearances.\n\n"
        "To tweak a thickness or a hole size: Modify > Change Parameters, "
        "User Parameters.\n"
        "To change the clearance list, run this script again after editing "
        "the settings at the top of the file."
    ).format(pocket=POCKET_CLEARANCE)
    if path:
        note += "\n\nSTL saved to:\n{}".format(path)
    ui.messageBox(note)


def stop(_context: str) -> None:
    """Scripts do not stay loaded. Add-ins would clean up here."""


def build_fit_test_block() -> str:
    settings = fit_layout.Settings(
        body_length=BODY_LENGTH,
        body_width=BODY_WIDTH,
        body_depth=BODY_DEPTH,
        pocket_clearance=POCKET_CLEARANCE,
        test_clearances=tuple(TEST_CLEARANCES),
        wall_thickness=WALL_THICKNESS,
        floor_thickness=FLOOR_THICKNESS,
        pocket_depth=POCKET_DEPTH,
        feature_pad_thickness=FEATURE_PAD_THICKNESS,
    )
    layout = fit_layout.compute_layout(settings)

    app = adsk.core.Application.get()
    doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    design = adsk.fusion.Design.cast(app.activeProduct)
    design.designType = adsk.fusion.DesignTypes.ParametricDesignType
    root = design.rootComponent
    root.name = "fit_test_block"
    _add_parameters(design, settings)

    base = _sketch_on(root, root.xYConstructionPlane, "base")
    _centered_rectangle(
        base,
        layout.outer_length,
        layout.outer_width,
        "outer_length",
        "outer_width",
    )
    extrudes = root.features.extrudeFeatures
    _extrude(
        extrudes,
        _only_profile(base, "base"),
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation,
        "block_height",
        adsk.fusion.ExtentDirections.PositiveExtentDirection,
    )

    # The pad overlaps the wall by 0.2 mm so the join is a real intersection.
    # That overlap sits inside the wall, so the outside shape does not change.
    pad = _sketch_on(root, root.xYConstructionPlane, "test pad")
    _add_strip(pad, layout)
    _extrude(
        extrudes,
        _only_profile(pad, "test pad"),
        adsk.fusion.FeatureOperations.JoinFeatureOperation,
        "feature_pad_thickness",
        adsk.fusion.ExtentDirections.PositiveExtentDirection,
    )

    top = _offset_plane(root, "block_height", "top of walls")
    pocket = _sketch_on(root, top, "pocket")
    _centered_rectangle(
        pocket,
        layout.pocket_length,
        layout.pocket_width,
        "pocket_length",
        "pocket_width",
    )
    pocket_profile = _only_profile(pocket, "pocket")
    _extrude(
        extrudes,
        pocket_profile,
        adsk.fusion.FeatureOperations.CutFeatureOperation,
        "pocket_depth",
        adsk.fusion.ExtentDirections.NegativeExtentDirection,
    )
    _fillet_pocket_corners(root, layout)

    pad_top = _offset_plane(root, "feature_pad_thickness", "top of test pad")
    holes = _sketch_on(root, pad_top, "screw holes")
    for circle in layout.circles:
        if circle.kind != "hole":
            continue
        _add_circle(holes, circle)
    for profile in _profiles(holes):
        _extrude(
            extrudes,
            profile,
            adsk.fusion.FeatureOperations.CutFeatureOperation,
            "feature_pad_thickness",
            adsk.fusion.ExtentDirections.NegativeExtentDirection,
        )

    slots = _sketch_on(root, pad_top, "slots")
    for slot in layout.slots:
        _add_slot(slots, slot)
    for profile in _profiles(slots):
        _extrude(
            extrudes,
            profile,
            adsk.fusion.FeatureOperations.CutFeatureOperation,
            "feature_pad_thickness",
            adsk.fusion.ExtentDirections.NegativeExtentDirection,
        )

    pegs = _sketch_on(root, pad_top, "pegs")
    for circle in layout.circles:
        if circle.kind != "peg":
            continue
        _add_circle(pegs, circle)
    for profile in _profiles(pegs):
        _extrude(
            extrudes,
            profile,
            adsk.fusion.FeatureOperations.JoinFeatureOperation,
            "peg_height",
            adsk.fusion.ExtentDirections.PositiveExtentDirection,
        )

    pad_labels = _sketch_on(root, pad_top, "clearance labels")
    for label in layout.labels:
        if label.where == "pad":
            _add_label(pad_labels, label)
    _cut_sketch_down(extrudes, pad_labels, "label_depth")

    floor = _offset_plane(root, "floor_thickness", "pocket floor")
    floor_labels = _sketch_on(root, floor, "pocket label")
    for label in layout.labels:
        if label.where == "pocket_floor":
            _add_label(floor_labels, label)
    _cut_sketch_down(extrudes, floor_labels, "label_depth")

    if not EXPORT_STL:
        return ""
    return _export_stl(design, root)


def _add_parameters(design: adsk.fusion.Design, settings: fit_layout.Settings) -> None:
    for name, expression, comment in fit_layout.parameter_definitions(settings):
        design.userParameters.add(
            name,
            adsk.core.ValueInput.createByString(expression),
            "mm",
            comment,
        )


def _sketch_on(root, plane, name: str):
    sketch = root.sketches.add(plane)
    sketch.name = name
    return sketch


def _offset_plane(root, expression: str, name: str):
    planes = root.constructionPlanes
    plane_input = planes.createInput()
    plane_input.setByOffset(
        root.xYConstructionPlane,
        adsk.core.ValueInput.createByString(expression),
    )
    plane = planes.add(plane_input)
    plane.name = name
    return plane


def _extrude(extrudes, profile, operation, distance_expr: str, direction) -> None:
    ext_input = extrudes.createInput(profile, operation)
    extent = adsk.fusion.DistanceExtentDefinition.create(
        adsk.core.ValueInput.createByString(distance_expr)
    )
    ext_input.setOneSideExtent(extent, direction)
    extrudes.add(ext_input)


def _cut_sketch_down(extrudes, sketch, distance_expr: str) -> None:
    profiles = _profiles(sketch)
    if not profiles:
        raise RuntimeError("Sketch '{}' has no text to cut.".format(sketch.name))
    for profile in profiles:
        _extrude(
            extrudes,
            profile,
            adsk.fusion.FeatureOperations.CutFeatureOperation,
            distance_expr,
            adsk.fusion.ExtentDirections.NegativeExtentDirection,
        )


def _centered_rectangle(sketch, length_mm: float, width_mm: float, length_expr: str, width_expr: str):
    """Rectangle centred on the sketch origin, driven by two parameter expressions."""

    half_x = length_mm / 20.0  # mm -> cm, then half
    half_y = width_mm / 20.0
    lines = sketch.sketchCurves.sketchLines
    rect = lines.addTwoPointRectangle(
        adsk.core.Point3D.create(-half_x, -half_y, 0),
        adsk.core.Point3D.create(half_x, half_y, 0),
    )
    x_axis, y_axis = _origin_axes(sketch)
    left, right, bottom, top = _sides(rect)
    constraints = sketch.geometricConstraints
    constraints.addSymmetry(left, right, y_axis)
    constraints.addSymmetry(bottom, top, x_axis)
    _distance(sketch, bottom, "horizontal", length_expr, adsk.core.Point3D.create(0, -half_y - 0.4, 0))
    _distance(sketch, left, "vertical", width_expr, adsk.core.Point3D.create(-half_x - 0.4, 0, 0))
    return rect


def _add_strip(sketch, layout: fit_layout.Layout) -> None:
    """Test pad on +Y. It overlaps the wall by 0.2 mm so the two solids join."""

    overlap_mm = 0.2
    x0 = layout.xmin / 10.0
    x1 = layout.xmax / 10.0
    y0 = (layout.outer_width / 2 - overlap_mm) / 10.0
    y1 = layout.ymax / 10.0
    rect = sketch.sketchCurves.sketchLines.addTwoPointRectangle(
        adsk.core.Point3D.create(x0, y0, 0),
        adsk.core.Point3D.create(x1, y1, 0),
    )
    left, right, _bottom, top = _sides(rect)
    constraints = sketch.geometricConstraints
    _x_axis, y_axis = _origin_axes(sketch)
    constraints.addSymmetry(left, right, y_axis)
    _distance(
        sketch,
        top,
        "horizontal",
        "outer_length",
        adsk.core.Point3D.create(0, y1 + 0.4, 0),
    )
    _distance(
        sketch,
        left,
        "vertical",
        "feature_strip_width + {:.1f} mm".format(overlap_mm),
        adsk.core.Point3D.create(x0 - 0.5, (y0 + y1) / 2, 0),
    )
    # Top edge of the pad, measured from the origin. Together with the height
    # above, this plants the pad against the pocket box.
    mid = sketch.sketchPoints.add(adsk.core.Point3D.create(0, y1, 0))
    constraints.addMidpoint(mid, top)
    dim = sketch.sketchDimensions.addDistanceDimension(
        sketch.originPoint,
        mid,
        adsk.fusion.DimensionOrientations.VerticalDimensionOrientation,
        adsk.core.Point3D.create(0.6, y1 * 0.5, 0),
    )
    dim.parameter.expression = "outer_width / 2 + feature_strip_width"


def _origin_axes(sketch):
    """Long construction lines through the origin, reused if already drawn.

    Other construction lines (slot centerlines) are ignored: an axis is the
    long one whose midpoint is the origin.
    """

    x_axis = None
    y_axis = None
    curves = sketch.sketchCurves.sketchLines
    for index in range(curves.count):
        line = curves.item(index)
        if not line.isConstruction:
            continue
        start = line.startSketchPoint.geometry
        end = line.endSketchPoint.geometry
        span = abs(start.x - end.x) + abs(start.y - end.y)
        mid_x = (start.x + end.x) / 2
        mid_y = (start.y + end.y) / 2
        if span < 5 or abs(mid_x) > 1e-3 or abs(mid_y) > 1e-3:
            continue
        if abs(start.y - end.y) < 1e-4:
            x_axis = line
        elif abs(start.x - end.x) < 1e-4:
            y_axis = line
    if x_axis is not None and y_axis is not None:
        return x_axis, y_axis

    x_axis = curves.addByTwoPoints(
        adsk.core.Point3D.create(-10, 0, 0),
        adsk.core.Point3D.create(10, 0, 0),
    )
    y_axis = curves.addByTwoPoints(
        adsk.core.Point3D.create(0, -10, 0),
        adsk.core.Point3D.create(0, 10, 0),
    )
    x_axis.isConstruction = True
    y_axis.isConstruction = True
    sketch.geometricConstraints.addHorizontal(x_axis)
    sketch.geometricConstraints.addVertical(y_axis)
    sketch.geometricConstraints.addCoincident(sketch.originPoint, x_axis)
    sketch.geometricConstraints.addCoincident(sketch.originPoint, y_axis)
    return x_axis, y_axis


def _sides(rect):
    horizontal = []
    vertical = []
    for index in range(rect.count):
        line = rect.item(index)
        start = line.startSketchPoint.geometry
        end = line.endSketchPoint.geometry
        if abs(start.y - end.y) <= abs(start.x - end.x):
            horizontal.append(line)
        else:
            vertical.append(line)
    if len(horizontal) != 2 or len(vertical) != 2:
        raise RuntimeError("Could not find the four sides of a rectangle.")
    horizontal.sort(key=lambda line: (line.startSketchPoint.geometry.y + line.endSketchPoint.geometry.y) / 2)
    vertical.sort(key=lambda line: (line.startSketchPoint.geometry.x + line.endSketchPoint.geometry.x) / 2)
    bottom, top = horizontal
    left, right = vertical
    return left, right, bottom, top


def _distance(sketch, line, orientation: str, expression: str, text_point) -> None:
    if orientation == "horizontal":
        orient = adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation
    else:
        orient = adsk.fusion.DimensionOrientations.VerticalDimensionOrientation
    dim = sketch.sketchDimensions.addDistanceDimension(
        line.startSketchPoint,
        line.endSketchPoint,
        orient,
        text_point,
    )
    dim.parameter.expression = expression


def _profiles(sketch) -> list:
    return [sketch.profiles.item(index) for index in range(sketch.profiles.count)]


def _only_profile(sketch, name: str):
    profiles = _profiles(sketch)
    if len(profiles) != 1:
        raise RuntimeError("Sketch '{}' should have one profile, has {}.".format(name, len(profiles)))
    return profiles[0]


def _fillet_pocket_corners(root, layout: fit_layout.Layout) -> None:
    body = root.bRepBodies.item(0)
    half_l = layout.pocket_length / 20.0
    half_w = layout.pocket_width / 20.0
    edges = adsk.core.ObjectCollection.create()
    for edge in body.edges:
        if edge.geometry.objectType != adsk.core.Line3D.classType():
            continue
        start = edge.startVertex.geometry
        end = edge.endVertex.geometry
        if abs(start.x - end.x) > 1e-4 or abs(start.y - end.y) > 1e-4:
            continue
        x = (start.x + end.x) / 2
        y = (start.y + end.y) / 2
        if abs(abs(x) - half_l) < 0.05 and abs(abs(y) - half_w) < 0.05:
            edges.add(edge)
    if edges.count != 4:
        raise RuntimeError("Expected 4 pocket corners to round, found {}.".format(edges.count))
    fillets = root.features.filletFeatures
    fillet_input = fillets.createInput()
    fillet_input.addConstantRadiusEdgeSet(
        edges,
        adsk.core.ValueInput.createByString("pocket_corner_radius"),
        False,
    )
    fillets.add(fillet_input)


def _add_circle(sketch, feature: fit_layout.CircleFeature) -> None:
    circle = sketch.sketchCurves.sketchCircles.addByCenterRadius(
        adsk.core.Point3D.create(feature.x / 10.0, feature.y / 10.0, 0),
        feature.diameter / 20.0,
    )
    _pin_point(
        sketch,
        circle.centerSketchPoint,
        feature.x,
        feature.y,
        feature.x_abs_expr,
        feature.y_expr,
    )
    dim = sketch.sketchDimensions.addDiameterDimension(
        circle,
        adsk.core.Point3D.create(feature.x / 10.0 + 0.5, feature.y / 10.0 + 0.4, 0),
    )
    dim.parameter.expression = feature.diameter_expr


def _add_slot(sketch, slot: fit_layout.SlotFeature) -> None:
    """Obround slot along X. Length includes the round ends."""

    cx = slot.x / 10.0
    cy = slot.y / 10.0
    radius = slot.width / 20.0
    half_straight = (slot.length - slot.width) / 20.0
    arcs = sketch.sketchCurves.sketchArcs
    right = arcs.addByThreePoints(
        adsk.core.Point3D.create(cx + half_straight, cy + radius, 0),
        adsk.core.Point3D.create(cx + half_straight + radius, cy, 0),
        adsk.core.Point3D.create(cx + half_straight, cy - radius, 0),
    )
    left = arcs.addByThreePoints(
        adsk.core.Point3D.create(cx - half_straight, cy - radius, 0),
        adsk.core.Point3D.create(cx - half_straight - radius, cy, 0),
        adsk.core.Point3D.create(cx - half_straight, cy + radius, 0),
    )
    lines = sketch.sketchCurves.sketchLines
    top = lines.addByTwoPoints(right.startSketchPoint, left.endSketchPoint)
    lines.addByTwoPoints(left.startSketchPoint, right.endSketchPoint)
    sketch.geometricConstraints.addHorizontal(top)
    sketch.geometricConstraints.addEqual(right, left)

    radius_dim = sketch.sketchDimensions.addRadialDimension(
        right,
        adsk.core.Point3D.create(cx + half_straight + radius, cy + radius, 0),
    )
    radius_dim.parameter.expression = "({}) / 2".format(slot.width_expr)
    length_dim = sketch.sketchDimensions.addDistanceDimension(
        right.centerSketchPoint,
        left.centerSketchPoint,
        adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation,
        adsk.core.Point3D.create(cx, cy + radius + 0.3, 0),
    )
    length_dim.parameter.expression = "slot_length - ({})".format(slot.width_expr)

    center = sketch.sketchPoints.add(adsk.core.Point3D.create(cx, cy, 0))
    bridge = lines.addByTwoPoints(right.centerSketchPoint, left.centerSketchPoint)
    bridge.isConstruction = True
    sketch.geometricConstraints.addMidpoint(center, bridge)
    _pin_point(sketch, center, slot.x, slot.y, slot.x_abs_expr, slot.y_expr)


def _pin_point(sketch, point, x_mm: float, y_mm: float, x_abs_expr: str, y_expr: str) -> None:
    """Lock a sketch point to the origin with parameter expressions."""

    origin = sketch.originPoint
    geo = point.geometry
    _x_axis, y_axis = _origin_axes(sketch)
    if abs(x_mm) < 1e-6:
        sketch.geometricConstraints.addCoincident(point, y_axis)
    else:
        dim = sketch.sketchDimensions.addDistanceDimension(
            origin,
            point,
            adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation,
            adsk.core.Point3D.create(geo.x * 0.5, geo.y + 0.25, 0),
        )
        dim.parameter.expression = x_abs_expr
    dim_y = sketch.sketchDimensions.addDistanceDimension(
        origin,
        point,
        adsk.fusion.DimensionOrientations.VerticalDimensionOrientation,
        adsk.core.Point3D.create(geo.x + 0.35, geo.y * 0.5, 0),
    )
    dim_y.parameter.expression = y_expr


def _add_label(sketch, label: fit_layout.Label) -> None:
    texts = sketch.sketchTexts
    text_input = texts.createInput2(label.text, label.height / 10.0)
    text_input.fontName = "Arial"
    # A box a bit wider than the letters, centred on the label point.
    half_w = (0.7 * label.height * len(label.text)) / 20.0
    half_h = label.height / 20.0
    text_input.setAsMultiLine(
        adsk.core.Point3D.create(label.x / 10.0 - half_w, label.y / 10.0 - half_h, 0),
        adsk.core.Point3D.create(label.x / 10.0 + half_w, label.y / 10.0 + half_h, 0),
        adsk.core.HorizontalAlignments.CenterHorizontalAlignment,
        adsk.core.VerticalAlignments.MiddleVerticalAlignment,
        0,
    )
    texts.add(text_input)


def _export_stl(design: adsk.fusion.Design, root) -> str:
    if STL_PATH:
        path = str(Path(STL_PATH).expanduser())
    else:
        path = str(_SCRIPT_DIR / "fit_test_block.stl")
    export = design.exportManager
    options = export.createSTLExportOptions(root, path)
    options.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementMedium
    options.isBinaryFormat = True
    options.sendToPrintUtility = False
    export.execute(options)
    return path
