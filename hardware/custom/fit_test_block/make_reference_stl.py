"""Build ``fit_test_block_reference.stl`` with CadQuery.

This is the same layout the Fusion script builds. Fusion itself is not
installed here, so this file is the headless check. It imports CadQuery only
when you run it, not when pytest imports ``fit_layout``.

    python make_reference_stl.py
"""

from __future__ import annotations

from pathlib import Path

from fit_layout import Layout, compute_layout


OUT_PATH = Path(__file__).resolve().parent / "fit_test_block_reference.stl"


def build_solid(layout: Layout):
    import cadquery as cq

    settings = layout.settings
    parts = []

    parts.append(
        _box(
            cq,
            layout.xmin,
            layout.ymin,
            0.0,
            layout.xmax,
            layout.ymin + layout.outer_width,
            layout.block_height,
        )
    )
    parts.append(
        _box(
            cq,
            layout.xmin,
            layout.outer_width / 2,
            0.0,
            layout.xmax,
            layout.ymax,
            settings.feature_pad_thickness,
        )
    )

    pocket_tool = (
        cq.Workplane("XY")
        .workplane(offset=settings.floor_thickness - 0.02)
        .rect(layout.pocket_length, layout.pocket_width)
        .extrude(settings.pocket_depth + 0.04)
        .edges("|Z")
        .fillet(settings.pocket_corner_radius)
    )

    solid = parts[0]
    for part in parts[1:]:
        solid = solid.union(part)
    solid = solid.cut(pocket_tool)

    for circle in layout.circles:
        if circle.kind != "hole":
            continue
        solid = solid.cut(_cylinder(cq, circle.x, circle.y, circle.diameter / 2, -0.02, settings.feature_pad_thickness + 0.02))

    for slot in layout.slots:
        solid = solid.cut(
            cq.Workplane("XY")
            .center(slot.x, slot.y)
            .slot2D(slot.length, slot.width)
            .extrude(settings.feature_pad_thickness + 0.02)
        )

    for circle in layout.circles:
        if circle.kind != "peg":
            continue
        solid = solid.union(
            _cylinder(
                cq,
                circle.x,
                circle.y,
                circle.diameter / 2,
                settings.feature_pad_thickness,
                settings.feature_pad_thickness + settings.peg_height,
            )
        )

    for label in layout.labels:
        if label.where == "pad":
            plane_z = settings.feature_pad_thickness
        else:
            plane_z = settings.floor_thickness
        text = (
            cq.Workplane("XY")
            .workplane(offset=plane_z)
            .center(label.x, label.y)
            .text(
                label.text,
                label.height,
                -label.depth,
                font="Arial",
                halign="center",
                valign="center",
            )
        )
        solid = solid.cut(text)

    return solid


def export_stl(layout: Layout | None = None, path: Path = OUT_PATH) -> Path:
    import cadquery as cq

    layout = layout or compute_layout()
    solid = build_solid(layout)
    cq.exporters.export(solid, str(path), tolerance=0.02, angularTolerance=0.1)
    return path


def _box(cq, xmin, ymin, zmin, xmax, ymax, zmax):
    return (
        cq.Workplane("XY")
        .center((xmin + xmax) / 2, (ymin + ymax) / 2)
        .workplane(offset=(zmin + zmax) / 2)
        .box(xmax - xmin, ymax - ymin, zmax - zmin)
    )


def _cylinder(cq, x, y, radius, zmin, zmax):
    return (
        cq.Workplane("XY")
        .center(x, y)
        .workplane(offset=zmin)
        .circle(radius)
        .extrude(zmax - zmin)
    )


if __name__ == "__main__":
    out = export_stl()
    print(out)
