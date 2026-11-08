"""Virtual sections: in-memory computation and training-oriented file output."""

from __future__ import annotations

import colorsys
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
from matplotlib.path import Path as MatplotlibPath

from .tessellation import (
    Plane,
    SectionPolygon,
    Tessellation,
    avoid_vertices,
    intersect_polyhedra,
    plane_coordinate,
    projected_bounds,
    read_tessellation,
)

FORMAT_VERSION = 2


class SectionGeometryError(ValueError):
    """Raised when the computed section does not tile the requested window."""


@dataclass(frozen=True)
class Section:
    """A rasterized planar section of a 3D tessellation.

    ``cell_ids`` and ``parent_cell_ids`` are ``(resolution, resolution)`` int32
    arrays with the raster origin at the top-left corner of ``window``;
    ``orientations`` holds the three orientation values of the cell under every
    pixel (NaN where the cell has no orientation).
    """

    plane: Plane
    axis: str
    position_relative: float
    plane_shifted: bool
    window: tuple[float, float, float, float]
    resolution: int
    cell_ids: np.ndarray
    parent_cell_ids: np.ndarray
    orientations: np.ndarray
    polygons: list[SectionPolygon]

    @property
    def polygon_count(self) -> int:
        return len(self.polygons)


@dataclass(frozen=True)
class GeneratedSection:
    """Paths of the files written for one section."""

    image_path: Path
    data_path: Path
    metadata_path: Path
    polygon_count: int


# ------------------------------------------------------------------- computation


def _rasterize(
    polygons: list[SectionPolygon],
    window: tuple[float, float, float, float],
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign every pixel center to the polygon containing it."""
    umin, umax, vmin, vmax = window
    pixel_width = (umax - umin) / resolution
    pixel_height = (vmax - vmin) / resolution
    cell_ids = np.zeros((resolution, resolution), dtype=np.int32)
    parent_ids = np.zeros_like(cell_ids)

    for polygon in polygons:
        min_u, min_v = polygon.points.min(axis=0)
        max_u, max_v = polygon.points.max(axis=0)
        column_start = max(0, int(np.floor((min_u - umin) / pixel_width)))
        column_end = min(resolution, int(np.ceil((max_u - umin) / pixel_width)))
        row_start = max(0, int(np.floor((vmax - max_v) / pixel_height)))
        row_end = min(resolution, int(np.ceil((vmax - min_v) / pixel_height)))
        if column_start >= column_end or row_start >= row_end:
            continue

        columns = np.arange(column_start, column_end)
        rows = np.arange(row_start, row_end)
        u_coordinates = umin + (columns + 0.5) * pixel_width
        v_coordinates = vmax - (rows + 0.5) * pixel_height
        grid_u, grid_v = np.meshgrid(u_coordinates, v_coordinates)
        sample_points = np.column_stack((grid_u.ravel(), grid_v.ravel()))
        inside = (
            MatplotlibPath(polygon.points)
            .contains_points(sample_points, radius=1e-12)
            .reshape(len(rows), len(columns))
        )
        cell_ids[row_start:row_end, column_start:column_end][inside] = polygon.cell_id
        parent_ids[row_start:row_end, column_start:column_end][inside] = (
            polygon.parent_cell_id
        )

    return cell_ids, parent_ids


def _polygon_area(points: np.ndarray) -> float:
    u, v = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1))))


def _validate_coverage(
    polygons: list[SectionPolygon],
    window: tuple[float, float, float, float],
    cell_ids: np.ndarray,
) -> None:
    """Check that the polygons tile the window: no gaps, no overlaps.

    The area test (sum of polygon areas equals the window area) and the raster
    test (every pixel center claimed) together rule out gaps and overlaps of
    measurable size.
    """
    umin, umax, vmin, vmax = window
    expected_area = (umax - umin) * (vmax - vmin)
    actual_area = sum(_polygon_area(polygon.points) for polygon in polygons)
    if not np.isclose(actual_area, expected_area, rtol=1e-7, atol=1e-10):
        raise SectionGeometryError(
            f"Section polygons cover {100.0 * actual_area / expected_area:.4f}% "
            "of the window area."
        )
    unassigned = int(np.count_nonzero(cell_ids == 0))
    if unassigned:
        raise SectionGeometryError(
            f"Section raster has {unassigned} unassigned pixels."
        )


def _orientation_map(
    cell_ids: np.ndarray, polygons: list[SectionPolygon]
) -> np.ndarray:
    result = np.full((*cell_ids.shape, 3), np.nan, dtype=np.float32)
    orientations = {
        polygon.cell_id: polygon.orientation
        for polygon in polygons
        if polygon.orientation is not None
    }
    for cell_id, orientation in orientations.items():
        result[cell_ids == cell_id] = orientation
    return result


def compute_section(
    tessellation: Tessellation,
    *,
    axis: str,
    position: float,
    resolution: int,
    tolerance: float = 1e-10,
) -> Section:
    """Section ``tessellation`` by the plane normal to ``axis`` at relative ``position``.

    ``position`` is relative to the tessellation bounds and must lie strictly in
    ``(0, 1)``.  The raster window is the full projected domain, so the section is
    validated to tile it exactly.
    """
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be one of: x, y, z")
    if not 0.0 < position < 1.0:
        raise ValueError("position must lie strictly inside (0, 1)")
    if resolution < 1:
        raise ValueError("resolution must be a positive integer")
    if tessellation.dimension != 3:
        raise ValueError("The source tessellation must be three-dimensional.")

    plane = Plane.axis_aligned(axis, plane_coordinate(tessellation, axis, position))
    plane, shifted = avoid_vertices(tessellation, plane, tolerance)
    polygons = intersect_polyhedra(tessellation, plane, tolerance=tolerance)
    window = projected_bounds(tessellation, axis)
    cell_ids, parent_ids = _rasterize(polygons, window, resolution)
    _validate_coverage(polygons, window, cell_ids)

    return Section(
        plane=plane,
        axis=axis,
        position_relative=position,
        plane_shifted=shifted,
        window=window,
        resolution=resolution,
        cell_ids=cell_ids,
        parent_cell_ids=parent_ids,
        orientations=_orientation_map(cell_ids, polygons),
        polygons=polygons,
    )


# ------------------------------------------------------------------------ output


def _identifier_color(identifier: int) -> np.ndarray:
    if identifier == 0:
        return np.asarray((255, 255, 255), dtype=np.uint8)
    hue = (identifier * 0.6180339887498949) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.58, 0.95)
    return np.asarray((red * 255, green * 255, blue * 255), dtype=np.uint8)


def debug_image(identifier_map: np.ndarray) -> np.ndarray:
    """Deterministic per-identifier colors with black boundaries (not IPF colors)."""
    maximum = int(identifier_map.max(initial=0))
    palette = np.vstack(
        [_identifier_color(identifier) for identifier in range(maximum + 1)]
    )
    image = palette[identifier_map]

    boundary = np.zeros(identifier_map.shape, dtype=bool)
    boundary[:, 1:] |= identifier_map[:, 1:] != identifier_map[:, :-1]
    boundary[:, :-1] |= identifier_map[:, 1:] != identifier_map[:, :-1]
    boundary[1:, :] |= identifier_map[1:, :] != identifier_map[:-1, :]
    boundary[:-1, :] |= identifier_map[1:, :] != identifier_map[:-1, :]
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    image[boundary] = 0
    return image


def _position_token(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def section_metadata(
    section: Section, tessellation: Tessellation, source: Path | None
) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "source_tessellation": str(source.resolve()) if source else None,
        "source_crystal_symmetry": tessellation.crystal_symmetry,
        "orientation_descriptor": tessellation.orientation_descriptor,
        "section": {
            "axis": section.axis,
            "position_relative": section.position_relative,
            "plane_normal": section.plane.normal.tolist(),
            "plane_offset": section.plane.offset,
            "plane_shifted_to_avoid_vertices": section.plane_shifted,
            "basis_u": section.plane.basis_u.tolist(),
            "basis_v": section.plane.basis_v.tolist(),
            "window": list(section.window),
            "resolution": [section.resolution, section.resolution],
        },
        "raster": {"array_origin": "top-left"},
        "polygons": [
            {
                "cell_id": polygon.cell_id,
                "parent_cell_id": polygon.parent_cell_id,
                "orientation": (
                    polygon.orientation.tolist()
                    if polygon.orientation is not None
                    else None
                ),
                "vertices": polygon.points.tolist(),
            }
            for polygon in section.polygons
        ],
    }


def write_section(
    section: Section,
    tessellation: Tessellation,
    output_directory: str | Path,
    *,
    stem: str | None = None,
    color_by: str = "cell-id",
    source: Path | None = None,
) -> GeneratedSection:
    """Write PNG (debug colors), NPZ (per-pixel arrays) and JSON (metadata, polygons)."""
    if color_by not in {"cell-id", "parent-id"}:
        raise ValueError("color_by must be one of: cell-id, parent-id")
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or (
        f"section-{section.axis}-{_position_token(section.position_relative)}"
        f"-r{section.resolution}"
    )
    image_path = output_dir / f"{stem}.png"
    data_path = output_dir / f"{stem}.npz"
    metadata_path = output_dir / f"{stem}.json"

    identifier_map = (
        section.cell_ids if color_by == "cell-id" else section.parent_cell_ids
    )
    mpimg.imsave(image_path, debug_image(identifier_map))
    np.savez_compressed(
        data_path,
        cell_ids=section.cell_ids,
        parent_cell_ids=section.parent_cell_ids,
        orientation_values=section.orientations,
    )
    metadata = section_metadata(section, tessellation, source)
    metadata["rendering"] = {
        "color_by": color_by,
        "meaning": "deterministic debug colors; not an EBSD/IPF color map",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return GeneratedSection(image_path, data_path, metadata_path, section.polygon_count)


def generate_section(
    source_path: str | Path,
    output_directory: str | Path,
    *,
    axis: str,
    position: float,
    resolution: int,
    color_by: str = "cell-id",
    output_stem: str | None = None,
) -> GeneratedSection:
    """Read a tessellation file, compute one section and write its artifacts."""
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"Tessellation file not found: {source}")
    tessellation = read_tessellation(source)
    section = compute_section(
        tessellation, axis=axis, position=position, resolution=resolution
    )
    return write_section(
        section,
        tessellation,
        output_directory,
        stem=output_stem,
        color_by=color_by,
        source=source,
    )
