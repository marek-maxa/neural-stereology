import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from neural_stereology.sections.generator import (
    SectionGeometryError,
    _polygon_area,
    compute_section,
    generate_section,
)
from neural_stereology.sections.tessellation import (
    Plane,
    avoid_vertices,
    intersect_polyhedra,
    plane_coordinate,
    read_tessellation,
)

REAL_TESSELLATION = (
    Path(__file__).resolve().parents[1] / "src" / "lamella" / "data" / "2scale.tess"
)


def _tess_text(cell_count, vertices, faces, polyhedra, orientations=None):
    """Build a minimal Neper .tess text from vertex/face/polyhedron lists."""
    lines = ["***tess", " **format", "   3.5", " **general", "   3 standard"]
    lines += [" **cell", f"  {cell_count}", "  *crysym", "   cubic"]
    if orientations:
        lines += ["  *ori", "   euler-bunge:passive"]
        lines += [f"   {a} {b} {c}" for a, b, c in orientations]
    lines += [" **vertex", f"  {len(vertices)}"]
    lines += [
        f"   {index} {x} {y} {z} 0" for index, (x, y, z) in enumerate(vertices, 1)
    ]
    lines += [" **face", f"  {len(faces)}"]
    for index, vertex_ids in enumerate(faces, 1):
        ids = " ".join(map(str, vertex_ids))
        lines += [
            f"   {index} {len(vertex_ids)} {ids}",
            f"     {len(vertex_ids)} {ids}",  # edge ids (unused by the reader)
            "     0 0 0 1",
            "     0 -1 0 0 0",
        ]
    lines += [" **polyhedron", f"  {len(polyhedra)}"]
    for index, face_ids in enumerate(polyhedra, 1):
        lines.append(f"   {index} {len(face_ids)} {' '.join(map(str, face_ids))}")
    lines.append("***end")
    return "\n".join(lines) + "\n"


def _prism_text(footprint, orientation):
    """Right prism over a polygon in the xy-plane, extruded over z in [0, 1]."""
    count = len(footprint)
    vertices = [(x, y, 0) for x, y in footprint] + [(x, y, 1) for x, y in footprint]
    bottom = tuple(range(1, count + 1))
    top = tuple(range(count + 1, 2 * count + 1))
    sides = [
        (index + 1, (index + 1) % count + 1, (index + 1) % count + 1 + count, index + 1 + count)
        for index in range(count)
    ]
    faces = [bottom, top, *sides]
    return _tess_text(1, vertices, faces, [tuple(range(1, len(faces) + 1))], [orientation])


@pytest.fixture
def cube_tessellation(tmp_path):
    path = tmp_path / "cube.tess"
    path.write_text(_prism_text([(0, 0), (1, 0), (1, 1), (0, 1)], (0.1, 0.2, 0.3)))
    return path


@pytest.fixture
def l_prism_tessellation(tmp_path):
    """Non-convex cell: unit square footprint with the (0.5,1)x(0.5,1) corner removed."""
    path = tmp_path / "l_prism.tess"
    footprint = [(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)]
    path.write_text(_prism_text(footprint, (0.4, 0.5, 0.6)))
    return path


@pytest.fixture
def l_and_box_tessellation(tmp_path):
    """Two cells tiling the unit cube: the L prism plus the box filling its notch."""
    footprint = [(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)]
    vertices = [(x, y, 0) for x, y in footprint] + [(x, y, 1) for x, y in footprint]
    vertices += [(1, 1, 0), (1, 1, 1)]  # 13, 14
    l_sides = [(1, 2, 8, 7), (2, 3, 9, 8), (3, 4, 10, 9), (4, 5, 11, 10), (5, 6, 12, 11), (6, 1, 7, 12)]
    faces = [tuple(range(1, 7)), tuple(range(7, 13)), *l_sides]  # 1..8
    faces += [(3, 13, 5, 4), (9, 14, 11, 10), (3, 13, 14, 9), (13, 5, 11, 14)]  # 9..12
    polyhedra = [tuple(range(1, 9)), (9, 10, 11, 12, 5, 6)]
    path = tmp_path / "l_and_box.tess"
    path.write_text(_tess_text(2, vertices, faces, polyhedra, [(0.4, 0.5, 0.6), (0.7, 0.8, 0.9)]))
    return path


def test_axis_aligned_intersection_of_cube(cube_tessellation):
    tessellation = read_tessellation(cube_tessellation)
    plane = Plane.axis_aligned("x", plane_coordinate(tessellation, "x", 0.25))

    polygons = intersect_polyhedra(tessellation, plane)

    assert plane.offset == pytest.approx(0.25)
    assert len(polygons) == 1
    assert set(map(tuple, np.round(polygons[0].points, 10))) == {
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    }
    assert polygons[0].orientation == pytest.approx([0.1, 0.2, 0.3])


def test_non_convex_cell_section_is_traced_exactly(l_prism_tessellation):
    tessellation = read_tessellation(l_prism_tessellation)
    plane = Plane.axis_aligned("z", 0.3)

    (polygon,) = intersect_polyhedra(tessellation, plane)

    assert len(polygon.points) == 6
    assert _polygon_area(polygon.points) == pytest.approx(0.75)
    # Vertices follow the boundary: consecutive points share a coordinate.
    for start, end in zip(polygon.points, np.roll(polygon.points, -1, axis=0)):
        assert np.isclose(start[0], end[0]) or np.isclose(start[1], end[1])


def test_plane_through_vertices_is_shifted_and_still_valid(l_prism_tessellation):
    tessellation = read_tessellation(l_prism_tessellation)
    plane = Plane.axis_aligned("x", 0.5)  # passes through the re-entrant edge

    with pytest.raises(ValueError, match="passes through a vertex"):
        intersect_polyhedra(tessellation, plane)

    shifted_plane, shifted = avoid_vertices(tessellation, plane, tolerance=1e-10)
    (polygon,) = intersect_polyhedra(tessellation, shifted_plane)

    assert shifted
    assert shifted_plane.offset == pytest.approx(0.5, abs=1e-8)
    assert _polygon_area(polygon.points) == pytest.approx(0.5, abs=1e-8)


def test_compute_section_rasterizes_with_top_left_origin(l_and_box_tessellation):
    tessellation = read_tessellation(l_and_box_tessellation)

    section = compute_section(tessellation, axis="z", position=0.5, resolution=8)

    assert section.polygon_count == 2
    assert section.cell_ids[7, 0] == 1  # bottom-left pixel (small x, small y) is the L
    assert section.cell_ids[0, 7] == 2  # top-right pixel (large x, large y) is the box
    assert np.count_nonzero(section.cell_ids == 2) == 16
    assert section.orientations[7, 0] == pytest.approx([0.4, 0.5, 0.6])
    assert section.orientations[0, 7] == pytest.approx([0.7, 0.8, 0.9])


def test_compute_section_rejects_incomplete_tiling(l_prism_tessellation):
    """The L prism alone cannot tile its bounding square; validation must notice."""
    tessellation = read_tessellation(l_prism_tessellation)
    with pytest.raises(SectionGeometryError, match="window area"):
        compute_section(tessellation, axis="z", position=0.5, resolution=8)


def test_generate_section_writes_training_arrays(cube_tessellation, tmp_path):
    result = generate_section(
        cube_tessellation, tmp_path / "sections", axis="z", position=0.5, resolution=16
    )

    image = mpimg.imread(result.image_path)
    with np.load(result.data_path) as data:
        assert data["cell_ids"].shape == (16, 16)
        assert data["parent_cell_ids"].shape == (16, 16)
        assert data["orientation_values"].shape == (16, 16, 3)
        assert np.all(data["cell_ids"] == 1)
        assert data["orientation_values"][8, 8] == pytest.approx([0.1, 0.2, 0.3])

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert image.shape[:2] == (16, 16)
    assert metadata["format_version"] == 2
    assert metadata["section"]["plane_normal"] == [0, 0, 1]
    assert metadata["rendering"]["meaning"].startswith("deterministic debug")


@pytest.mark.parametrize("position", [0.0, 1.0, -0.1, 1.1])
def test_generate_section_rejects_boundary_positions(cube_tessellation, tmp_path, position):
    with pytest.raises(ValueError, match="strictly inside"):
        generate_section(
            cube_tessellation, tmp_path, axis="z", position=position, resolution=16
        )


@pytest.mark.skipif(not REAL_TESSELLATION.is_file(), reason="generated 2scale.tess missing")
@pytest.mark.parametrize("axis,position", [("x", 0.37), ("y", 0.63), ("z", 0.5)])
def test_generated_multiscale_tessellation_tiles_every_section(axis, position):
    """Regularized Neper cells are non-convex; the section must still tile the square."""
    tessellation = read_tessellation(REAL_TESSELLATION)

    section = compute_section(tessellation, axis=axis, position=position, resolution=128)

    assert np.count_nonzero(section.cell_ids == 0) == 0
    assert section.polygon_count > 50
    assert set(np.unique(section.parent_cell_ids)) <= set(tessellation.parent_cell_ids.values())
