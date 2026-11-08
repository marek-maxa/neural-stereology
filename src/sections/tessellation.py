"""Minimal Neper ``.tess`` reader and exact planar sections of its cells.

The section geometry does not assume convex cells or planar faces.  Neper's
``-regularization`` and the lamellar subdivision produce slightly skew faces and
non-convex polyhedra, so the section polygon of a cell is built topologically:
every face contributes the chords where the plane crosses its boundary, and the
chords are chained through the shared cell edges into closed loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Tessellation:
    """Geometry and cell attributes needed to construct virtual sections."""

    dimension: int
    crystal_symmetry: str | None
    orientation_descriptor: str | None
    vertices: dict[int, np.ndarray]
    faces: dict[int, tuple[int, ...]]
    polyhedra: dict[int, tuple[int, ...]]
    cell_ids: dict[int, int]
    orientations: dict[int, np.ndarray]
    parent_cell_ids: dict[int, int]

    @property
    def bounds(self) -> np.ndarray:
        coordinates = np.vstack(tuple(self.vertices.values()))
        return np.stack((coordinates.min(axis=0), coordinates.max(axis=0)))


@dataclass(frozen=True)
class Plane:
    """Oriented plane ``normal . x = offset`` with an orthonormal in-plane basis.

    Points are mapped to section coordinates ``(x . basis_u, x . basis_v)``.
    """

    normal: np.ndarray
    offset: float
    basis_u: np.ndarray
    basis_v: np.ndarray

    @classmethod
    def axis_aligned(cls, axis: str, coordinate: float) -> Plane:
        """Plane normal to ``axis``; the remaining axes, in order, span the section."""
        if axis not in {"x", "y", "z"}:
            raise ValueError("axis must be one of: x, y, z")
        normal_index = "xyz".index(axis)
        horizontal, vertical = [index for index in range(3) if index != normal_index]
        return cls(
            normal=np.eye(3)[normal_index],
            offset=float(coordinate),
            basis_u=np.eye(3)[horizontal],
            basis_v=np.eye(3)[vertical],
        )

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points) @ self.normal - self.offset

    def project(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(points)
        return np.column_stack((points @ self.basis_u, points @ self.basis_v))

    def shifted(self, delta: float) -> Plane:
        return Plane(self.normal, self.offset + delta, self.basis_u, self.basis_v)


@dataclass(frozen=True)
class SectionPolygon:
    """One connected piece of a cell section, in plane coordinates (CCW order)."""

    cell_id: int
    parent_cell_id: int
    orientation: np.ndarray | None
    points: np.ndarray


# --------------------------------------------------------------------------- parsing


def _find_line(lines: list[str], marker: str) -> int | None:
    return next(
        (index for index, line in enumerate(lines) if line.strip() == marker), None
    )


def _next_nonempty(lines: list[str], index: int) -> int:
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        raise ValueError("Unexpected end of Neper tessellation file.")
    return index


def _read_counted_record(
    lines: list[str], index: int, *, has_identifier: bool
) -> tuple[int | None, tuple[int, ...], int]:
    """Read ``[id] count value...`` including any continuation lines."""
    index = _next_nonempty(lines, index)
    tokens = lines[index].split()
    prefix_length = 2 if has_identifier else 1
    if len(tokens) < prefix_length:
        raise ValueError(f"Invalid counted record at line {index + 1}.")

    identifier = int(tokens[0]) if has_identifier else None
    count = int(tokens[1] if has_identifier else tokens[0])
    values = [int(value) for value in tokens[prefix_length:]]
    index += 1

    while len(values) < count:
        index = _next_nonempty(lines, index)
        values.extend(int(value) for value in lines[index].split())
        index += 1

    if len(values) != count:
        raise ValueError(f"Invalid value count in record for entity {identifier}.")
    return identifier, tuple(values), index


def _read_cell_data(
    lines: list[str], cell_count: int
) -> tuple[str | None, str | None, dict[int, int], dict[int, np.ndarray]]:
    cell_start = _find_line(lines, "**cell")
    if cell_start is None:
        raise ValueError("Neper tessellation has no **cell section.")

    next_section = next(
        (
            index
            for index in range(cell_start + 1, len(lines))
            if lines[index].strip().startswith("**")
        ),
        len(lines),
    )
    section = lines[cell_start:next_section]

    symmetry = None
    symmetry_index = _find_line(section, "*crysym")
    if symmetry_index is not None:
        symmetry = section[_next_nonempty(section, symmetry_index + 1)].strip()

    local_to_actual = {cell_id: cell_id for cell_id in range(1, cell_count + 1)}
    id_index = _find_line(section, "*id")
    if id_index is not None:
        values: list[int] = []
        cursor = id_index + 1
        while len(values) < cell_count:
            cursor = _next_nonempty(section, cursor)
            values.extend(int(value) for value in section[cursor].split())
            cursor += 1
        if len(values) != cell_count:
            raise ValueError(
                "Invalid number of cell identifiers in Neper tessellation."
            )
        local_to_actual = dict(enumerate(values, start=1))

    descriptor = None
    orientations: dict[int, np.ndarray] = {}
    orientation_index = _find_line(section, "*ori")
    if orientation_index is not None:
        cursor = _next_nonempty(section, orientation_index + 1)
        descriptor = section[cursor].strip()
        cursor += 1
        for local_id in range(1, cell_count + 1):
            cursor = _next_nonempty(section, cursor)
            values = [float(value) for value in section[cursor].split()]
            if len(values) < 3:
                raise ValueError(f"Invalid orientation for cell {local_id}.")
            orientations[local_to_actual[local_id]] = np.asarray(
                values[:3], dtype=float
            )
            cursor += 1

    return symmetry, descriptor, local_to_actual, orientations


def _read_faces(lines: list[str]) -> dict[int, tuple[int, ...]]:
    section_index = _find_line(lines, "**face")
    if section_index is None:
        raise ValueError("Neper tessellation has no **face section.")

    cursor = _next_nonempty(lines, section_index + 1)
    face_count = int(lines[cursor].split()[0])
    cursor += 1
    faces: dict[int, tuple[int, ...]] = {}

    for _ in range(face_count):
        face_id, vertex_ids, cursor = _read_counted_record(
            lines, cursor, has_identifier=True
        )
        _, _, cursor = _read_counted_record(lines, cursor, has_identifier=False)
        cursor = _next_nonempty(lines, cursor) + 1  # plane equation
        cursor = _next_nonempty(lines, cursor) + 1  # face state and representative point
        assert face_id is not None
        faces[face_id] = vertex_ids

    return faces


def _read_polyhedra(lines: list[str]) -> dict[int, tuple[int, ...]]:
    section_index = _find_line(lines, "**polyhedron")
    if section_index is None:
        return {}

    cursor = _next_nonempty(lines, section_index + 1)
    polyhedron_count = int(lines[cursor].split()[0])
    cursor += 1
    polyhedra: dict[int, tuple[int, ...]] = {}
    for _ in range(polyhedron_count):
        polyhedron_id, face_ids, cursor = _read_counted_record(
            lines, cursor, has_identifier=True
        )
        assert polyhedron_id is not None
        polyhedra[polyhedron_id] = face_ids
    return polyhedra


def _read_parent_ids(lines: list[str], cell_count: int) -> dict[int, int]:
    parents = {cell_id: cell_id for cell_id in range(1, cell_count + 1)}
    scale_index = _find_line(lines, "**scale")
    if scale_index is None:
        return parents

    cell_id_index = next(
        (
            index
            for index in range(scale_index + 1, len(lines))
            if lines[index].strip() == "*cellid"
        ),
        None,
    )
    if cell_id_index is None:
        return parents

    cursor = cell_id_index + 1
    for _ in range(cell_count):
        cursor = _next_nonempty(lines, cursor)
        values = [int(value) for value in lines[cursor].split()]
        if len(values) < 2:
            raise ValueError(f"Invalid multiscale cell mapping at line {cursor + 1}.")
        parents[values[0]] = values[1]
        cursor += 1
    return parents


def read_tessellation(path: str | Path) -> Tessellation:
    """Read the subset of Neper's scalar tessellation format used by this project."""
    tessellation_path = Path(path)
    lines = tessellation_path.read_text(encoding="utf-8").splitlines()

    general_index = _find_line(lines, "**general")
    if general_index is None:
        raise ValueError("Neper tessellation has no **general section.")
    general_values = lines[_next_nonempty(lines, general_index + 1)].split()
    dimension = int(general_values[0])

    cell_index = _find_line(lines, "**cell")
    if cell_index is None:
        raise ValueError("Neper tessellation has no **cell section.")
    cell_count = int(lines[_next_nonempty(lines, cell_index + 1)].split()[0])

    vertex_index = _find_line(lines, "**vertex")
    if vertex_index is None:
        raise ValueError("Neper tessellation has no **vertex section.")
    cursor = _next_nonempty(lines, vertex_index + 1)
    vertex_count = int(lines[cursor].split()[0])
    cursor += 1
    vertices: dict[int, np.ndarray] = {}
    for _ in range(vertex_count):
        cursor = _next_nonempty(lines, cursor)
        values = lines[cursor].split()
        if len(values) < 4:
            raise ValueError(f"Invalid vertex at line {cursor + 1}.")
        vertices[int(values[0])] = np.asarray(values[1:4], dtype=float)
        cursor += 1

    symmetry, descriptor, cell_ids, orientations = _read_cell_data(lines, cell_count)
    return Tessellation(
        dimension=dimension,
        crystal_symmetry=symmetry,
        orientation_descriptor=descriptor,
        vertices=vertices,
        faces=_read_faces(lines),
        polyhedra=_read_polyhedra(lines),
        cell_ids=cell_ids,
        orientations=orientations,
        parent_cell_ids=_read_parent_ids(lines, cell_count),
    )


# ------------------------------------------------------------------- plane helpers


def plane_coordinate(
    tessellation: Tessellation, axis: str, relative_position: float
) -> float:
    """Convert a position in ``(0, 1)`` to the tessellation's physical coordinate."""
    axis_index = "xyz".index(axis)
    lower, upper = tessellation.bounds[:, axis_index]
    return float(lower + relative_position * (upper - lower))


def projected_bounds(
    tessellation: Tessellation, axis: str
) -> tuple[float, float, float, float]:
    """Return ``(umin, umax, vmin, vmax)`` of the domain for an axis-aligned section."""
    plane = Plane.axis_aligned(axis, 0.0)
    corners = plane.project(tessellation.bounds)
    return (
        float(corners[:, 0].min()),
        float(corners[:, 0].max()),
        float(corners[:, 1].min()),
        float(corners[:, 1].max()),
    )


def avoid_vertices(
    tessellation: Tessellation, plane: Plane, tolerance: float
) -> tuple[Plane, bool]:
    """Shift ``plane`` by a negligible amount if it passes through a vertex.

    Vertices exactly in the plane would need special-cased topology (the plane
    touching an edge or a face).  A shift of a few ``tolerance`` lengths changes
    the section by an immeasurable amount and keeps the algorithm single-cased.
    """
    coordinates = np.vstack(tuple(tessellation.vertices.values()))
    shifted = False
    for _ in range(100):
        if np.abs(plane.signed_distance(coordinates)).min() > tolerance:
            return plane, shifted
        plane = plane.shifted(4.0 * tolerance)
        shifted = True
    raise ValueError("Could not move the section plane away from tessellation vertices.")


# ---------------------------------------------------------------- section geometry


def _face_chords(
    vertex_ids: tuple[int, ...],
    vertices: dict[int, np.ndarray],
    distances: dict[int, float],
    plane: Plane,
) -> list[tuple[tuple[int, int], tuple[int, int], np.ndarray, np.ndarray]]:
    """Chords of ``plane`` across one face: ``(edge_a, edge_b, point_a, point_b)``.

    Crossings are found where the signed distance changes sign along the face
    boundary.  Two crossings form one chord.  Four or more (a skew or non-convex
    face) are paired by their order along the intersection line, which is the
    even-odd rule for a polygon crossed by a line.
    """
    crossings: list[tuple[tuple[int, int], np.ndarray]] = []
    count = len(vertex_ids)
    for index in range(count):
        start, end = vertex_ids[index], vertex_ids[(index + 1) % count]
        distance_start, distance_end = distances[start], distances[end]
        if distance_start * distance_end >= 0.0:
            continue
        weight = distance_start / (distance_start - distance_end)
        point = vertices[start] + weight * (vertices[end] - vertices[start])
        crossings.append(((min(start, end), max(start, end)), point))

    if not crossings:
        return []
    if len(crossings) % 2:
        raise ValueError("Odd number of plane crossings on a face boundary.")

    if len(crossings) > 2:
        face_points = np.vstack([vertices[vertex_id] for vertex_id in vertex_ids])
        face_normal = np.linalg.svd(face_points - face_points.mean(axis=0))[2][2]
        direction = np.cross(face_normal, plane.normal)
        if np.linalg.norm(direction) < 1e-12:
            direction = plane.basis_u
        crossings.sort(key=lambda crossing: float(crossing[1] @ direction))

    return [
        (crossings[index][0], crossings[index + 1][0], crossings[index][1], crossings[index + 1][1])
        for index in range(0, len(crossings), 2)
    ]


def _chain_chords(
    chords: list[tuple[tuple[int, int], tuple[int, int], np.ndarray, np.ndarray]],
    cell_id: int,
) -> list[list[np.ndarray]]:
    """Chain chords sharing cell edges into closed loops of 3D points.

    Every crossed cell edge belongs to exactly two faces of the cell, so every
    crossing point is an endpoint of exactly two chords; following the chords
    through their shared edges therefore traces the section boundary.
    """
    by_edge: dict[tuple[int, int], list[int]] = {}
    for index, (edge_a, edge_b, _, _) in enumerate(chords):
        by_edge.setdefault(edge_a, []).append(index)
        by_edge.setdefault(edge_b, []).append(index)
    bad_edges = [edge for edge, users in by_edge.items() if len(users) != 2]
    if bad_edges:
        raise ValueError(
            f"Cell {cell_id}: section chords do not close "
            f"({len(bad_edges)} edge(s) used by != 2 faces)."
        )

    unused = set(range(len(chords)))
    loops: list[list[np.ndarray]] = []
    while unused:
        current = unused.pop()
        start_edge, next_edge, point_a, _ = chords[current]
        loop = [point_a]
        while next_edge != start_edge:
            (following,) = [index for index in by_edge[next_edge] if index != current]
            if following not in unused:
                raise ValueError(f"Cell {cell_id}: broken chord chain.")
            unused.remove(following)
            current = following
            edge_a, edge_b, chord_point_a, chord_point_b = chords[current]
            if edge_a == next_edge:
                loop.append(chord_point_a)
                next_edge = edge_b
            else:
                loop.append(chord_point_b)
                next_edge = edge_a
        loops.append(loop)
    return loops


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def intersect_polyhedra(
    tessellation: Tessellation,
    plane: Plane,
    *,
    tolerance: float = 1e-10,
) -> list[SectionPolygon]:
    """Exact section of every cell by ``plane`` (which must avoid all vertices).

    Each cell may yield several polygons if the plane cuts it into disconnected
    pieces (possible for non-convex regularized cells).  Polygons are returned in
    counter-clockwise order in plane coordinates.
    """
    if tessellation.dimension != 3 or not tessellation.polyhedra:
        raise ValueError("Sectioning requires a 3D tessellation with polyhedra.")

    distances = {
        vertex_id: float(plane.signed_distance(point))
        for vertex_id, point in tessellation.vertices.items()
    }
    if min(abs(distance) for distance in distances.values()) <= tolerance:
        raise ValueError(
            "The section plane passes through a vertex; use avoid_vertices() first."
        )

    polygons: list[SectionPolygon] = []
    for local_cell_id, signed_face_ids in tessellation.polyhedra.items():
        chords = []
        for signed_face_id in signed_face_ids:
            chords.extend(
                _face_chords(
                    tessellation.faces[abs(signed_face_id)],
                    tessellation.vertices,
                    distances,
                    plane,
                )
            )
        if not chords:
            continue

        actual_cell_id = tessellation.cell_ids.get(local_cell_id, local_cell_id)
        for loop in _chain_chords(chords, actual_cell_id):
            if len(loop) < 3:
                continue
            points = plane.project(np.vstack(loop))
            if _signed_area(points) < 0:
                points = points[::-1]
            polygons.append(
                SectionPolygon(
                    cell_id=actual_cell_id,
                    parent_cell_id=tessellation.parent_cell_ids.get(
                        actual_cell_id, actual_cell_id
                    ),
                    orientation=tessellation.orientations.get(actual_cell_id),
                    points=points,
                )
            )

    if not polygons:
        raise ValueError("The section plane does not intersect any cells.")
    return polygons
