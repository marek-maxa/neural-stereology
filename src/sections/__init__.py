"""Virtual 2D sections of generated 3D microstructures."""

from .generator import (
    GeneratedSection,
    Section,
    SectionGeometryError,
    compute_section,
    generate_section,
    write_section,
)
from .tessellation import Plane, SectionPolygon, Tessellation, read_tessellation

__all__ = [
    "GeneratedSection",
    "Plane",
    "Section",
    "SectionGeometryError",
    "SectionPolygon",
    "Tessellation",
    "compute_section",
    "generate_section",
    "read_tessellation",
    "write_section",
]
