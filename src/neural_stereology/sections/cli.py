"""Command-line interface for virtual section generation."""

from __future__ import annotations

import argparse
import sys

from .generator import generate_section


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an axis-aligned virtual section of a Neper tessellation."
    )
    parser.add_argument("--tess", required=True, help="Input 3D Neper .tess file.")
    parser.add_argument("--axis", required=True, choices=("x", "y", "z"))
    parser.add_argument(
        "--position",
        required=True,
        type=float,
        help="Relative position inside the domain; must lie strictly inside (0, 1).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Square raster resolution in pixels (default: 256).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sections",
        help="Directory for PNG, NPZ and JSON outputs.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Optional common file name without an extension.",
    )
    parser.add_argument(
        "--color-by",
        choices=("cell-id", "parent-id"),
        default="cell-id",
        help="Debug coloring only; this is not EBSD/IPF coloring.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = generate_section(
            args.tess,
            args.output_dir,
            axis=args.axis,
            position=args.position,
            resolution=args.resolution,
            color_by=args.color_by,
            output_stem=args.output_stem,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Section generation failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Generated {result.polygon_count} polygons.\n"
        f"Image: {result.image_path}\n"
        f"Data: {result.data_path}\n"
        f"Metadata: {result.metadata_path}"
    )
    return 0
