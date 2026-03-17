#!/usr/bin/env python3

"""
Generate a small synthetic input dataset for the lamella structure-generation pipeline.

This script creates:
- tessellation
- inner_cells
- orientation

The files are intended for smoke testing and development only.
"""

from pathlib import Path
import argparse
import math
import random


def sample_points(count, min_distance, rng):
    points = []
    attempts = 0

    while len(points) < count and attempts < count * 500:
        attempts += 1
        point = (
            rng.uniform(0.08, 0.92),
            rng.uniform(0.08, 0.92),
            rng.uniform(0.08, 0.92),
        )

        if all(
            math.dist(point, existing) >= min_distance
            for existing in points
        ):
            points.append(point)

    if len(points) != count:
        raise RuntimeError("Could not place enough seed points. Try fewer cells.")

    return points


def write_tessellation(path, points, rng):
    with path.open("w", encoding="utf-8") as handle:
        for idx, (x, y, z) in enumerate(points, start=1):
            radius = rng.uniform(0.06, 0.14)
            handle.write(f"{idx} {x:.6f} {y:.6f} {z:.6f} {radius:.6f}\n")


def write_inner_cells(path, count):
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(1, count + 1):
            handle.write(f"{idx}\n")


def write_orientations(path, count, rng):
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(count):
            phi1 = rng.uniform(0.0, 2 * math.pi)
            phi = rng.uniform(0.0, math.pi)
            phi2 = rng.uniform(0.0, 2 * math.pi)
            handle.write(f"{phi1:.6f} {phi:.6f} {phi2:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate mock lamella inputs.")
    parser.add_argument("--output-dir", default="src/lamella", help="Directory to write the input files into.")
    parser.add_argument("--cells", type=int, default=24, help="Number of synthetic grains.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    points = sample_points(args.cells, min_distance=0.12, rng=rng)

    write_tessellation(output_dir / "tessellation", points, rng)
    write_inner_cells(output_dir / "inner_cells", len(points))
    write_orientations(output_dir / "orientation", len(points), rng)

    print(f"Generated mock inputs in {output_dir}")


if __name__ == "__main__":
    main()
