#!/usr/bin/env python3

"""
Create a visualization scene from a Neper tessellation.

The script always tries to export a POV-Ray scene via Neper.
If `povray` is available, it also attempts to render a PNG.
"""

from pathlib import Path
import argparse
import json
import math
import re
import shutil
import subprocess
import sys


def run_command(command, cwd=None, allow_failure=False):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Command failed")
    return result


def _extract_cell_count(tess_path):
    lines = tess_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "**cell":
            try:
                return int(lines[index + 1].strip())
            except (IndexError, ValueError) as exc:
                raise RuntimeError(f"Could not read cell count from {tess_path}") from exc
    raise RuntimeError(f"Could not find '**cell' section in {tess_path}")


def _extract_multiscale_group_sizes(tess_path):
    lines = tess_path.read_text(encoding="utf-8").splitlines()
    lam_numbers = []
    reading_lam = False

    for line in lines:
        stripped = line.strip()
        if stripped == "*lam":
            reading_lam = True
            continue

        if not reading_lam:
            continue

        if stripped.startswith("*"):
            break

        lam_numbers.extend(int(token) for token in stripped.split())

    if not lam_numbers:
        raise RuntimeError(f"Could not read '*lam' section from {tess_path}")

    group_sizes = []
    current_size = 0
    previous = None

    for number in lam_numbers:
        if previous is not None and number <= previous:
            group_sizes.append(current_size)
            current_size = 0
        current_size += 1
        previous = number

    if current_size:
        group_sizes.append(current_size)

    return group_sizes


def _extract_multiscale_orientations(tess_path):
    lines = tess_path.read_text(encoding="utf-8").splitlines()
    reading_ori = False
    orientations = []

    for line in lines:
        stripped = line.strip()
        if stripped == "*ori":
            reading_ori = True
            continue

        if not reading_ori:
            continue

        if stripped == "euler-bunge:passive":
            continue

        if stripped.startswith("*"):
            break

        orientations.append(tuple(float(token) for token in stripped.split()))

    if not orientations:
        raise RuntimeError(f"Could not read '*ori' section from {tess_path}")

    return orientations


def _angles_close(a, b, tol=1e-6):
    return all(math.isclose(x, y, rel_tol=0.0, abs_tol=tol) for x, y in zip(a, b))


def _interpolate_rgb(color_a, color_b, weight):
    return tuple(
        round((1.0 - weight) * channel_a + weight * channel_b)
        for channel_a, channel_b in zip(color_a, color_b)
    )


def propensity_to_rgb(value, min_value=0.0, max_value=0.5):
    value = max(min_value, min(max_value, float(value)))
    midpoint = (min_value + max_value) / 2.0
    blue = (0, 0, 255)
    white = (255, 255, 255)
    red = (255, 0, 0)

    if value <= midpoint:
        weight = 0.0 if midpoint == min_value else (value - min_value) / (midpoint - min_value)
        return _interpolate_rgb(blue, white, weight)

    weight = 0.0 if max_value == midpoint else (value - midpoint) / (max_value - midpoint)
    return _interpolate_rgb(white, red, weight)


def build_propensity_data_file(tess_path, results_json_path, output_stem):
    with open(results_json_path, "r", encoding="utf-8") as file:
        cells = json.load(file)

    group_sizes = _extract_multiscale_group_sizes(tess_path)
    if len(group_sizes) != len(cells):
        raise RuntimeError(
            "Propensity source size does not match the number of parent cells in the tessellation: "
            f"{len(cells)} values for {len(group_sizes)} parent cells."
        )

    values = []
    for cell, group_size in zip(cells, group_sizes):
        values.extend([float(cell["twinning_propensity"])] * group_size)

    expected_count = _extract_cell_count(tess_path)
    if len(values) != expected_count:
        raise RuntimeError(
            "Propensity map size does not match tessellation cell count: "
            f"{len(values)} values for {expected_count} cells."
        )

    data_path = output_stem.parent / f"{output_stem.name}.propensity.dat"
    with open(data_path, "w", encoding="utf-8") as file:
        for value in values:
            file.write(f"{value:.12f}\n")

    return data_path


def build_propensity_color_file(tess_path, results_json_path, output_stem, lamella_color="160:160:160"):
    with open(results_json_path, "r", encoding="utf-8") as file:
        cells = json.load(file)

    group_sizes = _extract_multiscale_group_sizes(tess_path)
    orientations = _extract_multiscale_orientations(tess_path)

    if len(group_sizes) != len(cells):
        raise RuntimeError(
            "Color source size does not match the number of parent cells in the tessellation: "
            f"{len(cells)} values for {len(group_sizes)} parent cells."
        )

    if sum(group_sizes) != len(orientations):
        raise RuntimeError(
            "Orientation count does not match multiscale group sizes: "
            f"{len(orientations)} orientations for {sum(group_sizes)} groups."
        )

    colors = []
    offset = 0
    for cell, group_size in zip(cells, group_sizes):
        parent_orientation = tuple(float(value) for value in cell["orientation"])
        twin_orientation = tuple(float(value) for value in cell["lamella_orientation"])
        parent_color = ":".join(str(channel) for channel in propensity_to_rgb(cell["twinning_propensity"]))

        for orientation in orientations[offset:offset + group_size]:
            if _angles_close(orientation, parent_orientation):
                colors.append(parent_color)
            elif _angles_close(orientation, twin_orientation):
                colors.append(lamella_color)
            else:
                raise RuntimeError(
                    "Could not classify multiscale orientation as parent or twin: "
                    f"{orientation} for cell {cell['cid']}"
                )
        offset += group_size

    data_path = output_stem.parent / f"{output_stem.name}.colors.dat"
    with open(data_path, "w", encoding="utf-8") as file:
        for color in colors:
            file.write(f"{color}\n")

    return data_path


def find_scale_png(output_stem):
    pattern = re.compile(rf"^{re.escape(output_stem.name)}-scale.*\.png$")
    candidates = [
        path for path in output_stem.parent.iterdir()
        if path.is_file() and pattern.match(path.name)
    ]
    return sorted(candidates)[-1] if candidates else None


def try_embed_scale(output_stem):
    png_path = output_stem.with_suffix(".png")
    scale_path = find_scale_png(output_stem)
    if not png_path.exists() or scale_path is None:
        return None, False

    magick = shutil.which("magick")
    convert = shutil.which("convert")

    if magick:
        command = [magick, str(png_path), str(scale_path), "-gravity", "East", "-composite", str(png_path)]
    elif convert:
        command = [convert, str(png_path), str(scale_path), "-gravity", "East", "-composite", str(png_path)]
    else:
        return scale_path, False

    run_command(command, allow_failure=True)
    return scale_path, True


def cleanup_outputs(output_stem, keep_temp=False, keep_scale=False):
    if keep_temp:
        return

    candidates = [
        output_stem.with_suffix(".propensity.dat"),
        output_stem.with_suffix(".colors.dat"),
        output_stem.with_suffix(".pov"),
    ]
    candidates.extend(output_stem.parent.glob(f"{output_stem.name}-scale*.pov"))
    if not keep_scale:
        candidates.extend(output_stem.parent.glob(f"{output_stem.name}-scale*.png"))
    candidates.extend(output_stem.parent.glob(f"{output_stem.name}-legend*"))

    for path in candidates:
        if path.exists():
            path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Visualize a Neper tessellation.")
    parser.add_argument(
        "--tess",
        default="src/lamella/data/2scale.tess",
        help="Path to the .tess file.",
    )
    parser.add_argument(
        "--output-stem",
        default="src/lamella/visualization/2scale",
        help="Output path stem without extension.",
    )
    parser.add_argument(
        "--color-by",
        choices=("id", "propensity"),
        default="id",
        help="Color cells by their identifiers or by twinning propensity.",
    )
    parser.add_argument(
        "--results-json",
        default="src/lamella/data/results.json",
        help="Path to simulation results used to build the propensity map.",
    )
    parser.add_argument(
        "--colormap",
        default="custom(blue,white,red)",
        help="Neper colormap to use for real-valued propensity rendering.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate files such as auxiliary data and scale renders.",
    )
    parser.add_argument(
        "--lamella-color",
        default="160:160:160",
        help="Fixed RGB color used for twin lamellae when rendering propensity.",
    )
    args = parser.parse_args()

    tess_path = Path(args.tess)
    if not tess_path.exists():
        raise RuntimeError(f"Tessellation file not found: {tess_path}")

    output_stem = Path(args.output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    neper_command = ["neper", "-V", str(tess_path)]
    scale_path = None

    if args.color_by == "id":
        neper_command.extend(["-datacellcol", "id"])
    else:
        results_json_path = Path(args.results_json)
        if not results_json_path.exists():
            raise RuntimeError(f"Results file not found: {results_json_path}")

        values_path = build_propensity_data_file(tess_path, results_json_path, output_stem)
        colors_path = build_propensity_color_file(
            tess_path,
            results_json_path,
            output_stem,
            lamella_color=args.lamella_color,
        )
        neper_command.extend(["-datacellcol", f"file({colors_path})"])

        legend_stem = output_stem.parent / f"{output_stem.name}-legend"
        legend_result = run_command(
            [
                "neper", "-V", str(tess_path),
                "-showtess", "0",
                "-showscale", "1",
                "-datacellcol", f"real:file({values_path})",
                "-datacellcolscheme", args.colormap,
                "-datacellscale", "0.0:0.5",
                "-datacellscaletitle", "Propensity for twinning",
                "-print", str(legend_stem),
            ],
            allow_failure=True,
        )
        scale_candidate = find_scale_png(legend_stem)
        if scale_candidate is not None:
            target_scale = output_stem.parent / f"{output_stem.name}-scale3.png"
            if target_scale.exists():
                target_scale.unlink()
            scale_candidate.rename(target_scale)
        else:
            stderr = legend_result.stderr.strip() or legend_result.stdout.strip()
            raise RuntimeError(f"Neper did not produce a scale image for propensity rendering.\n{stderr}")

    # Neper may still exit non-zero if PNG rendering fails, but it often writes the POV scene.
    neper_result = run_command(
        neper_command + ["-print", str(output_stem)],
        allow_failure=True,
    )

    pov_path = output_stem.with_suffix(".pov")
    png_path = output_stem.with_suffix(".png")

    if not pov_path.exists() and not png_path.exists():
        stderr = neper_result.stderr.strip() or neper_result.stdout.strip()
        raise RuntimeError(f"Neper did not produce a POV scene or PNG render.\n{stderr}")

    if pov_path.exists():
        print(f"POV scene written to {pov_path}")

    if png_path.exists():
        scale_path, embedded_scale = try_embed_scale(output_stem)
        cleanup_outputs(output_stem, keep_temp=args.keep_temp, keep_scale=bool(scale_path) and not embedded_scale)
        print(f"PNG render written to {png_path}")
        if scale_path and not embedded_scale:
            print(f"Scale image written to {scale_path}")
        return

    povray = shutil.which("povray")
    if not povray:
        print("POV-Ray is not installed. PNG rendering was skipped.")
        print(f"You can still use the POV scene: {pov_path}")
        return

    render_result = run_command(
        [
            povray,
            f"+I{pov_path}",
            f"+O{png_path}",
            "+W1200",
            "+H900",
            "-D",
        ],
        allow_failure=True,
    )

    if png_path.exists():
        scale_path, embedded_scale = try_embed_scale(output_stem)
        cleanup_outputs(output_stem, keep_temp=args.keep_temp, keep_scale=bool(scale_path) and not embedded_scale)
        print(f"PNG render written to {png_path}")
        if scale_path and not embedded_scale:
            print(f"Scale image written to {scale_path}")
        return

    stderr = render_result.stderr.strip() or render_result.stdout.strip()
    print("POV scene was generated, but PNG rendering failed.")
    if stderr:
        print(stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
