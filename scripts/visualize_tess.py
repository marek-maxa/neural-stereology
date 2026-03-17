#!/usr/bin/env python3

"""
Create a visualization scene from a Neper tessellation.

The script always tries to export a POV-Ray scene via Neper.
If `povray` is available, it also attempts to render a PNG.
"""

from pathlib import Path
import argparse
import shutil
import subprocess
import sys


def run_command(command, cwd=None, allow_failure=False):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Command failed")
    return result


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
    args = parser.parse_args()

    tess_path = Path(args.tess)
    if not tess_path.exists():
        raise RuntimeError(f"Tessellation file not found: {tess_path}")

    output_stem = Path(args.output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    # Neper may still exit non-zero if PNG rendering fails, but it often writes the POV scene.
    neper_result = run_command(
        ["neper", "-V", str(tess_path), "-print", str(output_stem)],
        allow_failure=True,
    )

    pov_path = output_stem.with_suffix(".pov")
    png_path = output_stem.with_suffix(".png")

    if pov_path.exists():
        print(f"POV scene written to {pov_path}")
    else:
        stderr = neper_result.stderr.strip() or neper_result.stdout.strip()
        raise RuntimeError(f"Neper did not produce a POV file.\n{stderr}")

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
        print(f"PNG render written to {png_path}")
        return

    stderr = render_result.stderr.strip() or render_result.stdout.strip()
    print("POV scene was generated, but PNG rendering failed.")
    if stderr:
        print(stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
