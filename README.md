# neural-stereology

Research codebase for inverse stereology from 2D sections.

## Project Aim

The long-term goal of the project is to infer properties of a 3D polycrystalline microstructure from observations in planar sections.

The working idea is:

- define a parametric model of a random 3D microstructure with orientations and defects
- generate a large dataset of synthetic 3D microstructures with systematic parameter variation
- obtain virtual 2D images by slicing these 3D structures
- compute a feature vector from the 2D sections
- train a neural network on 2D data with known 3D model parameters
- use the trained model to fit real 2D observations to the underlying 3D model

## Current Status

The currently implemented part of the repository is the 3D structure-generation component.

## Repository Structure

- `src/lamella/` — synthetic 3D structure generation (inherited generator, run as a script)
- `src/neural_stereology/` — installable package with the new pipeline stages (currently `sections`)
- `scripts/` — thin command-line entry points
- `tests/` — fast sanity and unit tests
- `pyproject.toml` — Python environment and dependency definition
- `README.md` — project overview

## Environment

The project uses `uv` for dependency and virtual-environment management.

```bash
uv sync --python 3.12
```

## Development Checks

Install the development tools and enable the local pre-commit hooks:

```bash
uv sync --group dev --python 3.12
uv run pre-commit install
```

The hooks run critical Ruff checks and a small, fast sanity-test suite. They do not run the
full structure generation or external Neper computations. Run the same checks manually with:

```bash
uv run --group dev pytest -q
uv run pre-commit run --all-files
```

## Current Usage

The currently available workflow is run from the repository root:

Prepare these runtime inputs in `src/lamella/` or adjust the paths in `src/lamella/model/config.json`:

- `tessellation`
- `inner_cells`
- `orientation`
- `model/NiTi_pm3m.cif`
- `model/NiTiB19p.cif`

The NiTi CIF files are part of the repository. The tessellation, inner-cell list, and orientation sample
must be supplied for each run.

Then run the generator:

```bash
uv run --directory src/lamella python main.py --config ./model/config.json
```

The generator now reports progress for the main workflow stages and for active-cell lamella growth, including elapsed time and an ETA estimate.

Parallel lamella growth can be configured in `src/lamella/model/config.json`:

- `parallel_workers: 0` uses all available CPU cores
- `progress_report_interval` controls how often active-cell progress is logged
- `log_level` controls console and file verbosity

These settings can also be overridden from the command line:

```bash
uv run --directory src/lamella python main.py --config ./model/config.json --workers 8 --progress-interval 10
```

## Visualization

The generated tessellation can be visualized from:

- `src/lamella/data/2scale.tess`

Example:

```bash
neper -V src/lamella/data/2scale.tess -print img
```

For image rendering, `povray` may be required as a system dependency.

A helper script is also available:

```bash
python scripts/visualize_tess.py --tess src/lamella/data/2scale.tess --output-stem src/lamella/visualization/2scale
```

The helper can also color the final multiscale tessellation by the simulated twinning propensity
stored in `src/lamella/data/results.json`:

```bash
python scripts/visualize_tess.py \
  --tess src/lamella/data/2scale.tess \
  --output-stem src/lamella/visualization/2scale-propensity \
  --color-by propensity
```

The propensity render uses a blue-white-red color map by default and removes temporary helper files after the image is created.
Pass `--keep-temp` if you want to keep the auxiliary files for debugging or manual post-processing.

## Virtual 2D Sections

The sectioning module (`src/neural_stereology/sections/`) creates an axis-aligned cut at a relative position
inside the 3D tessellation and rasterizes it at a configurable resolution:

```bash
uv run python scripts/generate_sections.py \
  --tess src/lamella/data/2scale.tess \
  --axis z \
  --position 0.5 \
  --resolution 256
```

Sections are computed by an internal exact algorithm: every cell face contributes the chords
where the plane crosses its boundary, and the chords are chained through shared cell edges into
closed polygons. The method does not assume convex cells or planar faces, which matters because
Neper's regularization and the lamellar subdivision produce slightly non-convex cells. Each section
is validated to tile the domain exactly (polygon area and full pixel coverage); a plane that would
pass through a vertex is shifted by a negligible amount and the shift is recorded in the metadata.

Each section writes:

- a square PNG with deterministic debug colors
- a compressed NPZ containing subcell IDs, parent-cell IDs, and orientation values per pixel
- JSON metadata and vector polygons describing the exact plane

The PNG colors identify cells only. They are explicitly **not** physical EBSD/IPF colors;
the physical rendering will be added once the experimental map type, crystal symmetry, reference
direction, and color key are confirmed.

The module can also be used programmatically to cut one loaded tessellation many times:

```python
from neural_stereology.sections import read_tessellation, compute_section

tessellation = read_tessellation("src/lamella/data/2scale.tess")
section = compute_section(tessellation, axis="x", position=0.37, resolution=256)
section.cell_ids, section.parent_cell_ids, section.orientations  # numpy arrays
```

## Notes

- the repository is at an early stage
- only part of the intended full workflow is implemented so far
