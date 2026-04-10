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

- `src/lamella/` — current module for synthetic 3D structure generation
- `scripts/` — helper scripts for bootstrap and development workflows
- `pyproject.toml` — Python environment and dependency definition
- `README.md` — project overview

## Environment

The project uses `uv` for dependency and virtual-environment management.

```bash
uv sync --python 3.12
```

## Current Usage

The currently available workflow is run from the repository root:

Smoke-test bootstrap:

```bash
python scripts/generate_mock_inputs.py --cells 8
```

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

## Notes

- the repository is at an early stage
- only part of the intended full workflow is implemented so far
- a mock mode is available for smoke testing without material-specific CIF inputs
