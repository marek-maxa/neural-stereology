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
- `pyproject.toml` — Python environment and dependency definition
- `README.md` — project overview

## Environment

The project uses `uv` for dependency and virtual-environment management.

```bash
uv sync --python 3.12
```

## Current Usage

The currently available workflow is run from the repository root:

```bash
uv run --directory src/lamella python main.py --config ./model/config.json
```

## Notes

- the repository is at an early stage
- only part of the intended full workflow is implemented so far
- runtime input data are not stored in the repository
