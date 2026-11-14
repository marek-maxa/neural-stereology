# Notebooks

Exploratory work: visual checks, prototyping, inspecting samples.

- A notebook only imports `neural_stereology` and visualizes. Code that is needed twice moves into the
  package and gets a test.
- Name notebooks `NN_topic.ipynb` (e.g. `01_sections_overview.ipynb`).
- Outputs are stripped on commit by the `nbstripout` pre-commit hook; large artifacts go to `outputs/`.

Start JupyterLab from the repository root:

```bash
uv run --group dev jupyter lab
```
