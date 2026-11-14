# Configs

Versioned JSON configurations of pipeline runs (dataset designs, training runs, rendering settings).
Every run copies the config it used, together with the git commit hash, into its `outputs/<run>/`
directory, so each result can be traced back to its exact inputs.

Naming: `<purpose>_<variant>.json`, e.g. `dataset_pilot.json`, `train_baseline.json`.

The inherited lamella generator still reads its own `src/lamella/model/config.json`.
