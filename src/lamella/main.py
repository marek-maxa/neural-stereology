"""
=========================================================
File: main.py
---------------------------------------------------------
Description:
    Entry point for the Lamella simulation. It handles
    loading configuration, running the simulation, and 
    saving the results. Optionally result assessment.
    
Author:
    Oleksandr Kornijcuk <oleksandr.kornijcuk@proton.me>

Created:
    03-02-2025

License:
    General Public License
=========================================================
"""


import argparse
import inspect
import logging
import os
import sys
from pathlib import Path
import core.runner as runner
import core.tools as tools
import json


def load_config(config_path):
    """
    Loads configuration from a JSON file.
    """
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise RuntimeError(f"Configuration file {config_path} not found.")
    except json.JSONDecodeError:
        raise RuntimeError(f"Error parsing the JSON configuration file: {config_path}.")


def setup_logging(log_file='simulation.log', log_level="INFO"):
    """
    Setup logger for the simulation.
    """
    return tools.setup_logger(log_file=log_file, level=tools.normalize_log_level(log_level))


def validate_required_inputs(config):
    """
    Validate that the required runtime inputs are present before the simulation starts.
    """
    required_paths = {
        "tessellation_path": config.get("tessellation_path"),
        "inner_cells_path": config.get("inner_cells_path"),
        "orientation_sample_path": config.get("orientation_sample_path"),
        "model/NiTi_pm3m.cif": "./model/NiTi_pm3m.cif",
        "model/NiTiB19p.cif": "./model/NiTiB19p.cif",
    }

    missing = []
    for label, raw_path in required_paths.items():
        if not raw_path:
            missing.append(f"{label} (not set)")
            continue

        if not Path(raw_path).exists():
            missing.append(raw_path)

    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise RuntimeError(
            "Missing required input files:\n"
            f"{missing_list}\n"
            "Lamella simulation always requires tessellation, orientation, inner-cell, "
            "and NiTi CIF inputs.\n"
            "Provide these files before running the structure-generation workflow."
        )


def run_simulation(config, strain, logger):
    """
    Run the simulation using the provided configuration and logger.
    """
    try:
        logger.info("Starting the deformation simulation.")
        config = dict(config)
        config.setdefault("strain", strain)
        config.pop("log_level", None)
        allowed_keys = set(inspect.signature(runner.deform_tessellation).parameters)
        removed_unknown_keys = sorted(key for key in config if key not in allowed_keys)
        for key in removed_unknown_keys:
            config.pop(key, None)
        if removed_unknown_keys:
            logger.warning(
                "Ignoring unsupported config keys: %s",
                ", ".join(removed_unknown_keys),
            )
        validate_required_inputs(config)
        cells = runner.deform_tessellation(**config, logger=logger)
        logger.info("Simulation completed successfully.")
        return cells
    except Exception as e:
        logger.error("Simulation failed: %s", e)
        raise


def save_results(cells, logger, output_path="./data/results.json"):
    """
    Save simulation results to a JSON file.
    """
    logger.info("Saving simulation results.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tools.save_simulation_results(cells, output_path)
    logger.info("Simulation results saved successfully.")


def assess_results(cells, segments, logger):
    """
    Assess simulation results (e.g., check precision).
    """
    logger.info("Assessing simulation results.")
    tools.check_precision(cells, segments, logger)


def main(
    config_path,
    twinning_strain='StrainSymGLTwin',
    assess_results_flag=False,
    get_results=False,
    workers=None,
    progress_interval=None,
    log_level=None,
):
    """
    Main function to orchestrate the simulation run.
    """
    # Load config
    config = load_config(config_path)

    if workers is not None:
        config["parallel_workers"] = workers
    if progress_interval is not None:
        config["progress_report_interval"] = progress_interval

    logger = setup_logging(log_level=log_level or config.get("log_level", "INFO"))
    logger.info("Loading configuration from %s.", config_path)

    # Run the simulation
    cells, segments = run_simulation(config, twinning_strain, logger)

    # Save results
    save_results(cells, logger)

    # Optionally assess the results
    if assess_results_flag:
        assess_results(cells, segments, logger)
    if get_results:
        return cells, segments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a deformation simulation.")
    parser.add_argument(
        "--config", type=str, default="./model/config.json", help="Path to the configuration JSON file."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for lamella growth. Use 0 to auto-detect CPU cores.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=None,
        help="Log progress after this many processed active cells.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Console/file log level override (e.g. INFO, WARNING, DEBUG).",
    )
    parser.add_argument(
        "--assess", action="store_true", help="Assess the results after simulation."
    )
    args = parser.parse_args()
    try:
        main(
            config_path=args.config,
            assess_results_flag=args.assess,
            workers=args.workers,
            progress_interval=args.progress_interval,
            log_level=args.log_level,
        )
    except Exception:
        sys.exit(1)
