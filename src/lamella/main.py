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
import logging
import os
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


def setup_logging(log_file='simulation.log'):
    """
    Setup logger for the simulation.
    """
    return tools.setup_logger(log_file=log_file, level=logging.ERROR)


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
        validate_required_inputs(config)
        cells = runner.deform_tessellation(**config, logger=logger)
        logger.info("Simulation completed successfully.")
        return cells
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
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


def main(config_path, twinning_strain='StrainSymGLTwin', assess_results_flag=False, get_results=False):
    """
    Main function to orchestrate the simulation run.
    """
    logger = setup_logging()

    # Load config
    logger.info("Loading configuration.")
    config = load_config(config_path)

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
        "--assess", action="store_true", help="Assess the results after simulation."
    )
    args = parser.parse_args()
    main(config_path=args.config, assess_results_flag=args.assess)
