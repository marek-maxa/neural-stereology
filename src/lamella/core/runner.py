"""
=========================================================
File: runner.py
---------------------------------------------------------
Description:
    Contains the main functions to run the deformation simulation.
    It handles the tessellation data import, computation of twinning 
    parameters, lamellae growth, and the overall deformation process. 
    The results are processed and prepared for output.
    
Author:
    Oleksandr Kornijcuk <oleksandr.kornijcuk@proton.me>

Created:
    03-02-2025

License:
    General Public License
=========================================================
"""


import os
import shutil
import math
import subprocess
import time
import random
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

import core.classes as classes
import core.solver as solver
import core.tools as tools
import core.sa as sa

# Constants
from core.const import (FERET_PARTITION_NUMBER, MAX_ITERATION, TOLERANCE, FIXED_LAMELLAR_NUMBER)

def import_tessellation(tessellation_path,inner_cells_path):
    """
    Imports tessellation data (generators, radii, inner cells) from pre-defined paths.

    Returns:
        tuple: generators (list), radii (list), inner_cells (list)
    """
    generators, radii, inner_cells = [], [], []
    os.makedirs('./data', exist_ok=True)

    # Copy tessellation and inner cells to the data folder
    shutil.copy2(tessellation_path, './data')

    with open(tessellation_path, 'r') as file:
        for line in file:
            this_line = line.split()
            generators.append([float(this_line[1]), float(this_line[2]), float(this_line[3])])
            radii.append(float(this_line[4]))

    # Import inner cells
    with open(inner_cells_path, 'r') as file:
        for line in file:
            inner_cells.append(int(line.split()[0]) - 1)

    return generators, radii, inner_cells


def import_orientation(orientation_sample_path):
    """
    Imports orientation data from a given file path.

    Args:
        orientation_sample_path (str): Path to the orientation sample file.

    Returns:
        list: List of orientation data.
    """
    orientations = []
    with open(orientation_sample_path, 'r', encoding='utf-8') as file:
        for line in file:
            split_line = line.split(" ")
            orientations.append([float(split_line[0]), float(split_line[1]), float(split_line[2])])
    return orientations


def compute_twinning_parameters(macroscopic_strain, orientation_sample, strain, data_directory='./data',
                                use_mock_twinning=False, mock_active_fraction=0.25,
                                mock_volume_fraction_base=0.05, mock_volume_fraction_jitter=0.01,
                                logger=None):
    """
    Computes twinning parameters (normals, volume fractions, Schmid factors) based on
    macroscopic strain and orientation data. Saves the results to text files.

    Args:
        macroscopic_strain (float): Macroscopic strain value.
        orientation_sample (list): List of orientation sample data.
        data_directory (str): Directory to save results.
        logger

    Returns:
        tuple: lamella_orientations (list), volume_fractions (list), normals (list), propensity (list)
    """
    # Ensure the data directory exists
    os.makedirs(data_directory, exist_ok=True)

    # Generate twinning parameters using provided data
    if use_mock_twinning:
        df = tools.generate_mock_twin_parameters(
            macroscopic_strain,
            orientation_sample,
            strain,
            mock_active_fraction=mock_active_fraction,
            mock_volume_fraction_base=mock_volume_fraction_base,
            mock_volume_fraction_jitter=mock_volume_fraction_jitter,
            logger=logger,
        )
    else:
        df = tools.generate_twin_parameters(macroscopic_strain, orientation_sample, strain, logger)

    # Save the generated data to text files
    save_twinning_data(df, data_directory)

    # Create lamella orientations list
    lamella_orientations = [
        [phi1_l, PHI_l, phi2_l]
        for phi1_l, PHI_l, phi2_l in zip(df['phi1_l'], df['PHI_l'], df['phi2_l'])
    ]

    return lamella_orientations, df['Twin volume fraction'].tolist(), df[['n_x', 'n_y', 'n_z']].values.tolist(), df[
        'Schmid factor'].tolist(),df[['twinning_strain_xx','twinning_strain_yy','twinning_strain_zz','twinning_strain_xy','twinning_strain_yz','twinning_strain_zx']].values.tolist()



def save_twinning_data(df, data_directory):
    """
    Saves the twinning data (normals, volume fractions, Schmid factors) to text files.

    Args:
        df (DataFrame): The DataFrame containing the twinning data.
        data_directory (str): The directory to save the files.
    """
    normals = df[['n_x', 'n_y', 'n_z']].values.tolist()
    write_to_file(normals, os.path.join(data_directory, 'normals'))

    volume_fractions = df['Twin volume fraction'].tolist()
    write_to_file(volume_fractions, os.path.join(data_directory, 'volume_fractions'))

    propensity = df['Schmid factor'].tolist()
    write_to_file(propensity, os.path.join(data_directory, 'propensities'))


def write_to_file(data, path):
    """
    Writes the provided data to a file at the specified path.

    Args:
        data (list): Data to write.
        path (str): File path to write the data to.
    """
    with open(path, 'w', encoding='utf-8') as file:
        for idx, item in enumerate(data, start=1):
            if isinstance(item, (list, tuple)):
                # If item is a list or tuple, join its elements
                file.write(f"{idx} {' '.join(map(str, item))}\n")
            else:
                # If item is not iterable (e.g., a float), just write it directly
                file.write(f"{idx} {str(item)}\n")


def generate_feret(tessellation_path, normals_path, num_of_cells):
    """
    Runs the Feret C++ code to compute the Feret projections and volume function approximations.

    Args:
        tessellation_path (str): Path to the tessellation file.
        normals_path (str): Path to the normals file.
        num_of_cells (int): Number of cells in the tessellation

    Returns:
        tuple: Feret projection endpoints (list), volume function (list)
    """
    executable_path = './cpp/Feret'
    try:
        subprocess.run([executable_path, tessellation_path, normals_path], capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running Feret C++ code: {e.stderr}") from e

    # Read the results
    a,b = read_feret_data('./data/feret')
    vol = read_volume_function('./data/volume_function', num_of_cells)

    return a, b, vol


def read_feret_data(feret_path):
    """
    Reads Feret projection data from a file.

    Args:
        feret_path (str): Path to the Feret file.

    Returns:
        tuple: Points a and b from the Feret projections.
    """
    points_a, points_b = [], []
    with open(feret_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split(" ")
            points_a.append(float(values[0]))
            points_b.append(float(values[1]))
    return points_a, points_b


def read_volume_function(volf_path,  num_of_cells):
    """
    Reads the volume function data from a file.

    Args:
        volf_path (str): Path to the volume function file.
        num_of_cells (int): Number of cells in the tessellation

    Returns:
        np.ndarray: Volume function matrix.
    """
    volume_matrix = np.zeros((num_of_cells, FERET_PARTITION_NUMBER))
    with open(volf_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            values = line.strip().split(" ")
            for j in range(FERET_PARTITION_NUMBER):
                volume_matrix[i, j] = float(values[j])
    return volume_matrix


def twinning_threshold(cells, min_loc=0.4, max_loc=0.2):
    """
    Computes and updates the twinning threshold for each cell in the tessellation
    based on its volume using the Hall-Petch function.

    Args:
        cells (list): List of `Cell` objects representing the tessellation.
        min_loc (float): Minimum threshold location for twinning.
        max_loc (float): Maximum threshold location for twinning.

    Returns:
        list: Updated list of `Cell` objects with twinning thresholds.
    """

    # Hall-Petch function for computing twinning behavior
    def f(x):
        return 1 / ((6 * x / math.pi) ** (1 / 6))

    # Compute minimum and maximum volume in the tessellation
    vmin = min(cell.volume for cell in cells)
    vmax = max(cell.volume for cell in cells)

    # Update twinning threshold for each cell
    for cell in cells:
        # Use Hall-Petch function to update the twinning threshold
        cell.twinning_threshold = min_loc + ((f(cell.volume) - f(vmin)) / (f(vmax) - f(vmin))) * (max_loc - min_loc)

    return cells


def _should_twin(cell):
    """
    Decide whether the cell should go through lamella growth.
    """
    return bool(cell.is_inner and cell.twinning_propensity > cell.twinning_threshold)


def _resolve_parallel_workers(parallel_workers):
    """
    Resolve worker count. A value of 0 means automatic CPU-core detection.
    """
    if parallel_workers is None:
        return 1

    parallel_workers = int(parallel_workers)
    if parallel_workers == 0:
        return max(1, os.cpu_count() or 1)

    return max(1, parallel_workers)


def _seed_rng_for_cell(base_seed, cell_id):
    """
    Seed Python and NumPy RNGs per cell so parallel workers do not share RNG state.
    """
    if base_seed is None:
        seed = int.from_bytes(os.urandom(8), "little") ^ (os.getpid() << 16) ^ int(cell_id)
    else:
        seed = int(base_seed) + int(cell_id)

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    return seed


def _process_cell_task(task):
    """
    Worker entry point for one cell.
    """
    index, cell, max_lamellae_per_cell, use_simul_annealing, max_feret, random_seed = task
    seed = _seed_rng_for_cell(random_seed, cell.cid)
    start_time = time.perf_counter()

    metadata = {
        "eligible": True,
        "solver": "simulated_annealing" if use_simul_annealing else "dynamic_placement",
        "seed": seed,
        "status": "started",
    }

    if use_simul_annealing:
        tol = 0.01
        N = 1000
        M = 100
        D = 0.001
        L = 0.0005
        T0 = 0.01
        alpha = 0.99

        state = sa.simulated_annealing(cell, tol, N, M, D, L, T0, alpha)
        points = [lam.center for lam in state]
        widths = [lam.width for lam in state]
        cell.lamellae = sa.save_lamellae(cell, points, widths)

        target_volume = cell.volume * cell.volume_fraction
        actual_volume = sum(lamella.volume for lamella in cell.lamellae)
        metadata.update({
            "selected_lamellae": len(cell.lamellae),
            "target_volume": float(target_volume),
            "best_relative_error": (
                float(abs(actual_volume - target_volume) / max(target_volume, np.finfo(float).eps))
                if target_volume >= 0
                else None
            ),
        })
    else:
        cell.lamellae, solver_metadata = solver.grow(
            cell,
            MAX_ITERATION,
            max_lamellae_per_cell,
            max_feret,
            TOLERANCE,
            FIXED_LAMELLAR_NUMBER,
            logger=None,
            return_metadata=True,
        )
        metadata.update(solver_metadata)

    metadata.update({
        "status": "grown" if cell.lamellae else "eligible_without_lamellae",
        "duration_seconds": float(time.perf_counter() - start_time),
        "lamellae_count": len(cell.lamellae),
    })
    cell.runtime_metadata = metadata

    return index, cell


def _log_growth_progress(logger, completed, total, elapsed, grown_cells):
    """
    Log lamella-growth progress together with an ETA estimate.
    """
    if not logger or total == 0:
        return

    processing_rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = total - completed
    eta_seconds = remaining / processing_rate if processing_rate > 0 else None
    eta_text = tools.format_duration(eta_seconds) if eta_seconds is not None else "n/a"
    logger.info(
        "Lamella growth progress: %s/%s active cells done (%.1f%%), grown=%s, elapsed=%s, ETA=%s.",
        completed,
        total,
        (completed / total) * 100,
        grown_cells,
        tools.format_duration(elapsed),
        eta_text,
    )


def _log_growth_summary(cells, logger, elapsed_seconds):
    """
    Log summary statistics to highlight where lamella growth spends time.
    """
    if not logger:
        return

    eligible_cells = [cell for cell in cells if cell.runtime_metadata.get("eligible")]
    if not eligible_cells:
        logger.info("Lamella growth finished in %s. No active cells required twinning.", tools.format_duration(elapsed_seconds))
        return

    durations = [cell.runtime_metadata["duration_seconds"] for cell in eligible_cells]
    grown_cells = [cell for cell in eligible_cells if cell.lamellae]
    slowest_cells = sorted(
        eligible_cells,
        key=lambda cell: cell.runtime_metadata.get("duration_seconds", 0.0),
        reverse=True,
    )[:5]

    logger.info(
        "Lamella growth finished in %s. Eligible cells: %s, cells with lamellae: %s, mean active-cell time: %s, median: %s, max: %s.",
        tools.format_duration(elapsed_seconds),
        len(eligible_cells),
        len(grown_cells),
        tools.format_duration(statistics.fmean(durations)),
        tools.format_duration(statistics.median(durations)),
        tools.format_duration(max(durations)),
    )
    logger.info(
        "Slowest active cells: %s",
        ", ".join(
            f"cell {cell.cid}={tools.format_duration(cell.runtime_metadata['duration_seconds'])} "
            f"({cell.runtime_metadata.get('lamellae_count', 0)} lamellae)"
            for cell in slowest_cells
        ),
    )


def perform_twinning(
    cells,
    max_lamellae_per_cell,
    use_simul_annealing,
    parallel_workers=0,
    progress_report_interval=10,
    random_seed=None,
    logger=None,
):

    """
    Performs the twinning operation on the cells in the tessellation. If a cell
    has a twinning propensity greater than its threshold, lamellae are grown for that cell.

    Args:
        cells (list): List of `Cell` objects.
        max_lamellae_per_cell (int): Maximum number of lamellae to grow per cell.
        logger

    Returns:
        list: Updated list of `Cell` objects with grown lamellae (if applicable).
    """
    # Update the twinning thresholds for all cells
    cells = twinning_threshold(cells)

    max_feret = np.max(np.array([cell.volume_fraction for cell in cells]))
    progress_report_interval = max(1, int(progress_report_interval))
    worker_count = _resolve_parallel_workers(parallel_workers)

    active_indices = []
    for index, cell in enumerate(cells):
        if _should_twin(cell):
            active_indices.append(index)
        else:
            cell.lamellae = []
            cell.runtime_metadata = {
                "eligible": False,
                "status": "skipped",
                "duration_seconds": 0.0,
                "lamellae_count": 0,
            }

    if logger:
        logger.info(
            "Starting lamella growth for %s active cells out of %s total. Solver=%s, workers=%s, progress interval=%s.",
            len(active_indices),
            len(cells),
            "simulated_annealing" if use_simul_annealing else "dynamic_placement",
            worker_count,
            progress_report_interval,
        )

    if not active_indices:
        return cells

    start_time = time.perf_counter()
    completed = 0
    grown_cells = 0

    if worker_count == 1 or len(active_indices) == 1:
        for index in active_indices:
            _, processed_cell = _process_cell_task(
                (index, cells[index], max_lamellae_per_cell, use_simul_annealing, max_feret, random_seed)
            )
            cells[index] = processed_cell
            completed += 1
            grown_cells += int(bool(processed_cell.lamellae))
            if completed % progress_report_interval == 0 or completed == len(active_indices):
                _log_growth_progress(logger, completed, len(active_indices), time.perf_counter() - start_time, grown_cells)
    else:
        tasks = [
            (index, cells[index], max_lamellae_per_cell, use_simul_annealing, max_feret, random_seed)
            for index in active_indices
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(_process_cell_task, task): task[0] for task in tasks}
            for future in as_completed(future_map):
                index, processed_cell = future.result()
                cells[index] = processed_cell
                completed += 1
                grown_cells += int(bool(processed_cell.lamellae))
                if completed % progress_report_interval == 0 or completed == len(active_indices):
                    _log_growth_progress(logger, completed, len(active_indices), time.perf_counter() - start_time, grown_cells)

    _log_growth_summary(cells, logger, time.perf_counter() - start_time)

    return cells

def deform_tessellation(macroscopic_strain, orientation_sample_path, max_lamellae_per_cell, min_distance_from_endpoints,
                        min_distance_among_lamellae, min_lamella_width, max_lamella_width, growth_rates,
                        tessellation_path, inner_cells_path, strain, use_simul_annealing,
                        parallel_workers=0, progress_report_interval=10, random_seed=None,
                        use_mock_twinning=False, mock_active_fraction=0.25,
                        mock_volume_fraction_base=0.05, mock_volume_fraction_jitter=0.01,
                        logger=None):
    """
    Main function to deform the tessellation by growing lamellae based on given parameters.

    Args:
        macroscopic_strain (float): The macroscopic strain value.
        orientation_sample_path (str): Path to the orientation sample file.
        max_lamellae_per_cell (int): Maximum number of lamellae per cell.
        min_distance_from_endpoints (float): Minimum distance between lamellae and projection endpoints.
        min_distance_among_lamellae (float): Minimum distance between lamellae in a cell.
        min_lamella_width (float): Minimum width of lamellae.
        max_lamella_width (float): Maximum width of lamellae.
        growth_rates (list): Growth rates for lamellae.
        logger (logging.Logger, optional): Logger for logging progress.

    Returns:
        list: List of `Cell` objects after performing the deformation and growing lamellae.
    """

    # Step 1: Import data from files
    workflow_start = time.perf_counter()
    step_start = time.perf_counter()
    generators, radii, inner_cells = import_tessellation(tessellation_path,inner_cells_path)
    orientations = import_orientation(orientation_sample_path)
    logger.info("Step 1/6 finished: imported input files in %s.", tools.format_duration(time.perf_counter() - step_start))

    # Step 2: Compute twinning parameters
    step_start = time.perf_counter()
    lamellae_orientations, volume_fractions, normals, propensity, strain = compute_twinning_parameters(macroscopic_strain,
                                                                                               orientations, strain,
                                                                                               "./data",
                                                                                               use_mock_twinning,
                                                                                               mock_active_fraction,
                                                                                               mock_volume_fraction_base,
                                                                                               mock_volume_fraction_jitter,
                                                                                               logger)
    #print(f'volfrac:{np.where(np.array(volume_fractions)<0.)}')
    #print(f'volfrac:{volume_fractions}')
    #print(strain)
    #print('======================')
    logger.info("Step 2/6 finished: computed twinning parameters in %s.", tools.format_duration(time.perf_counter() - step_start))

    # Step 3: Generate Feret projection and volume function approximation
    step_start = time.perf_counter()
    a, b, volume_functions = generate_feret('./data/tessellation', './data/normals', len(generators))
    logger.info("Step 3/6 finished: computed Feret projection in %s.", tools.format_duration(time.perf_counter() - step_start))
    #print(inner_cells_path)
    #print(inner_cells)
    # Step 4: Initialize cells in the tessellation
    step_start = time.perf_counter()
    cells = initialize_cells(generators, radii, a, b, volume_fractions, normals, propensity, volume_functions,
                             orientations, lamellae_orientations, inner_cells, min_distance_from_endpoints,
                             min_distance_among_lamellae, min_lamella_width, max_lamella_width, growth_rates,strain)
    logger.info("Step 4/6 finished: initialized %s cells in %s.", len(cells), tools.format_duration(time.perf_counter() - step_start))

    # Step 5: Perform twinning on cells
    logger.info("Step 5/6 started: performing lamellar growth model computations.")
    step_start = time.perf_counter()
    cells = perform_twinning(
        cells,
        max_lamellae_per_cell,
        use_simul_annealing,
        parallel_workers=parallel_workers,
        progress_report_interval=progress_report_interval,
        random_seed=random_seed,
        logger=logger,
    )
    logger.info("Step 5/6 finished in %s.", tools.format_duration(time.perf_counter() - step_start))
    for cell in cells:
        if cell.volume_fraction<0:
            print(cell.volume_fraction)

    # Step 6: Prepare the cells for the Neper tool
    step_start = time.perf_counter()
    segments, small_cells = tools.prepare_for_neper(cells, logger)
    logger.info(
        "Step 6/6 finished: prepared Neper inputs and reduced to %s small cells in %s.",
        len(small_cells),
        tools.format_duration(time.perf_counter() - step_start),
    )
    logger.info("Whole deformation workflow finished in %s.", tools.format_duration(time.perf_counter() - workflow_start))

    return small_cells, segments


def initialize_cells(generators, radii, a, b, volume_fractions, normals, propensity, volume_functions,
                     orientations, lamellae_orientations, inner_cells, min_distance_from_endpoints,
                     min_distance_among_lamellae, min_lamella_width, max_lamella_width, growth_rates,twinning_strain):
    """
    Initializes the cells for the tessellation with the provided parameters.

    Args:
        generators (list): List of generator points.
        radii (list): List of radii for the cells.
        a (list): Feret projection left endpoints.
        b (list): Feret projection right endpoints.
        volume_fractions (list): Volume fractions for twinning.
        normals (list): List of twinning normals.
        propensity (list): List of twinning propensities.
        volume_functions (list): Volume function approximations.
        orientations (list): Orientation data for the cells.
        lamellae_orientations (list): Lamellae orientation data.
        inner_cells (list): List of inner cell indices.
        min_distance_from_endpoints (float): Minimum distance from Feret projection endpoints.
        min_distance_among_lamellae (float): Minimum distance between lamellae.
        min_lamella_width (float): Minimum lamella width.
        max_lamella_width (float): Maximum lamella width.
        growth_rates (list): Growth rates for lamellae.
        twinning_strain: twinning strain

    Returns:
        list: Initialized list of `Cell` objects.
    """
    cells = []
    for i in range(len(generators)):
        cell = classes.Cell(i + 1, generators[i], radii[i], a[i], b[i], volume_fractions[i], normals[i],
                            propensity[i], volume_functions[i],{'xx':twinning_strain[i][0],'yy':twinning_strain[i][1],'zz':twinning_strain[i][2],'xy':twinning_strain[i][3],'yz':twinning_strain[i][4],'zx':twinning_strain[i][5]})

        # Update cell properties based on the input parameters
        cell.min_distance_from_endpoints = abs(cell.b - cell.a) * min_distance_from_endpoints
        cell.min_distance_among_lamellae = abs(cell.b - cell.a) * min_distance_among_lamellae
        cell.min_lamellae_width = abs(cell.b - cell.a) * min_lamella_width
        cell.max_lamellae_width = abs(cell.b - cell.a) * max_lamella_width
        cell.growth_rates = growth_rates
        cell.orientation = orientations[i]
        cell.lamella_orientation = lamellae_orientations[i]

        # Mark inner cells
        if (i) in inner_cells:
            #print(i+1)
            cell.is_inner = True

        cells.append(cell)

    return cells
