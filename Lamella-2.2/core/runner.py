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


def compute_twinning_parameters(macroscopic_strain, orientation_sample, strain, data_directory='./data', logger=None):
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
    # Generate twinning parameters using provided data
    df = tools.generate_twin_parameters(macroscopic_strain, orientation_sample, strain, logger)

    # Ensure the data directory exists
    os.makedirs(data_directory, exist_ok=True)

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


def perform_twinning(cells, max_lamellae_per_cell, use_simul_annealing, logger=None):

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

    # Perform the twinning operation for cells with a propensity greater than their threshold
    if use_simul_annealing:
        print('Using simulated annealing/',end='')
    else:
        print('Dynamic placement/',end='')
    for cid,cell in enumerate(cells):
        #print(f'{cell.cid}, is inner:{cell.is_inner}')
        if cell.is_inner and cell.twinning_propensity > cell.twinning_threshold:
            # Grow lamellae for inner cells with sufficient twinning propensity
            #print(max_lamellae_per_cell)
            #print('new cell',end=',')
            if use_simul_annealing:
                tol = 0.01
                N = 1000
                M = 100
                D = 0.001
                L = 0.0005
                T0 = 0.01
                alpha = 0.99

                state = sa.simulated_annealing(cell, tol, N, M, D, L, T0, alpha)

                # Save lamellae
                points = [lam.center for lam in state]
                widths = [lam.width for lam in state]
                cell.lamellae = sa.save_lamellae(cell, points, widths)
                #print(cell.number_of_lamellae(),end=',')

            else:
                cell.lamellae = solver.grow(cell, MAX_ITERATION, max_lamellae_per_cell, max_feret, TOLERANCE, FIXED_LAMELLAR_NUMBER,
                                            logger)
        else:
            # Clear lamellae for other cells
            cell.lamellae = []

    return cells

def deform_tessellation(macroscopic_strain, orientation_sample_path, max_lamellae_per_cell, min_distance_from_endpoints,
                        min_distance_among_lamellae, min_lamella_width, max_lamella_width, growth_rates,tessellation_path, inner_cells_path, target_path,param_path,kappa,us,vs,use_moving_average,strain,neighbor_matrix_path,use_simul_annealing, logger=None,):
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

    generators, radii, inner_cells = import_tessellation(tessellation_path,inner_cells_path)
    orientations = import_orientation(orientation_sample_path)
    logger.info("Imported data from input files.")

    # Step 2: Compute twinning parameters
    lamellae_orientations, volume_fractions, normals, propensity, strain = compute_twinning_parameters(macroscopic_strain,
                                                                                               orientations,strain,
                                                                                               "./data", logger)
    #print(f'volfrac:{np.where(np.array(volume_fractions)<0.)}')
    #print(f'volfrac:{volume_fractions}')
    #print(strain)
    #print('======================')
    logger.info("Computed twinning parameters based on the input.")

    # Step 3: Generate Feret projection and volume function approximation
    a, b, volume_functions = generate_feret('./data/tessellation', './data/normals', len(generators))
    logger.info("Feret projection computed using CPP code.")
    #print(inner_cells_path)
    #print(inner_cells)
    # Step 4: Initialize cells in the tessellation
    cells = initialize_cells(generators, radii, a, b, volume_fractions, normals, propensity, volume_functions,
                             orientations, lamellae_orientations, inner_cells, min_distance_from_endpoints,
                             min_distance_among_lamellae, min_lamella_width, max_lamella_width, growth_rates,strain)
    logger.info("Tessellation cells initialized.")

    # Step 5: Perform twinning on cells
    logger.info("Performing lamellar growth model computations.")
    cells = perform_twinning(cells, max_lamellae_per_cell, use_simul_annealing, logger)
    for cell in cells:
        if cell.volume_fraction<0:
            print(cell.volume_fraction)

    # Step 6: Prepare the cells for the Neper tool
    segments, small_cells = tools.prepare_for_neper(cells, logger)

    # Step 7: Copy results to the result directory
    tools.copy_data_to_result()

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
