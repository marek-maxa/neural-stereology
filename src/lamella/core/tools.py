"""
=========================================================
File: tools.py
---------------------------------------------------------
Description:
    Contains utility functions for the simulation, including 
    functions to set up the logger, read and write data files, 
    handle file I/O for the Neper software, and generate various 
    types of data required for the simulation, such as twinning 
    parameters and simulation results.
    
Author:
    Oleksandr Kornijcuk <oleksandr.kornijcuk@proton.me>

Created:
    03-02-2025

License:
    General Public License
=========================================================
"""


import os
import subprocess
import logging
import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from crystals import Crystal
from core.crystal import lattice_vec, B19p_B2_lattice_correspondence, def_gradient_stressfree, niti_twinning
from core.crystal import get_twinningdata, eu2mat
import pickle

def setup_logger(log_file=None, level=logging.INFO):
    """
    Function that sets up a logger to track all
    changes during twinning simulation.
    """
    # Create a custom logger
    logger = logging.getLogger("Twinning")
    logger.setLevel(level)
    logger.handlers.clear()

    # Create handlers (console and file, if specified)
    console_handler = logging.StreamHandler()
    handlers = [console_handler]

    if log_file:
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    # Set the logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Attach formatter to each handler
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def generate_twin_parameters(epsilon, orientation_sample, strain, logger=None):
    """
    Generates twinning parameters and returns them in a DataFrame.

    Args:
        epsilon (float): Macroscopic strain.
        orientation_sample (list): List of Euler angles for orientation samples.
        logger (logging.Logger, optional): Logger instance. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the generated twinning parameters.
    """
    logger = logger or logging.getLogger("Twinning")
    os.makedirs("./data", exist_ok=True)

    oris = [eu2mat(eu) for eu in orientation_sample]

    logger.info("Initializing lattice parameters and computing tensors.")
    parent_params, product_params = initialize_lattice_parameters()
    l_a, lr_a = calculate_lattice_tensors(parent_params)
    l_m, lr_m = calculate_lattice_tensors(product_params)

    logger.info("Loading crystal structures.")
    b2, b19p = load_crystal_structures()

    logger.info("Computing lattice correspondence and deformation gradient.")
    c_d, ci_d, c_p, ci_p = B19p_B2_lattice_correspondence(notation='Waitz')
    _, uv, _, _, r_am = def_gradient_stressfree(c_d, l_a, l_m)

    logger.info("Generating twinning systems.")
    twin_systems = generate_twinning_systems(b2, b19p, uv, l_a, lr_a, l_m, lr_m, ci_d, ci_p, r_am)

    logger.info("Extracting twinning data.")
    twinning_data = get_twinningdata_from_orientation(oris, orientation_sample, twin_systems, epsilon, logger)
    # Save twinning data for future
    fileTD = open('./data/TwinningDataAll.pckl', 'wb')
    pickle.dump(twinning_data, fileTD)
    fileTD.close()

    # Prepare rows for the DataFrame
    data = prepare_rows(twinning_data, orientation_sample,strain)

    logger.info("Twinning parameters generated successfully.")
    return pd.DataFrame(data)


def generate_mock_twin_parameters(epsilon, orientation_sample, strain, logger=None):
    """
    Generate synthetic twinning parameters for smoke testing the geometry pipeline
    without crystallographic CIF inputs.
    """
    logger = logger or logging.getLogger("Twinning")
    rng = np.random.default_rng(42)

    logger.warning(
        "Using mock twinning mode. The generated lamellae are suitable for testing the "
        "pipeline, not for scientific interpretation."
    )

    rows = []
    for gi, eu in enumerate(orientation_sample):
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)

        lamella_orientation = [
            float((eu[0] + rng.uniform(-0.15, 0.15)) % (2 * np.pi)),
            float(np.clip(eu[1] + rng.uniform(-0.1, 0.1), 0, np.pi)),
            float((eu[2] + rng.uniform(-0.15, 0.15)) % (2 * np.pi)),
        ]

        volume_fraction = float(np.clip(0.05 + 0.01 * np.sin(gi), 0.03, 0.08))
        propensity = 0.55 if gi % 4 == 0 else 0.05

        twinning_strain = np.array([
            [epsilon, 0.0, 0.0],
            [0.0, -epsilon / 2, 0.0],
            [0.0, 0.0, -epsilon / 2],
        ])

        rows.append(
            {
                "grain_id": gi,
                "n_x": float(normal[0]),
                "n_y": float(normal[1]),
                "n_z": float(normal[2]),
                "Schmid factor": propensity,
                "Twin volume fraction": volume_fraction,
                "phi1_g": eu[0],
                "PHI_g": eu[1],
                "phi2_g": eu[2],
                "phi1_l": lamella_orientation[0],
                "PHI_l": lamella_orientation[1],
                "phi2_l": lamella_orientation[2],
                "twinning_strain_xx": float(twinning_strain[0, 0]),
                "twinning_strain_yy": float(twinning_strain[1, 1]),
                "twinning_strain_zz": float(twinning_strain[2, 2]),
                "twinning_strain_xy": 0.0,
                "twinning_strain_yz": 0.0,
                "twinning_strain_zx": 0.0,
            }
        )

    return pd.DataFrame(rows)


def initialize_lattice_parameters():
    """Initializes lattice parameters for parent and product phases."""
    parent_params = {'type': 'cubic', 'a': 3.015}
    product_params = {'type': 'monoclinic', 'a': 2.889, 'b': 4.12, 'c': 4.622, 'beta': np.deg2rad(96.8)}
    return parent_params, product_params


def calculate_lattice_tensors(params):
    """Calculates lattice and reciprocal tensors."""
    lattice = np.array(lattice_vec(params)).T
    reciprocal = np.linalg.inv(lattice).T
    return lattice, reciprocal


def load_crystal_structures():
    """Load crystal structures from CIF files."""
    b2 = Crystal.from_cif(Path("./model/NiTi_pm3m.cif"))
    b19p = Crystal.from_cif(Path('./model/NiTiB19p.cif'))
    return b2, b19p


def generate_twinning_systems(b2, b19p, uv, l_a, lr_a, l_m, lr_m, ci_d, ci_p, r_am):
    """Generate twinning systems using the provided parameters."""
    b2_symops = [sym[:3, :3] for sym in b2.symmetry_operations()]
    b2_recsymops = [sym[:3, :3] for sym in b2.reciprocal_symmetry_operations()]

    twin_systems = niti_twinning(
        b2, b19p, uv, l_a, lr_a, l_m, lr_m, ci_d, ci_p, r_am,
        b2_symops, b2_recsymops, miller='no'
    )
    return twin_systems


def get_twinningdata_from_orientation(orientation_sample, euangles , twin_systems, epsilon, logger=None):
    """Extract twinning data based on orientation samples and twinning systems."""

    # Select twinning system
    twt = '114'
    phase = 'a'
    eps_tw = twin_systems[twt]['s'][0]

    ldir_css = [0, 0, 1]
    twinning_data = get_twinningdata(orientation_sample, euangles, ldir_css, twin_systems, twt, phase)

    # Calculate twin volume fraction
    twinning_data['Eps_tw'] = eps_tw
    twinning_data['twin_volfrac'] = epsilon / np.array(twinning_data['StrainLdirSymGl'])
    twinning_data['twin_volfrac'][np.where(twinning_data['twin_volfrac']>1)]=1.
    twinning_data['twin_volfrac'][np.where(twinning_data['twin_volfrac']<0)]=0.

    return twinning_data


def prepare_rows(twinning_data, orientation_sample,strain):
    """Prepares rows for DataFrame creation."""
    return [
        {
            'grain_id': gi,
            'n_x': n_css[0], 'n_y': n_css[1], 'n_z': n_css[2],
            'Schmid factor': SF,
            'Twin volume fraction': volfrac,
            'phi1_g': eu[0], 'PHI_g': eu[1], 'phi2_g': eu[2],
            'phi1_l': eul[0], 'PHI_l': eul[1], 'phi2_l': eul[2],
            'twinning_strain_xx': STR[0,0],
            'twinning_strain_yy': STR[1,1],
            'twinning_strain_zz': STR[2,2],
            'twinning_strain_xy': STR[0,1],
            'twinning_strain_yz': STR[1,2],
            'twinning_strain_zx': STR[2,0]

        }
        for gi, n_css, eu, eul, volfrac, SF,STR in zip(
            range(len(twinning_data['n_css'])),
            twinning_data['n_css'],
            orientation_sample,
            twinning_data['neweus'],
            twinning_data['twin_volfrac'],
            twinning_data['SF'],
            twinning_data[strain]
        )
    ]


def prepare_for_neper(cells, logger=None):
    """
    Prepares data for Neper and generates tessellations with lamellae.
    This involves running the Feret tool, reading results, creating auxiliary files,
    and running Neper twice to generate two tessellation files.

    Args:
        cells (list): List of `Cell` objects representing the simulation data.
        logger

    Returns:
        tuple: (small_segments, small_cells) - Processed segments and cells.
    """
    # Step 1: Run Feret code to get projection data
    run_feret_code()

    # Step 2: Read Feret data
    small_a, small_b = read_feret_data("./data/feret_small")

    # Step 3: Process the Feret data to extract small cell details
    small_cells, small_generators, small_radii, new_a, new_b = process_feret_data(small_a, small_b, cells)

    # Step 4: Write auxiliary files for Neper
    write_auxiliary_files(small_cells, small_generators, small_radii)

    # Step 5: Create the first tessellation using Neper
    create_first_tessellation(small_generators, logger)

    # Step 6: Recompute lamellae based on new Feret data
    small_segments = recompute_lamellae(small_cells, new_a, new_b)

    # Step 7: Write additional files for Neper
    write_additional_files_for_neper(small_cells, small_segments)

    # Step 8: Create the second tessellation using Neper
    create_second_tessellation(small_cells, logger)

    return small_segments, small_cells


def run_feret_code():
    """Runs the Feret code to generate Feret projections."""
    executable_path = './cpp/Feret_tiny'
    try:
        subprocess.run([executable_path, "./data/tessellation", "./data/normals"],
                       capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running Feret C++ code: {e.stderr}") from e


def read_feret_data(feret_file_path):
    """Reads Feret projection data from a file."""
    small_a, small_b = [], []
    with open(feret_file_path, 'r', encoding='utf-8') as feret_file:
        for line in feret_file:
            values = line.strip().split(" ")
            small_a.append(float(values[0]))
            small_b.append(float(values[1]))
    return small_a, small_b


def process_feret_data(small_a, small_b, cells):
    """Processes Feret data to extract small cells and their details."""
    small_cells, small_generators, small_radii, new_a, new_b = [], [], [], [], []

    for i in range(len(small_a)):
        if small_a[i] != 0 and small_b[i] != 0:
            small_generators.append(cells[i].generator)
            small_radii.append(cells[i].radius)
            small_cells.append(cells[i])
            new_a.append(small_a[i])
            new_b.append(small_b[i])

    return small_cells, small_generators, small_radii, new_a, new_b


def write_auxiliary_files(small_cells, small_generators, small_radii):
    """Write auxiliary files ('seed', 'weight', 'ori2', 'lam_pos', etc.) for Neper."""
    with open("./data/seed", "w", encoding='utf-8') as file_seed, \
         open("./data/weight", "w", encoding='utf-8') as file_weight:
        for generator, radius in zip(small_generators, small_radii):
            x, y, z = generator
            file_seed.write(f"{x} {y} {z}\n")
            file_weight.write(f"{radius}\n")

    small_ori = [cell.orientation for cell in small_cells]

    with open("./data/ori2", "w", encoding='utf-8') as file_ori2, \
         open("./data/lam_pos", "w", encoding='utf-8') as file_pos:
        for i, ori in enumerate(small_ori):
            file_pos.write(f"{i + 1} start\n")
            file_ori2.write(f"{i + 1} file(cell{i + 1},des=euler-bunge:passive)\n")

    with open("./data/ori", "w", encoding='utf-8') as file_ori_tess:
        for eu in small_ori:
            file_ori_tess.write(f"{eu[0]} {eu[1]} {eu[2]}\n")

    with open("./data/small_normals", "w", encoding='utf-8') as small_normals_file:
        for i, cell in enumerate(small_cells):
            small_normals_file.write(f"{i + 1} {cell.twinning_normal[0]} {cell.twinning_normal[1]} {cell.twinning_normal[2]}\n")


def create_first_tessellation(small_generators, logger=None):
    """Create the first tessellation using Neper."""
    command1 = [
        "neper", "-T", "-n", str(len(small_generators)), "-morphooptiini",
        "coo:file(data/seed),weight:file(data/weight)",
        "-crysym", "cubic", "-morpho", "voronoi", "-ori", "file(data/ori,des=euler-bunge:passive)",
        "-oridescriptor", "euler-bunge:passive", "-o", "small"
    ]
    try:
        subprocess.run(command1, capture_output=True, text=True, check=True)
        shutil.move('small.tess', './data/small.tess')
        logger.info("Successfully created the .tess file for the original tessellation.")
    except subprocess.CalledProcessError as e:
        logger.info("Error in creating .tess file for the original tessellation:")
        #print(e.stderr)


def recompute_lamellae(small_cells, new_a, new_b):
    """Recompute lamellae for the new Feret data."""
    small_segments = []
    for i, cell in enumerate(small_cells):
        new_seg = segment_lengths_in_subinterval(cell.lamellae, cell.a, cell.b, new_a[i], new_b[i])
        small_segments.append(new_seg)
    return small_segments


def write_additional_files_for_neper(small_cells, small_segments):
    """Write additional files for Neper, including 'cell' and 'lam_wid'."""
    for i, cell in enumerate(small_cells):
        file_name = f"./data/cell{i + 1}"
        with open(file_name, "w", encoding='utf-8') as file:
            segments = small_segments[i]
            for seg in segments:
                if seg[1] == "gap":
                    file.write(f"{cell.orientation[0] } {cell.orientation[1] } {cell.orientation[2] }\n")
                else:
                    file.write(
                        f"{cell.lamella_orientation[0] } {cell.lamella_orientation[1] } {cell.lamella_orientation[2] }\n")

    with open("./data/lam_wid", "w", encoding='utf-8') as file_wid:
        for i, cell in enumerate(small_cells):
            file_wid.write(f"{i + 1} ")
            segments = small_segments[i]
            if len(segments) == 1:
                seg = segments[0]
                file_wid.write(f"{seg[0]}:0")
            else:
                for j, seg in enumerate(segments):
                    # Skip very small widths
                    if seg[0] >= 0.001:
                        if j == len(segments) - 1:
                            file_wid.write(f"{seg[0]}")
                        else:
                            file_wid.write(f"{seg[0]}:")

            file_wid.write("\n")


def create_second_tessellation(small_cells, logger=None):
    """Create the second tessellation using Neper."""
    data_dir = os.path.join(os.getcwd(), 'data')
    original_directory = os.getcwd()
    os.chdir(data_dir)
    '''
    command2 = [
        "neper", "-T", "-n", f"{len(small_cells)}::from_morpho",
        f"file(data/small.tess)", "-morpho", "voronoi::lamellar(w=msfile(data/lam_wid),"
        f"v=msfile(data/small_normals),pos=msfile(data/lam_pos))",
        f"-ori", f"random::file(data/ori2,des=euler-bunge:passive)", "-oridescriptor", "euler-bunge:passive", "-o", "2scale"
    ]
    '''

    command2 = [
        "neper",
        "-T","-regularization","1",
        "-n", f"{len(small_cells)}::from_morpho",
        "-morphooptiini", f"file({os.path.join(data_dir, 'small.tess')})",
        "-morpho", f"voronoi::lamellar(w=msfile({os.path.join(data_dir, 'lam_wid')}),"
                   f"v=msfile({os.path.join(data_dir, 'small_normals')}),"
                   f"pos=msfile({os.path.join(data_dir, 'lam_pos')}))",
        "-ori", f"random::file({os.path.join(data_dir, 'ori2')},des=euler-bunge:passive)",
        "-oridescriptor", "euler-bunge:passive", "-o", "2scale"
    ]
    #print(command2)
    try:
        subprocess.run(command2, capture_output=True, text=True, check=True)
        logger.info("Successfully created the .tess file for the tessellation with lamellae.")
    except subprocess.CalledProcessError as e:
        logger.info("Error in creating .tess file for the tessellation with lamellae:")
        #print(e.stderr)
        #print(e.stdout)

    os.chdir(original_directory)


def segment_lengths_in_subinterval(lamellae, a, b, c, d):
    """
    Generate segment lengths between given boundaries, clipped to a subinterval (c, d).

    Args:
        lamellae (list): List of Lamella objects to be processed.
        a (float): Left boundary of the interval.
        b (float): Right boundary of the interval.
        c (float): Start of the clipping subinterval.
        d (float): End of the clipping subinterval.

    Returns:
        list: List of tuples (segment_length, label) where label is either 'gap' or 'lamella'.
    """
    boundaries = [a]
    labels = ["gap"]  # The first segment is always a gap before the first lamella

    # Add lamella boundaries and labels
    for L in lamellae:
        boundaries.append(L.center - L.width)
        labels.append("lamella")  # The lamella itself
        boundaries.append(L.center + L.width)
        labels.append("gap")  # The space after the lamella

    boundaries.append(b)

    # Generate segment lengths and clip them to (c, d)
    segment_lengths = []
    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        seg_label = labels[i]  # "lamella" or "gap"

        # Clip the segment to (c, d)
        clipped_start = max(seg_start, c)
        clipped_end = min(seg_end, d)
        clipped_length = clipped_end - clipped_start

        if clipped_length > 0:
            segment_lengths.append((clipped_length, seg_label))

    return segment_lengths


def save_simulation_results(cells, file_path, logger=None):
    """
    Save the simulation results as a JSON file.

    Args:
        cells (list): List of Cell objects representing the simulation results.
        file_path (str): Path to the output JSON file.
        logger (logging.Logger, optional): Logger for logging actions. Defaults to None.
    """
    logger = logger or logging.getLogger("Twinning")

    # Convert each Cell object to a dictionary
    cells_as_dicts = [cell_to_dict(cell) for cell in cells]

    # Write the dictionary to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(cells_as_dicts, json_file, indent=4)

    if logger:
        logger.info(f"Simulation results successfully saved to {file_path}.")
    return None


def check_precision(cells, segments, logger=None):
    result_dir = Path(os.getcwd()) / 'data'

    try:
        os.chdir(result_dir)  # Change to the result directory

        # Run Neper command
        command = ['neper', '-T', '-loadtess', '2scale.tess', '-statcell', 'vol']
        subprocess.run(command, check=True, capture_output=True, text=True)

        if logger:
            logger.info("Neper volume analysis completed successfully.")
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"Error in running Neper: {e.stderr}")
        raise RuntimeError(f"Error in running Neper: {e.stderr}")
    except Exception as e:
        if logger:
            logger.error(f"Error during Neper command execution: {e}")
        raise RuntimeError(f"Error during Neper command execution: {e}")

    volume_file = "2scale.stcell"
    vfs = [cell.volume_fraction for cell in cells]

    # Read the volume data from file
    with open(volume_file, "r", encoding="utf-8") as f:
        volumes = [float(line.strip()) for line in f.readlines()]

    quotients = []
    index = 0  # Tracks position in the volume list

    for segment in segments:
        lamella_volume = 0
        total_volume = 0

        for (length, label), vf in zip(segment, vfs):
            volume = volumes[index]  # Fetch corresponding volume
            total_volume += volume
            if label == "lamella":
                lamella_volume += volume
            index += 1  # Move to next volume entry

            # Compute quotient for this segment
            if total_volume > 0:
                quotients.append(abs((lamella_volume / total_volume) - vf))

    # Plot the histogram
    plt.hist(quotients, bins=20, edgecolor='k', alpha=0.7)
    plt.xlabel("Deviation from Target Lamellae Volume")
    plt.ylabel("Frequency")
    plt.title("Histogram of Lamellae Volume Accuracy")
    plt.savefig('dev.png')
    if logger:
        logger.info("Accuracy histogram saved as dev.png.")

    return None


def convert_ndarray_to_list(data):
    """Helper function to convert ndarray to list if necessary."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


def cell_to_dict(cell):
    """
    Converts a Cell object to a dictionary format for serialization.

    Args:
        cell (Cell): The cell object to convert.

    Returns:
        dict: Dictionary representation of the cell object.
    """
    cell_dict = {
        'cid': cell.cid,
        'generator': convert_ndarray_to_list(cell.generator),  # Convert ndarray if present
        'radius': convert_ndarray_to_list(cell.radius),  # Convert ndarray if present
        'a': cell.a,
        'b': cell.b,
        'volume_fraction': cell.volume_fraction,
        'twinning_normal': convert_ndarray_to_list(cell.twinning_normal),  # Convert ndarray if present
        'twinning_strain':cell.twinning_strain,
        'twinning_propensity': cell.twinning_propensity,
        'volume_function_approximation': convert_ndarray_to_list(cell.volume_function_approximation),  # Convert ndarray if present
        'volume': cell.volume,
        'is_inner': cell.is_inner,
        'twinning_threshold': cell.twinning_threshold,
        'min_distance_among_lamellae': cell.min_distance_among_lamellae,
        'min_distance_from_endpoints': cell.min_distance_from_endpoints,
        'min_lamellae_width': cell.min_lamellae_width,
        'max_lamellae_width': cell.max_lamellae_width,
        'growth_rates': convert_ndarray_to_list(cell.growth_rates),  # Convert ndarray if present
        'orientation': convert_ndarray_to_list(cell.orientation),  # Convert ndarray if present
        'lamella_orientation': convert_ndarray_to_list(cell.lamella_orientation),  # Convert ndarray if present
        'lamellae': [lamella_to_dict(lamella) for lamella in cell.lamellae]  # Assuming you also want to convert lamellae to dicts
    }
    return cell_dict


def lamella_to_dict(lamella):
    """
    Converts a Lamella object to a dictionary format.

    Args:
        lamella (Lamella): The lamella object to convert.

    Returns:
        dict: Dictionary representation of the lamella object.
    """
    return {
        'center': lamella.center,
        'width': lamella.width,
        'volume': lamella.volume
    }
