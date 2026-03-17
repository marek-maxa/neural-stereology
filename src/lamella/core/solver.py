"""
=========================================================
File: solver.py
---------------------------------------------------------
Description:
    Implements the computational model for lamellae growth 
    based on the deformation process. This includes functions 
    for calculating lamellae volumes, placing lamellae centers, 
    and performing twinning operations using a Poisson distribution 
    strategy. It also handles the Feret and volume approximation 
    processes for each cell.

Author:
    Oleksandr Kornijcuk <oleksandr.kornijcuk@proton.me>

Created:
    03-02-2025

License:
    General Public License
=========================================================
"""


import random
import numpy as np
from core.classes import Cell, Lamella
from scipy.stats import poisson
from core.const import MAX_ATTEMPTS_TO_PLACE_POINTS, TARGET_VOLUME_PARTITION


def truncated_poisson(lambda_, max_value, size=1):
    """
    Sample from a Poisson distribution truncated from the right at `max_value`.

    Args:
        lambda_ (float): The intensity parameter (rate) of the Poisson distribution.
        max_value (int): The maximum allowed value (inclusive).
        size (int, optional): The number of samples to generate. Defaults to 1.

    Returns:
        np.ndarray: Array of samples from the truncated Poisson distribution.
    """
    #print(f'volfrac:{lambda_}')
    if max_value < 1:
        raise ValueError("Maximum value must be an integer greater than or equal to 1.")
        
    if lambda_ <= 0:
        return np.array([0])
    
    # Compute the un-normalized probabilities up to `max_value`
    probabilities = [poisson.pmf(k, lambda_) for k in range(max_value + 1)]

    # Normalize probabilities to sum to 1
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()

    # Generate samples using the normalized probabilities
    values = np.arange(max_value + 1)
    samples = np.random.choice(values, size=size, p=probabilities)

    return samples


def generate_num_of_lamellae(cell, max_lamellae, max_feret, strategy=None):
    """
    Generate the number of lamellae for a given cell.

    Args:
        cell (Cell): The cell object.
        max_lamellae (int): Maximum number of lamellae allowed.
        strategy (callable, optional): A custom function to determine the number
                                       of lamellae. Should accept (cell, max_lamellae)
                                       and return an integer.

    Returns:
        int: The number of lamellae.
    """
    if strategy:
        return strategy(cell, max_lamellae, max_feret)
    #print('rand')
    return random.randint(1, max_lamellae)


def poisson_strategy(cell, max_lamellae, max_feret):
    """
    Poisson strategy for generating the number of lamellae.

    Args:
        cell (Cell): The cell object.
        max_lamellae (int): Maximum number of lamellae.

    Returns:
        int: Number of lamellae.
    """
    C = max_lamellae
    rate = 0
    
    if cell.volume_fraction <= 0.5:
        rate = cell.volume_fraction / max_feret
    elif cell.volume_fraction > 0.5:
        rate = 2 * (1 - cell.volume_fraction)
    #print(f'rate:{rate}')
    res = truncated_poisson(C * rate, max_lamellae - 1, 1).item()
    res = res + 1
    #print(res,end=',')
    return res


def feret_linear_interpolation(cell, x):
    """
    Linearly interpolates values at `x` in the interval [a, b].

    Args:
        cell (Cell): The cell object.
        x (float): The x-coordinate where interpolation is to be done.

    Returns:
        float: The interpolated value at x.
    """
    # Initialization of variables
    alpha = cell.a
    beta = cell.b
    interval_length = abs(beta - alpha)

    # Volume function approximation for the cell
    volume_function_approximation = cell.volume_function_approximation
    k = len(volume_function_approximation)

    # Calculate the spacing between points
    step = interval_length / (k - 1)

    # Find the segment that x belongs to
    index = int((x - alpha) / step)

    # Clamp index to valid range
    if index < 0:
        return volume_function_approximation[0]

    if index >= k - 1:
        return cell.volume

    # Get the values and positions for interpolation
    x0 = alpha + index * step
    x1 = alpha + (index + 1) * step
    y0 = volume_function_approximation[index]
    y1 = volume_function_approximation[index + 1]

    # Linear interpolation formula
    interpolated_value = y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    return interpolated_value


def volume_of_lamellae(cell, centers, widths):
    """
    Computes the volume of lamellae within the given cell.

    Args:
        cell (Cell): The cell object.
        centers (list): The list of centers of lamellae.
        widths (list): The list of widths of lamellae.

    Returns:
        float: Total volume of all lamellae.
    """
    # Initialize result
    total_volume = 0

    # Compute the volume for each lamella
    for center, width in zip(centers, widths):
        upper_bound = center + width
        lower_bound = center - width
        total_volume += feret_linear_interpolation(cell, upper_bound) - feret_linear_interpolation(cell, lower_bound)

    return total_volume


def place_lamellar_centers(cell, m, max_attempts):
    """
    Generates m points inside the cell using a Matern hard-core point process.
    If placement fails after max_attempts, falls back to equidistant points.

    Args:
        cell (Cell): The cell object.
        m (int): Number of lamellae to place.
        max_attempts (int): Maximum number of attempts to place the points.

    Returns:
        list: Sorted list of lamella centers.
    """
    # Local variable initialization
    interval_length = abs(cell.b - cell.a)

    # Check if there is sufficient space for m points
    available_space = interval_length - 2 * (cell.min_distance_from_endpoints + cell.min_lamellae_width)
    min_required_space = (m - 1) * (cell.min_distance_among_lamellae + 2 * cell.min_lamellae_width)
    if available_space < min_required_space:
        return []

    # Attempt to place points using Matern hard-core process
    points = []
    for attempt in range(max_attempts):
        new_point = random.uniform(cell.a + cell.min_distance_from_endpoints + cell.min_lamellae_width,
                                   cell.b - cell.min_distance_from_endpoints - cell.min_lamellae_width)

        # Validate the new point against existing points
        if all(abs(new_point - existing) > (cell.min_distance_among_lamellae + 2 * cell.min_lamellae_width)
               for existing in points):
            points.append(new_point)

        # Stop if we have enough points
        if len(points) == m:
            break

    # If random placement fails, fall back to equidistant points
    if len(points) < m:
        points = np.linspace(cell.a + cell.min_distance_from_endpoints + cell.min_lamellae_width,
                             cell.b - cell.min_distance_from_endpoints - cell.min_lamellae_width, m).tolist()

    return sorted(points)


def grow(cell, max_attempts, max_lamellae, max_feret, solution_tolerance, fixed_num_of_lamellae_points=False, logger=None):
    """
    Tries to place lamellae inside the cell using a dynamic placement method.

    Args:
        cell (Cell): The cell object.
        max_attempts (int): Maximum number of attempts.
        max_lamellae (int): Maximum number of lamellae.
        solution_tolerance (float): Tolerance for solution error.
        fixed_num_of_lamellae_points (bool, optional): Whether to use a fixed number of lamellae.
        logger

    Returns:
        list: List of Lamella objects.
    """
    # Compute target volume of lamellae
    target_volume = cell.volume * cell.volume_fraction
    #print(fixed_num_of_lamellae_points,end='')

    # Variables to track the best solution
    best_width = 0
    best_points = []
    best_vfs = float('inf')


    #fixed_num_of_lamellae_points=True

    #poisson_strategy=None

    # Fixed number of lamellae if required
    
    m = generate_num_of_lamellae(cell, max_lamellae, max_feret, poisson_strategy) if fixed_num_of_lamellae_points else None
    #m=random.randint(1, max_lamellae)
    #print(max_lamellae,end='-')
    #print(m,end=',')
    #print('==========================')
    #print('==========================')
    max_attempts=10000
    #print(max_attempts)
    #solution_tolerance = 0.01
    for attempt in range(1, max_attempts + 1):
        # Generate a random number of lamellae if not fixed
        if not fixed_num_of_lamellae_points:
            m = generate_num_of_lamellae(cell, max_lamellae, max_feret, poisson_strategy)
            #print(poisson_strategy)
        # Attempt to place lamellae centers
        points = place_lamellar_centers(cell, m, MAX_ATTEMPTS_TO_PLACE_POINTS)
        if not points:
            continue

        # Compute bounds for the width
        min_width = cell.min_lamellae_width
        max_width = min(points[0] - cell.a - cell.min_distance_from_endpoints,
                        cell.b - cell.min_distance_from_endpoints - points[-1], cell.max_lamellae_width)
        max_width = _compute_max_width(points, cell.growth_rates, cell.min_distance_among_lamellae, max_width)
        #print(f'target volume:{target_volume}')
        #print(f'{(volume_of_lamellae(cell, points, [min_width] * len(points)))}<={target_volume}<={volume_of_lamellae(cell, points, [max_width] * len(points))}')
        #print(m,end=',')
        # Check if valid width bounds exist
        if (volume_of_lamellae(cell, points, [min_width] * len(points)) <= target_volume <=
                volume_of_lamellae(cell, points, [max_width] * len(points))):

            # Find the optimal width
            optimal_width, vfs_closest = _find_optimal_width(cell, points, min_width, max_width, target_volume)

            # Update the best solution
            if vfs_closest < best_vfs:
                best_vfs = vfs_closest
                best_width = optimal_width
                best_points = points

            # Stop early if within tolerance
            if vfs_closest <= solution_tolerance * target_volume:
                #print(f'Lam m={m}||',end='||')
                #print('==========================')
                #print(m)
                #print('==========================')
                #print(vfs_closest,end=',')
                #print(_generate_lamellae(cell, points, optimal_width),end='||')
                return _generate_lamellae(cell, points, optimal_width)

    # Save the best solution if no exact match was found
    logger.warning("Maximum number of attempts reached.")
    #print(f'Number of lamella {m}')
    if best_points:
        #print("Maximum number of attempts reached.",end='')
        return _generate_lamellae(cell, best_points, best_width)
    else:
        #if no solution found we fix number of lamella to 1
        #print('Number of lamella forced to 1')
        for attempt in range(1, max_attempts + 1):
            # Generate a random number of lamellae if not fixed
            m=1
                #print(poisson_strategy)
            # Attempt to place lamellae centers
            points = place_lamellar_centers(cell, m, MAX_ATTEMPTS_TO_PLACE_POINTS)
            if not points:
                continue
    
            # Compute bounds for the width
            min_width = cell.min_lamellae_width
            max_width = min(points[0] - cell.a - cell.min_distance_from_endpoints,
                            cell.b - cell.min_distance_from_endpoints - points[-1], cell.max_lamellae_width)
            max_width = _compute_max_width(points, cell.growth_rates, cell.min_distance_among_lamellae, max_width)
            #print(f'target volume:{target_volume}')
            #print(f'{(volume_of_lamellae(cell, points, [min_width] * len(points)))}<={target_volume}<={volume_of_lamellae(cell, points, [max_width] * len(points))}')
            #print(m,end=',')
            # Check if valid width bounds exist
            if (volume_of_lamellae(cell, points, [min_width] * len(points)) <= target_volume <=
                    volume_of_lamellae(cell, points, [max_width] * len(points))):
    
                # Find the optimal width
                optimal_width, vfs_closest = _find_optimal_width(cell, points, min_width, max_width, target_volume)
    
                # Update the best solution
                if vfs_closest < best_vfs:
                    best_vfs = vfs_closest
                    best_width = optimal_width
                    best_points = points
    
                # Stop early if within tolerance
                if vfs_closest <= solution_tolerance * target_volume:
                    #print(f'Lam m={m}||',end='||')
                    #print('==========================')
                    #print(m)
                    #print('==========================')
                    #print(vfs_closest,end=',')
                    #print(_generate_lamellae(cell, points, optimal_width),end='||')
                    return _generate_lamellae(cell, points, optimal_width)
        #if not best_points:
        #    print(f"No lam,m={m}",end='||')
        #    print(f'{(volume_of_lamellae(cell, points, [min_width] * len(points)))}<={target_volume}<={volume_of_lamellae(cell, points, [max_width] * len(points))}')
        
        return []


def _compute_max_width(points, theta, gamma, initial_max_width):
    """
    Computes the maximum allowable width based on spacing and theta differences.

    Args:
        points (list): List of lamella centers.
        theta (list): Growth rates for lamellae.
        gamma (float): Minimum distance between lamellae.
        initial_max_width (float): Initial maximum width.

    Returns:
        float: The computed maximum width.
    """
    if len(points) < 2:
        return initial_max_width

    min_width = initial_max_width
    for j in range(1, len(points)):
        numerator = points[j] - points[j - 1] - gamma
        denominator = theta[j] + theta[j - 1]
        min_width = min(min_width, numerator / denominator)

    return min_width


def _find_optimal_width(cell, points, min_width, max_width, target_volume):
    """
    Finds the optimal lamella width that minimizes the volume fraction error.

    Args:
        cell (Cell): The cell object.
        points (list): The list of lamella centers.
        min_width (float): Minimum width.
        max_width (float): Maximum width.
        target_volume (float): Target volume of lamellae.

    Returns:
        tuple: Optimal width and volume fraction error.
    """
    widths = np.linspace(min_width, max_width, TARGET_VOLUME_PARTITION)
    volumes = [volume_of_lamellae(cell, points, [theta * w for theta in cell.growth_rates]) for w in widths]
    errors = [abs(v - target_volume) for v in volumes]

    closest_index = np.argmin(errors)

    return widths[closest_index], errors[closest_index]


def _generate_lamellae(cell, points, width):
    """
    Generates and returns the list of Lamella objects based on the given parameters.

    Args:
        cell (Cell): The cell object.
        points (list): The list of lamella centers.
        width (float): The width of the lamellae.

    Returns:
        list: List of Lamella objects.
    """
    lamellae = []
    for p, th in zip(points, cell.growth_rates):
        vol = feret_linear_interpolation(cell, p + width * th) - feret_linear_interpolation(cell, p - width * th)
        lamellae.append(Lamella(p, width * th, vol))

    return lamellae
