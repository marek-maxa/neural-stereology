from typing import List
import math
from . import classes
import random
from . import solver
import numpy as np
from .classes import Lamella
from .solver import poisson_strategy


class Lam:
    def __init__(self, center, width):
        self.center = center
        self.width = width


def simulated_annealing(cell: classes.Cell, tol, N: int, M: int, D: int, L: int, T0, alpha):
    """
    Simulated annealing main loop.

    Parameters:
      cell  : cell in which we generate lamellae
      tol   : tolerance within which we accept the solution
      N     : number of temperature steps
      M     : number of neighbor trials at each temperature
      D     : perturbation magnitudes for centers
      L     : perturbation magnitudes for  widths
      T0    : initial temperature
      alpha : cooling rate for the temperature schedule

    Returns:
      A state that meets the stopping criteria.
    """
    # Initialize first state
    s = initialize_state(cell)

    # If already acceptable, return it
    if is_Acceptable(s, cell, tol):
        return s

    # The main SA loop
    for k in range(1, N + 1):
        # Compute temperature in step k
        T = temperature(k, T0, alpha)

        # Run for M times at temperature T(k)
        for _ in range(M):

            # Generate a new neighbor of s
            s_new = neighbor(s, D, L, cell)

            # No valid neighbor found; try the next trial.
            if s_new is None:
                continue

            # Compute energy for two states
            E_current = energy(s, cell)
            E_new = energy(s_new, cell)

            # Compute a sample from uniform distribution
            u = random.uniform(0, 1)

            # Compute acceptance probability
            if acceptance_probability(E_current, E_new, T) >= u:
                s = s_new

                # If acceptable, return it
                if is_Acceptable(s, cell, tol):
                    return s

    # Return the last attempt
    return s


def initialize_state(cell):
    # Generate the number of lamellae
    m = solver.generate_num_of_lamellae(
        cell,
        3,
        max(cell.volume_fraction, np.finfo(float).eps),
        poisson_strategy,
    )
    #print(m)

    # Basic parameters of a cell
    xi, eta1, gamma = cell.min_distance_from_endpoints, cell.min_lamellae_width, cell.min_distance_among_lamellae
    alpha, beta = cell.a, cell.b

    # Generate evenly spaced points
    points = np.linspace(alpha + xi + eta1, beta - xi - eta1, m).tolist()

    # Initialize with minimal possible widths
    state = []
    for p in points:
        state.append(Lam(p, eta1))

    return state


def energy(state: List[Lam], cell):
    # Target volume
    target_volume = cell.volume * cell.volume_fraction

    # Compute volume of the state
    volume = solver.volume_of_lamellae(cell, [lam.center for lam in state], [lam.width for lam in state])

    return abs(volume - target_volume)


def neighbor(state, D, L, cell, max_attempts=100):
    """
    Generate a neighboring state by perturbing each lamella center and width.
    Returns a new state if valid, otherwise None.
    """
    for _ in range(max_attempts):
        new_state = []
        for lam in state:
            new_center = lam.center + random.uniform(-D, D)
            new_width = lam.width + random.uniform(-L, L)
            new_state.append(Lam(new_center, new_width))
        # Check if the new state is valid (ALS)
        if is_ALS(new_state, cell):
            return new_state
    return None


def is_ALS(state: List[Lam], cell):
    xi, eta1, gamma = cell.min_distance_from_endpoints, cell.min_lamellae_width, cell.min_distance_among_lamellae
    eta2 = cell.max_lamellae_width
    alpha, beta = cell.a, cell.b

    # Check first condition: boundaries
    if state[0].center - state[0].width < alpha + xi:
        return False
    if state[-1].center + state[-1].width > beta - xi:
        return False

    # Check second condition: spacing between lamellae
    for i in range(len(state) - 1):
        if state[i + 1].center - state[i + 1].width - (state[i].center + state[i].width) < gamma:
            return False

    # Check third condition: width constraints
    for lam in state:
        if lam.width < eta1 or lam.width > eta2:
            return False

    return True


def acceptance_probability(E_current, E_new, T):
    if E_new < E_current:
        return 1.0
    else:
        return math.exp(-(E_new - E_current) / T)


def temperature(k, T0, alpha):
    """
    Example: Exponential cooling schedule.
    T0 is the initial temperature and alpha is the cooling rate (0 < alpha < 1).
    """
    return T0 * (alpha ** k)


def is_Acceptable(state, cell, tol):
    # Target volume minus state volume
    accuracy = energy(state, cell)

    # Target volume
    target_volume = cell.volume_fraction * cell.volume

    # Check relative tolerance
    if accuracy < tol * target_volume:
        return True
    else:
        return False


def save_lamellae(cell, points, widths):
    """
    Generates and returns the list of Lamella objects based on the given parameters.

    Args:
        cell (Cell): The cell object.
        points (list): The list of lamella centers.
        widths (list): The width of the lamellae.

    Returns:
        list: List of Lamella objects.
    """
    lamellae = []
    for p, w in zip(points, widths):
        vol = solver.feret_linear_interpolation(cell, p + w) - solver.feret_linear_interpolation(cell, p - w)
        lamellae.append(Lamella(p, w, vol))

    return lamellae
