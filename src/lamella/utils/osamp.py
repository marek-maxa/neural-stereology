"""
tess.py

Module that generates cell orientations based on tessellation,
sampling model, and texture parameters.
"""

import numpy as np
from scipy.stats import truncnorm

# Set random seed for reproducibility
np.random.seed(42)

# Constant for 2π
TWO_PI = 2 * np.pi



def euler_to_quaternion(phi1, phi, phi2):
    """
    Convert Euler angles to a quaternion.

    Args:
        phi1 (float): First Euler angle.
        phi (float): Second Euler angle.
        phi2 (float): Third Euler angle.

    Returns:
        list: Quaternion representation.
    """
    q0 = np.cos(phi / 2) * np.cos((phi1 + phi2) / 2)
    q1 = -np.sin(phi / 2) * np.cos((phi1 - phi2) / 2)
    q2 = -np.sin(phi / 2) * np.sin((phi1 - phi2) / 2)
    q3 = -np.cos(phi / 2) * np.sin((phi1 + phi2) / 2)

    return [q0, q1, q2, q3]


def quaternion_to_euler(q):
    """
    Compute Euler angles from a quaternion.

    Args:
        q (list): Quaternion representation.

    Returns:
        list: Euler angles corresponding to the quaternion.
    """
    q0, q1, q2, q3 = q
    chi = np.sqrt((q0 ** 2 + q3 ** 2) * (q1 ** 2 + q2 ** 2))

    if chi > 0:
        # Case 1: Regular case
        phi1 = np.arctan2(-q0 * q2 + q1 * q3, -q0 * q1 - q2 * q3)
        phi = np.arccos(q0 ** 2 + q3 ** 2 - q1 ** 2 - q2 ** 2)
        phi2 = np.arctan2(q0 * q2 + q1 * q3, -q0 * q1 + q2 * q3)

    elif q0 ** 2 + q3 ** 2 == 0:
        # Case 2: Singular case
        phi1 = np.arctan2(2 * q1 * q2, q1 ** 2 - q2 ** 2)
        phi = np.pi
        phi2 = 0

    else:
        # Case 3: Handle edge cases
        phi1 = np.arctan2(-2 * q0 * q3, q0 ** 2 - q3 ** 2)
        phi = 0
        phi2 = 0

    return [phi1, phi, phi2]


def matrix_to_euler(m):
    """
    Transform a rotation matrix into Euler angles.

    Args:
        m (np.ndarray): Rotation matrix.

    Returns:
        list: Euler angles corresponding to the rotation matrix.
    """
    C = m[2, 2]
    Phi = np.arccos(np.clip(C, -1, 1))  # Avoid potential NaNs from arccos

    if C == 1:
        phi1 = np.arccos(np.clip(m[0, 0], -1, 1))
        phi2 = 0

    elif C == -1:
        phi1 = np.arccos(np.clip(m[0, 0], -1, 1))
        phi2 = np.pi

    else:
        phi1 = np.arctan2(m[2, 0], -m[2, 1])
        phi2 = np.arctan2(m[0, 2], m[1, 2])

    return [phi1, Phi, phi2]


def euler_to_matrix(eus):
    """
    Transform Euler angles into rotation matrices.

    Args:
        eus (list): Euler angles [phi1, Phi, phi2].

    Returns:
        np.ndarray: Rotation matrix corresponding to the Euler angles.
    """
    phi1, Phi, phi2 = eus

    # Rotation matrix components
    g11 = np.cos(phi1) * np.cos(phi2) - np.sin(phi1) * np.sin(phi2) * np.cos(Phi)
    g12 = np.sin(phi1) * np.cos(phi2) + np.cos(phi1) * np.sin(phi2) * np.cos(Phi)
    g13 = np.sin(phi2) * np.sin(Phi)
    g21 = -np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(Phi)
    g22 = -np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(Phi)
    g23 = np.cos(phi2) * np.sin(Phi)
    g31 = np.sin(phi1) * np.sin(Phi)
    g32 = -np.cos(phi1) * np.sin(Phi)
    g33 = np.cos(Phi)

    matrix = np.array([[g11, g12, g13], [g21, g22, g23], [g31, g32, g33]])
    return matrix


def symmetries_matrix():
    """
    Generate rotation matrices in the octahedral symmetry group.

    Returns:
        list: List of rotation matrices.
    """
    matrices = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
        np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
        np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
        np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
        np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),
        np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
        np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
        np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
        np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
        np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
        np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]),
        np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
        np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
    ]
    return matrices


def tilt(g, v, u):
    """
    Computes tilt or orientation g (in rotation matrix representation)
    with respect to directions u and v.

    Args:
        g (np.ndarray): Rotation matrix.
        v (np.ndarray): Direction vector v.
        u (np.ndarray): Direction vector u.

    Returns:
        float: Calculated tilt.
    """
    c = -np.inf
    syms = symmetries_matrix()

    for S in syms:
        val = (v.dot(S.dot(g)).dot(u)) / (np.linalg.norm(u) * np.linalg.norm(v))
        c = max(c, val)

    return c


def preferential_sample(us, vs, kap, n):
    """
    Generates a sample of orientations based on preferential sampling.

    Args:
        us (list): List of direction vectors u.
        vs (list): List of direction vectors v.
        kap (list): List of concentration parameters.
        n (int): Number of orientations to generate.

    Returns:
        list: Sample of orientations as Euler angles.
    """
    # Normalize the input vectors
    us = [np.array(u) for u in us]
    vs = [np.array(v) for v in vs]

    # Compute the exponential constant in the probability distribution function
    const = np.exp(sum(abs(kappa) for kappa in kap))

    sample = []

    for time in range(n):
        Done = False
        while not Done:
            x1, x2, x3 = np.random.uniform(0, 1, 3)

            # Generate an orientation from the uniform distribution
            phi1 = TWO_PI * x1
            phi2 = TWO_PI * x2
            Phi = np.arccos(2 * x3 - 1)

            try_ori = [phi1, Phi, phi2]

            U = np.random.rand()

            f_star = np.exp(sum(kappa * tilt(euler_to_matrix(try_ori), v, u) for kappa, v, u in zip(kap, vs, us)))

            if U <= f_star / const:
                Done = True
                sample.append(try_ori)

    return sample


def check_fundamental_zone(phi1, phi, phi2):
    """
    Check whether (phi1, phi, phi2) is in the fundamental zone.

    Args:
        phi1 (float): First Euler angle.
        phi (float): Second Euler angle.
        phi2 (float): Third Euler angle.

    Returns:
        bool: True if in fundamental zone, False otherwise.
    """
    phi1 %= TWO_PI
    phi2 %= TWO_PI

    Phi_l = np.arccos(min(np.cos(phi2) / np.sqrt(1 + np.cos(phi2) ** 2),
                          np.sin(phi2) / np.sqrt(1 + np.sin(phi2) ** 2)))

    return (0 <= phi1 <= TWO_PI) and (Phi_l <= phi <= np.pi / 2) and (0 <= phi2 <= np.pi / 2)


def read_vector(prompt):
    """
    Function to read a 3D vector from user input.
    Returns a list of floats representing the vector.
    """
    while True:
        vector_input = input(prompt).strip().split()
        if len(vector_input) == 3:
            try:
                vector = [float(value) for value in vector_input]
                return vector
            except ValueError:
                print("Invalid input. Please enter three numerical values separated by spaces.")
        else:
            print("Invalid input. Please enter exactly three values.")


def load_matrix(file_path):
    """
    Function to load a matrix from a given file path.
    Raises IOError if loading fails.
    """
    try:
        return np.loadtxt(file_path, dtype=int)
    except IOError as e:
        print(f"Error loading matrix from '{file_path}': {e}")
        raise


def save_orientations_to_file(sample, file_path):
    """
    Function to save orientations to a specified file.
    """
    try:
        with open(file_path, 'w') as file:
            for eu in sample:
                file.write(f'{eu[0]} {eu[1]} {eu[2]}\n')
        print(f'Orientations saved in {file_path}.')
    except IOError as e:
        print(f"Error saving to file '{file_path}': {e}")


import numpy as np
import sys

def read_vector(vector_str):
    """
    Parses a 3D vector from a string input formatted as "x y z".
    """
    try:
        vector = [float(value) for value in vector_str.strip().split()]
        if len(vector) == 3:
            return vector
        else:
            raise ValueError("Vector must contain exactly three values.")
    except ValueError as e:
        print(f"Error reading vector: {e}")
        sys.exit(1)

def load_matrix(file_path):
    """
    Loads a matrix from a specified file path.
    """
    try:
        return np.loadtxt(file_path, dtype=int)
    except IOError as e:
        print(f"Error loading matrix from '{file_path}': {e}")
        sys.exit(1)

def save_orientations_to_file(sample, file_path):
    """
    Saves the orientations to a specified file.
    """
    try:
        with open(file_path, 'w') as file:
            for eu in sample:
                file.write(f'{eu[0]} {eu[1]} {eu[2]}\n')
        print(f'Orientations saved in {file_path}.')
    except IOError as e:
        print(f"Error saving to file '{file_path}': {e}")
        sys.exit(1)

def main(use_moving_average, m_file,ori_file, samplefile = None, kappa=None, u=None, v=None):
    """
    Main function to handle orientation sampling based on input arguments.
    """
    if use_moving_average:
        # Load the matrix of neighboring relations in the tessellation
        M = load_matrix(m_file)

        # Sample the orientations for kappa = 0 (uniform random sampling)
        if samplefile is None:
            sample = preferential_sample(u, v, kappa, M.shape[0])
        else:
            sample = [list(ori) for ori in np.loadtxt(samplefile)]
            #print(sample[0])
        # Store orientations in the fundamental zone
        sample_FD = []
        Syms = symmetries_matrix()  # Matrices in the octahedral group

        # Transfer to the fundamental zone
        for eu in sample:
            mat = euler_to_matrix(eu)

            # Check all equivalent representations in the quotient group
            for S in Syms:
                matS = np.dot(S, mat)
                euS = matrix_to_euler(matS)

                if check_fundamental_zone(euS[0], euS[1], euS[2]):
                    sample_FD.append(euS)
                    break  # Only keep the first valid representation

        # Assign fundamental zone Euler angles to the sample
        sample = sample_FD

        # Transform Euler angles to quaternions and normalize
        Qs = [euler_to_quaternion(s[0], s[1], s[2]) for s in sample]
        new_Qs = M.dot(Qs)
        sample_MV = [Q / np.linalg.norm(Q) for Q in new_Qs]

        # Store moving average sample in the form of Euler angles
        EU_MV = [quaternion_to_euler(item) for item in sample_MV]

        # Save to a file 'ori'
        save_orientations_to_file(EU_MV, ori_file)
    else:
        # Validate required parameters for non-moving average
        if kappa is None or u is None or v is None:
            print("Error: For non-moving average, kappa, u, and v must be specified.")
            sys.exit(1)

        # Load the matrix from file 'M'
        M = load_matrix(m_file)

        # Generate a sample for random sampling
        sample = preferential_sample(u, v, kappa, M.shape[0])

        # Save the sample to a file 'ori'
        save_orientations_to_file(sample, ori_file)
        #print(sample)
    return sample
#if __name__ == "__main__":
#    
#    use_moving_average = False
#    
#    kappa = 0
#    u = [0, 0, 1]
#    v = [1, 1, 1]
#    
#    # Run the main function with parsed arguments
#    main(use_moving_average, kappa, u, v)

