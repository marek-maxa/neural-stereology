"""
=========================================================
File: classes.py
---------------------------------------------------------
Description:
    Defines the `Cell` and `Lamella` classes, which are 
    used to represent the cells and lamellae in the tessellation.
    These classes store properties such as radius, orientation, 
    and volume for each cell, as well as the lamellae formed 
    within the cells during the simulation.
    
Author:
    Oleksandr Kornijcuk <oleksandr.kornijcuk@proton.me>

Created:
    03-02-2025

License:
    General Public License
=========================================================
"""


import numpy as np


class Cell:
    """
    Represents a single cell in a tessellation structure.

    Attributes:
        cid (int): Unique identifier for the cell.
        generator (list): Coordinates of the generator point of the cell.
        radius (float): Radius of the cell in the tessellation.
        a (float): Left endpoint of the Feret projection.
        b (float): Right endpoint of the Feret projection.
        volume_fraction (float): Volume fraction for twinning.
        twinning_normal (list): Vector representing the twinning normal.
        twinning_propensity (float): Propensity for twinning in the cell.
        volume_function_approximation (list): Approximated volume function for the cell.
        volume (float): Volume of the cell.
        is_inner (bool): Flag indicating if the cell is an inner cell.
        twinning_threshold (float): Threshold for twinning.
        min_distance_among_lamellae (float): Minimum distance between lamellae.
        min_distance_from_endpoints (float): Minimum distance from Feret projection endpoints.
        min_lamellae_width (float): Minimum lamella width.
        max_lamellae_width (float): Maximum lamella width.
        growth_rates (list): Growth rates for lamellae.
        orientation (list): Orientation of the cell in Euler angles.
        lamella_orientation (list): Orientation of lamellae in Euler angles.
        lamellae (list): List of `Lamella` objects associated with this cell.
        twinning_strain: twinning_strain
    """

    def __init__(self, cid, generator, radius, a, b, volume_fraction, twinning_normal, twinning_propensity,
                 volume_function_approximation,twinning_strain):
        self.cid = cid
        self.generator = generator
        self.radius = radius
        self.a = a
        self.b = b
        self.volume_fraction = volume_fraction
        self.twinning_normal = twinning_normal
        self.twinning_strain = twinning_strain
        self.twinning_propensity = twinning_propensity
        self.volume_function_approximation = volume_function_approximation
        self.volume = volume_function_approximation[-1]  # Assuming the last entry is the volume
        self.is_inner = False
        self.twinning_threshold = 0
        self.min_distance_among_lamellae = 0
        self.min_distance_from_endpoints = 0
        self.min_lamellae_width = 0
        self.max_lamellae_width = 0
        self.growth_rates = []
        self.orientation = []
        self.lamella_orientation = []
        self.lamellae = []
        self.runtime_metadata = {}

    def number_of_lamellae(self):
        """
        Returns the number of lamellae in the cell.
        """
        return len(self.lamellae)


class Lamella:
    """
    Represents a lamella (layer) within a cell in the tessellation.

    Attributes:
        center (float): Center position of the lamella in the Feret projection.
        width (float): Width of the lamella.
        volume (float): Volume of the lamella.
    """

    def __init__(self, center, width, volume):
        self.center = center
        self.width = width
        self.volume = volume
