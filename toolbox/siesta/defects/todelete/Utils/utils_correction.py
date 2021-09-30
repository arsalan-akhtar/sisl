# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
#from __future__ import absolute_import

import numpy as np
#from aiida import orm
#from aiida.engine import calcfunction

#from qe_tools.constants import bohr_to_ang
#from qe_tools import CONSTANTS # bohr_to_ang


"""
Utility functions for the gaussian countercharge workchain
"""

def calc_correction(isolated_energy, model_energy):
    """
    Get the energy correction for each model size

    Parameters
    ----------
    isolated_energy: orm.Float
        The estimated energy of the isolated model
    *model_energies: orm.Float objects
        Any number of calculated model energies
    """
    correction_energy = isolated_energy - model_energy

    return correction_energy



def create_model_structure(base_structure, scale_factor):
    """
    Prepare a model structure with a give scale factor, based on a base_structure
    """
    base_cell = base_structure
    model_structure = base_cell.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2)

    return model_structure



def get_gaussian_3d(grid, position, sigma):
    """
    Calculate 3D Gaussian on grid
    NOTE: Minus sign at front give negative values of charge density throughout the cell
    """
    x = grid[0] - position[0]
    y = grid[1] - position[1]
    z = grid[2] - position[2]
    gaussian = -np.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2)) / (
        (2.0 * np.pi)**1.5 * np.sqrt(sigma))

    return gaussian


def get_grids (limits,dimensions):

    i = np.linspace(0, limits[0], dimensions[0])
    j = np.linspace(0, limits[1], dimensions[1])
    k = np.linspace(0, limits[2], dimensions[2])
    grid = np.meshgrid(i, j, k)

    return grid

def get_integral(data, dimensions, limits):
    """
    Get the integral of a uniformly spaced 3D data array
    """
    limits = np.array(limits)
    dimensions = np.array(dimensions)
    volume_element = np.prod(limits / dimensions)
    return np.sum(data) * volume_element

def get_charge_model_sigma_cutoff(limits,
                  dimensions,
                  defect_position,
                  sigma,
                  charge):
    """
    For a given system charge, create a model charge distribution.
    The charge model for now is a Gaussian.
    Grid = coord grid
    TODO: Change of basis
    """
    limits = limits
    dimensions = dimensions
    defect_position = defect_position
    sigma = sigma
    charge = charge
    print("utils_correction.get_charge_model DEBUG: Dimensions = {}".format(dimensions))
    print("utils_correction.get_charge_model DEBUG: Limits = {}".format(limits))

    i = np.linspace(0, limits[0], dimensions[0])
    j = np.linspace(0, limits[1], dimensions[1])
    k = np.linspace(0, limits[2], dimensions[2])
    grid = np.meshgrid(i, j, k)

    # Get the gaussian at the defect position
    #g = get_gaussian_3d(grid, defect_position, sigma)
    g = get_gaussian_3d(grid, defect_position, sigma)
    # Get the offsets
    offsets = np.zeros(3)
    for axis in range(3):
        # Capture the offset needed for an image
        if defect_position[axis] > limits[axis] / 2.0:
            offsets[axis] = -limits[axis]
        else:
            offsets[axis] = +limits[axis]

    # Apply periodic boundary conditions
    g = 0
    for dim0 in range(2):
        for dim1 in range(2):
            for dim2 in range(2):
                image_offset = [dim0, dim1, dim2] * offsets
                g = g + get_gaussian_3d(grid, defect_position + image_offset, sigma=sigma)
                #g = get_gaussian_3d_old(grid, defect_position + image_offset, sigma=sigma)

    # Scale the charge density to the desired charge
    #int_charge_density = np.trapz(np.trapz(np.trapz(g, i), j), k)
    int_charge_density = get_integral(g, dimensions, limits)
    print(
        "DEBUG: Integrated charge density (g) = {}".format(int_charge_density))
    g = g / (int_charge_density / charge)

    # Compensating jellium background
    print("DEBUG: Integrated charge density (scaled_g) = {}".format(
        get_integral(g, dimensions, limits)))

    #scaled_g = scaled_g - np.sum(scaled_g)/np.prod(scaled_g.shape)
    print("DEBUG: Integrated charge density (jellium) = {}".format(
        get_integral(g, dimensions, limits)))

    # Pack the array
    #model_charge_array = orm.ArrayData()
    #model_charge_array.set_array('model_charge', g)

    return g #model_charge_array


def get_charge_model_sigma(limits,
                  dimensions,
                  defect_position,
                  sigma,
                  charge):
    """
    For a given system charge, create a model charge distribution.
    The charge model for now is a Gaussian.
    Grid = coord grid
    TODO: Change of basis
    """
    limits = limits
    dimensions = dimensions
    defect_position = defect_position
    sigma = sigma
    charge = charge
    print("utils_correction.get_charge_model_sigma DEBUG: Dimensions = {}".format(dimensions))
    print("utils_correction.get_charge_model_sigma DEBUG: Limits = {}".format(limits))

    i = np.linspace(0, limits[0], dimensions[0])
    j = np.linspace(0, limits[1], dimensions[1])
    k = np.linspace(0, limits[2], dimensions[2])
    grid = np.meshgrid(i, j, k)

    # Get the gaussian at the defect position
    #g = get_gaussian_3d(grid, defect_position, sigma)
    g = get_gaussian_3d(grid, defect_position, sigma)
    # Get the offsets
    offsets = np.zeros(3)
    for axis in range(3):
        # Capture the offset needed for an image
        if defect_position[axis] > limits[axis] / 2.0:
            offsets[axis] = -limits[axis]
        else:
            offsets[axis] = +limits[axis]

    # Apply periodic boundary conditions
    g = 0
    for dim0 in range(2):
        for dim1 in range(2):
            for dim2 in range(2):
                image_offset = [dim0, dim1, dim2] * offsets
                g = g + get_gaussian_3d(grid, defect_position + image_offset, sigma=sigma)
                #g = get_gaussian_3d_old(grid, defect_position + image_offset, sigma=sigma)

    # Scale the charge density to the desired charge
    #int_charge_density = np.trapz(np.trapz(np.trapz(g, i), j), k)
    int_charge_density = get_integral(g, dimensions, limits)
    print("utils_correction.get_charge_model_sigma DEBUG: Integrated charge density (g) = {}".format(int_charge_density))
    g = g / (int_charge_density / charge)

    # Compensating jellium background
    print("utils_correction.get_charge_model_sigma DEBUG: Integrated charge density (scaled_g) = {}".format(
        get_integral(g, dimensions, limits)))

    #scaled_g = scaled_g - np.sum(scaled_g)/np.prod(scaled_g.shape)
    print("utils_correction.get_charge_model_sigma DEBUG: Integrated charge density (jellium) = {}".format(
        get_integral(g, dimensions, limits)))

    # Pack the array
    #model_charge_array = orm.ArrayData()
    #model_charge_array.set_array('model_charge', g)

    return g #model_charge_array


def fit_energies(dimensions_dict, energies_dict):
    """
    Fit the model energies

    Parameters
    ----------
    energies_dict : Dict (orm.StructureData : orm.Float)
        AiiDA dictionary of the form: structure : energy
    """

    from scipy.optimize import curve_fit

    def fitting_func(x, a, b, c):
        """
        Function to fit:
        E = a*Omega^(-3) + b*Omega^(-1) + c
        Where:
            Omega is the volume of the cell
            a,b,c are parameters to be fitted

        Parameters
        ----------
        x: Float
            Linear cell dimension
        a,b,c: Float
            Parameters to fit
        """
        return a * x + b * x**3 + c

    #dimensions_dict = dimensions_dict.get_dict()
    #energies_dict = energies_dict.get_dict()

    # Sort these scale factors so that they are in ascending order
    #keys_list = dimensions_dict.keys()
    #keys_lis.sort()

    linear_dim_list = []
    energy_list = []
    # Unpack the dictionaries:
    for key in sorted(dimensions_dict.keys()):
        linear_dim_list.append(dimensions_dict[key])
        energy_list.append(energies_dict[key])

    # Fit the data to the function

    fit_params = curve_fit(fitting_func, np.array(linear_dim_list),
                           np.array(energy_list))[0]

    # Get the isolated model energy at linear dimension = 0.0
    isolated_energy = fitting_func(0.0, *fit_params)

    return isolated_energy


def get_reciprocal_grid(cell_matrix, cutoff):
    """
    Prepare a reciprocal space grid to achieve a given planewave energy cutoff
    cutoff (Ry)

    Parameters
    ----------
    cell_matrix: 3x3 array
        Cell matrix of the reciprocal-space cell
    cutoff: float
        Desired kinetic energy cutoff in Rydberg

    Returns
    -------
    grid_max
        A numpy vector of grid dimensions for the given cutoff

    """

    # Radius of reciprocal space sphere containing planewaves with a given kinetic energy
    G_max = 2.0 * np.sqrt(cutoff)  # Ry

    # Get the number of G-vectors needed along each cell vector
    # Note, casting to int always rounds down so we add one
    grid_max = (G_max / np.linalg.norm(cell_matrix, axis=1)).astype(int) + 1

    # For convenience later, ensure the grid is odd valued
    for axis in range(3):
        if grid_max[axis] % 2 == 0:
            grid_max[axis] += 1

    return grid_max.tolist()

