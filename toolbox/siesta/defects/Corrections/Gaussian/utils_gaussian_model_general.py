# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################

import numpy as np

def get_charge_model_sigma(limits,
                  dimensions,
                  defect_position,
                  sigma,
                  charge,
                  gamma,
                  eta,
                  ):
    """
    For a given system charge, create a model charge distribution.
    The charge model for now is a Gaussian.
    Grid = coord grid
    TODO: Change of basis
    """
    limits = limits
    dimensions = dimensions
    #defect_position = defect_position
    defect_position = [defect_position[1],defect_position[0],defect_position[2]]
    sigma = sigma
    charge = charge
    print("DEBUG: Dimensions = {}".format(dimensions))
    print("DEBUG: Limits = {}".format(limits))
    print(f"DEBUG: Defect Position {defect_position}")

    i = np.linspace(0, limits[0], dimensions[0])
    j = np.linspace(0, limits[1], dimensions[1])
    k = np.linspace(0, limits[2], dimensions[2])
    grid = np.meshgrid(i, j, k)

    # Get the gaussian at the defect position
    g = get_gaussian_3d(grid, defect_position, sigma,charge,gamma,eta)
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
                g = g + get_gaussian_3d(grid, defect_position + image_offset, sigma=sigma,charge=charge,gamma=gamma,eta=eta)
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




def get_reciprocal_grid(rcell_matrix, cutoff):
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
    grid_max = (G_max / np.linalg.norm(rcell_matrix, axis=1)).astype(int) + 1

    # For convenience later, ensure the grid is odd valued
    for axis in range(3):
        if grid_max[axis] % 2 == 0:
            grid_max[axis] += 1

    #return orm.List(list=grid_max.tolist())
    return grid_max







#from qe_tools.constants import bohr_to_ang
#from qe_tools import CONSTANTS # bohr_to_ang


def get_interpolation(input_array, target_shape):
    """
    Interpolate an array into a larger array of size, `target_size`

    Parameters
    ----------
    array: orm.ArrayData
        Array to interpolate
    target_shape: orm.List
        The target shape to interpolate the array to

    Returns
    -------
    interpolated_array
        The calculated difference of the two potentials
    """

    from scipy.ndimage.interpolation import map_coordinates

    # Unpack
    #array = input_array.get_array(input_array.get_arraynames()[0])
    #target_shape = target_shape.get_list()
    array = input_array #.get_array(input_array.get_arraynames()[0])

    # It's a bit complicated to understand map_coordinates
    # The coordinates used to understand the data are the matrix coords of the data
    # The coords passed are the new coords you want to interpolate for
    # So, we meshgrid a new set of coords in units of the matrix coords of the data
    i = np.linspace(0, array.shape[0]-1, target_shape[0])
    j = np.linspace(0, array.shape[1]-1, target_shape[1])
    k = np.linspace(0, array.shape[2]-1, target_shape[2])

    ii,jj,kk = np.meshgrid(i,j,k)
    target_coords = np.array([ii,jj,kk])
    interp_array = map_coordinates(input=np.real(array), coordinates=target_coords)

    #interpolated_array = orm.ArrayData()
    #interpolated_array.set_array('interpolated_array', interp_array)

    return interp_array #interpolated_array


"""
Utility functions for the gaussian countercharge workchain
"""
def create_model_structure(base_structure, scale_factor):
    """
    Prepare a model structure with a give scale factor, based on a base_structure
    """
    base_cell = base_structure
    model_structure = base_cell.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2)

    return model_structure



def get_gaussian_3d(grid, position, sigma,charge,gamma,eta):
    """
    Calculate 3D Gaussian on grid
    NOTE: Minus sign at front give negative values of charge density throughout the cell
    """
    x = grid[0] - position[0]
    y = grid[1] - position[1]
    z = grid[2] - position[2]
    
    beta = sigma
    N_gamma = (8.0*np.pi*(gamma**3.0))**(-1.0)
    N_beta = (np.pi*(beta**2.0))**(-3.0/2.0)

    #w= 0.3
    #eta = 1.0 - w
    w = 1.0 - eta
    print(f"DEBUG: w is {w}")
    print(f"DEBUG: eta is {eta}")
    print(f"DEBUG: gamma is {gamma}")

    #gaussian = charge * w * N_gamma *(np.exp(-(x+y+z)/gamma)) + charge*(eta)*N_beta*(np.exp(-1.0*(x**2 + y**2 + z**2) / (beta**2))) 
    
    gaussian = charge * w * N_gamma *(np.exp(-(x+y+z)**2/gamma)) + charge*(eta)*N_beta*(1.0*np.exp(-1.0*(x**2 + y**2 + z**2) / (beta**2))) 
    #gaussian = w * N_gamma *(np.exp(-(x+y+z)**2/gamma)) + (eta)*N_beta*(np.exp(-1.0*(x**2 + y**2 + z**2) / (beta**2))) 
    #gaussian = w*N_gamma( (np.exp(-(x+y+z)**1/gamma))/2.0 - (np.exp(-(x+y+z)**1/gamma))/2.0 ) + (eta)*N_beta*(np.exp(-1.0*(x**2 + y**2 + z**2) / (beta**2))) 

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




