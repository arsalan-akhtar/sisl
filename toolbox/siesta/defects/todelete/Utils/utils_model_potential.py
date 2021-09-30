# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal


#from qe_tools.constants import hartree_to_ev, bohr_to_ang
from qe_tools import constants # hartree_to_ev, bohr_to_ang

hartree_to_ev = constants.hartree_to_ev
bohr_to_ang = constants.bohr_to_ang


def get_cell_matrix(structure):
    """
    Get the cell matrix object

    Parameters
    ----------
    structure: Sisl StructureData
        The structure object of interest

    Returns
    -------
    cell_matrix
        3x3 cell matrix array in units of Bohr

    """
    cell_matrix = np.array(structure.cell) # / bohr_to_ang  # Angstrom to Bohr
    return cell_matrix


def get_reciprocal_cell(cell_matrix):
    """
    For a given cell_matrix, compute the reciprocal cell matrix

    Parameters
    ----------
    cell_matrix: 3x3 array
        Cell matrix of the real space cell

    Returns
    -------
    reciprocal_cell
        3x3 cell matrix array in reciprocal units
    """
    from numpy.linalg import inv
    reciprocal_cell = (2 * np.pi * inv(cell_matrix)).transpose()  # Alternative definition (2pi)

    return reciprocal_cell


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

    return orm.List(list=grid_max.tolist())    


def get_xyz_coords(cell_matrix, dimensions):
    """
    For a given array, generate an array of xyz coordinates in the cartesian basis
    """

    # Generate a grid of crystal coordinates
    i = np.linspace(0., 1., dimensions[0])
    j = np.linspace(0., 1., dimensions[1])
    k = np.linspace(0., 1., dimensions[2])
    # Generate NxN arrays of crystal coords
    iii, jjj, kkk = np.meshgrid(i, j, k, indexing='ij')
    # Flatten this to a 3xNN array
    ijk_array = np.array([iii.ravel(), jjj.ravel(), kkk.ravel()])
    # Change the crystal basis to a cartesian basis
    xyz_array = np.dot(cell_matrix.T, ijk_array)

    return xyz_array



def get_integral(data, cell_matrix):
    """
    Get the integral of a uniformly spaced 3D data array by rectangular rule.
    Works better than trapezoidal or Simpson's rule for sharpely peaked coarse grids.
    """
    a = cell_matrix[0]
    b = cell_matrix[1]
    c = cell_matrix[2]
    cell_vol = np.dot(np.cross(a, b), c)
    element_volume = cell_vol / np.prod(data.shape)
    return np.sum(data) * element_volume


def get_fft(grid):
    """
    Get the FFT of a grid
    """
    return np.fft.fftshift(np.fft.fftn(grid))


def get_inverse_fft(fft_grid):
    """
    Get the inverse FFT of an FFT grid
    """
    return np.fft.ifftn(np.fft.ifftshift(fft_grid))


def get_model_potential(cell_matrix, dimensions, charge_density, epsilon):
    """
    3D possion solver

    Parameters
    ----------
    cell_matrix: 3x3 array
        The reciprocal space cell matrix
    dimensions: vector-like
        The dimensions required for the reciprocal space grid
    charge_density:  array
        The calculated model charge density on a 3-dimensional real space grid
    epsilon: float
        The value of the dielectric constant

    Returns
    -------
    V_model_r
        The calculated model potential in real space array
    """

    dimensions = np.array(dimensions)
    #cell_matrix = cell_matrix.get_array('cell_matrix')
    #charge_density = charge_density.get_array('model_charge')
    #epsilon = epsilon.value

    # Set up a reciprocal space grid for the potential
    # Prepare coordinates in a 3D array of ijk vectors
    # (Note: Indexing is column major order, but the 4th dimension vectors remain ijk)
    dimensions = dimensions // 2  #floor division

    #ijk_array = np.mgrid[
    #    -dimensions[0]:dimensions[0] + 1, 
    #    -dimensions[1]:dimensions[1] + 1, 
    #    -dimensions[2]:dimensions[2] + 1].T
    
    ijk_array = np.mgrid[
        -dimensions[0]:dimensions[0] , 
        -dimensions[1]:dimensions[1] , 
        -dimensions[2]:dimensions[2] ].T

    # Get G vectors
    G_array = np.dot(ijk_array, (cell_matrix.T))

    # Calculate the square modulus
    G_sqmod_array = (np.linalg.norm(G_array, axis=3)**2).T

    # Get the reciprocal space charge density
    charge_density_g = get_fft(charge_density)

    # Compute the model potential
    v_model = np.divide(
        charge_density_g, G_sqmod_array, where=G_sqmod_array != 0.0)
    V_model_g = v_model * 4. * np.pi / epsilon

    V_model_g[dimensions[0] + 1, dimensions[1] + 1, dimensions[2] + 1] = 0.0

    # Get the model potential in real space
    V_model_r = get_inverse_fft(V_model_g)

    # Pack up the array
    #V_model_array = orm.ArrayData()
    #V_model_array.set_array('model_potential', V_model_r)

    return V_model_r#V_model_array




def get_energy(potential, charge_density, cell_matrix):
    """
    Calculate the total energy for a given model potential
    """
    hartree_to_ev = 27.2114
    energy = np.real(0.5 * get_integral(charge_density*potential, cell_matrix) * hartree_to_ev)
    #energy = np.real(0.5 * get_integral(charge_density*potential, cell_matrix))
    return (energy)




def get_model_potential_cutoff(cell_matrix, dimensions, charge_density, epsilon):
    """
    3D possion solver

    Parameters
    ----------
    cell_matrix: 3x3 array
        The reciprocal space cell matrix
    dimensions: vector-like
        The dimensions required for the reciprocal space grid
    charge_density:  array
        The calculated model charge density on a 3-dimensional real space grid
    epsilon: float
        The value of the dielectric constant

    Returns
    -------
    V_model_r
        The calculated model potential in real space array
    """

    dimensions = np.array(dimensions)
    #cell_matrix = cell_matrix.get_array('cell_matrix')
    #charge_density = charge_density.get_array('model_charge')
    #epsilon = epsilon.value

    # Set up a reciprocal space grid for the potential
    # Prepare coordinates in a 3D array of ijk vectors
    # (Note: Indexing is column major order, but the 4th dimension vectors remain ijk)
    dimensions = dimensions // 2  #floor division
    print(dimensions)
    #Even Odd Program using Modulus Operator.
    #a=int(dimensions[0]);
    #if(a%2==0):
    #    print("This Number is Even")
    #    ijk_array = np.mgrid[
    #        -dimensions[0]:dimensions[0] +1 , 
    #        -dimensions[1]:dimensions[1] +1 , 
    #        -dimensions[2]:dimensions[2] +1].T
    #else:
    #    print("This Number is Odd")
    #    ijk_array = np.mgrid[
    #           -dimensions[0]:dimensions[0] , 
    #           -dimensions[1]:dimensions[1] , 
    #           -dimensions[2]:dimensions[2] ].T

    ijk_array = np.mgrid[
            -dimensions[0]:dimensions[0]  , 
            -dimensions[1]:dimensions[1]  , 
            -dimensions[2]:dimensions[2] ].T



   
    # Get G vectors
    G_array = np.dot(ijk_array, (cell_matrix.T))

    # Calculate the square modulus
    G_sqmod_array = (np.linalg.norm(G_array, axis=3)**2).T

    # Get the reciprocal space charge density
    charge_density_g = get_fft(charge_density)
    #----------------------------------------------
    # Debug
    #-----------------------------------------------
    print(charge_density_g.shape)
    # Compute the model potential
    v_model = np.divide(
        charge_density_g, G_sqmod_array, where=G_sqmod_array != 0.0)
    V_model_g = v_model * 4. * np.pi / epsilon

    V_model_g[dimensions[0] + 1, dimensions[1] + 1, dimensions[2] + 1] = 0.0

    # Get the model potential in real space
    V_model_r = get_inverse_fft(V_model_g)

    # Pack up the array
    #V_model_array = orm.ArrayData()
    #V_model_array.set_array('model_potential', V_model_r)

    return V_model_r#V_model_array



