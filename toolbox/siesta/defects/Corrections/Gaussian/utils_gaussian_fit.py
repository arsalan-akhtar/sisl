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
Utility functions for the gaussian fit countercharge workchain
"""

def get_gaussian_3d(grid, origin, covar):
    """
    Compute anisotropic 3D Gaussian on grid

    Parameters
    ----------
    grid: array
        Array on which to compute gaussian
    origin: array
        Centre of gaussian
    covar: 3x3 array 
        Covariance matrix of gaussian
        
    Returns
    -------
    gaussian
        anisotropic Gaussian on grid
    """
    
    from scipy.stats import multivariate_normal

    origin = origin.ravel()
    gaussian = multivariate_normal.pdf(grid, origin, covar)

    return gaussian


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

def generate_charge_model(cell_matrix, peak_charge):
    """
    Return a function to compute a periodic gaussian on a grid.
    The returned function can be used for fitting. 

    Parameters
    ----------
    cell_matrix: 3x3 array
        Cell matrix of the real space cell
    peak_charge: float
        The peak charge density at the centre of the gaussian.
        Used for scaling the result.

    Returns
    -------
    compute_charge
        A function that will compute a periodic gaussian on a grid 
        for a given cell and peak charge intensity
    """

    def compute_charge(
        xyz_real,
        x0, y0, z0,
        sigma_x, sigma_y, sigma_z,
        cov_xy, cov_xz, cov_yz):
        """
        For a given system charge, create a model charge distribution using 
        an anisotropic periodic 3D gaussian.
        The charge model for now is a Gaussian.

        NOTE: 
        The values for sigma and cov are not the values used in construction 
        of the Gaussian. After the covariance matrix is constructed, its 
        transpose is multiplied by itself (that is to construct a Gram matrix) 
        to ensure that it is positive-semidefinite. It is this matrix which is 
        the real covariance matrix. This transformation is to allow this 
        function to be used directly by the fitting algorithm without a danger 
        of crashing.  

        Parameters
        ----------
        xyz_real: 3xN array 
            Coordinates to compute the Gaussian for in cartesian coordinates.
        x0, y0, z0: float
            Center of the Gaussian in crystal coordinates.
        sigma_x, sigma_y, sigma_z: float
            Spread of the Gaussian (not the real values used, see note above).
        cov_xy, cov_xz, cov_yz: float
            Covariance values controlling the rotation of the Gaussian 
            (not the real values used, see note above).

        Returns
        -------
        g
            Values of the Gaussian computed at all of the desired coordinates and 
            scaled by the value of charge_integral.

        """

        # Construct the pseudo-covariance matrix
        V = np.array([[sigma_x, cov_xy, cov_xz],[cov_xy, sigma_y, cov_yz], [cov_xz, cov_yz, sigma_z]])
        # Construct the actual covariance matrix in a way that is always positive semi-definite
        # Construct the actual covariance matrix in a way that is always positive semi-definite
        covar = np.dot(V.T, V)

        gauss_position = np.array([x0, y0, z0])

        # Apply periodic boundary conditions
        g = 0
        for ii in [-1, 0, 1]:
            for jj in [-1, 0, 1]:
                for kk in [-1, 0, 1]:
                    # Compute the periodic origin in crystal coordinates
                    origin_crystal = (gauss_position + np.array([ii, jj, kk])).reshape(3,1)
                    # Convert this to cartesian coordinates
                    origin_real = np.dot(cell_matrix.T, origin_crystal)
                    # Compute the Gaussian centred at this position
                    g = g + get_gaussian_3d(xyz_real.T, origin_real, covar)
                    

        print("DEBUG: Integrated charge density (unscaled) = {}".format(get_integral(g, cell_matrix)))

        print("DEBUG: g.max()  = {}".format(g.max()))
        # Scale the result to match the peak charge density
        g = g * (peak_charge / g.max())
        print("DEBUG: Peak Charge target  = {}".format(peak_charge))
        print("DEBUG: Peak Charge scaled  = {}".format(g.max()))
        print("DEBUG: Integrated charge density (scaled) = {}".format(get_integral(g, cell_matrix)))

        return g

    return compute_charge


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

def get_cell_matrix(structure):
    """
    Get the cell matrix (in bohr) from an AiiDA StructureData object

    Parameters
    ----------
    structure: AiiDA StructureData
        The structure object of interest

    Returns
    -------
    cell_matrix
        3x3 cell matrix array in units of Bohr

    """
    #cell_matrix = np.array(structure.cell) / bohr_to_ang  # Angstrom to Bohr
    cell_matrix = structure.cell
    return cell_matrix



def get_charge_model_fit(rho_defect_q0, rho_defect_q, structure):
    """
    Fit the charge model to the defect data

    Parameters
    ----------
    model_correction: orm.Float
        The correction energy derived from the electrostatic model
    total_alignment: orm.Float
        The correction energy derived from the alignment of the DFT difference
        potential and the model potential, and alignment of the defect potential
        in the q=0 charge state and the potential of the pristine host structure

    Returns
    -------
    total_correction
        The calculated correction, including potential alignment

    """

    from scipy.optimize import curve_fit
    #from .model_potential.utils import generate_charge_model, get_xyz_coords, get_cell_matrix

    # Get the cell matrix
    cell_matrix = structure.cell

    # Compute the difference in charge density between the host and defect systems
    #rho_defect_q_data = rho_defect_q.get_array(rho_defect_q.get_arraynames()[0])
    #rho_host_data = rho_host.get_array(rho_host.get_arraynames()[0])

    rho_defect_q0_data = rho_defect_q0
    rho_defect_q_data = rho_defect_q
    #rho_host_data = rho_host.read_grid().grid
    #rho_defect_q0_data = rho_defect_q.grid

    # Charge density from QE is in e/cubic-bohr, so convert if necessary
    # TODO: Check if the CUBE file format is strictly Bohr or if this is a QE thing
    #rho_diff = (rho_host_data - rho_defect_q_data)/(bohr_to_ang**3)
    #rho_diff = rho_host_data - rho_defect_q_data
    rho_diff = rho_defect_q_data - rho_defect_q0_data

    # Detect the centre of the charge in the data
    max_pos_mat = np.array(np.unravel_index(rho_diff.argmax(), rho_diff.shape)) # matrix coords
    max_pos_ijk = (max_pos_mat*1.)/(np.array(rho_diff.shape)-1) # Compute crystal coords
    max_i = max_pos_ijk[0]
    max_j = max_pos_ijk[1]
    max_k = max_pos_ijk[2]

    # Generate cartesian coordinates for a grid of the same size as the charge data
    xyz_coords = get_xyz_coords(cell_matrix, rho_diff.shape)

    # Set up some safe parameters for the fitting
    guesses = [max_i, max_j, max_k, 1., 1., 1., 0., 0., 0.]
    bounds = (
        [0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [1., 1., 1., np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    peak_charge = rho_diff.max()

    # Do the fitting
    fit, covar_fit = curve_fit(
        generate_charge_model(cell_matrix, peak_charge),
        xyz_coords,
        rho_diff.ravel(),
        p0=guesses,
        bounds=bounds)

    # Compute the one standard deviation errors from the 9x9 covariance array
    fit_error = np.sqrt(np.diag(covar_fit))

    fitting_results = {}

    fitting_results['fit'] = fit.tolist()
    fitting_results['peak_charge'] = peak_charge
    fitting_results['error'] = fit_error.tolist()

    return fitting_results #orm.Dict(dict=fitting_results)


def fit_charge_model(fit,charge_fit_tolerance,strict_fit):
    """
    Fit an anisotropic gaussian to the charge state electron density
    """

    #fit = get_charge_model_fit(
    #    self.inputs.rho_host,
    #    self.inputs.rho_defect_q,
    #    self.inputs.host_structure)

    fitted_params = fit['fit']
    peak_charge = fit['peak_charge']

    for parameter in fit['error']:
        if parameter > charge_fit_tolerance:
            print("Charge fitting parameter worse than allowed tolerance")
            if strict_fit:
                print ("the mode fit to charge density is exceeds tolerance")
        else:
            comment="Fitting Is Done"
    print (comment)




def get_charge_model(cell_matrix, peak_charge, defect_charge, dimensions, gaussian_params):
    """
    For a given system charge, create a model charge distribution.

    Parameters
    ----------
    cell_matrix: 3x3 array
        Cell matrix of the real space cell.
    peak_charge : float
        The peak charge density at the centre of the gaussian.
    defect_charge : float
        Charge state of the defect
    dimensions: 3x1 array-like
        Dimensions of grid to compute charge on.
    gaussian_params: list (length 6)
        Parameters determining the distribution position and shape obtained
        by the fitting procedure.

    Returns
    -------
    model_charge_array
        The grid with the charge data as an AiiDA ArrayData object

    """

    #cell_matrix = cell_matrix.get_array('cell_matrix')
    #peak_charge = peak_charge.value
    #defect_charge = defect_charge.value
    dimensions = np.array(dimensions)
    #gaussian_params = gaussian_params.get_list()

    xyz_coords = get_xyz_coords(cell_matrix, dimensions)

    get_model = generate_charge_model(cell_matrix, peak_charge)
    g = get_model(xyz_coords, *gaussian_params)

    # Unflatten the array
    g = g.reshape(dimensions)

    print("DEBUG: fit params: {}".format(gaussian_params))

    # Rescale to defect charge
    print("DEBUG: Integrated charge density target  = {}".format(defect_charge))
    g = g * (defect_charge / get_integral(g, cell_matrix))
    print("DEBUG: Integrated charge density (scaled) = {}".format(get_integral(g, cell_matrix)))

    # Compensating jellium background
    g = g - np.sum(g)/np.prod(g.shape)
    print("DEBUG: Integrated charge density (jellium) = {}".format(get_integral(g, cell_matrix)))

    # Pack the array
    #model_charge_array = orm.ArrayData()
    #model_charge_array.set_array('model_charge', g)

    return g #model_charge_array
    
