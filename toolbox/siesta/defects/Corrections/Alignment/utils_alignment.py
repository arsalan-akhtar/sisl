# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################

import numpy as np

"""
Utility functions for the potential alignment workchain
"""
def get_potential_difference(first_potential, second_potential):
    """
    Calculate the difference of two potentials that have the same size

    Parameters
    ----------
    first_potential: ArrayData
        The first potential array
    second_potential: ArrayData
        The second potential, to be subtracted from the first

    Returns
    -------
    difference_potential
        The calculated difference of the two potentials
    """

    #first_array = first_potential.grid 
    #second_array = second_potential.grid 

    #difference_array = first_array - second_array
    #difference_potential =  first_potential.grid - second_potential.grid    #orm.ArrayData()
    difference_potential =  first_potential - second_potential    #orm.ArrayData()
    #difference_potential.set_array('difference_potential', difference_array)

    return difference_potential

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
    array = input_array # .get_array(input_array.get_arraynames()[0])
    #target_shape = target_shape.get_list()

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

    return interp_array




def get_alignment(potential_difference, charge_density, tolerance):
    """
    Compute the density-weighted potential alignment
    """
    # Unpack

    #charge_density = charge_density.get_array(
    #    charge_density.get_arraynames()[0])

    # Get array mask based on elements' charge exceeding the tolerance.
    mask = np.ma.greater(np.abs(charge_density), tolerance)

    # Apply this mask to the diff array
    v_diff_masked = np.ma.masked_array(potential_difference , mask=mask)

    # Check if any values are left after masking
    if v_diff_masked.count() == 0:
        print ("SOMETHING IS WROOOOOOOOOOOONGGGGGGGG")
        #raise AllValuesMaskedError

    # Compute average alignment
    alignment = np.average(np.abs(v_diff_masked))

    return alignment


def get_total_alignment(alignment_dft_model, alignment_q0_host, charge):
    """
    Calculate the total potential alignment

    Parameters
    ----------
    alignment_dft_model:
    The correction energy derived from the alignment of the DFT difference
          potential and the model potential
    alignment_q0_host:
        The correction energy derived from the alignment of the defect
        potential in the q=0 charge state and the potential of the pristine host structure
    charge:  The charge state of the defect

    Returns
    -------
    total_alignment
        The calculated total potential alignment

    """

    total_alignment = -1.0*(charge * alignment_dft_model) + (charge * alignment_q0_host)

    return total_alignment


def get_total_correction(model_correction, total_alignment):
    """
    Calculate the total correction, including the potential alignments

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

    total_correction = model_correction - total_alignment  #Maybe A BUG!!
    #total_correction = model_correction + total_alignment

    return total_correction

