# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
#from __future__ import absolute_import

import numpy as np

#from qe_tools.constants import bohr_to_ang
#from qe_tools import CONSTANTS # bohr_to_ang


"""
Utility functions for the gaussian countercharge workchain
"""
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
        #return a * x + b * x**3  +  c

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


