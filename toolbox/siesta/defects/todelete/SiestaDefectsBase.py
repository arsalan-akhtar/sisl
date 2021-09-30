# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from .Utils.utils_defects import (
    get_raw_formation_energy,
    get_corrected_formation_energy,
    get_corrected_aligned_formation_energy,
)

class FormationEnergyWorkchainBase():
    """
    The base class to compute the formation energy for a given defect, containing the
    generic, code-agnostic methods, error codes, etc.

    Any computational code can be used to calculate the required energies and relative permittivity.
    However, different codes must be setup in specific ways, and so separate classes are used to implement these
    possibilities. This is an abstract class and should not be used directly, but rather the
    concrete code-specific classes should be used instead.
    """
    def __init__(self,
                 host_path,
                 neutral_defect_path,
                 charge_defect_path,
                 host_structure,
                 defect_structure_neutral,
                 defect_structure_charge,
                 add_or_remove,
                 host_VT,
                 defect_q0_VT,
                 defect_q_VT,
                 defect_site,
                 defect_charge,
                 chemical_potential,
                 fermi_level,
                 epsilon,
                 correction_scheme,
                 sigma = 0.5,
                 rho_host = None,
                 rho_defect_q = None,
                 host_fdf_name =None,
                 neutral_defect_fdf_name = None,
                 charge_defect_fdf_name = None,
                 use_siesta_mesh_cutoff = True,
                 cutoff = None,
                 ):

        self.host_path = host_path
        self.neutral_defect_path = neutral_defect_path
        self.charge_defect_path = charge_defect_path
        
        self.host_fdf_name = host_fdf_name
        self.neutral_defect_fdf_name = neutral_defect_fdf_name 
        self.charge_defect_fdf_name = charge_defect_fdf_name

        self.host_structure = host_structure
        self.defect_structure_neutral = defect_structure_neutral
        self.defect_structure_charge = defect_structure_charge
        self.add_or_remove = add_or_remove
        self.host_VT = host_VT
        self.defect_q0_VT = defect_q0_VT
        self.defect_q_VT = defect_q_VT
        self.defect_site = defect_site 
        self.defect_charge = defect_charge
        self.chemical_potential = chemical_potential
        self.fermi_level = fermi_level
        self.epsilon = epsilon
        self.correction_scheme = correction_scheme 
        self.sigma = sigma
        self.rho_host = rho_host 
        self.rho_defect_q = rho_defect_q 
        self.use_siesta_mesh_cutoff = use_siesta_mesh_cutoff
        self.cutoff = cutoff




    def setup(self):
        """
        Setup the workchain
        """

        #print(" Check if correction scheme is valid ...")
        correction_schemes_available = ["gaussian-model","gaussian-rho", "point","none","rho"]
        if self.correction_scheme is not None:
            if self.correction_scheme not in correction_schemes_available:
                print("NOT IMPLEMENTED")
            else:
                return self.correction_scheme
                print("Correction scheme is: {}".format(self.correction_scheme))
       
    def check_cutoff(self):
        if self.cutoff is not None:
            print("The Cutoff is provided and will NOT USE SIESTA MESH GRID SIZE")
            self.use_siesta_mesh_cutoff = False
        else:
            print("The Cutoff is Not provided and will USE SIESTA MESH GRID SIZE")
            self.use_siesta_mesh_cutoff = True
