# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
import numpy as np

from .utils_alignment import get_potential_difference
from .utils_alignment import get_alignment_density_weighted
from .utils_alignment import get_alignment_FNV
from .utils_alignment import get_alignment_classic
from sisl import unit_convert

from .utils_alignment import get_density_weighted_potential

from .utils_alignment import get_interpolation, get_interpolation_sisl_from_array
from qe_tools import constants
hartree_to_ev = constants.hartree_to_ev 
import sys
class PotentialAlignment():
    """
    Align two electrostatic potentials according to a specified method.
    """
    def __init__(self,
                 first_potential,
                 second_potential,
                 charge_density=None,
                 tolerance = 1.0e-3 ,
                 scheme = 'density_weighted',
                 allow_interpolation = False,
                 pot_site = None,
                 avg_plane = 'xy'
            ):

        self.allow_interpolation = allow_interpolation
        self.scheme = scheme
        self.first_potential = first_potential
        self.second_potential = second_potential
        self.charge_density = charge_density 
        self.tolerance = tolerance 
        self.pot_site = pot_site
        self.avg_plane = avg_plane

    def run(self):
        """
        The Main Method for Running 
        """
        print ("-------------------- Potential Alignment Class Start --------------------")
        self.setup()
        
        return  self.alignment
        print ("-------------------- Potential Alignment Class Finshed --------------------")


    def setup(self):
        """
        Input validation and context setup
        """
        print ("Potential Alignment Scheme is : {}".format(self.scheme))
        print ("Charge Tolerence if DW Scheme is : {}".format(self.tolerance))
        print ("Shape of first potential is   : {}".format(self.first_potential.shape)) 
        print ("Shape of second potential is  : {}".format(self.second_potential.shape)) 
        if self.charge_density is not None:
            print ("Shape of charge density is    : {}".format(self.charge_density.shape)) 
            if self.charge_density.shape != self.first_potential.shape:
                print("Need  Interpolation")
                print("Interpolating...!")
                #self.charge_density = get_interpolation(self.charge_density,self.first_potential.shape) AiiDA interpolation method
                self.charge_density = get_interpolation_sisl_from_array(self.charge_density,self.first_potential.shape)
                print ("Shape of charge density NOW is   : {}".format(self.charge_density.shape))
        if self.second_potential.shape != self.first_potential.shape:
            #self.second_potential = get_interpolation(self.second_potential,self.first_potential.shape)
            self.second_potential = get_interpolation_sisl_from_array(self.second_potential,self.first_potential.shape)
            print ("Shape of second potential NOW is : {}".format(self.second_potential.shape))



        if self.scheme == 'lany_zunger':
            print ("Not Implemented YET! BYE")
            sys.exit("Exiting")
        
        if self.scheme == 'FNV':
            print("Using FNV alignment scheme")
            self.compute_difference()
            self.calculate_alignment_FNV()
            self.results()

        if  self.scheme == 'density_weighted':
            print("Using Density Weighted alignment scheme")
            self.compute_difference()
            self._get_density_weighted_potential()
            self.calculate_alignment_density_weighted()
            self.results()
        
        if self.scheme == 'FNV_DW':
            print("Using FNV_DW alignment scheme")
            self.compute_difference()
            self.calculate_alignment_density_weighted()
            self.results()

        if self.scheme == 'Classic':
            self.compute_difference()
            self.compute_alignment_classic()
            self.results()

    def calculate_alignment_lany_zunger(self):
        """
        Calculate the alignment according to the requested scheme
        """

    def compute_difference(self):
        """
        """
        print ("Computing First and Second Potential Difference")
        self.potential_difference = get_potential_difference( first_potential = self.first_potential,
                                                              second_potential = self.second_potential
                                                             )

    def _get_density_weighted_potential(self):
        """
        """
        self.potential_dw = get_density_weighted_potential(self.potential_difference,
                                                 self.charge_density,
                                                 self.tolerance)


    def calculate_alignment_density_weighted(self):
        """
        """
        print ("Density Weighted Alignment Starts")
        self.alignment = get_alignment_density_weighted(self.potential_difference,
                                                        self.charge_density,
                                                        self.tolerance)

    def calculate_alignment_FNV(self):
        """
        """
        #self.potential_difference = self.first_potential.grid - self.second_potential.grid
        self.alignment = get_alignment_FNV (self.potential_difference)

    def compute_alignment_classic(self):
        """
        """
        print("Classic Alignemt ...!")
        print(f"DEBUG:{self.pot_site}")
        self.alignment = get_alignment_classic (V_hq = self.first_potential , 
                V_model = self.second_potential /unit_convert("Bohr","Ang")**3 ,
                pot_site =  self.pot_site,
                avg_plane = self.avg_plane
                ) 


    def results(self):
        """
        Collect results
        """
        #print("Completed alignment. An alignment of {} eV is required".format(self.alignment * hartree_to_ev/2.0 ))
        print("Completed alignment. An alignment of {} eV is required".format(self.alignment ))
