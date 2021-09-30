# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np

from ..Utils.utils_alignment import get_potential_difference
from ..Utils.utils_density_weighted import get_alignment
from ..Utils.utils_gaussian_rho  import get_interpolation
from qe_tools import constants
hartree_to_ev = constants.hartree_to_ev 

class PotentialAlignment():
    """
    Align two electrostatic potentials according to a specified method.
    """
    def __init__(self,
                 first_potential,
                 second_potential,
                 charge_density,
                 tolerance = 1.0e-3 ,
                 scheme = 'density_weighted',
                 allow_interpolation = False,
            ):

        self.allow_interpolation = allow_interpolation
        self.scheme = scheme
        self.first_potential = first_potential
        self.second_potential = second_potential
        self.charge_density = charge_density 
        self.tolerance = tolerance 

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
        print ("Potential Alignment Scheme is {} ".format(self.scheme))
        print ("Shape of first potential is {}".format(self.first_potential.shape)) 
        print ("Shape of second potential is {}".format(self.second_potential.shape)) 
        print ("Shape of charge density is {}".format(self.charge_density.shape)) 
        if self.charge_density.shape != self.first_potential.shape:
            print("Need  Interpolation")
            self.charge_density = get_interpolation(self.charge_density,self.first_potential.shape)
            print ("Shape of charge density NOW is {}".format(self.charge_density.shape))
        if self.second_potential.shape != self.first_potential.shape:
            self.second_potential = get_interpolation(self.second_potential,self.first_potential.shape)
            print ("Shape of second potential NOW is {}".format(self.second_potential.shape))



        if self.scheme == 'lany_zunger':
            print ("Not Implemented YET! BYE")

        if  self.scheme == 'density_weighted':
            print ("Computing First and Second Potential Difference")
            self.compute_difference()
            #print ("DEBUG : DONE Computing First and Second Potential Difference ")
            self.calculate_alignment_density_weighted()
            self.results()





    def calculate_alignment_lany_zunger(self):
        """
        Calculate the alignment according to the requested scheme
        """
    

    def compute_difference(self):
        """
        """
        #from utils_alignment import get_potential_difference
        self.potential_difference = get_potential_difference( first_potential = self.first_potential,
                                                              second_potential = self.second_potential
                                                             )


    def calculate_alignment_density_weighted(self):
        """
        """
        print ("Density Weighted Alignment Starts")
        
        #from utils_density_weighted import get_alignment
        self.alignment = get_alignment(self.potential_difference,
                                       self.charge_density,
                                       self.tolerance)


    def results(self):
        """
        Collect results
        """
        #print("Completed alignment. An alignment of {} eV is required".format(self.alignment * hartree_to_ev/2.0 ))
        print("Completed alignment. An alignment of {} eV is required".format(self.alignment ))
