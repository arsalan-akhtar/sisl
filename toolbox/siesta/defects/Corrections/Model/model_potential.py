# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
import numpy as np
#from qe_tools.constants import bohr_to_ang
from qe_tools import  constants

bohr_to_ang = constants.bohr_to_ang
from .utils_model_potential import get_model_potential, get_energy,get_model_potential_rho


class ModelPotential():
    """
    Workchain to compute the model electrostatic potential
    """
    def __init__(self,
                 charge_density, 
                 scale_factor, 
                 structure, 
                 grid,
                 epsilon,
                 use_siesta_mesh_cutoff = None,
                 rho=False,
                 ):
              
        self.charge_density = charge_density 
        self.scale_factor = scale_factor
        self.structure = structure
        self.grid = grid
        self.epsilon = epsilon
        self.use_siesta_mesh_cutoff = use_siesta_mesh_cutoff   
        self.rho = rho 
        print("\n")
        print("--------------------ModelPotential Class Start--------------------\n")
        print("ModelPotential Class Debug: grid dimension is {}".format(self.grid[self.scale_factor]))
        print("ModelPotential Class Debug: Computing model potential for scale factor {}".format(self.scale_factor))
        self.cell = self.structure.cell
        self.r_cell = self.structure.rcell
        print("ModelPotential Class DEBUG: cell is :\n {}".format(self.cell))
        print("ModelPotential Class DEBUG: rec cell is :\n {}".format(self.r_cell))
 



    def run(self):
        """
        Run ModelPotential 
        """
        #print("\n")
        #print("--------------------ModelPotential Class Start--------------------\n")
        #print("ModelPotential Class Debug: grid dimension is {}".format(self.grid[self.scale_factor]))
        #print("Siesta grid is {}".format(self.siesta_grid[self.scale_factor]))
        
        # Dict to store model energies
        #self.model_energies = {}

        #self.compute_model_potential()
        #self.model_energies [self.scale_factor] = self.compute_energy()
        
        self.compute_model_potential()
        self.model_energies = self.compute_energy()


        #print("ModelPotential Class Debug: Finished successfully for scale {}\n".format(self.scale_factor))
        #print("--------------------ModelPotential Class Finished--------------------\n")
        return self.model_energies


    def compute_model_potential(self):
        """
        Compute the model potential according to the Gaussian Counter Charge scheme
        """
        #print("ModelPotential Class Debug: Computing model potential for scale factor {}".format(self.scale_factor))
        #self.cell = get_cell_matrix(self.host_structure[self.scale_factor])
        
        #self.cell = get_cell_matrix(self.host_structure)
        #self.r_cell = get_reciprocal_cell(self.cell)
        #print("ModelPotential Class DEBUG: cell is :\n {}".format(self.cell))
        #print("ModelPotential Class DEBUG: rec cell is :\n {}".format(self.r_cell))
        
        if self.rho:
            self.model_potential = get_model_potential_rho( self.r_cell,
                                                dimensions= self.grid[self.scale_factor],
                                                charge_density= self.charge_density[self.scale_factor],
                                                epsilon=self.epsilon                                      
                                               )
        else:
            if self.use_siesta_mesh_cutoff:
                self.model_potential = get_model_potential ( self.r_cell,
                                                dimensions= self.grid[self.scale_factor],
                                                charge_density= self.charge_density[self.scale_factor],
                                                epsilon=self.epsilon                                      
                                               )
            else:
                print("ModelPotential Class DEBUG: Model Potential with cutoff! ")
                self.model_potential = get_model_potential ( rcell_matrix = self.r_cell,
                                                dimensions = self.grid[self.scale_factor],
                                                charge_density = self.charge_density[self.scale_factor],
                                                epsilon = self.epsilon
                                               )


        return self.model_potential 

    def compute_energy(self):
        """
        Compute the model energy
        """
        print("ModelPotential Class Debug: Computing model energy for scale factor {}".format(self.scale_factor))
        hartree_to_eV = 27.2114
        self.model_energy = get_energy(potential = self.model_potential,
                                       charge_density = self.charge_density[self.scale_factor],
                                       cell_matrix = self.cell)

        print("ModelPotential Class Debug: Calculated model energy: {} eV".format(self.model_energy))
        print("ModelPotential Class Debug: Finished successfully for scale {}\n".format(self.scale_factor))
        print("--------------------ModelPotential Class Finished--------------------\n")


        return self.model_energy



