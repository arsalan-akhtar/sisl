# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
import numpy as np
from qe_tools import constants #bohr_to_ang

bohr_to_ang = constants.bohr_to_ang
from .utils_gaussian_model import get_charge_model_sigma 
from ..Model.model_potential import ModelPotential
from ..Alignment.potential_alignment import PotentialAlignment
from ..Model.utils_model_potential import get_integral
from .utils_gaussian_rho import cut_rho
import sisl
from toolbox.siesta.defects.Corrections.Gaussian.utils_gaussian_rho import shift_prepare

from sisl import unit_convert
Ang_to_Bohr = unit_convert("Ang","Bohr")
One_Ang_to_Bohr = 1.0 / Ang_to_Bohr

class GaussianCharge():
    """
    Compute the electrostatic correction for charged defects according to the
    Guassian counter-charge method.
    Here we implement the Komsa-Pasquarello method (https://doi.org/10.1103/PhysRevLett.110.095505),
    which is itself based on the Freysoldt method
    (https://doi.org/10.1103/PhysRevLett.102.016402).
    """
    def __init__(self,
                 v_host, 
                 v_defect_q0, 
                 v_defect_q, 
                 defect_charge, 
                 defect_site,
                 host_structure, 
                 scheme  ,
                 epsilon ,
                 model_iterations_required ,
                 sigma = None ,
                 gamma = None,
                 eta = None,
                 use_siesta_mesh_cutoff = None,
                 siesta_grid = None,
                 rho_host = None,
                 rho_defect_q0 = None,
                 rho_defect_q = None,
                 cutoff = None,
                 fit_params = None,
                 potential_align_scheme = None, 
                 charge_dw_tol = None,
                 E_iso_type = 'original',
                 avg_plane = None,
                 #defect_structure_charge=None,
                 ):

        self.v_host = v_host 
        self.v_defect_q0 = v_defect_q0
        self.v_defect_q = v_defect_q
        self.defect_charge =  defect_charge
        self.defect_site =  defect_site
        self.host_structure = host_structure
        self.epsilon = epsilon
        self.model_iterations_required = model_iterations_required
        self.use_siesta_mesh_cutoff = use_siesta_mesh_cutoff
        self.sigma = sigma
        self.gamma = gamma
        self.eta = eta
        self.scheme = scheme
        self.siesta_grid = siesta_grid
        self.rho_host = rho_host
        self.rho_defect_q0 = rho_defect_q0
        self.rho_defect_q = rho_defect_q
        self.cutoff = cutoff 
        self.fit_params = fit_params
        self.potential_align_scheme = potential_align_scheme 
        self.charge_dw_tol = charge_dw_tol 
        self.E_iso_type = E_iso_type
        self.avg_plane = avg_plane 
        #self.run()
    def run(self):

        if self.scheme =="gaussian-rho":
            if self.fit_params is not None:
                self.is_fit = False
                print("The Fitting Params given by user!")
                self.is_fit = False
                #self.peak_charge = self.fit_params[0]['peakcharge']
                #self.fitted_params  = self.fit_params[0]['fit_params']
                self.peak_charge = self.fit_params['peak_charge']
                self.fitted_params  = self.fit_params['fit']
                print ("The Fitting Paramers are {}".format(self.fitted_params))
                print ("The Peak Charge is {}".format(self.peak_charge))
            else:
                print("Need to fit gaussian...!")
                self.is_fit = True
        is_mesh = self.use_siesta_mesh_cutoff
        self.__setup()
        
        #=======================================================================================
        #                                Gaussian Model Sigma
        #=======================================================================================

        if self.scheme == 'gaussian-model': 
            print("--------------------********************************--------------------")
            print("                    Starting Gaussian Model Scheme                      ")
            print("--------------------********************************--------------------\n")
            print ("Computing MODEL INFO")
            print ("Gaussian Sigma is {}".format(self.sigma))
            while (self.__should_run_model()):
                self.model_iteration +=1
                print ("Calculations For Scale Factor {}".format(self.model_iteration))
                # if using SIESTA MESH TO GENERATE THE MODEL
                if is_mesh:
                    print("DEBUG: Using Siesta Mesh to Calculate the Model Charge")
                    charge_model_info = self._compute_model_charge_with_siesta_mesh()
                    self.model_energies [self.model_iteration] = self._compute_model_potential(charge_model_info)
                else:
                    print("DEBUG: Generating the Mesh to Calculating the Model Charge ")
                    charge_model_info = self._compute_model_charge_with_cutoff()

                    #print(f"DEBUG: Normalize Charge density Gaussian is { get_integral(self.model_charges[self.model_iteration]*Ang_to_Bohr,self.model_structures[self.model_iteration].cell*Ang_to_Bohr) }")

                    print ("Computing MODEL INFO")
                    self.model_energies [self.model_iteration] = self._compute_model_potential(charge_model_info)
        
            self.check_model_potential_energies()
            print ("All Scaled Energies Are DONE\n")

        #=======================================================================================
        #                                Gaussian Model Sigma with tail
        #=======================================================================================

        if self.scheme == 'gaussian-model-tail': 
            print("--------------------********************************--------------------")
            print("                  Starting Gaussian Model Tail Scheme                   ")
            print("--------------------********************************--------------------\n")
            print ("Computing MODEL INFO")
            print ("Gaussian Sigma/beta is {}".format(self.sigma))
            while (self.__should_run_model()):
                self.model_iteration +=1
                print ("Calculations For Scale Factor {}".format(self.model_iteration))
                # if using SIESTA MESH TO GENERATE THE MODEL
                if is_mesh:
                    print("DEBUG: Using Siesta Mesh to Calculate the Model Charge")
                    charge_model_info = self._compute_model_charge_with_siesta_mesh()
                    self.model_energies [self.model_iteration] = self._compute_model_potential(charge_model_info)
                else:
                    print("DEBUG: Generating the Mesh to Calculating the Model Charge ")
                    charge_model_info = self._compute_model_charge_tail_with_cutoff()
                    print ("Computing MODEL INFO")
                    self.model_energies [self.model_iteration] = self._compute_model_potential(charge_model_info)
        
            self.check_model_potential_energies()
            print ("All Scaled Energies Are DONE\n")

        #=======================================================================================
        #                                Gaussian Model Sigma General with tail
        #=======================================================================================

        if self.scheme == 'gaussian-model-general': 
            print("--------------------********************************--------------------")
            print("             Starting Gaussian Model General Tail Scheme                 ")
            print("--------------------********************************--------------------\n")
            print ("Computing MODEL INFO")
            print ("Gaussian Sigma/beta is {}".format(self.sigma))
            while (self.__should_run_model()):
                self.model_iteration +=1
                print ("Calculations For Scale Factor {}".format(self.model_iteration))
                # if using SIESTA MESH TO GENERATE THE MODEL
                if is_mesh:
                    print("DEBUG: Using Siesta Mesh to Calculate the Model Charge")
                    charge_model_info = self._compute_model_charge_with_siesta_mesh()
                    self.model_energies [self.model_iteration] = self._compute_model_potential(charge_model_info)
                else:
                    print("DEBUG: Generating the Mesh to Calculating the Model Charge ")
                    charge_model_info = self._compute_model_charge_general_with_cutoff()
                    print ("Computing MODEL INFO")
                    self.model_energies [self.model_iteration] = self._compute_model_potential(charge_model_info)
        
            self.check_model_potential_energies()
            print ("All Scaled Energies Are DONE\n")


        #=======================================================================================
        #                                Gaussian Model Rho
        #=======================================================================================
 
        if self.scheme == 'gaussian-rho':
            print("--------------------********************************--------------------")
            print("                 Starting Fitting Gaussian RHO Scheme                   ")
            print("--------------------********************************--------------------\n")
            self.rho_host_array = self.rho_host.read_grid().grid
            self.rho_defect_q0_array = self.rho_defect_q0.read_grid().grid
            self.rho_defect_q_array = self.rho_defect_q.read_grid().grid
            if self.is_fit:
                print ("Start Fitting Gaussian...")
                self.fit_charge_model()
                print ("Fitting Gaussian DONE! ")
            else:
                print("Using Users Parameters of Fit to Generate Charge Model")
            print ("------------------------------------------------------------\n")
            while (self.__should_run_model()):
                self.model_iteration +=1
                print ("Calculations For Scale Factor {} ...".format(self.model_iteration))
                print ("Computing CHARGE MODEL for Scale Factor {}".format(self.model_iteration))
                if is_mesh:
                    print("DEBUG: Using Siesta Mesh to Calculate the Model Charge")
                    charge_model_info = self._compute_model_charge_fit_with_siesta_mesh()
                    self.model_energies [self.model_iteration] = self._compute_model_potential(charge_model_info)
                else:
                    print ("Computing POTENTIAL MODEL for Scale Factor {}".format(self.model_iteration))
                    charge_model_info = self._compute_model_charge_fit_with_cutoff()
                    self.model_energies [self.model_iteration] = self._compute_model_potential(charge_model_info)
                    print ("Done For Scale Factor {}".format(self.model_iteration))
                    print ("------------------------------------------------------------\n")
            self.check_model_potential_energies()
            print ("All Scaled Energies Are DONE\n")

        #=======================================================================================
        #                                          Rho
        #=======================================================================================
 
        if self.scheme == 'rho':
            print("--------------------********************************---------------------")
            print("                           Starting RHO Scheme                          ")
            print("--------------------********************************--------------------\n")
            #self.rho_host_array_up = self.rho_host.read_grid(spin=0)
            #self.rho_host_array_down = self.rho_host.read_grid(spin=1)
            
            #self.rho_host_array = self.rho_host.read_grid(spin=0) + self.rho_host.read_grid(spin=1)
            #self.rho_defect_q0_array = self.rho_defect_q0.read_grid(spin=0) + self.rho_defect_q0.read_grid(spin=1)
            #self.rho_defect_q_array = self.rho_defect_q.read_grid(spin=0) + self.rho_defect_q.read_grid(spin=1)
            
            self.rho_host_array = self.rho_host.read_grid(spin=[1,1]) 
            self.rho_defect_q0_array = self.rho_defect_q0.read_grid(spin=[1,1]) 
            self.rho_defect_q_array = self.rho_defect_q.read_grid(spin=[1,1]) 
 
            rho_norm =  self.rho_defect_q0_array - self.rho_defect_q_array  #THis Works
            #rho_norm =  (self.rho_defect_q_array - self.rho_defect_q0_array ) * (-1)



            #self.rho_host_array = self.rho_host.read_grid()
            #self.rho_defect_q0_array = self.rho_defect_q0.read_grid()
            #self.rho_defect_q_array = self.rho_defect_q.read_grid()
            #self.rho_defect_q_q0_array_diff = self.rho_defect_q_array - self.rho_defect_q0_array
            #self.rho_defect_q0_q_array_diff = self.rho_defect_q0_array - self.rho_defect_q_array
            #self.rho_defect_q_q0_array_abs = abs(self.rho_defect_q_array - self.rho_defect_q0_array)

            

            #================
            # At least for e
            #================
            #if self.defect_charge < 0.0 :
            #    print("DEBUG: Charge is electron ... !!!!")
            #    rho_sum = self.rho_defect_q_q0_array_abs
            #    rho_norm = (rho_sum / get_integral(rho_sum.grid,rho_sum.cell)) * self.defect_charge
            #    self.rho_sum = rho_sum
            #    self.rho_norm = rho_norm 
            #================
            # At least for h
            #================
            #if self.defect_charge > 0.0 :
            #    print("DEBUG: Charge is hole ... !!!!")
            #    #rho_sum = self.rho_defect_q_q0_array_abs + self.rho_defect_q_q0_array_diff # is working
            #    rho_sum = self.rho_defect_q_q0_array_abs # Working for DW
            #    #rho_sum = self.rho_defect_q_q0_array_abs + self.rho_defect_q0_q_array_diff
            #    #rho_sum = self.rho_defect_q0_array - self.rho_defect_q0_q_array_diff
            #    #rho_sum = self.rho_defect_q_array - self.rho_defect_q0_q_array_diff
            #    #rho_sum = self.rho_defect_q_q0_array_diff + self.rho_defect_q0_q_array_diff +self.rho_defect_q_q0_array_abs
            #    rho_norm = (rho_sum / get_integral(rho_sum.grid,rho_sum.cell)) * self.defect_charge
            #    self.rho_sum = rho_sum
            #    self.rho_norm = rho_norm 
 
            
            #============
            # Without Cut
            #rho_sum = self.rho_defect_q_q0_array_diff + self.rho_defect_q_q0_array_abs
            #rho_norm = (rho_sum/ get_integral(rho_sum.grid,rho_sum.cell)) * self.defect_charge
            #============
            # To check Worked
            #self.rho_defect_q_q0_array_abs = abs(self.rho_defect_q_array - self.rho_defect_q0_array)
            #rho_sum =  self.rho_defect_q_q0_array_abs + self.rho_defect_q_q0_array_diff
            #rho_norm = (rho_sum/ get_integral(rho_sum.grid,rho_sum.cell)) * self.defect_charge
             
            ##rho_norm = self.rho_defect_q_q0_array_diff
            
            #===========
            # To check version 2
            

            #self.rho_defect_q_q0_array_abs1 = abs(self.rho_defect_q_array - self.rho_defect_q0_array)
            #self.rho_defect_q_q0_array_abs2 = abs(self.rho_defect_q0_array - self.rho_defect_q_array)
            #self.rho_defect_q_q0_array_abs2 = self.rho_defect_q_array - self.rho_defect_q0_array
            #rho_sum = self.rho_defect_q_q0_array_abs1 #+ self.rho_defect_q_q0_array_abs2
            
            #rho_sum = abs(self.rho_defect_q_array ) - abs(self.rho_defect_q0_array)
            
            #rho_norm = (rho_sum / get_integral(rho_sum.grid,rho_sum.cell)) * self.defect_charge
            #===============================
            #norm = np.linalg.norm(rho_sum.grid)
            #rho_norm_temp = rho_sum.grid / norm
            #rho_norm = sisl.Grid(shape=rho_norm_temp.shape, geometry = self.host_structure)
            #rho_norm.grid = rho_norm_temp
            #=================================
            #self.rho_sum = rho_sum
            #self.rho_norm = rho_norm 
            #if self.defect_charge > 0 :
            #    self.rho_defect_q_q0_array_diff = self.rho_defect_q_array - self.rho_defect_q0_array
            #    self.rho_defect_q_q0_array_abs = abs(self.rho_defect_q_array - self.rho_defect_q0_array)
            #    rho_sum =  self.rho_defect_q_q0_array_abs + self.rho_defect_q_q0_array_diff
            #else:
            #    self.rho_defect_q_q0_array_diff = self.rho_defect_q_array - self.rho_defect_q0_array
            #    self.rho_defect_q_q0_array_abs = abs(self.rho_defect_q_array - self.rho_defect_q0_array)
            #    rho_sum =  self.rho_defect_q_q0_array_abs + self.rho_defect_q_q0_array_diff
            #rho_sum = self.rho_defect_q_q0_array_diff #*2.0
            #rho_norm = (rho_sum/ get_integral(rho_sum.grid,rho_sum.cell)) * self.defect_charge

            #===============
            # To cut the rho
            #rho_sum_no_cut = self.rho_defect_q_q0_array_diff + self.rho_defect_q_q0_array_abs
            #rho_sum = cut_rho(rho_sum_no_cut,1.0e-1,self.host_structure)
            #rho_norm = (rho_sum/ get_integral(rho_sum.grid,rho_sum.cell)) * self.defect_charge
            # ============

            print(f"Normalize Charge density rho is { get_integral(rho_norm.grid,rho_norm.cell) }")
            self.rho_defect_q_q0_array = rho_norm  

            #self.rho_defect_q_q0_array_diff + self.rho_defect_q_q0_array_abs
            #self.rho_defect_q_q0_array= self.rho_defect_q_array - self.rho_defect_q0_array
            while (self.__should_run_model()):
                self.model_iteration +=1
                print ("Calculations For Scale Factor {} ...".format(self.model_iteration))
                print ("Computing CHARGE MODEL for Scale Factor {}".format(self.model_iteration))
                if is_mesh:
                    print("DEBUG: Using Siesta Mesh to Calculate the Model Charge")
                    charge_model_info = self._compute_model_charge_rho_q_q0_with_siesta_mesh()
                    self.model_energies [self.model_iteration] = self._compute_model_potential(charge_model_info,rho=True)
                else:
                    print ("Computing POTENTIAL MODEL for Scale Factor {}".format(self.model_iteration))
                    #charge_model_info = self._compute_model_charge_rho_q_q0_with_cutoff()
                    charge_model_info = self._compute_model_charge_rho_q_q0_with_cutoff_debug()
                    self.model_energies [self.model_iteration] = self._compute_model_potential(charge_model_info,rho=True)
                    print ("Done For Scale Factor {}".format(self.model_iteration))
                    print ("------------------------------------------------------------\n")
            self.check_model_potential_energies()
            print ("All Scaled Energies Are DONE\n")
            print ("DEBUG : Model for Charge Alignemt ...")
            self._compute_model_potential_for_alignment()

   
        
    def __setup(self):
        """
        Setup the calculation
        """
        print("Checking Correction  ({}) Scheme ...".format(self.scheme))
        ## Verification
        if self.model_iterations_required < 3:
           print('The requested number of iterations, {}, is too low. At least 3 are required to achieve an #adequate data fit'.format(self.model_iterations_required))

        # Track iteration number
        self.model_iteration = 0 

        #self.v_host_array = self.v_host.read_grid()
        #self.v_defect_q0_array = self.v_defect_q0.read_grid()
        #self.v_defect_q_array = self.v_defect_q.read_grid() 
        
        # with spin 

        #self.v_host_array = self.v_host.read_grid(spin=[1,1])
        #self.v_defect_q0_array = self.v_defect_q0.read_grid(spin=[1,1])
        #self.v_defect_q_array = self.v_defect_q.read_grid(spin=[1,1]) 

        # with VH
        self.v_host_array = self.v_host.read_grid()
        self.v_defect_q0_array = self.v_defect_q0.read_grid()
        self.v_defect_q_array = self.v_defect_q.read_grid() 


        # FROM Methods 
        # Compute the difference in the DFT potentials for the cases of q=q and q=0
        #self.v_defect_q_q0 = self.v_defect_q_array.grid - self.v_defect_q0_array.grid
        self.v_defect_q_q0_array = self.v_defect_q_array - self.v_defect_q0_array
        self.v_defect_q_q0 = self.v_defect_q_q0_array.grid

        # Compute the difference in the DFT potentials for the cases of host and q=0
        # NOTE: I haven't Used this
        self.v_host_q0 = self.v_host_array.grid - self.v_defect_q0_array.grid
        
        # for Classic USE THIS
        self.v_host_q_array = ( self.v_host_array - self.v_defect_q_array) * 2

        #print (self.v_defect_q0_array)
        # Dict to store model energies
        self.model_energies = {}

        # Dict to store model structures
        self.model_structures = {}

        # Dict to store correction energies
        self.model_correction_energies = {}
        
        ## Dict to store model charges
        self.model_charges = {}
        
        # Dict to store model charges grid
        self.grid_info ={}

        # Dict to store model potentials
        self.v_model={}

        #return

    def __should_run_model(self):
        """
        Return whether a model workchain should be run, which is dependant on the number of model energies computed
        with respect to to the total number of model energies needed.
        """
        return self.model_iteration < self.model_iterations_required

    #================================================
    #  FOR GUASSIAN MODEL  (SIGMA)
    #================================================

    def _compute_model_charge_with_siesta_mesh(self):
        """
        IF SCHEME IS GAUSSIAN-MODEL
        Compute the potential for the system using a model charge distribution
        """
        print("-----------------------------------------")
        print(" Computing Model Charge with SIESTA MESH ") 
        print("-----------------------------------------")
        scale_factor = self.model_iteration
        self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2) 
        dimension = self.v_defect_q0_array.shape
        grid = (dimension[0] * scale_factor,
                dimension[1] * scale_factor,
                dimension[2] * scale_factor)
        if grid[0]%2==0:
            print(f'Siesta has EVEN mesh {grid[0]} changing to odd')
            self.grid_dimension = (grid[0]-1,grid[1]-1,grid[2]-1)
        else:
            print(f'Siesta has ODD mesh {grid[0]} Great ... !')
            self.grid_dimension = grid
        
        self.grid_info[scale_factor] = self.grid_dimension
        self.limits = np.array([self.model_structures [scale_factor].cell[0][0],
                                self.model_structures [scale_factor].cell[1][1],
                                self.model_structures [scale_factor].cell[2][2],
                               ])

        print("Gaussian Class DEBUG: Computing Model Charge for scale factor {}".format(scale_factor))
        print("Gaussian Class DEBUG: Dimension of Grid is {}".format(self.grid_dimension))
        print("Dimension of Grid is {}".format(self.grid_info))
        print ("Gaussian Class DEBUG: limits is {}".format(self.limits))
  
        charge = get_charge_model_sigma (limits = self.limits,
                                        dimensions =  self.grid_dimension,
                                        defect_position =  self.defect_site,
                                        sigma =  self.sigma,
                                        charge = self.defect_charge
                                        )

        self.model_charges[scale_factor] = charge
        
        self.write_model_charge(scale_factor)
        return self.model_charges , self.grid_info

    
    def _compute_model_charge_with_cutoff(self):
       """
       """
       from .utils_gaussian_model import get_reciprocal_grid
       from .utils_gaussian_model import get_charge_model_sigma
       Ang_to_Bohr = unit_convert("Ang","Bohr") #1.88973
       One_Ang_to_Bohr = 1.0 / Ang_to_Bohr
       scale_factor = self.model_iteration
 
       cell = self.host_structure.cell * Ang_to_Bohr
       r_cell = self.host_structure.rcell   * (One_Ang_to_Bohr)

       print("Gaussian Class DEBUG: cell is :\n {}".format(cell))
       print("Gaussian Class DEBUG: rec cell is :\n {}".format(r_cell))
       
       self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2)
       self.grid_dimension = get_reciprocal_grid(self.model_structures [scale_factor].rcell * One_Ang_to_Bohr, self.cutoff)
       print("Gaussian Class DEBUG: The Grid Size is {}".format(self.grid_dimension))

       self.grid_info[scale_factor] = self.grid_dimension
       self.limits = np.array([self.model_structures [scale_factor].cell[0][0]*Ang_to_Bohr,
                               self.model_structures [scale_factor].cell[1][1]*Ang_to_Bohr,
                               self.model_structures [scale_factor].cell[2][2]*Ang_to_Bohr,
                              ])

       print("Gaussian Class DEBUG: Computing Model Charge for scale factor {}".format(scale_factor))
       print("Dimension of Grid is {}".format(self.grid_info))
       print ("Gaussian Class DEBUG: limits is {}".format(self.limits))
       print("Model Structure cell {}".format(self.model_structures[scale_factor].cell*Ang_to_Bohr))
       
       charge = get_charge_model_sigma (limits = self.limits,
                                        dimensions =  self.grid_dimension,
                                        defect_position =  self.defect_site*Ang_to_Bohr,
                                        sigma =  self.sigma,#*Ang_to_Bohr,
                                        charge = self.defect_charge,
                                        cell_matrix = cell
                                        )

       self.model_charges[scale_factor] = charge
       #if scale_factor==1:
       #    self.charge_for_align = self.model_charges[scale_factor]
       #    self.write_model_charge(scale_factor)
       
       self.charge_for_align = self.model_charges[scale_factor]
       self.write_model_charge(scale_factor)



    
       return self.model_charges , self.grid_info

    def _compute_model_charge_tail_with_cutoff(self):
       """
       """
       from .utils_gaussian_model_tail import get_reciprocal_grid
       from .utils_gaussian_model_tail import get_charge_model_sigma
       Ang_to_Bohr = unit_convert("Ang","Bohr") #1.88973
       One_Ang_to_Bohr = 1.0 / Ang_to_Bohr
       scale_factor = self.model_iteration
 
       cell = self.host_structure.cell * Ang_to_Bohr
       r_cell = self.host_structure.rcell   * (One_Ang_to_Bohr)

       print("Gaussian Class DEBUG: cell is :\n {}".format(cell))
       print("Gaussian Class DEBUG: rec cell is :\n {}".format(r_cell))
       
       self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2)
       self.grid_dimension = get_reciprocal_grid(self.model_structures [scale_factor].rcell * One_Ang_to_Bohr, self.cutoff)
       print("Gaussian Class DEBUG: The Grid Size is {}".format(self.grid_dimension))

       self.grid_info[scale_factor] = self.grid_dimension
       self.limits = np.array([self.model_structures [scale_factor].cell[0][0]*Ang_to_Bohr,
                               self.model_structures [scale_factor].cell[1][1]*Ang_to_Bohr,
                               self.model_structures [scale_factor].cell[2][2]*Ang_to_Bohr,
                              ])

       print("Gaussian Class DEBUG: Computing Model Charge for scale factor {}".format(scale_factor))
       print("Dimension of Grid is {}".format(self.grid_info))
       print ("Gaussian Class DEBUG: limits is {}".format(self.limits))
       print("Model Structure cell {}".format(self.model_structures[scale_factor].cell*Ang_to_Bohr))
       
       charge = get_charge_model_sigma (limits = self.limits,
                                        dimensions =  self.grid_dimension,
                                        defect_position =  self.defect_site*Ang_to_Bohr,
                                        sigma =  self.sigma,#*Ang_to_Bohr,
                                        charge = self.defect_charge,
                                        volume = self.model_structures[scale_factor].volume*Ang_to_Bohr**3,
                                        eta = self.eta,
                                        )

       self.model_charges[scale_factor] = charge
       if scale_factor==1:
           self.charge_for_align = self.model_charges[scale_factor]
           self.write_model_charge(scale_factor)
       
       #self.charge_for_align = self.model_charges[scale_factor]
       #self.write_model_charge(scale_factor)



    
       return self.model_charges , self.grid_info


    def _compute_model_charge_general_with_cutoff(self):
       """
       """
       from .utils_gaussian_model_general import get_reciprocal_grid
       from .utils_gaussian_model_general import get_charge_model_sigma
       Ang_to_Bohr = unit_convert("Ang","Bohr") #1.88973
       One_Ang_to_Bohr = 1.0 / Ang_to_Bohr
       scale_factor = self.model_iteration
 
       cell = self.host_structure.cell * Ang_to_Bohr
       r_cell = self.host_structure.rcell   * (One_Ang_to_Bohr)

       print("Gaussian Class DEBUG: cell is :\n {}".format(cell))
       print("Gaussian Class DEBUG: rec cell is :\n {}".format(r_cell))
       
       self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2)
       self.grid_dimension = get_reciprocal_grid(self.model_structures [scale_factor].rcell * One_Ang_to_Bohr, self.cutoff)
       print("Gaussian Class DEBUG: The Grid Size is {}".format(self.grid_dimension))

       self.grid_info[scale_factor] = self.grid_dimension
       self.limits = np.array([self.model_structures [scale_factor].cell[0][0]*Ang_to_Bohr,
                               self.model_structures [scale_factor].cell[1][1]*Ang_to_Bohr,
                               self.model_structures [scale_factor].cell[2][2]*Ang_to_Bohr,
                              ])

       print("Gaussian Class DEBUG: Computing Model Charge for scale factor {}".format(scale_factor))
       print("Dimension of Grid is {}".format(self.grid_info))
       print ("Gaussian Class DEBUG: limits is {}".format(self.limits))
       print("Model Structure cell {}".format(self.model_structures[scale_factor].cell*Ang_to_Bohr))
       
       charge = get_charge_model_sigma (limits = self.limits,
                                        dimensions =  self.grid_dimension,
                                        defect_position =  self.defect_site*Ang_to_Bohr,
                                        sigma =  self.sigma,#*Ang_to_Bohr,
                                        charge = self.defect_charge,
                                        gamma = self.gamma,
                                        eta =self.eta,
                                        )

       self.model_charges[scale_factor] = charge
       if scale_factor==1:
           self.charge_for_align = self.model_charges[scale_factor]
           self.write_model_charge(scale_factor)
       #self.charge_for_align = self.model_charges[scale_factor]
       #self.write_model_charge(scale_factor)



    
       return self.model_charges , self.grid_info





    #================================================
    #  FOR GUASSIAN MODEL-RHO  (FITTING) From REAL
    #  RHO
    #  First Try to FIT!..
    #================================================

    def fit_charge_model(self):       
        """ 
        Fit an anisotropic gaussian to the charge state electron density
        """
        from .utils_gaussian_fit import get_charge_model_fit
        import time
        start_time = time.time()

        fit = get_charge_model_fit(rho_defect_q0  = self.rho_defect_q0_array,
                                   rho_defect_q =  self.rho_defect_q_array,
                                   structure = self.host_structure)

        self.fitted_params = fit['fit']
        self.peak_charge = fit['peak_charge']

        print("Gaussian Shape Fitted via RHO {}".format(fit))
        
        print("--- %s min ---" % ((time.time() - start_time)/60.0))
        print("--- %s seconds ---" % (time.time() - start_time))



    def _compute_model_charge_fit_with_cutoff(self):
       """
       """
       from .utils_gaussian_model import get_reciprocal_grid
       from .utils_gaussian_fit import get_charge_model

       scale_factor = self.model_iteration
 

       cell = self.host_structure.cell * Ang_to_Bohr
       r_cell = self.host_structure.rcell * One_Ang_to_Bohr

       print("Gaussian Class DEBUG: cell is :\n {}".format(cell))
       print("Gaussian Class DEBUG: rec cell is :\n {}".format(r_cell))
 
       
       self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2)
       self.grid_dimension = get_reciprocal_grid(self.model_structures [scale_factor].rcell*One_Ang_to_Bohr , self.cutoff)
       print("Gaussian Class DEBUG: The Grid Size is {}".format(self.grid_dimension))


       self.grid_info[scale_factor] = self.grid_dimension
       
       #self.limits = np.array([self.model_structures [scale_factor].cell[0][0],#*Ang_to_Bohr,
       #                        self.model_structures [scale_factor].cell[1][1],#*Ang_to_Bohr,
       #                        self.model_structures [scale_factor].cell[2][2],#*Ang_to_Bohr,
       #                      ])

       print("Gaussian Class DEBUG: Computing Model Charge for scale factor {}".format(scale_factor))
       print("Dimension of Grid is {}".format(self.grid_info))
       #print ("Gaussian Class DEBUG: limits is {}".format(self.limits))
       print("Model Structure cell {}".format(self.model_structures[scale_factor].cell))
       
       charge = get_charge_model (cell_matrix = self.model_structures [scale_factor].cell*Ang_to_Bohr,
                                  defect_charge = self.defect_charge,
                                  dimensions =  self.grid_dimension,
                                  gaussian_params = (np.array(self.fitted_params)),#*scale_factor ,
                                  peak_charge = self.peak_charge* scale_factor#*Ang_to_Bohr,
                                        )

       self.model_charges[scale_factor] = charge

       self.write_model_charge(scale_factor)

       return self.model_charges , self.grid_info


    def _compute_model_charge_fit_with_siesta_mesh(self):
        """
        """
        from .utils_gaussian_fit import get_charge_model
        scale_factor = self.model_iteration
        
        self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2) 
        dimension = self.v_defect_q0_array.shape

        grid = (dimension[0] * scale_factor,
                dimension[1] * scale_factor,
                dimension[2] * scale_factor)

        if grid[0]%2==0:
            print(f'Siesta has EVEN mesh {grid[0]} changing to odd')
            self.grid_dimension = (grid[0]-1,grid[1]-1,grid[2]-1)
        else:
            print(f'Siesta has ODD mesh {grid[0]} Great ... !')
            self.grid_dimension = grid
        
        self.grid_info[scale_factor] = self.grid_dimension

        rho = self.rho_defect_q_array - self.rho_defect_q0_array

        charge = get_charge_model (cell_matrix = self.model_structures [scale_factor].cell,
                                   peak_charge = self.peak_charge,
                                   defect_charge = self.defect_charge,
                                   dimensions =  self.grid_dimension,
                                   gaussian_params = self.fitted_params
                                        )

        print ('MODEL CHARGE FROM RHO DONE!')

        self.model_charges[scale_factor] = charge

        self.write_model_charge(scale_factor)

        return self.model_charges , self.grid_info

    #=================================================
    #  FOR Direct RHO 
    #  WITH MOVING IN CENTER WITHOUT DOING EXTRA STUFF 
    #=================================================

    def _compute_model_charge_rho_q_q0_with_siesta_mesh(self,shift=(90,90,90)):
        ""
        "For now just moving charge in  in center of bulk if its in corner"
        ""
        from .utils_gaussian_rho import get_shift_initial_rho_model,get_rho_model , shift_prepare
        
        print("DEBUG: with siesta mesh...")
        scale_factor = self.model_iteration
        self.charge_for_align = self.rho_defect_q_q0_array 
        shift = shift_prepare(self.defect_site , self.rho_defect_q_q0_array)
        self.init_charge = get_shift_initial_rho_model( charge_grid = self.rho_defect_q_q0_array.grid, 
                                                        shift_grid = shift,
                                                        geometry =  self.host_structure )
        sub_shift = ((0,scale_factor * self.init_charge.shape[0] - self.init_charge.shape[0]),
                     (0,scale_factor * self.init_charge.shape[1] - self.init_charge.shape[0]),
                     (0,scale_factor * self.init_charge.shape[2] - self.init_charge.shape[0]))
        print(f"DEBUG: scale factor {scale_factor} with sub grid shift {sub_shift}")

        self._charge = get_rho_model ( charge_grid = self.init_charge,
                                geometry = self.host_structure,
                                scale_f =  scale_factor,
                                sub_grid_shift = sub_shift,
                                write_out = True,
                                )
        dimension = self._charge.shape
        grid = (dimension[0], # * scale_factor,
               dimension[1], # * scale_factor,
               dimension[2], # * scale_factor
                )

        self.grid_dimension = grid
        self.grid_info[scale_factor] = self.grid_dimension
       
        self.model_charges[scale_factor] = self._charge.grid / Ang_to_Bohr**3
        self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2)
        print("Gaussian Class DEBUG: Computing Model Charge for scale factor {}".format(scale_factor))
        print("Dimension of Grid is {}".format(self.grid_info))
        print("Model Structure cell {}".format(self.model_structures[scale_factor].cell))


        print ('MODEL CHARGE FROM RHO (Direct) DONE!')

        return self.model_charges , self.grid_info

    def _compute_model_charge_rho_q_q0_with_cutoff_debug(self,shift=(90,90,90)):
        """
        """
        from .utils_gaussian_rho import get_shift_initial_rho_model,get_rho_model , shift_prepare
        from .utils_gaussian_model import get_reciprocal_grid 
        from ..Alignment.utils_alignment import get_interpolation_sisl 
        
        print("DEBUG: with cutoff mesh...")
        
        self.charge_for_align = self.rho_defect_q_q0_array 
        scale_factor = self.model_iteration
        if scale_factor ==1:
            print("Reading RHO from DFT ... initially ...")
            self._charge = self.rho_defect_q_q0_array *-1.0       

        
        else:
            print("Reading RHO from DFT and shift to the center...")
            shift = shift_prepare(self.defect_site , self.rho_defect_q_q0_array)
            self.init_charge = get_shift_initial_rho_model( charge_grid = self.rho_defect_q_q0_array.grid, 
                                                        shift_grid = shift,
                                                        geometry =  self.host_structure )
            sub_shift = ((0,scale_factor * self.init_charge.shape[0] - self.init_charge.shape[0]),
                         (0,scale_factor * self.init_charge.shape[1] - self.init_charge.shape[0]),
                         (0,scale_factor * self.init_charge.shape[2] - self.init_charge.shape[0]))
            print(f"DEBUG: scale factor {scale_factor} with sub grid shift {sub_shift}")

            self._charge = get_rho_model ( charge_grid = self.init_charge,
                                    geometry = self.host_structure,
                                    scale_f =  scale_factor,
                                    sub_grid_shift = sub_shift,
                                    write_out = False,
                                    )
        
        print ("DEBUG Preparing For LOWER GRID")
       
        self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2)
        r_cell = self.model_structures[scale_factor].rcell #* One_Ang_to_Bohr
        if scale_factor ==1:
            lower_grid_dimension = self.rho_defect_q_q0_array.shape
        else:
            lower_grid_dimension = get_reciprocal_grid(r_cell, self.cutoff)
        #if lower_grid_dimension[0]%2==0:
        #    print(f'Siesta has EVEN mesh {lower_grid_dimension} Great...!')
        #
        #else:
        #    print(f'Siesta has ODD mesh {lower_grid_dimension} changing to Even ... !')
        #    lower_grid_dimension = (lower_grid_dimension[0]-1,
        #                            lower_grid_dimension[1]-1,
        #                            lower_grid_dimension[2]-1)
        if lower_grid_dimension[0]%2==0:
            print(f'Siesta has EVEN mesh {lower_grid_dimension} Changing to Odd ...!')
            lower_grid_dimension = (lower_grid_dimension[0]-1,
                                    lower_grid_dimension[1]-1,
                                    lower_grid_dimension[2]-1)
        else:
            print(f'Siesta has ODD mesh {lower_grid_dimension} Great ... !')

        print(f"lower gird is {lower_grid_dimension}")
       
        self._charge_lower_grid = get_interpolation_sisl(self._charge,lower_grid_dimension)
        # Writing in sisl grid 
        self.charge_lower_grid_sisl = sisl.Grid(shape =self._charge_lower_grid.shape,
                #geometry = self.model_structures [scale_factor]
                #geometry = self.defect_structure_charge 
                )
        self.charge_lower_grid_sisl.grid = self._charge_lower_grid.grid
        #if scale_factor ==1:
        #    self.charge_lower_grid_sisl.write(f"Test-{scale_factor}.XSF")
        #self.charge_lower_grid_sisl.write(f"Test-{scale_factor}.XSF")

        dimension = self._charge_lower_grid.shape
        grid = (dimension[0], 
               dimension[1], 
               dimension[2],
                )

        self.grid_dimension = grid
        self.grid_info[scale_factor] = self.grid_dimension
       
        self.model_charges[scale_factor] = -1.0 * self._charge_lower_grid.grid #/ Ang_to_Bohr**3
        #self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2)
        print("Gaussian Class DEBUG: Computing Model Charge for scale factor {}".format(scale_factor))
        print("Dimension of Grid is {}".format(self.grid_info))
        print("Model Structure cell {}".format(self.model_structures[scale_factor].cell))

        return self.model_charges , self.grid_info

        #charge_lower_grid = get_interpolation_sisl(self.rho_defect_q_q0_array ,lower_grid_dimension)
       
        #self.init_charge_before_shift = sisl.Grid ( lower_grid_dimension, geometry = self.host_structure)
        #self.init_charge_before_shift.grid = charge_lower_grid.grid
       
        #shift = shift_prepare( defect_site = self.defect_site , #* Ang_to_Bohr,
        #                       grid = self.init_charge_before_shift)
        #print(f"Grid Shift is {shift}")
        #self.init_charge = get_shift_initial_rho_model( charge_grid = charge_lower_grid.grid, 
        #                                                 shift_grid = shift,
        #                                                 geometry =  self.host_structure )
        #sub_shift = ((0,scale_factor * self.init_charge.shape[0] - self.init_charge.shape[0]),
        #              (0,scale_factor * self.init_charge.shape[1] - self.init_charge.shape[0]),
        #              (0,scale_factor * self.init_charge.shape[2] - self.init_charge.shape[0]))
        #
        #print(f"DEBUG: scale factor {scale_factor} with sub grid shift {sub_shift}")
        #
        #charge = get_rho_model ( charge_grid = self.init_charge,
        #                         geometry = self.host_structure,
        #                         scale_f =  scale_factor,
        #                         sub_grid_shift = sub_shift,
        #                         write_out = True,
        #                         )
        #dimension = charge.shape
        #grid = (dimension[0], # * scale_factor,
        #        dimension[1], # * scale_factor,
        #        dimension[2], # * scale_factor
        #         )
        #
        #self.grid_dimension = grid
        #self.grid_info[scale_factor] = self.grid_dimension
        #
        #self.model_charges[scale_factor] = charge.grid
        #self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2)
        #print("Gaussian Class DEBUG: Computing Model Charge for scale factor {}".format(scale_factor))
        #print("Dimension of Grid is {}".format(self.grid_info))
        #print("Model Structure cell {}".format(self.model_structures[scale_factor].cell))


        #print ('MODEL CHARGE FROM RHO LOWER (Direct) DONE!')
       
        #return self.model_charges , self.grid_info



    def _compute_model_charge_rho_q_q0_with_cutoff(self,shift=(90,90,90)):
       """
       """
       import sisl
       from .utils_gaussian_rho import get_shift_initial_rho_model,get_rho_model,shift_prepare 
       from .utils_gaussian_model import get_reciprocal_grid
       from ..Alignment.utils_alignment import get_interpolation_sisl 
       print ("DEBUG q_q0 LOWER GRID")
       #shift = (36,72,72) # Need to fix this
       scale_factor = self.model_iteration
       
       #self.charge_for_align = self.rho_defect_q_q0_array 
       self.charge_for_align = self.rho_defect_q_q0_array.grid
       
       r_cell = self.host_structure.rcell #* One_Ang_to_Bohr
       lower_grid_dimension = get_reciprocal_grid(r_cell, self.cutoff)
       if lower_grid_dimension[0]%2==0:
            print(f'Siesta has EVEN mesh {lower_grid_dimension} Great...!')

       else:
            print(f'Siesta has ODD mesh {lower_grid_dimension} changing to Even ... !')
            lower_grid_dimension = (lower_grid_dimension[0]-1,lower_grid_dimension[1]-1,lower_grid_dimension[2]-1)
            #lower_grid_dimension = grid
 
       print(f"lower gird is {lower_grid_dimension}")
       charge_lower_grid = get_interpolation_sisl(self.rho_defect_q_q0_array ,lower_grid_dimension)
       
       self.init_charge_before_shift = sisl.Grid ( lower_grid_dimension, geometry = self.host_structure)
       #self.init_charge_before_shift = sisl.Grid ( lower_grid_dimension)
       self.init_charge_before_shift.grid = charge_lower_grid
       
       #shift=( (int(self.defect_site[0]/self.init_charge_before_shift.dcell[0][0]+1 )) - lower_grid_dimension[0]/2,
       
       #shift=(lower_grid_dimension[0]/2,
       #       lower_grid_dimension[1]/2 ,
       #       lower_grid_dimension[2]/2 )
       shift = shift_prepare( defect_site = self.defect_site ,
                              grid = self.init_charge_before_shift)
                               

       print(f"Grid Shift is {shift}")

       self.init_charge = get_shift_initial_rho_model( charge_grid = charge_lower_grid.grid, #self.rho_defect_q_q0_array, 
                                                        shift_grid = shift,
                                                        geometry =  self.host_structure )
       sub_shift = ((0,scale_factor * self.init_charge.shape[0] - self.init_charge.shape[0]),
                     (0,scale_factor * self.init_charge.shape[1] - self.init_charge.shape[0]),
                     (0,scale_factor * self.init_charge.shape[2] - self.init_charge.shape[0]))

       print(f"DEBUG: scale factor {scale_factor} with sub grid shift {sub_shift}")

       charge = get_rho_model ( charge_grid = self.init_charge,
                                geometry = self.host_structure,
                                scale_f =  scale_factor,
                                sub_grid_shift = sub_shift,
                                write_out = False,
                                )
       dimension = charge.shape
       grid = (dimension[0], # * scale_factor,
               dimension[1], # * scale_factor,
               dimension[2], # * scale_factor
                )

       self.grid_dimension = grid
       self.grid_info[scale_factor] = self.grid_dimension
       
       self.model_charges[scale_factor] = charge.grid
       self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2)
       print("Gaussian Class DEBUG: Computing Model Charge for scale factor {}".format(scale_factor))
       print("Dimension of Grid is {}".format(self.grid_info))
       print("Model Structure cell {}".format(self.model_structures[scale_factor].cell))


       print ('MODEL CHARGE FROM RHO LOWER (Direct) DONE!')
       return self.model_charges , self.grid_info

    #=================================================
    #  Calculating The Potential in PBC to Calculate 
    #  the Long Range Interactions
    #=================================================

    def _compute_model_potential(self,charge_model_info,rho=False):
        """
        Main Method For Calculating From CHARGE_MODEL the Potential in
        PBC
        """
        model_potential =  ModelPotential(charge_density = charge_model_info[0],
                       scale_factor = self.model_iteration,
                       structure = self.model_structures[self.model_iteration] ,
                       grid = charge_model_info[1],
                       epsilon = self.epsilon,
                       use_siesta_mesh_cutoff=self.use_siesta_mesh_cutoff,
                       rho = rho)       

        out = model_potential.run()
        self.v_model [self.model_iteration] = model_potential.compute_model_potential()

        return out
        #return self.v_model
    
    # ======================
    # TO DELETE this one
    #========================
    def _compute_model_potential_for_alignment_BUG(self,rho=True):
        """
        """
        import sisl
        from .utils_gaussian_rho import get_shift_initial_rho_model,get_rho_model,shift_prepare 
 
        grid_align = {}
        grid_align[1] = self.charge_for_align.shape
        model_potential =  ModelPotential(charge_density = self.charge_for_align,
                       scale_factor = 1 , #self.model_iteration,
                       structure = self.model_structures[1] ,
                       grid = grid_align , #charge_model_info[1],  #self.charge_for_align.shape, 
                       epsilon = self.epsilon,
                       use_siesta_mesh_cutoff=self.use_siesta_mesh_cutoff,
                       rho = rho)       

        out = model_potential.run()
        #self.v_model ['rho'] = model_potential.compute_model_potential()
        
        v_model_rho =  model_potential.compute_model_potential()
        
        shifted = sisl.Grid(v_model_rho.shape,geometry=self.host_structure)
        shifted.grid = v_model_rho

        self.v_model ['rho_shifted'] = shifted

        shift=shift_prepare( defect_site = self.defect_site ,
                             grid = shifted)

        self.v_model['rho'] = get_shift_initial_rho_model( charge_grid = shifted.grid,
                                                           shift_grid = shift,
                                     geometry =  self.host_structure )

    def _compute_model_potential_for_alignment(self):
        """
        """
        import sisl
        from .utils_gaussian_rho import get_shift_initial_rho_model
        from .utils_gaussian_rho import get_rho_model
        from .utils_gaussian_rho import reverse_shift_prepare  
        
        for_shift = sisl.Grid(self.v_model[1].shape,geometry=self.model_structures[1])
        for_shift.grid = self.v_model[1]
        
        #shift=reverse_shift_prepare( defect_site = self.defect_site ,
        #                             grid = for_shift)
        #self.v_model['long'] = get_shift_initial_rho_model( charge_grid = for_shift.grid,
        #                                                    shift_grid = shift,
        #                                                     geometry =  self.model_structures[1])
        
        self.v_model ['long'] = for_shift

    #=================================================
    #  Calculating The Energy of the 
    #  Long Range Interactions
    #  Scale to Fit with Markov-Payne Equation 
    #  L^{-1}+L^{-3}
    #  & Getting Correction Required for Scaled System 
    #=================================================

    def check_model_potential_energies(self):
        """
        Check if the model potential workchains have finished correctly.
        If yes, assign the outputs to the context
        """
        for ii in range(self.model_iterations_required):
            scale_factor = ii + 1
            print("Gaussian DEBUG: model energies {}".format(self.model_energies[scale_factor]))
            #self.model_structures [scale_factor] = create_model_structure(self.host_structure,scale_factor)
            print("Gaussian DEBUG: model structures {}".format(self.model_structures ))

    def get_isolated_energy(self):
        """
        Fit the calculated model energies and obtain an estimate for the isolated model energy
        """
        from ..utils_correction import fit_energies
        from ..utils_correction import fit_energies_original
        # Get the linear dimensions of the structures
        linear_dimensions = {}

        for scale, structure in self.model_structures.items():
            volume = structure.volume
            linear_dimensions[scale] = 1 / (volume**(1 / 3.))
            print ("Gaussian Class DEBUG: Scale {} Fitting volume {} ".format(scale,volume))
            print ("Gaussian Class DEBUG: Scale {} Fitting linear_dimensions {} ".format(scale,linear_dimensions[scale]))

        print("Fitting the model energies to obtain the model energy for the isolated case")
        if self.E_iso_type == "original":
            print("Fitting with L^{-1}+L^{-3}")
            self.isolated_energy = fit_energies_original(linear_dimensions,
                                                         self.model_energies)
            print("The isolated model energy is {} eV".format(self.isolated_energy))
        if self.E_iso_type == "1/L":
            print("Fitting with L^{-1}")
            self.isolated_energy = fit_energies(linear_dimensions,
                                            self.model_energies,
                                            self.epsilon)
            print("The isolated model energy is {} eV with epsilon {}".format(self.isolated_energy,self.epsilon))


        return self.isolated_energy

    def get_model_corrections(self):
        """
        Get the energy corrections for each model size
        """
        from ..utils_correction import calc_correction
        print("Gaussian Class DEBUG: Computing the required correction for each model size")
        for scale_factor,model_energy in self.model_energies.items():
             self.model_correction_energies [scale_factor] = calc_correction (self.isolated_energy,
                    model_energy )

 
    #=================================================
    #  Calculating The Alignments Stuff
    #  This Part is Tricky!!!
    #=================================================
    #def _compute_dft_difference_potential_host_q0(self):
    #    """
    #    Compute the difference in the DFT potentials for the cases of host and q=0
    #    """
    #    print( "..........................................................................")
    #    print ("Computing Difference in the DFT potentials for the cases of host and q=0 \n")
    #    print( "..........................................................................")
    #    self.v_host_q0 = self.v_host_array.grid - self.v_defect_q0_array.grid
    #    return self.v_host_q0

    #def compute_dft_difference_potential_q_q0(self):
    #    """
    #    Compute the difference in the DFT potentials for the cases of q=q and q=0
    #    """
    #    print( "..........................................................................")
    #    print ("Computing Difference in the DFT potentials for the cases of q=q and q=0 \n")
    #    print( "..........................................................................")
    #    self.v_defect_q_q0 = self.v_defect_q_array.grid - self.v_defect_q0_array.grid
    #    return self.v_defect_q_q0

    
    def compute_alignment_host_q0(self):
        """
        Align the electrostatic potential of the defective material in the q=0 charge
        state with the pristine host system
        """
        print( "..........................................................................")
        print( "Starting Alignment for DEFECT q=0 and PRISTINE \n" )
        #print( "..........................................................................")
        if self.scheme == 'rho':
            #test_1 = self.rho_defect_q0_array.grid - self.rho_defect_q_array.grid
            self._potential_align_host_q0 = PotentialAlignment(first_potential= self.v_defect_q0_array.grid,
                                                               second_potential= self.v_host_array.grid,
                                                               charge_density= self.charge_for_align, #self.model_charges[1],
                                                               scheme = self.potential_align_scheme,
                                                               tolerance = self.charge_dw_tol,
                                                               )
            self.align_host_q0 = self._potential_align_host_q0.run()

            return self.align_host_q0


        else:
            self._potential_align_host_q0 = PotentialAlignment(first_potential= self.v_defect_q0_array.grid,
                                                     second_potential= self.v_host_array.grid,
                                                     charge_density= self.model_charges[1],
                                                     scheme = self.potential_align_scheme,
                                                     tolerance = self.charge_dw_tol,
                                                     )
            self.align_host_q0 = self._potential_align_host_q0.run()

            return self.align_host_q0

    def compute_alignment_model_q_q0(self):
        """
        Align the relative electrostatic potential of the defective material in the q_q0 
        state with the pristine host system
        """
        print( "..........................................................................")
        print( "Starting Alignment for DFT q_q0 with model_V")
        #print( "..........................................................................")
        if self.scheme=='rho':
            print("DEBUG: DFT q_q0 with rho(model) ")
            self._potential_align_model_q_q0 = PotentialAlignment(first_potential= self.v_defect_q_q0,
                                                     #second_potential= self.v_model['rho_shifted'].grid,
                                                     second_potential= self.v_model['long'].grid,
                                                     charge_density= self.charge_for_align,
                                                     scheme = 'FNV', #self.potential_align_scheme,
                                                     tolerance = self.charge_dw_tol,
                                                     #defect_site = self.defect_site,
                                                     )
            self.align_model_q_q0 = self._potential_align_model_q_q0.run()

            return self.align_model_q_q0

        else:
            self._potential_align_model_q_q0 = PotentialAlignment(first_potential= self.v_defect_q_q0,
                                                     second_potential= self.v_model[1],
                                                     charge_density= self.model_charges[1],
                                                     scheme = 'FNV', #self.potential_align_scheme,
                                                     tolerance = self.charge_dw_tol,
                                                     ) 
            self.align_model_q_q0 = self._potential_align_model_q_q0.run()

            return self.align_model_q_q0

    def compute_alignment_q_q0(self):
        """
        Align the relative electrostatic potential of the defective material in the q 
        state with the defective material in the q0 
        """

        print( "..........................................................................")
        print( "Starting Alignment for DFT q with q0")
        #print( "..........................................................................")
        if self.scheme=='rho':
            print("DEBUG: q_q0 with rho(model) ")
            self._potential_align_q_q0 = PotentialAlignment(first_potential= self.v_defect_q0_array.grid,
                                                     second_potential= self.v_defect_q_array.grid,
                                                     charge_density= self.charge_for_align,
                                                     scheme = self.potential_align_scheme,
                                                     tolerance = self.charge_dw_tol,
                                                     )
            self.align_q_q0 = self._potential_align_q_q0.run()

            return self.align_q_q0

        else:
            self._potential_align_q_q0 = PotentialAlignment(first_potential= self.v_defect_q0_array.grid,
                                                     second_potential= self.v_defect_q_array.grid,
                                                     charge_density= self.model_charges[1],
                                                     scheme = self.potential_align_scheme,
                                                     tolerance = self.charge_dw_tol,
                                                     ) 
            self.align_q_q0 = self._potential_align_q_q0.run()

            return self.align_q_q0


    def compute_alignment_dft_model(self):
        """
        Testing DFT to Model
        """
        if self.scheme == 'rho':
            print("DEBUG: dft with rho(model) ")
            potential_alignment_dft_model = PotentialAlignment(first_potential = self.v_defect_q_array.grid,
                                                           second_potential = self.v_model['rho'],
                                                           charge_density = self.charge_for_align,
                                                           scheme = self.potential_align_scheme)

            self.align_dft_mdoel = potential_alignment_dft_model.run()
            return self.align_dft_model
        else:
            potential_alignment_dft_model = PotentialAlignment(first_potential = self.v_defect_q_array.grid,
                                                           second_potential = self.v_model[1],
                                                           charge_density = self.model_charges[1],
                                                           scheme = self.potential_align_scheme)
            self.align_dft_mdoel = potential_alignment_dft_model.run()
            return self.align_dft_model

    def compute_alignment_h_q_model(self):
        """
        Align the relative electrostatic potential of the defective material in the q_q0 
        state with the pristine host system
        """
        print( "..........................................................................")
        print( "Starting Alignment for DFT h_q with model_V")
        #print( "..........................................................................")
        V_mid = shift_prepare(self.defect_site,self.v_host_q_array)


        if self.scheme=='rho':
            print("DEBUG: DFT h_q_ with V_model ")

            self._potential_align_h_q_model = PotentialAlignment(first_potential= self.v_host_q_array.grid,
                                                     second_potential= self.v_model['long'].grid,
                                                     #charge_density= self.charge_for_align,
                                                     scheme = 'Classic', 
                                                     pot_site = V_mid[2],
                                                     avg_plane = self.avg_plane,
                                                     )
            self.align_host_q_model = self._potential_align_h_q_model.run()

            return self.align_host_q_model 

        else:
            self._potential_align_h_q_model = PotentialAlignment(first_potential= self.v_host_q_array.grid,
                                                     second_potential= self.v_model[1],
                                                     #charge_density= self.model_charges[1],
                                                     scheme = 'Classic', 
                                                     pot_site = V_mid[2],
                                                     avg_plane = self.avg_plane,
                                                     ) 
            self.align_host_q_model = self._potential_align_h_q_model.run()

            return self.align_host_q_model 



    #=================================================
    #  Costumized Method for  Calculating / Writining
    #  and Plotting Stuff for Charge and Potentials
    #=================================================

    def write_model_charge(self,scale,name='charge_gaussian'):
        """
        """
        import sisl
        name = f"{name}-{scale}{scale}{scale}.XSF"
        grid = sisl.Grid(self.model_charges[scale].shape)
        grid.grid = self.model_charges[scale]
        grid.set_geometry(self.model_structures [scale])
        grid.write(name)
        print(f"DEBUG file {name} ")


    def plot_eta(self,
            save = False,
            q_q0 = True,
            host_q = False,
            host_q0 = False,
            **kwargs):
        """
        """
        import matplotlib.pyplot as plt
        epsilon = self.epsilon
        eta = self.eta        
        dpi_size=800
        name_graph="eta-charge"
        #title_name_graph = r" " 
        for k,v in kwargs.items():
            if k == 'epsilon':
                epsilon = v
            if k == 'eta':
                eta = v
            if k == 'dpi_size':
                dpi_size = v
            if k == 'name_graph':
                name_graph == v 
            #if k == 'title_name_graph':
            #    title_name_graph = v
            if k == 'Charge':
                Charge_q_q0 = v
                
        charge_eta = (1-0.6*(1/epsilon))*self.defect_charge
        charge_epsilon = (1-1*(1/epsilon))*self.defect_charge
        print(f" charge eta:{charge_eta}\n charge epsilon {charge_epsilon}")


        if q_q0:
            c_avg = np.average(self.rho_defect_q_q0_array.grid*self.host_structure.volume,(0,1))
            #c_avg_mask = np.average(Charge_q_q0_masked,(0,1))
            fig = plt.figure()
            #plt.title(title_name_graph)
            plt.title(r"$q-q_0$")
            ax = plt.subplot(111)
            ax.plot(c_avg,label=r'$\rho_{DFT}$',linestyle="-",alpha=1)
            ax.axhline(charge_eta,label=rf"$(1-\eta/\epsilon)q, \epsilon = {epsilon},\eta = {eta}$",linestyle="-.",color='red')
            ax.axhline(charge_epsilon,label=rf"$(1-1/\epsilon)q, \epsilon = {epsilon}, \eta = 1 $",linestyle="--",color='green')
            ax.set_xlabel(r"$Postition\ z\ [grid]$")
            ax.set_ylabel(r"$Charge\ Density\ [e\ / Volume]$")
            ax.legend()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            #ax.set(ylim=(-0.25,0.25))
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if save:
                plt.savefig(f"{name_graph}-q-q0.jpeg", dpi=dpi_size, bbox_inches='tight')
                plt.savefig(f"{name_graph}-q-q0.png", dpi=dpi_size, bbox_inches='tight')

            plt.show()
        if host_q:
            self.rho_host_q_array = self.rho_host_array - self.rho_defect_q_array              
            c_avg = np.average(self.rho_host_q_array.grid*self.host_structure.volume,(0,1))
            #c_avg = np.average(self.rho_defect_q_q0_array.grid*self.host_structure.volume,(0,1))
            fig = plt.figure()
            plt.title(r"$host-q$")
            #plt.title(title_name_graph)
            ax = plt.subplot(111)
            ax.plot(c_avg,label=r'$\rho_{DFT}$',linestyle="-",alpha=1)
            ax.axhline(charge_eta,label=r"$(1-\eta/\epsilon)q$",linestyle="-.",color='red')
            ax.axhline(charge_epsilon,label=r"$(1-1/\epsilon)q$",linestyle="--",color='green')
            ax.set_xlabel(r"$Postition\ z\ [grid]$")
            ax.set_ylabel(r"$Charge\ Density\ [e\ / V]$")
            ax.legend()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            #ax.set(ylim=(-0.25,0.25))
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if save:
                plt.savefig(f"{name_graph}-h-q.jpeg", dpi=dpi_size, bbox_inches='tight')
                plt.savefig(f"{name_graph}-h-q.png", dpi=dpi_size, bbox_inches='tight')

            plt.show()
        if host_q0:
            self.rho_host_q0_array = self.rho_host_array - self.rho_defect_q0_array              
            c_avg = np.average(self.rho_host_q0_array.grid*self.host_structure.volume,(0,1))
            #c_avg = np.average(self.rho_defect_q_q0_array.grid*self.host_structure.volume,(0,1))
            fig = plt.figure()
            plt.title(r"$host-q_0$")
            #plt.title(title_name_graph)
            ax = plt.subplot(111)
            ax.plot(c_avg,label=r'$\rho_{DFT}$',linestyle="-",alpha=1)
            ax.axhline(charge_eta,label=r"$(1-\eta/\epsilon)q$",linestyle="-.",color='red')
            ax.axhline(charge_epsilon,label=r"$(1-1/\epsilon)q$",linestyle="--",color='green')
            ax.set_xlabel(r"$Postition\ z\ [grid]$")
            ax.set_ylabel(r"$Charge\ Density\ [e\ / V]$")
            ax.legend()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            #ax.set(ylim=(-0.25,0.25))
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if save:
                plt.savefig(f"{name_graph}-h-q0.jpeg", dpi=dpi_size, bbox_inches='tight')
                plt.savefig(f"{name_graph}-h-q0.png", dpi=dpi_size, bbox_inches='tight')

            plt.show()
 
