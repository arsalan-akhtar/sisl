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
from .model_potential import ModelPotential
from .potential_alignment import PotentialAlignment
from ..Utils.utils_correction import calc_correction
from ..Utils.utils_correction import create_model_structure 
from ..Utils.utils_correction import get_charge_model_sigma_cutoff  
from ..Utils.utils_correction import get_charge_model_sigma  
from ..Utils.utils_correction import fit_energies 
from ..Utils.utils_correction import get_reciprocal_grid
from ..Utils.utils_gaussian_fit  import  get_charge_model_fit
from ..Utils.utils_gaussian_rho  import get_interpolation
from ..Utils.utils_gaussian_fit  import get_charge_model
from ..Utils.utils_model_potential import get_cell_matrix , get_reciprocal_cell

class GaussianCounterChargeWorkchain():
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
                 sigma = None ,
                 model_iterations_required = 3,
                 use_siesta_mesh_cutoff = True,
                 siesta_grid = None,
                 rho_host = None,
                 rho_defect_q = None,
                 cutoff = None,
                 fit_params = None,
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
        self.scheme = scheme
        self.siesta_grid = siesta_grid
        self.rho_host = rho_host
        self.rho_defect_q = rho_defect_q
        self.cutoff = cutoff 
        self.fit_params = fit_params
    
    def run(self):

        if self.fit_params is not None:
            print("The Fitting Params given by user!")
            is_fit = False
            
            self.peak_charge = self.fit_params[0]['peakcharge']
            self.fitted_params  = self.fit_params[0]['fit_params']
            print ("The Fitting Paramers are {}".format(self.fitted_params))
            print ("The Peak Charge is {}".format(self.peak_charge))
        else:
            is_fit = True

        is_mesh = self.is_use_siesta_mesh_cutoff()
        self.setup()

        if self.scheme == 'gaussian-model': 
            
            print("--------------------********************************--------------------")
            print("                    Starting Gaussian Model Scheme                      ")
            print("--------------------********************************--------------------\n")
            
            print ("Computing MODEL INFO")
            print ("Gaussian Sigma is {}".format(self.sigma))
            
            while (self.should_run_model()):
            
                self.model_iteration +=1
                print ("Calculations For Scale Factor {}".format(self.model_iteration))
                if is_mesh:
                    charge_model_info = self.compute_model_charge_with_siesta_mesh()
                    self.model_energies [self.model_iteration] = self.compute_model_potential(charge_model_info)
                else:
                    print("balba")
                    charge_model_info = self.compute_model_charge_with_cutoff()
                    #print ("Computing MODEL INFO")
                    self.model_energies [self.model_iteration] = self.compute_model_potential(charge_model_info)
        

            self.check_model_potential_energies()
            print ("DONE\n")

            #self.get_isolated_energy()
            #return charge_model_info,self.model_energies,self.v_model            
        #return self.model_energies


        if self.scheme == 'gaussian-rho':
            
            print("--------------------********************************--------------------")
            print("                 Starting Fitting Gaussian RHO Scheme                   ")
            print("--------------------********************************--------------------\n")
            
            if is_fit:

                print ("Start Fitting Gaussian...")
                self.fit_charge_model()
                print ("Fitting Gaussian DONE! ")
            else:
                print("Using Users Parameters of Fit")


 

            print ("------------------------------------------------------------\n")
            while (self.should_run_model()):
                
                self.model_iteration +=1

                print ("Start Calculations For Scale Factor {} ...".format(self.model_iteration))
                print ("Computing CHARGE MODEL for Scale Factor {}".format(self.model_iteration))
                charge_model_info = self.compute_model_charge_with_siesta_mesh_and_rho()
                print ("Computing POTENTIAL MODEL for Scale Factor {}".format(self.model_iteration))
                self.model_energies [self.model_iteration] = self.compute_model_potential(charge_model_info)
                print ("Done For Scale Factor {}".format(self.model_iteration))
                print ("------------------------------------------------------------\n")
            self.check_model_potential_energies()
            print ("All Scaled Energies Are DONE\n")

            #self.get_isolated_energy()
            #return charge_model_info,self.model_energies

        if self.scheme == 'rho':
            
            print("--------------------********************************--------------------")
            print("                           Starting RHO Scheme                          ")
            print("--------------------********************************--------------------\n")
            
            self.rho_host_array = self.rho_host.read_grid()
            #self.rho_defect_q0_array = self.rho_defect_q0.read_grid()
            self.rho_defect_q_array = self.rho_defect_q.read_grid()


            
            while (self.should_run_model()):
                #print ("Computting CHARGE MODEL INFO for Scale Factor {}".format())
                charge_model_info = self.compute_model_charge_with_siesta_mesh_and_rho_interpolate()
                print ("Computing MODEL INFO")
                self.model_energies [self.model_iteration] = self.compute_model_potential(charge_model_info)
                #self.model_v                 
            self.check_model_potential_energies()
            print ("DONE\n")


        
    def is_use_siesta_mesh_cutoff(self):
        """

        """
        if self.use_siesta_mesh_cutoff :
            print("Using Siesta Mesh grid Size To Generate The Model Potential")
            return True
        else:
            print("Using {} Ry Cutoff To Generate The Model Potential".format(self.cutoff))
            return False
    
        
    def setup(self):
        """
        Setup the calculation
        """
        print("Checking Correction  ({}) Scheme ...".format(self.scheme))
        ## Verification
        if self.model_iterations_required < 3:
           print('The requested number of iterations, {}, is too low. At least 3 are required to achieve an #adequate data fit'.format(self.model_iterations_required))

        # Track iteration number
        self.model_iteration = 0 

        self.v_host_array = self.v_host.read_grid()
        self.v_defect_q0_array = self.v_defect_q0.read_grid()
        self.v_defect_q_array = self.v_defect_q.read_grid() 

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

        return

    def should_run_model(self):
        """
        Return whether a model workchain should be run, which is dependant on the number of model energies computed
        with respect to to the total number of model energies needed.
        """
        return self.model_iteration < self.model_iterations_required

    
    def compute_model_charge_with_cutoff(self):
       """
       """
       self.model_iteration += 1
       scale_factor = self.model_iteration
 

       cell = get_cell_matrix(self.host_structure)
       r_cell = get_reciprocal_cell(cell)

       print("Gaussian Class DEBUG: cell is :\n {}".format(cell))
       print("Gaussian Class DEBUG: rec cell is :\n {}".format(r_cell))
 
       
       grid_dimensions = get_reciprocal_grid(r_cell, self.cutoff)
       print("Gaussian Class DEBUG: The Grid Size is {}".format(grid_dimensions))

       #dimension = self.v_defect_q0_array.read_grid().shape 
       #grid_dimension = self.v_defect_q0_array.shape
       a=int(grid_dimensions[0]*scale_factor);
       if(a%2==0):
           print("This Number is Even {}".format(a))
           grid = ((grid_dimensions[0] * scale_factor) ,
                   (grid_dimensions[1] * scale_factor) ,
                   (grid_dimensions[2] * scale_factor) )
       else:
           print("Odddddddddd  {}".format(a))
           grid = ((grid_dimensions[0] * scale_factor)+1 ,
                   (grid_dimensions[1] * scale_factor)+1 ,
                   (grid_dimensions[2] * scale_factor)+1 )

        # To Check grids Effects
        #grid = (dimension[0] , 
        #        dimension[1] ,
        #        dimension[2] )


       self.grid_info[scale_factor] = grid

       limits = np.array([self.host_structure.cell[0][0]  ,
                           self.host_structure.cell[1][1] ,
                           self.host_structure.cell[2][2]] )*scale_factor

       self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2)
       
       #limits = np.array([self.model_structures[scale_factor].cell[0][0], 
       #                  self.model_structures[scale_factor].cell[1][1],
       #                  self.model_structures[scale_factor].cell[2][2]]) * bohr_to_ang

       print("Gaussian Class DEBUG: Computing Model Charge for scale factor {}".format(scale_factor))
       print("Gaussian Class DEBUG: Dimension of Grid is {}".format(grid))
       #print("Dimension of Grid is {}".format(self.grid_info))
       print ("Gaussian Class DEBUG: limits is {}".format(limits))

       charge = get_charge_model_sigma_cutoff (limits,
                              grid,
                              self.defect_site,
                              self.sigma,
                              self.defect_charge
                              )

       #print (charge)
       self.model_charges[scale_factor] = charge

       return self.model_charges , self.grid_info



    def compute_model_charge_with_siesta_mesh(self):
        """
        Compute the potential for the system using a model charge distribution
        """
        print("-----------------------------------------")
        print(" Computing Model Charge with SIESTA MESH ") 
        print("-----------------------------------------")
        #self.model_iteration += 1
        scale_factor = self.model_iteration
        #dimension = self.v_defect_q0_array.read_grid().shape 
        dimension = self.v_defect_q0_array.shape 
        grid = (dimension[0] * scale_factor, 
                dimension[1] * scale_factor,
                dimension[2] * scale_factor)

        # To Check grids Effects
        #grid = (dimension[0] , 
        #        dimension[1] ,
        #        dimension[2] )


        self.grid_info[scale_factor] = grid
        limits = np.array([self.host_structure.cell[0][0],
                           self.host_structure.cell[1][1],
                           self.host_structure.cell[2][2]])*scale_factor

        self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2) 
        print("Gaussian Class DEBUG: Computing Model Charge for scale factor {}".format(scale_factor))
        print("Gaussian Class DEBUG: Dimension of Grid is {}".format(grid))
        #print("Dimension of Grid is {}".format(self.grid_info))
        print ("Gaussian Class DEBUG: limits is {}".format(limits))
        
        charge = get_charge_model_sigma (limits, 
                              grid,
                              self.defect_site,
                              self.sigma,
                              self.defect_charge
                              )

        #print (charge)
        self.model_charges[scale_factor] = charge
        
        return self.model_charges , self.grid_info

    
    def compute_model_charge_with_siesta_mesh_and_rho(self):
        """
        """
        

        #self.model_iteration += 1
        scale_factor = self.model_iteration
        
        #dimension = self.v_defect_q0_array.read_grid().shape
        dimension = self.v_defect_q0_array.shape
        #grid = (dimension[0] * scale_factor,
        #        dimension[1] * scale_factor,
        #        dimension[2] * scale_factor)

        # To Check grids Effects
        grid = (dimension[0] ,
                dimension[1] ,
                dimension[2] )


        self.grid_info[scale_factor] = grid
        limits = np.array([self.host_structure.cell[0][0],
                           self.host_structure.cell[1][1],
                           self.host_structure.cell[2][2]])*scale_factor

        self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2) 
        #cell = self.host_structure.cell
        cell = self.model_structures [scale_factor].cell
        #self.charge = get_charge_model( cell , 
        #                           self.peak_charge,
        #                           self.defect_charge,
        #                           grid,
        #                           self.fitted_params
        #        )
        
        self.charge = np.array(get_charge_model( cell , 
                                   self.peak_charge,
                                   self.defect_charge,
                                   grid,
                                   self.fitted_params),dtype=np.uint32)


        print ('MODEL CHARGE FROM RHO DONE!')

        self.model_charges[scale_factor] = self.charge

        #return charge
        return self.model_charges , self.grid_info


    def compute_model_charge_with_siesta_mesh_and_rho_interpolate(self):
        """
        """

        self.model_iteration += 1
        scale_factor = self.model_iteration
        
        #dimension = self.v_defect_q0_array.read_grid().shape
        dimension = self.v_defect_q0_array.shape
        grid = (dimension[0] * scale_factor,
                dimension[1] * scale_factor,
                dimension[2] * scale_factor)

        # To Check grids Effects
        #grid = (dimension[0] ,
        #        dimension[1] ,
        #        dimension[2] )

       
        self.grid_info[scale_factor] = grid
        self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2) 


        #cell = self.host_structure.cell

        rho = self.rho_defect_q_array.grid - self.rho_host_array.grid
        charge = get_interpolation( rho,
                                   grid
                )

        print ('MODEL CHARGE FROM RHO DONE!')

        self.model_charges[scale_factor] = charge

        #return charge
        return self.model_charges , self.grid_info


    def fit_charge_model(self):       
        """ 
        Fit an anisotropic gaussian to the charge state electron density
        """
        import time
        start_time = time.time()

        fit = get_charge_model_fit(
            self.rho_host,
            self.rho_defect_q,
            self.host_structure)

        self.fitted_params = fit['fit']
        self.peak_charge = fit['peak_charge']

        print("Gaussian Shape Fitted via RHO {}".format(fit))
        
        print("--- %s min ---" % ((time.time() - start_time)/60.0))
        print("--- %s seconds ---" % (time.time() - start_time))



    def compute_model_potential(self,charge_model_info):
        """
        """
        model_potential =  ModelPotential(charge_density = charge_model_info[0],
                       scale_factor = self.model_iteration,
                       host_structure = self.model_structures[self.model_iteration] ,
                       #host_structure = self.host_structure ,
                       grid = charge_model_info[1],
                       epsilon = self.epsilon,
                       use_siesta_mesh_cutoff=self.use_siesta_mesh_cutoff
                )       

        out = model_potential.run()

        self.v_model [self.model_iteration] = model_potential.compute_model_potential()

        return out
        #return self.v_model
    

    def check_model_potential_energies(self):
        """
        Check if the model potential workchains have finished correctly.
        If yes, assign the outputs to the context
        """
        for ii in range(self.model_iterations_required):
            scale_factor = ii + 1
            print("Gaussian DEBUG: model energies {}".format(self.model_energies[scale_factor]))
            self.model_structures [scale_factor] = create_model_structure(self.host_structure,scale_factor)


    def get_isolated_energy(self):
        """
        Fit the calculated model energies and obtain an estimate for the isolated model energy
        """
        # Get the linear dimensions of the structures
        linear_dimensions = {}

        for scale, structure in self.model_structures.items():
            volume = structure.volume
            linear_dimensions[scale] = 1 / (volume**(1 / 3.))
            print ("Gaussian Class DEBUG: Scale {} Fitting volume {} ".format(scale,volume))
            print ("Gaussian Class DEBUG: Scale {} Fitting linear_dimensions {} ".format(scale,linear_dimensions[scale]))

        print("Fitting the model energies to obtain the model energy for the isolated case")
        self.isolated_energy = fit_energies(linear_dimensions,
                                            self.model_energies)
        print("The isolated model energy is {} eV".format(self.isolated_energy))

        return self.isolated_energy


    def get_model_corrections(self):
        """
        Get the energy corrections for each model size
        """
        print("Gaussian Class DEBUG: Computing the required correction for each model size")
        for scale_factor,model_energy in self.model_energies.items():
             self.model_correction_energies [scale_factor] = calc_correction (self.isolated_energy,
                    model_energy )

             #self.model_correction_energies [scale_factor] = calc_correction (self.get_isolated_energy(),
             #       model_energy )




    def compute_dft_difference_potential_q_q0(self):
        """
        Compute the difference in the DFT potentials for the cases of q=q and q=0
        """
        self.v_defect_q_q0 = self.v_defect_q_array.grid - self.v_defect_q0_array.grid

        print ("Computing Difference in the DFT potentials for the cases of q=q and q=0 \n")

        return self.v_defect_q_q0


    def compute_alignment_host_q0(self):
        """
        Align the electrostatic potential of the defective material in the q=0 charge
        state with the pristine host system
        """
        potential_align_host_q0 = PotentialAlignment(first_potential= self.v_defect_q0_array.grid,
                                                     second_potential= self.v_host_array.grid,
                                                     charge_density= self.model_charges[1],
                                                     )
        self.align_host_q0 = potential_align_host_q0.run()

        return self.align_host_q0

    def compute_alignment_model_q_q0(self):
        """
        Align the relative electrostatic potential of the defective material in the q_q0 
        state with the pristine host system
        """

        potential_align_model_q_q0 = PotentialAlignment(first_potential= self.v_defect_q_q0,
                                                     second_potential= self.v_model[1],
                                                     charge_density= self.model_charges[1],
                                                     )
        self.align_model_q_q0 = potential_align_model_q_q0.run()

        return self.align_model_q_q0



