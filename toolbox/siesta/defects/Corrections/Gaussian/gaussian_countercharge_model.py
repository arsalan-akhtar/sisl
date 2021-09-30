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
                 sigma = None ,
                 model_iterations_required = 3,
                 use_siesta_mesh_cutoff = None,
                 siesta_grid = None,
                 rho_host = None,
                 rho_defect_q0 = None,
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
        self.rho_defect_q0 = rho_defect_q0
        self.rho_defect_q = rho_defect_q
        self.cutoff = cutoff 
        self.fit_params = fit_params
    
    def run(self):

        if self.fit_params is not None:
            print("The Fitting Params given by user!")
            is_fit = False
            
            #self.peak_charge = self.fit_params[0]['peakcharge']
            #self.fitted_params  = self.fit_params[0]['fit_params']
            self.peak_charge = self.fit_params['peak_charge']
            self.fitted_params  = self.fit_params['fit']
            print ("The Fitting Paramers are {}".format(self.fitted_params))
            print ("The Peak Charge is {}".format(self.peak_charge))
        else:
            is_fit = True

        is_mesh = self.use_siesta_mesh_cutoff
        self.setup()
        
        #=======================================================================================
        #                                Gaussian Model Sigma
        #=======================================================================================
        if self.scheme == 'gaussian-model': 
            
            print("--------------------********************************--------------------")
            print("                    Starting Gaussian Model Scheme                      ")
            print("--------------------********************************--------------------\n")
            
            print ("Computing MODEL INFO")
            print ("Gaussian Sigma is {}".format(self.sigma))
            
            while (self.should_run_model()):
            
                self.model_iteration +=1
                print ("Calculations For Scale Factor {}".format(self.model_iteration))
                # if using SIESTA MESH TO GENERATE THE MODEL
                if is_mesh:
                    print("DEBUG: Using Siesta Mesh to Calculate the Model Charge")
                    charge_model_info = self.compute_model_charge_with_siesta_mesh()
                    self.model_energies [self.model_iteration] = self.compute_model_potential(charge_model_info)
                else:
                    print("DEBUG: Generating the Mesh to Calculating the Model Charge ")
                    charge_model_info = self.compute_model_charge_with_cutoff()
                    print ("Computing MODEL INFO")
                    self.model_energies [self.model_iteration] = self.compute_model_potential(charge_model_info)
        

            self.check_model_potential_energies()
            print ("DONE\n")


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

           
            if is_fit:

                print ("Start Fitting Gaussian...")
                self.fit_charge_model()
                print ("Fitting Gaussian DONE! ")
            else:
                print("Using Users Parameters of Fit to Generate Charge Model")
                

 

            print ("------------------------------------------------------------\n")
            while (self.should_run_model()):
                
                self.model_iteration +=1

                print ("Calculations For Scale Factor {} ...".format(self.model_iteration))
                print ("Computing CHARGE MODEL for Scale Factor {}".format(self.model_iteration))
                if is_mesh:
                    print("DEBUG: Using Siesta Mesh to Calculate the Model Charge")
                    charge_model_info = self.compute_model_charge_fit_with_siesta_mesh()
                    self.model_energies [self.model_iteration] = self.compute_model_potential(charge_model_info)
                else:
                    print ("Computing POTENTIAL MODEL for Scale Factor {}".format(self.model_iteration))
                    charge_model_info = self.compute_model_charge_fit_with_cutoff()
                    self.model_energies [self.model_iteration] = self.compute_model_potential(charge_model_info)
                    print ("Done For Scale Factor {}".format(self.model_iteration))
                    print ("------------------------------------------------------------\n")
            self.check_model_potential_energies()
            print ("All Scaled Energies Are DONE\n")

        #=======================================================================================
        #                                Gaussian Model Rho
        #=======================================================================================
 
        if self.scheme == 'rho':
            
            print("--------------------********************************---------------------")
            print("                           Starting  RHO Scheme                          ")
            print("--------------------********************************--------------------\n")
            self.rho_host_array = self.rho_host.read_grid().grid
            self.rho_defect_q0_array = self.rho_defect_q0.read_grid().grid
            self.rho_defect_q_array = self.rho_defect_q.read_grid().grid
            self.rho_defect_q_q0_array= self.rho_defect_q_array - self.rho_defect_q0_array
            while (self.should_run_model()):
                
                self.model_iteration +=1

                print ("Calculations For Scale Factor {} ...".format(self.model_iteration))
                print ("Computing CHARGE MODEL for Scale Factor {}".format(self.model_iteration))
                if is_mesh:
                    print("DEBUG: Using Siesta Mesh to Calculate the Model Charge")
                    charge_model_info = self.compute_model_charge_rho_q_q0_with_siesta_mesh()
                    self.model_energies [self.model_iteration] = self.compute_model_potential(charge_model_info,rho=True)
                else:
                    print ("Computing POTENTIAL MODEL for Scale Factor {}".format(self.model_iteration))
                    charge_model_info = self.compute_model_charge_rho_q_q0_with_cutoff()
                    self.model_energies [self.model_iteration] = self.compute_model_potential(charge_model_info,rho=True)
                    print ("Done For Scale Factor {}".format(self.model_iteration))
                    print ("------------------------------------------------------------\n")
            self.check_model_potential_energies()
            print ("All Scaled Energies Are DONE\n")

   

    
        
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



    def compute_model_charge_with_siesta_mesh(self):
        """
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
        
        return self.model_charges , self.grid_info

 
    
    def compute_model_charge_with_cutoff(self):
       """
       """
       from .utils_gaussian_model import get_reciprocal_grid
       from .utils_gaussian_model import get_charge_model_sigma

       scale_factor = self.model_iteration
 

       cell = self.host_structure.cell
       r_cell = self.host_structure.rcell

       print("Gaussian Class DEBUG: cell is :\n {}".format(cell))
       print("Gaussian Class DEBUG: rec cell is :\n {}".format(r_cell))
 
       
       self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2)
       self.grid_dimension = get_reciprocal_grid(self.model_structures [scale_factor].rcell, self.cutoff)
       print("Gaussian Class DEBUG: The Grid Size is {}".format(self.grid_dimension))

       self.grid_info[scale_factor] = self.grid_dimension
       self.limits = np.array([self.model_structures [scale_factor].cell[0][0],
                               self.model_structures [scale_factor].cell[1][1],
                               self.model_structures [scale_factor].cell[2][2],
                             ])


       print("Gaussian Class DEBUG: Computing Model Charge for scale factor {}".format(scale_factor))
       print("Dimension of Grid is {}".format(self.grid_info))
       print ("Gaussian Class DEBUG: limits is {}".format(self.limits))
       print("Model Structure cell {}".format(self.model_structures[scale_factor].cell))
       
       charge = get_charge_model_sigma (limits = self.limits,
                                        dimensions =  self.grid_dimension,
                                        defect_position =  self.defect_site,
                                        sigma =  self.sigma,
                                        charge = self.defect_charge
                                        )

       self.model_charges[scale_factor] = charge

       return self.model_charges , self.grid_info

    def compute_model_charge_fit_with_cutoff(self):
       """
       """
       from .utils_gaussian_model import get_reciprocal_grid
       from .utils_gaussian_fit import get_charge_model

       scale_factor = self.model_iteration
 

       cell = self.host_structure.cell
       r_cell = self.host_structure.rcell

       print("Gaussian Class DEBUG: cell is :\n {}".format(cell))
       print("Gaussian Class DEBUG: rec cell is :\n {}".format(r_cell))
 
       
       self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2)
       self.grid_dimension = get_reciprocal_grid(self.model_structures [scale_factor].rcell, self.cutoff)
       print("Gaussian Class DEBUG: The Grid Size is {}".format(self.grid_dimension))


       self.grid_info[scale_factor] = self.grid_dimension
       
       self.limits = np.array([self.model_structures [scale_factor].cell[0][0],
                               self.model_structures [scale_factor].cell[1][1],
                               self.model_structures [scale_factor].cell[2][2],
                             ])

       print("Gaussian Class DEBUG: Computing Model Charge for scale factor {}".format(scale_factor))
       print("Dimension of Grid is {}".format(self.grid_info))
       print ("Gaussian Class DEBUG: limits is {}".format(self.limits))
       print("Model Structure cell {}".format(self.model_structures[scale_factor].cell))
       
       charge = get_charge_model (cell_matrix = self.model_structures [scale_factor].cell,
                                  peak_charge = self.peak_charge,
                                  defect_charge = self.defect_charge,
                                  dimensions =  self.grid_dimension,
                                  gaussian_params = self.fitted_params
                                        )

       self.model_charges[scale_factor] = charge

       return self.model_charges , self.grid_info





   

    def compute_model_charge_fit_with_siesta_mesh(self):
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

        return self.model_charges , self.grid_info



    def compute_model_charge_rho_q_q0_with_siesta_mesh(self,shift=(90,90,90)):
        ""
        "For now just moving charge in  in center of bulk if its in corner"
        ""
        from .utils_gaussian_rho import get_shift_initial_rho_model,get_rho_model
        
        print("DEBUG: with siesta mesh...")
        scale_factor = self.model_iteration
        self.charge_for_align = self.rho_defect_q_q0_array 
        self.init_charge = get_shift_initial_rho_model( charge_grid = self.rho_defect_q_q0_array, 
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
                                write_out = True,
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


        print ('MODEL CHARGE FROM RHO (Direct) DONE!')

        return self.model_charges , self.grid_info


    def compute_model_charge_rho_q_q0_with_cutoff(self):
       """
       """
       print ("DEBUG q_q0")
       from .utils_gaussian_rho import get_rho_model
       scale_factor = self.model_iteration
 

       cell = self.host_structure.cell
       r_cell = self.host_structure.rcell

       print("Gaussian Class DEBUG: cell is :\n {}".format(cell))
       print("Gaussian Class DEBUG: rec cell is :\n {}".format(r_cell))
       dimension = self.rho_defect_q_q0_array.shape
       grid = (dimension[0], # * scale_factor,
               dimension[1], # * scale_factor,
               dimension[2], # * scale_factor
                )

       #if grid[0]%2==0:
       #    print(f'Siesta has EVEN mesh {grid[0]} changing to odd')
       #    #self.grid_dimension = (grid[0]-1,grid[1]-1,grid[2]-1)
       #    self.grid_dimension = (grid[0]+1,grid[1]+1,grid[2]+1)
       #else:
       #    print(f'Siesta has ODD mesh {grid[0]} Great ... !')
       #    self.grid_dimension = grid
       self.grid_dimension = grid
       #self.grid_dimension = (grid[0]+1,grid[1]+1,grid[2]+1)
        
       
       self.model_structures [scale_factor] = self.host_structure.tile(scale_factor,0).tile(scale_factor,1).tile(scale_factor,2)
       #self.grid_dimension = get_reciprocal_grid(self.model_structures [scale_factor].rcell, self.cutoff)
       #print("Gaussian Class DEBUG: The Grid Size is {}".format(self.grid_dimension))


       self.grid_info[scale_factor] = self.grid_dimension
       
       print("Gaussian Class DEBUG: Computing Model Charge for scale factor {}".format(scale_factor))
       print("Dimension of Grid is {}".format(self.grid_info))
       print("Model Structure cell {}".format(self.model_structures[scale_factor].cell))
       charge = get_rho_model ( charge_grid = self.rho_defect_q_q0_array, 
                                geometry = self.host_structure,
                                scale_f =  scale_factor,
                                write_out = True,  
                                )
       
       self.model_charges[scale_factor] = charge.grid

       print ('MODEL CHARGE FROM RHO (Direct) DONE!')
       
       return self.model_charges , self.grid_info



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



    def compute_model_potential(self,charge_model_info,rho=False):
        """
        """
        model_potential =  ModelPotential(charge_density = charge_model_info[0],
                       scale_factor = self.model_iteration,
                       structure = self.model_structures[self.model_iteration] ,
                       grid = charge_model_info[1],
                       epsilon = self.epsilon,
                       use_siesta_mesh_cutoff=self.use_siesta_mesh_cutoff,
                       rho = rho
                )       

        out = model_potential.run()

        self.v_model [self.model_iteration] = model_potential.compute_model_potential()


        return out
        #return self.v_model
    
    def compute_model_potential_for_alignment(self,rho=True):
        """
        """
        grid_align = {}
        grid_align[1] = self.charge_for_align.shape
        model_potential =  ModelPotential(charge_density = self.charge_for_align,
                       scale_factor = 1 , #self.model_iteration,
                       structure = self.model_structures[1] ,
                       grid = grid_align , #charge_model_info[1],  #self.charge_for_align.shape, 
                       epsilon = self.epsilon,
                       use_siesta_mesh_cutoff=self.use_siesta_mesh_cutoff,
                       rho = rho
                )       

        out = model_potential.run()

        self.v_model ['rho'] = model_potential.compute_model_potential()
     
 
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
        from ..utils_correction import calc_correction
        print("Gaussian Class DEBUG: Computing the required correction for each model size")
        for scale_factor,model_energy in self.model_energies.items():
             self.model_correction_energies [scale_factor] = calc_correction (self.isolated_energy,
                    model_energy )




    def compute_dft_difference_potential_q_q0(self):
        """
        Compute the difference in the DFT potentials for the cases of q=q and q=0
        """
        print( "..........................................................................")
        print ("Computing Difference in the DFT potentials for the cases of q=q and q=0 \n")
        print( "..........................................................................")
        self.v_defect_q_q0 = self.v_defect_q_array.grid - self.v_defect_q0_array.grid
        return self.v_defect_q_q0


    def compute_alignment_host_q0(self):
        """
        Align the electrostatic potential of the defective material in the q=0 charge
        state with the pristine host system
        """
        print( "..........................................................................")
        print( "Computing Defect q=0 and pristine \n" )
        print( "..........................................................................")
        if self.scheme == 'rho':
            self._potential_align_host_q0 = PotentialAlignment(first_potential= self.v_defect_q0_array.grid,
                                                     second_potential= self.v_host_array.grid,
                                                     charge_density= self.charge_for_align,#self.model_charges[1],
                                                     )
            self.align_host_q0 = self._potential_align_host_q0.run()

            return self.align_host_q0


        else:
            self._potential_align_host_q0 = PotentialAlignment(first_potential= self.v_defect_q0_array.grid,
                                                     second_potential= self.v_host_array.grid,
                                                     charge_density= self.model_charges[1],
                                                     )
            self.align_host_q0 = self._potential_align_host_q0.run()

            return self.align_host_q0

    def compute_alignment_host_q0_TEST(self):
        """
        Align the electrostatic potential of the defective material in the q=0 charge
        state with the pristine host system
        """
        print( "..........................................................................")
        print( "Computing Defect q=0 and pristine \n" )
        print( "..........................................................................")
        self._potential_align_host_q0 = PotentialAlignment(first_potential= self.v_defect_q0_array.grid,
                                                     second_potential= self.v_host_array.grid,
                                                     charge_density= self.model_charges[1],
                                                     )
        self.align_host_q0 = self._potential_align_host_q0.run()

        return self.align_host_q0


    def compute_alignment_model_q_q0(self):
        """
        Align the relative electrostatic potential of the defective material in the q_q0 
        state with the pristine host system
        """

        print( "..........................................................................")
        print( "Computing DFT q_q0 with model_V")
        print( "..........................................................................")
        if self.scheme=='rho':
            print("DEBUG: DFT q_q0 with rho(model) ")
            self._potential_align_model_q_q0 = PotentialAlignment(first_potential= self.v_defect_q_q0,
                                                     second_potential= self.v_model['rho'],
                                                     charge_density= self.charge_for_align,
                                                     )
            self.align_model_q_q0 = self._potential_align_model_q_q0.run()

            return self.align_model_q_q0

        else:
            self._potential_align_model_q_q0 = PotentialAlignment(first_potential= self.v_defect_q_q0,
                                                     second_potential= self.v_model[1],
                                                     charge_density= self.model_charges[1],
                                                     )
            self.align_model_q_q0 = self._potential_align_model_q_q0.run()

            return self.align_model_q_q0

    def compute_alignment_dft_model(self):
        """
        Testing DFT to Model
        """
        if self.scheme == 'rho':
            print("DEBUG: dft with rho(model) ")
            potential_alignment_dft_model = PotentialAlignment(first_potential = self.v_defect_q_array.grid,
                                                           second_potential = self.v_model['rho'],
                                                           charge_density = self.charge_for_align)
            self.align_dft_mdoel = potential_alignment_dft_model.run()
            return self.align_dft_model
        else:
            potential_alignment_dft_model = PotentialAlignment(first_potential = self.v_defect_q_array.grid,
                                                           second_potential = self.v_model[1],
                                                           charge_density = self.model_charges[1])
            self.align_dft_mdoel = potential_alignment_dft_model.run()
            return self.align_dft_model



    def write_model_charge(self,scale,name='charge'):
        """
        """
        import sisl
        name = f"{name}-{scale}{scale}{scale}.XSF"
        grid = sisl.Grid(self.model_charges[scale].shape)
        grid.grid = self.model_charges[scale]
        grid.set_geometry(self.model_structures [scale])
        grid.write(name)
        print(f"DEBUG file {name} ")
