# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
import pathlib
import sisl
from .utils_defects import (
    get_raw_formation_energy,
    get_corrected_formation_energy,
    get_corrected_aligned_formation_energy,
    get_output_energy_manual,
    get_output_total_electrons_manual,
    get_vbm_siesta_manual_bands,
    get_fermi_siesta_from_fdf,
)

from .Corrections.Gaussian.gaussian_countercharge_model import GaussianCharge
from .Corrections.Alignment.utils_alignment import get_total_alignment_density_weighted 
from .Corrections.Alignment.utils_alignment import get_total_correction
from .Corrections.Alignment.utils_alignment import get_total_alignment_FNV 
from .Corrections.Alignment.utils_alignment import get_total_alignment_FNV_dft_model_part
import json

class DefectsFormationEnergyBase:
    """
    The base class to compute the formation energy for a given defect, 
    """
    def __init__(self,
                 host_path,
                 neutral_defect_path,
                 charge_defect_path,
                 add_or_remove,
                 defect_site,
                 defect_charge,
                 chemical_potential,
                 fermi_level,
                 epsilon,
                 correction_scheme,
                 model_iterations_required,
                 sigma = 0.5,
                 charge_dw_tol = 1.0e-3,
                 gamma = None,
                 eta = None,
                 host_fdf_name  = None,
                 neutral_defect_fdf_name = None,
                 charge_defect_fdf_name = None,
                 host_output_name = None,
                 neutral_defect_output_name = None,
                 charge_defect_output_name = None,
                 cutoff = None,
                 potential_align_scheme=None,
                 E_iso_type=None,
                 ):

        self.defect_site = defect_site 
        self.defect_charge = defect_charge
        self.chemical_potential = chemical_potential
        self.fermi_level = fermi_level
        self.epsilon = epsilon
        self.correction_scheme = correction_scheme 
        self.sigma = sigma
        self.gamma = gamma
        self.eta = eta
        self.cutoff = cutoff
        self.potential_align_scheme = potential_align_scheme 

        self.model_iterations_required = model_iterations_required
        self.host_path = pathlib.Path(host_path)
        self.neutral_defect_path = pathlib.Path(neutral_defect_path)
        self.charge_defect_path = pathlib.Path(charge_defect_path)
        
        self.host_fdf_name = host_fdf_name
        self.neutral_defect_fdf_name = neutral_defect_fdf_name 
        self.charge_defect_fdf_name = charge_defect_fdf_name
        
        self.host_output_name  = host_output_name
        self.neutral_defect_output_name = neutral_defect_output_name 
        self.charge_defect_output_name = charge_defect_output_name 
        
        self.charge_dw_tol = charge_dw_tol
        self.E_iso_type = E_iso_type

        # Structures
        print("DEBUG: initializing structures")
        self.host_structure = sisl.get_sile (self.host_path / host_fdf_name).read_geometry()
        self.host_structure_fdf = sisl.get_sile (self.host_path / host_fdf_name)

        self.host_systemlabel = self.host_structure_fdf.get("SystemLabel")
        if self.host_systemlabel is None:
            self.host_systemlabel = 'siesta'
        print(f"DEBUG: System Label for host {self.host_systemlabel}")
        self.defect_structure_neutral = sisl.get_sile (self.neutral_defect_path / neutral_defect_fdf_name).read_geometry()
        self.defect_structure_neutral_fdf = sisl.get_sile (self.neutral_defect_path / neutral_defect_fdf_name)
        self.defect_neutral_systemlabel  = self.defect_structure_neutral_fdf.get("SystemLabel")
        if self.defect_neutral_systemlabel is None:
            self.defect_neutral_systemlabel = 'siesta'
        print(f"DEBUG: System Label for host {self.defect_neutral_systemlabel}")
        self.defect_structure_charge = sisl.get_sile(self.charge_defect_path / charge_defect_fdf_name).read_geometry()
        self.defect_structure_charge_fdf = sisl.get_sile(self.charge_defect_path / charge_defect_fdf_name)
        self.defect_charge_sytemlabel = self.defect_structure_charge_fdf.get("SystemLabel") 
        if self.defect_charge_sytemlabel is None:
            self.defect_charge_sytemlabel = 'siesta'
        print(f"DEBUG: System Label for charge defect {self.defect_charge_sytemlabel}")
        self.add_or_remove = add_or_remove

        
        # VTs
        print("DEBUG: initializing VTs for host")
        self.host_VT =  sisl.get_sile(self.initialize_potential ( self.host_path , self.host_systemlabel ))
        print("DEBUG: initializing VTs for neutral defect")
        self.defect_q0_VT = sisl.get_sile( self.initialize_potential ( self.neutral_defect_path , self.defect_neutral_systemlabel ))
        print("DEBUG: initializing VTs for charge defect")
        self.defect_q_VT = sisl.get_sile(self.initialize_potential (self.charge_defect_path , self.defect_charge_sytemlabel) )
        
        # If Rho's Needed
        if self.correction_scheme == 'gaussian-rho' or self.correction_scheme == 'rho':
            print("DEBUG: initializing Rhos ...")
            self.rho_host = sisl.get_sile(self.initialize_rho( self.host_path, self.host_systemlabel))
            self.rho_defect_q0 = sisl.get_sile(self.initialize_rho ( self.neutral_defect_path , self.defect_neutral_systemlabel ))
            self.rho_defect_q = sisl.get_sile( self.initialize_rho (self.charge_defect_path , self.defect_charge_sytemlabel))

    def initialize_potential(self,path,label):
        """
        """
        #lVT = label+'.VT'
        #lnc = label+'.nc'
        #lnc = 'TotalPotential.grid.nc'
        lVT = label+'.VH'
        lnc = 'ElectrostaticPotential.grid.nc'
 
        if (path / lVT).exists():
            VT = path / lVT
            return VT
        elif (path/lnc).exists():
            VT = path / lnc
            return VT
        else:
            raise FileExistsError 
    
    def initialize_rho(self,path,label):
        """
        """
        lrho = label+'.RHO'
        lrhonc = label+'.RHOnc'
        if (path / lrho).exists():
            Rho = path / lrho
            return Rho
        elif (path/lrhonc).exists():
            Rho = path / lrhonc
            return Rho
        else:
            raise FileExistsError 


    def setup(self):
        """
        Setup the workchain
        """

        print("DEBUG: Check if correction scheme is valid ...")
        correction_schemes_available = ["gaussian-model",
                                        "gaussian-rho",
                                        "point",
                                        "none",
                                        "rho",
                                        "gaussian-model-tail",
                                        "gaussian-model-general"]
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



# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################

class DefectsFormationEnergy(DefectsFormationEnergyBase):
    """
    """
    def __init__(self,
                 host_path,
                 neutral_defect_path,
                 charge_defect_path,
                 add_or_remove,
                 defect_site,
                 defect_charge,
                 chemical_potential,
                 fermi_level,
                 epsilon,
                 correction_scheme,
                 model_iterations_required = 3 ,
                 sigma = 0.5,
                 gamma = None,
                 eta = None,
                 host_fdf_name ='input.fdf',
                 neutral_defect_fdf_name = 'input.fdf',
                 charge_defect_fdf_name = 'input.fdf',
                 host_output_name  = 'output.out',
                 neutral_defect_output_name = 'output.out',
                 charge_defect_output_name = 'output.out',
                 cutoff = None,
                 fit_params = None,
                 potential_align_scheme = 'FNV',
                 charge_dw_tol = 1.0e-3,
                 E_iso_type = "original",
                 ):
    


        super().__init__(host_path = host_path,
                 neutral_defect_path = neutral_defect_path ,
                 charge_defect_path  = charge_defect_path,
                 add_or_remove = add_or_remove,
                 defect_site = defect_site ,
                 defect_charge = defect_charge,
                 chemical_potential = chemical_potential,
                 fermi_level = fermi_level ,
                 epsilon = epsilon,
                 correction_scheme = correction_scheme,
                 sigma = sigma,
                 gamma = gamma,
                 eta = eta,
                 model_iterations_required = model_iterations_required,
                 host_fdf_name = host_fdf_name,
                 neutral_defect_fdf_name = neutral_defect_fdf_name ,
                 charge_defect_fdf_name = charge_defect_fdf_name ,
                 host_output_name  = host_output_name,
                 neutral_defect_output_name = neutral_defect_output_name,
                 charge_defect_output_name = neutral_defect_output_name,
                 cutoff = cutoff,
                 potential_align_scheme = potential_align_scheme,
                 charge_dw_tol = charge_dw_tol,
                 E_iso_type = E_iso_type 
                 )
                  
        self.fit_params = fit_params
        self.check_cutoff()
        self.Reading_SIESTA_Data()

    def run(self):
        print ("The Defect Correction Package for SIESTA...")
        print ("Starting ...")

        if  DefectsFormationEnergyBase.setup(self) == "none":
            self.Reading_inputs()
            print("There is No Correction Asked I will do Raw Formation Energy..")            
            
            self.Reading_SIESTA_Data()
            self.calculate_uncorrected_formation_energy()
 
            #self.Check()
        elif DefectsFormationEnergyBase.setup(self)=="gaussian-model":
            print("Starting Gaussian Model Correction ..")          
            self.Reading_inputs()
            self.Input_Gaussian = GaussianCharge(
                    v_host = self.host_VT,
                    v_defect_q0 =  self.defect_q0_VT,
                    v_defect_q  = self.defect_q_VT,
                    defect_charge =  self.defect_charge,
                    defect_site = self.defect_site,
                    host_structure = self.host_structure,
                    model_iterations_required = self.model_iterations_required,
                    scheme = 'gaussian-model',
                    sigma = self.sigma, 
                    epsilon = self.epsilon,
                    use_siesta_mesh_cutoff = self.use_siesta_mesh_cutoff,
                    cutoff = self.cutoff,
                    potential_align_scheme = self.potential_align_scheme,
                    charge_dw_tol = self.charge_dw_tol,
                    E_iso_type = self.E_iso_type
                    )

            self.Input_Gaussian.run()

            print("Calculating Isolated Defect Energy ...") 
            self.iso_energy = self.Input_Gaussian.get_isolated_energy()

            print("-------------------------") 
            print("...Starting Alignments...") 
            print("-------------------------") 
            
            # NOTE:
            # Put it in Setup of Gaussian Class
            #self.Input_Gaussian.compute_dft_difference_potential_q_q0()

            self.align_host_q0 = self.Input_Gaussian.compute_alignment_host_q0()
            self.align_model_q_q0 = self.Input_Gaussian.compute_alignment_model_q_q0()
            self.align_q_q0 = self.Input_Gaussian.compute_alignment_q_q0()

        elif DefectsFormationEnergyBase.setup(self)=="gaussian-model-tail":
            print("Starting Gaussian Model tail Correction ..")          
            print(f"Gaussian Model with eta :{self.eta} ..")          
            self.Reading_inputs()
            self.Input_Gaussian = GaussianCharge(
                    v_host = self.host_VT,
                    v_defect_q0 =  self.defect_q0_VT,
                    v_defect_q  = self.defect_q_VT,
                    defect_charge =  self.defect_charge,
                    defect_site = self.defect_site,
                    host_structure = self.host_structure,
                    model_iterations_required = self.model_iterations_required,
                    scheme = 'gaussian-model-tail',
                    sigma = self.sigma, 
                    eta = self.eta,
                    epsilon = self.epsilon,
                    use_siesta_mesh_cutoff = self.use_siesta_mesh_cutoff,
                    cutoff = self.cutoff,
                    potential_align_scheme = self.potential_align_scheme,
                    charge_dw_tol = self.charge_dw_tol,
                    E_iso_type = self.E_iso_type
                    )

            self.Input_Gaussian.run()

            print("Calculating Isolated Defect Energy ...") 
            self.iso_energy = self.Input_Gaussian.get_isolated_energy()

            print("-------------------------") 
            print("...Starting Alignments...") 
            print("-------------------------") 
            
            # NOTE:
            # Put it in Setup of Gaussian Class
            #self.Input_Gaussian.compute_dft_difference_potential_q_q0()

            self.align_host_q0 = self.Input_Gaussian.compute_alignment_host_q0()
            self.align_model_q_q0 = self.Input_Gaussian.compute_alignment_model_q_q0()
            self.align_q_q0 = self.Input_Gaussian.compute_alignment_q_q0()

        elif DefectsFormationEnergyBase.setup(self)=="gaussian-model-general":
            print("Starting Gaussian Model General with gamma and eta Correction ..")          
            print(f"Gaussian Model with gamma :{self.gamma} ..")          
            print(f"Gaussian Model with eta :{self.eta} ..")          
            self.Reading_inputs()
            self.Input_Gaussian = GaussianCharge(
                    v_host = self.host_VT,
                    v_defect_q0 =  self.defect_q0_VT,
                    v_defect_q  = self.defect_q_VT,
                    defect_charge =  self.defect_charge,
                    defect_site = self.defect_site,
                    host_structure = self.host_structure,
                    model_iterations_required = self.model_iterations_required,
                    scheme = 'gaussian-model-general',
                    sigma = self.sigma, 
                    gamma = self.gamma,
                    eta = self.eta,
                    epsilon = self.epsilon,
                    use_siesta_mesh_cutoff = self.use_siesta_mesh_cutoff,
                    cutoff = self.cutoff,
                    potential_align_scheme = self.potential_align_scheme,
                    charge_dw_tol = self.charge_dw_tol,
                    E_iso_type = self.E_iso_type
                    )

            self.Input_Gaussian.run()

            print("Calculating Isolated Defect Energy ...") 
            self.iso_energy = self.Input_Gaussian.get_isolated_energy()

            print("-------------------------") 
            print("...Starting Alignments...") 
            print("-------------------------") 
            
            # NOTE:
            # Put it in Setup of Gaussian Class
            #self.Input_Gaussian.compute_dft_difference_potential_q_q0()

            self.align_host_q0 = self.Input_Gaussian.compute_alignment_host_q0()
            self.align_model_q_q0 = self.Input_Gaussian.compute_alignment_model_q_q0()
            self.align_q_q0 = self.Input_Gaussian.compute_alignment_q_q0()


        elif DefectsFormationEnergyBase.setup(self)=="gaussian-rho":
            print("Starting Gaussian Rho Correction ..")          
            self.Reading_inputs()
            self.Input_Gaussian = GaussianCharge(
                    v_host = self.host_VT,
                    v_defect_q0 =  self.defect_q0_VT,
                    v_defect_q =  self.defect_q_VT,
                    defect_charge =  self.defect_charge,
                    defect_site =  self.defect_site,
                    host_structure =  self.host_structure,
                    scheme = 'gaussian-rho',
                    epsilon = self.epsilon,
                    sigma = None,
                    model_iterations_required = self.model_iterations_required,
                    cutoff = self.cutoff,
                    rho_host = self.rho_host ,
                    rho_defect_q0 = self.rho_defect_q0 , #self.rho_host, 
                    rho_defect_q = self.rho_defect_q ,
                    fit_params = self.fit_params,
                    potential_align_scheme = self.potential_align_scheme,
                    E_iso_type = self.E_iso_type,
                    )
            self.Input_Gaussian.run()

            print("Calculating Isolated Defect Energy ...") 
            self.iso_energy = self.Input_Gaussian.get_isolated_energy()

            print("-------------------------") 
            print("...Starting Alignments...") 
            print("-------------------------") 
 
            # NOTE:
            # Put it in Setup of Gaussian Class
            #self.Input_Gaussian.compute_dft_difference_potential_q_q0()

            self.align_host_q0 = self.Input_Gaussian.compute_alignment_host_q0()
            self.align_model_q_q0 = self.Input_Gaussian.compute_alignment_model_q_q0()
          
        elif DefectsFormationEnergyBase.setup(self)=="rho":
            print("Starting Rho Correction ..")          
            self.Reading_inputs()
            #self.Input_Gaussian = GaussianCounterChargeWorkchain(
            self.Input_Gaussian = GaussianCharge(
                    v_host = self.host_VT,
                    v_defect_q0 = self.defect_q0_VT,
                    v_defect_q = self.defect_q_VT,
                    defect_charge = self.defect_charge,
                    defect_site = self.defect_site,
                    host_structure = self.host_structure,
                    scheme = 'rho',
                    sigma = None,
                    epsilon = self.epsilon,
                    model_iterations_required = self.model_iterations_required,
                    use_siesta_mesh_cutoff = self.use_siesta_mesh_cutoff,
                    rho_host= self.rho_host,
                    rho_defect_q0 = self. rho_defect_q0, 
                    rho_defect_q= self.rho_defect_q ,
                    cutoff = self.cutoff,
                    potential_align_scheme = self.potential_align_scheme,
                    charge_dw_tol = self.charge_dw_tol,
                    E_iso_type = self.E_iso_type
                    )
            self.Input_Gaussian.run()

            print("Calculating Isolated Defect Energy ...") 
            self.iso_energy = self.Input_Gaussian.get_isolated_energy()

            print("-------------------------") 
            print("...Starting Alignments...") 
            print("-------------------------") 

            # NOTE:
            # Put it in Setup of Gaussian Class
            #self.Input_Gaussian.compute_dft_difference_potential_q_q0()

            self.Input_Gaussian._compute_model_potential_for_alignment()
            self.align_host_q0 = self.Input_Gaussian.compute_alignment_host_q0()
            self.align_q_q0 = self.Input_Gaussian.compute_alignment_q_q0()
            self.align_model_q_q0 = self.Input_Gaussian.compute_alignment_model_q_q0()


    def Check(self):
        print ("Checking ")
        print("To Check Defect Charged VBM {} ".format(self.defect_q_vbm))

    def Reading_inputs(self):
        """
        Just Printing User Defined Parameters
        """
        print ("====================================================")
        print("The Defect Charge (q) is  :{}".format(self.defect_charge))
        print("The Defect Site is        :{}".format(self.defect_site))
        print("The Correction Scheme is  :{}".format(self.correction_scheme))
        print("The Correction epsilon is :{}".format(self.epsilon))
        #if self.scheme is not 'gaussian-model':
        print(f"The mesh size is {self.host_VT.read_grid().shape}")
        if self.E_iso_type =="original":
            print(f"The fitting will be with 1/L+1/L^3")
        else:
            print(f"The fitting will be with 1/L")

        print ("====================================================")

    def Reading_SIESTA_Data(self):
        """
        """
        self.host_Energy = get_output_energy_manual(path_dir = self.host_path,
                                                    output_name = self.host_output_name)
        self.defect_q0_Energy = get_output_energy_manual(path_dir = self.neutral_defect_path,
                                                         output_name = self.neutral_defect_output_name) 
        self.defect_q_Energy = get_output_energy_manual( path_dir = self.charge_defect_path,
                                                         output_name = self.charge_defect_output_name)

        self.host_NE = get_output_total_electrons_manual(path_dir = self.host_path, 
                                                         output_name = self.neutral_defect_output_name )
        self.defect_q0_NE = get_output_total_electrons_manual(path_dir = self.neutral_defect_path,
                                                              output_name = self.neutral_defect_output_name)
        self.defect_q_NE = get_output_total_electrons_manual(path_dir = self.charge_defect_path,
                                                              output_name = self.charge_defect_output_name)

        self.host_vbm = get_vbm_siesta_manual_bands(path_dir = self.host_path, 
                                                    fdf_name = self.host_fdf_name,
                                                    NE = self.host_NE )
        self.defect_q0_vbm = get_vbm_siesta_manual_bands(path_dir = self.neutral_defect_path,
                                                         fdf_name = self.neutral_defect_fdf_name,
                                                         NE = self.defect_q0_NE )
        self.defect_q_vbm = get_vbm_siesta_manual_bands( path_dir = self.charge_defect_path,
                                                         fdf_name = self.charge_defect_fdf_name,
                                                         NE = self.defect_q0_NE )
        
        self.host_fermi = get_fermi_siesta_from_fdf( path_dir = self.host_path,
                                                     fdf_name = self.host_fdf_name)
        self.defect_q0_fermi = get_fermi_siesta_from_fdf( path_dir = self.neutral_defect_path,
                                                          fdf_name = self.neutral_defect_fdf_name)
        self.defect_q_fermi = get_fermi_siesta_from_fdf( path_dir = self.charge_defect_path,
                                                         fdf_name = self.charge_defect_fdf_name)


        print("======================================= SIESTA DATA =====================================") 
        print("Host Energy    : {} with Total Numeber of Electrons : {}".format(self.host_Energy,self.host_NE))
        print("Defect Neutral : {} with Total Numeber of Electrons : {}".format(self.defect_q0_Energy,self.defect_q0_NE))
        print("Defect Charged : {} with Total Numeber of Electrons : {}".format(self.defect_q_Energy, self.defect_q_NE))
        print(".........................................................................................")
        print("Host           VBM {} ".format(self.host_vbm))
        print("Neutral Defect VBM {} ".format(self.defect_q0_vbm))
        print("Charged Defect VBM {} ".format(self.defect_q_vbm))
        print(".........................................................................................")
        print("Host           Fermi Energy {} [eV] ".format(self.host_fermi))        
        print("Neutral Defect Fermi Energy {} [eV] ".format(self.defect_q0_fermi))        
        print("Charged Defect Fermi Energy {} [eV] ".format(self.defect_q_fermi))        
        # Now Read Geometry

        
    

    def set_fermi(self,fermi):
        """
        Overwriting fermi level
        """
        self.fermi_level = fermi


    def calculate_total_alignment_density_weighted(self):
        """

        """
        self.total_alignment = get_total_alignment_density_weighted(self.align_model_q_q0,
                                                   self.align_host_q0,
                                                   self.defect_charge)
 
        print ("Total Alignment Energy is {}".format(self.total_alignment))

        #return self.total_alignment
        self.align_host_q0_part = self.align_host_q0
        self.align_dft_model_part = self.align_model_q_q0
        self.align_q_q0_part = self.align_model_q_q0


        print(f"Alignment host_q0            part is = {self.align_host_q0_part}")
        print(f"Alignment q_q0               part is = {self.align_q_q0_part}")
        print(f"Alignment model model_(q_q0) part is = {self.align_dft_model_part}")
        print (f"Total Alignment Energy is            = {self.total_alignment}")
 
    def calculate_total_alignment_FNV_DEBUG(self):
        """

        """
        self.align_host_q0_part = self.align_host_q0
        self.align_dft_model_part = self.align_model_q_q0
        self.align_q_q0_part = self.align_model_q_q0
        print(f"Alignment host_q0            part is = {self.align_host_q0_part}")
        print(f"Alignment model model_(q_q0) part is = {self.align_dft_model_part}")
        print(f"Alignment q_q0               part is = {self.align_q_q0_part}")


        self.total_alignment = get_total_alignment_FNV(self.align_model_q_q0,
                                                       self.align_host_q0,
                                                       self.defect_charge)

                                                   #0.0, #self.align_host_q0,
                                                   #abs(self.defect_charge))
 
        #print ("Total Alignment Energy is {}".format(self.total_alignment))

        #return self.total_alignment

        print (f"Total Alignment Energy is            = {self.total_alignment}")
 
    def calculate_total_alignment_density_weighted_TEST(self):
        """

        """
        self.total_alignment = get_total_alignment_density_weighted(self.align_q_q0,
                                                   self.align_host_q0,
                                                   self.defect_charge)
 
        print ("Total Alignment Energy is {}".format(self.total_alignment))

        #return self.total_alignment
        self.align_host_q0_part = self.align_host_q0
        self.align_dft_model_part = 0.0
        self.align_q_q0_part = self.align_q_q0


        print(f"Alignment host_q0            part is = {self.align_host_q0_part}")
        print(f"Alignment q_q0               part is = {self.align_q_q0_part}")
        print(f"Alignment model model_(q_q0) part is = {self.align_dft_model_part}")
        print (f"Total Alignment Energy is            = {self.total_alignment}")
 
    def calculate_total_alignment_FNV(self,
                                      add_host_q0=False,
                                      add_q_q0=False,
                                      add_dfT_model_q_q0=False,
                                      add_diff=False
                                      ):
        """

        """
        self.total_alignment = 0
        self.align_host_q0_part = 0
        self.align_q_q0_part = 0
        self.align_dft_model_part = 0
        #host_q0 = get_total_alignment_FNV(self.align_host_q0,self.defect_charge)
        #q_q0 = get_total_alignment_FNV(self.align_q_q0,self.defect_charge)
        self.align_host_q0_part = get_total_alignment_FNV_dft_model_part(self.align_host_q0,self.defect_charge)
        self.align_q_q0_part = get_total_alignment_FNV_dft_model_part(self.align_q_q0,self.defect_charge)
        self.align_dft_model_part = get_total_alignment_FNV_dft_model_part(self.align_model_q_q0,self.defect_charge)
        #q_q0 = get_total_alignment_FNV_dft_model_part(self.align_q_q0,self.defect_charge)

        print(f"Alignment host_q0            part is = {self.align_host_q0_part}")
        print(f"Alignment q_q0               part is = {self.align_q_q0_part}")
        print(f"Alignment model model_(q_q0) part is = {self.align_dft_model_part}")
        if add_host_q0:
            self.total_alignment =  self.align_host_q0_part + self.total_alignment 
        if add_q_q0:
            self.total_alignment =  self.align_q_q0_part + self.total_alignment 
        if add_dfT_model_q_q0:
            self.total_alignment =  self.align_dft_model_part + self.total_alignment 

        if add_diff :
            self.total_alignment = self.align_q_q0_part - self.align_host_q0_part

        print (f"Total Alignment Energy is            = {self.total_alignment}")
    
    def calculate_total_alignment_manaul(self,value):
        """
        """
        self.total_alignment = value
        print (f"Total Alignment Energy is            = {self.total_alignment}")
    
    def calculate_total_correction(self):
        """

        """
        #from utils_alignment import get_total_correction
        
        #self.total_correction = get_total_correction(self.iso_energy,self.total_alignment )
        self.total_correction = get_total_correction( self.Input_Gaussian.model_correction_energies[1] , self.total_alignment )

        print ("Total Correction Energy (includeing Alignment) is {}".format(self.total_correction))

        #return self.total_correction


    def calculate_uncorrected_formation_energy(self):
        """
        Calulating uncorrected formation_energy
        """
        #vbm_diff = (self.defect_q_vbm -self.host_vbm)
        #fermi_diff = self.defect_q_fermi- self.host_fermi
        #fermi_shift = self.defect_q_fermi + vbm_diff - fermi_diff
        #self.fermi_level = self.fermi_level + fermi_shift  
        
        #self.valence_band_maximum = self.host_vbm 
        self.valence_band_maximum = self.host_vbm + self.host_fermi 
        self.uncorrected_fe = get_raw_formation_energy( defect_energy = self.defect_q_Energy, 
                                                   host_energy = self.host_Energy, 
                                                   add_or_remove = self.add_or_remove, 
                                                   chemical_potential = self.chemical_potential,
                                                   charge = self.defect_charge, 
                                                   fermi_energy = self.fermi_level, 
                                                   #valence_band_maximum = self.defect_q_vbm
                                                   #valence_band_maximum = self.host_vbm
                                                   valence_band_maximum = self.valence_band_maximum
                                                   )
        print ("Uncorrected Formation Energy : {} ".format(self.uncorrected_fe))
        self.fermi_level = 0.0

    def calculate_corrected_formation_energy(self):
        """
        Calulating Corrected formation_energy
        """
        self.corrected_fe = self.total_correction + self.uncorrected_fe
        
        print ("Corrected Formation Energy : {}".format(self.corrected_fe))



    def SinglePhaseDiagram(self,Emin,Emax,number_of_points=100,Save=True,font=None,specie_name=None,dump_data=True):
        """
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        Formation = np.array([])
        Energy= np.linspace(Emax,Emin,number_of_points)
        for i in Energy:
            self.set_fermi(i)
            self.calculate_uncorrected_formation_energy()
            self.calculate_corrected_formation_energy()
            Formation = np.append(Formation,self.corrected_fe)
        
        f,(ax1,ax2) = plt.subplots(1,2,gridspec_kw={'width_ratios': [5, 1]})
        ax1.plot(Energy,Formation)
        # Fermi line
        #ax1.axvline(x=0.0,color=fermi_color,ls=(fermi_line_style),lw=fermi_line_width)
        #ax1.set(xlabel= r'$E-E_f\ [eV]$',ylabel= r'$PDOS\ \ [states/eV]$')
        ax1.set_xlabel(xlabel=r'$\mu\ [eV]$',fontdict=font)
        ax1.set_ylabel(ylabel= r'$Formation Energy\ \ [eV]$',fontdict=font)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.axes.set_yticklabels([])
        ax2.set_yticks([])
        ax2.axes.set_xticklabels([])
        ax2.set_xticks([])
        f.tight_layout()
        if Save:
            if specie_name is None:
                print("Provide the Name of Defected Specie")
                specie_name=input()
            tos_d={}
            tos_d['name']=specie_name
            tos_d['Energy']=Energy
            tos_d['Formation']=Formation
            #np.save('tos',tos)
            tos=np.array([tos_d]) 
            if self.defect_charge > 0:
                filename = f'{specie_name}-{self.add_or_remove}-p{self.defect_charge}-PhaseData'
                np.save(filename,tos,allow_pickle=True)
                #np.save(specie_name+"-p"+str(self.defect_charge),tos,allow_pickle=True)
            if self.defect_charge < 0:
                filename = f'{specie_name}-{self.add_or_remove}-e{abs(self.defect_charge)}-PhaseData'
                np.save(filename,tos,allow_pickle=True)
                #np.save(specie_name+"-e"+str(self.defect_charge),tos,allow_pickle=True)

            print(f"Data Dumped in File {filename}!")
            print("DONE")

    def Write_info(self,specie,name='data'):
        """
        """

        info = {}
        if self.add_or_remove=='remove':
            info['Add_or_Remove']=str(specie)
        info ['Charge']= str(self.defect_charge)
        info ['Epsilon']= str(self.epsilon)
        #info ['Epsilon']= self.epsilon
        info ['Model_Corrections']= self.Input_Gaussian.model_correction_energies
        info ['Alignment']={'host_q0': str(self.align_host_q0_part),
                            'q_q0': str(self.align_q_q0_part),
                            'model_q_q0': str(self.align_dft_model_part),
                            'total': str(self.total_alignment)}
        info ['Uncorrected_Formation']= str(self.uncorrected_fe)
        info ['Corrected_Formation']= str(self.corrected_fe)
    
        y=json.dumps(info)
        if self.defect_charge>0:
            filename =f'{specie}-{self.add_or_remove}-p{self.defect_charge}-{name}.dat' 
        else:
            filename =f'{specie}-{self.add_or_remove}-e{abs(self.defect_charge)}-{name}.dat' 
        with open(filename, 'w') as outfile:
            json.dump(info, outfile,indent=4)

        print(f"Data Dumped in File {filename}!")
        print("DONE")
