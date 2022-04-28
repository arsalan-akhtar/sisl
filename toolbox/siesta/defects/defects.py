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
    get_formation_energy_neutral,
)

from .Corrections.Gaussian.gaussian_countercharge_model import GaussianCharge
from .Corrections.Alignment.utils_alignment import get_total_alignment_density_weighted 
from .Corrections.Alignment.utils_alignment import get_total_correction
from .Corrections.Alignment.utils_alignment import get_total_alignment_FNV 
from .Corrections.Alignment.utils_alignment import get_total_alignment_FNV_dft_model_part
import json

from sisl import unit_convert
from .Corrections.Alignment.utils_alignment import get_interpolation_sisl_from_array
import numpy as np
import matplotlib.pyplot as plt

from toolbox.siesta.defects.Corrections.Gaussian.utils_gaussian_rho import shift_prepare
from toolbox.siesta.defects.Corrections.Gaussian.utils_gaussian_rho import get_rotate_grid_sisl

class DefectsFormationEnergyBase:
    """
    The base class to compute the formation energy for a given defect, 
    """
    def __init__(self,
                 host_path,
                 neutralize_defect_path,
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
                 neutralize_defect_fdf_name = None,
                 charge_defect_fdf_name = None,
                 host_output_name = None,
                 neutral_defect_output_name = None,
                 neutralize_defect_output_name = None,
                 charge_defect_output_name = None,
                 cutoff = None,
                 potential_align_scheme=None,
                 E_iso_type=None,
                 avg_plane = None,
                 neutral_defect_path="./",
                 potential_type = "VH",
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
        self.neutralize_defect_path = pathlib.Path(neutralize_defect_path)
        self.charge_defect_path = pathlib.Path(charge_defect_path)
        
        self.host_fdf_name = host_fdf_name
        self.neutral_defect_fdf_name = neutral_defect_fdf_name 
        self.neutralize_defect_fdf_name = neutralize_defect_fdf_name 
        self.charge_defect_fdf_name = charge_defect_fdf_name
        
        self.host_output_name  = host_output_name
        self.neutral_defect_output_name = neutral_defect_output_name 
        self.neutralize_defect_output_name = neutralize_defect_output_name 
        self.charge_defect_output_name = charge_defect_output_name 
        
        self.charge_dw_tol = charge_dw_tol
        self.E_iso_type = E_iso_type
        self.avg_plane = avg_plane
        self.potential_type = potential_type
        #--------------------
        # Structure For Host
        #--------------------
        
        print("DEBUG: initializing structure for Host")
        self.host_structure = sisl.get_sile (self.host_path / host_fdf_name).read_geometry()
        self.host_structure_fdf = sisl.get_sile (self.host_path / host_fdf_name)
        self.host_systemlabel = self.host_structure_fdf.get("SystemLabel")
        if self.host_systemlabel is None:
            self.host_systemlabel = 'siesta'
        print(f"DEBUG: System Label for host {self.host_systemlabel}")
        
        #----------------------
        # Structure For Neutral
        #----------------------
        if self.neutral_defect_path.exists():
            print("DEBUG: initializing structure for Defect Neutral")
            self.defect_structure_neutral = sisl.get_sile (self.neutral_defect_path / neutral_defect_fdf_name).read_geometry(output=True)
            self.defect_structure_neutral_fdf = sisl.get_sile (self.neutral_defect_path / neutral_defect_fdf_name)
            self.defect_neutral_systemlabel  = self.defect_structure_neutral_fdf.get("SystemLabel")
            if self.defect_neutral_systemlabel is None:
                self.defect_neutral_systemlabel = 'siesta'
            print(f"DEBUG: System Label for host {self.defect_neutral_systemlabel}")
        else:
            print("Not Found Neutral Passing initializing structure for Defect Neutral ...!")

        #-------------------------
        # Structure For Neutralize
        #-------------------------
        print("DEBUG: initializing structure for Defect Neutralize")
        self.defect_structure_neutralize = sisl.get_sile (self.neutralize_defect_path / neutralize_defect_fdf_name).read_geometry(output=True)
        self.defect_structure_neutralize_fdf = sisl.get_sile (self.neutralize_defect_path / neutralize_defect_fdf_name)
        self.defect_neutralize_systemlabel  = self.defect_structure_neutralize_fdf.get("SystemLabel")
        if self.defect_neutralize_systemlabel is None:
            self.defect_neutralize_systemlabel = 'siesta'
        print(f"DEBUG: System Label for Defect Neutralize {self.defect_neutralize_systemlabel}")

        #----------------------
        # Structure For Defect
        #----------------------
 
        print("DEBUG: initializing structure for Host")
        self.defect_structure_charge = sisl.get_sile(self.charge_defect_path / charge_defect_fdf_name).read_geometry(output=True)
        self.defect_structure_charge_fdf = sisl.get_sile(self.charge_defect_path / charge_defect_fdf_name)
        self.defect_charge_sytemlabel = self.defect_structure_charge_fdf.get("SystemLabel") 
        if self.defect_charge_sytemlabel is None:
            self.defect_charge_sytemlabel = 'siesta'
        print(f"DEBUG: System Label for charge defect {self.defect_charge_sytemlabel}")
        self.add_or_remove = add_or_remove

        
        # VTs

        print(f"POTENTIAL TYPE IS {self.potential_type}")
        print("DEBUG: initializing VTs for host")
        self.host_VT =  sisl.get_sile(self.initialize_potential ( self.host_path , self.host_systemlabel ))
        print("DEBUG: initializing VTs for neutralize defect")
        self.defect_q0_VT = sisl.get_sile( self.initialize_potential ( self.neutralize_defect_path , self.defect_neutralize_systemlabel ))
        print("DEBUG: initializing VTs for charge defect")
        self.defect_q_VT = sisl.get_sile(self.initialize_potential (self.charge_defect_path , self.defect_charge_sytemlabel) )

        
        # If Rho's Needed
        #if self.correction_scheme == 'gaussian-rho' or self.correction_scheme == 'rho':
        print("DEBUG: initializing Rhos ...")
        self.rho_host = sisl.get_sile(self.initialize_rho( self.host_path, self.host_systemlabel))
        self.rho_defect_q0 = sisl.get_sile(self.initialize_rho ( self.neutralize_defect_path , self.defect_neutralize_systemlabel ))
        self.rho_defect_q = sisl.get_sile( self.initialize_rho (self.charge_defect_path , self.defect_charge_sytemlabel))

    def initialize_potential(self,path,label):
        """
        """
        if self.potential_type == "VT":

            lVT = label+'.VT'
            lnc = label+'.nc'
            lnc = 'TotalPotential.grid.nc'
        if self.potential_type == "VH":
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
                 neutralize_defect_path,
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
                 eta = 1.0,
                 host_fdf_name ='input.fdf',
                 neutral_defect_fdf_name = 'input.fdf',
                 neutralize_defect_fdf_name = 'input.fdf',
                 charge_defect_fdf_name = 'input.fdf',
                 host_output_name  = 'output.out',
                 neutral_defect_output_name = 'output.out',
                 neutralize_defect_output_name = 'output.out',
                 charge_defect_output_name = 'output.out',
                 cutoff = None,
                 fit_params = None,
                 potential_align_scheme = 'FNV',
                 charge_dw_tol = 1.0e-3,
                 E_iso_type = "original",
                 avg_plane = "xy",
                 neutral_defect_path="./",
                 potential_type = "VH",
                 ):
    


        super().__init__(host_path = host_path,
                 neutral_defect_path = neutral_defect_path ,
                 neutralize_defect_path = neutralize_defect_path ,
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
                 neutralize_defect_fdf_name = neutralize_defect_fdf_name ,
                 charge_defect_fdf_name = charge_defect_fdf_name ,
                 host_output_name  = host_output_name,
                 neutral_defect_output_name = neutral_defect_output_name,
                 neutralize_defect_output_name = neutralize_defect_output_name,
                 charge_defect_output_name = charge_defect_output_name,
                 cutoff = cutoff,
                 potential_align_scheme = potential_align_scheme,
                 charge_dw_tol = charge_dw_tol,
                 E_iso_type = E_iso_type,
                 avg_plane = avg_plane,
                 potential_type = potential_type
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
                    E_iso_type = self.E_iso_type,
                    avg_plane = self.avg_plane,
                    rho_host= self.rho_host,
                    rho_defect_q0 = self. rho_defect_q0,
                    rho_defect_q= self.rho_defect_q ,
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
            if self.potential_align_scheme == "Classic":
                self.align_h_q_model = self.Input_Gaussian.compute_alignment_h_q_model()
            else:
                #self.Input_Gaussian._compute_model_potential_for_alignment()
                #self.align_q_q0 = self.Input_Gaussian.compute_alignment_q_q0()
                #self.align_host_q0 = self.Input_Gaussian.compute_alignment_host_q0()
                #self.align_model_q_q0 = self.Input_Gaussian.compute_alignment_model_q_q0()
 
                #self.align_host_q0 = self.Input_Gaussian.compute_alignment_host_q0()
                #self.align_model_q_q0 = self.Input_Gaussian.compute_alignment_model_q_q0()
                #self.align_q_q0 = self.Input_Gaussian.compute_alignment_q_q0()


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
                    E_iso_type = self.E_iso_type,
                    avg_plane = self.avg_plane,
                    rho_host= self.rho_host,
                    rho_defect_q0 = self. rho_defect_q0,
                    rho_defect_q= self.rho_defect_q ,                    
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
            if self.potential_align_scheme == "Classic":
                self.align_h_q_model = self.Input_Gaussian.compute_alignment_h_q_model()
            else:
                #self.Input_Gaussian._compute_model_potential_for_alignment()
                #self.align_q_q0 = self.Input_Gaussian.compute_alignment_q_q0()
                #self.align_host_q0 = self.Input_Gaussian.compute_alignment_host_q0()
                #self.align_model_q_q0 = self.Input_Gaussian.compute_alignment_model_q_q0()
 
                #self.align_host_q0 = self.Input_Gaussian.compute_alignment_host_q0()
                #self.align_model_q_q0 = self.Input_Gaussian.compute_alignment_model_q_q0()
                #self.align_q_q0 = self.Input_Gaussian.compute_alignment_q_q0()


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
                    E_iso_type = self.E_iso_type,
                    avg_plane = self.avg_plane,
                    rho_host= self.rho_host,
                    rho_defect_q0 = self. rho_defect_q0, 
                    rho_defect_q= self.rho_defect_q ,
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

            if self.potential_align_scheme == "Classic":
                self.align_h_q_model = self.Input_Gaussian.compute_alignment_h_q_model()
            else:
                #self.Input_Gaussian._compute_model_potential_for_alignment()
                #self.align_q_q0 = self.Input_Gaussian.compute_alignment_q_q0()
                #self.align_host_q0 = self.Input_Gaussian.compute_alignment_host_q0()
                #self.align_model_q_q0 = self.Input_Gaussian.compute_alignment_model_q_q0()
 
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
                    avg_plane = self.avg_plane
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
            if self.potential_align_scheme == "Classic":
                self.align_h_q_model = self.Input_Gaussian.compute_alignment_h_q_model()
            else:
                #self.Input_Gaussian._compute_model_potential_for_alignment()
                #self.align_q_q0 = self.Input_Gaussian.compute_alignment_q_q0()
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
                    E_iso_type = self.E_iso_type,
                    avg_plane = self.avg_plane
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
            
            if self.potential_align_scheme == "Classic":
                self.align_h_q_model = self.Input_Gaussian.compute_alignment_h_q_model()
            else:
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
        if self.neutral_defect_path.exists():
            self.defect_0_Energy = get_output_energy_manual(path_dir = self.neutral_defect_path,
                                                        output_name = self.neutral_defect_output_name) 

        self.defect_q0_Energy = get_output_energy_manual(path_dir = self.neutralize_defect_path,
                                                         output_name = self.neutralize_defect_output_name) 
        self.defect_q_Energy = get_output_energy_manual( path_dir = self.charge_defect_path,
                                                         output_name = self.charge_defect_output_name)

        self.host_NE = get_output_total_electrons_manual(path_dir = self.host_path, 
                                                         output_name = self.host_output_name )
        # For Neutral Case
        if self.neutral_defect_path.exists():
            self.defect_0_NE = get_output_total_electrons_manual(path_dir = self.neutral_defect_path,
                                                              output_name = self.neutral_defect_output_name)
        # For Neutralize Case
        self.defect_q0_NE = get_output_total_electrons_manual(path_dir = self.neutralize_defect_path,
                                                              output_name = self.neutralize_defect_output_name)
        # For Charge Case
        self.defect_q_NE = get_output_total_electrons_manual(path_dir = self.charge_defect_path,
                                                              output_name = self.charge_defect_output_name)

        self.host_vbm = get_vbm_siesta_manual_bands(path_dir = self.host_path, 
                                                    fdf_name = self.host_fdf_name,
                                                    NE = self.host_NE )
        #self.defect_q0_vbm = get_vbm_siesta_manual_bands(path_dir = self.neutral_defect_path,
        #                                                 fdf_name = self.neutral_defect_fdf_name,
        #                                                 NE = self.defect_q0_NE )
        #self.defect_q_vbm = get_vbm_siesta_manual_bands( path_dir = self.charge_defect_path,
        #                                                 fdf_name = self.charge_defect_fdf_name,
        #                                                 NE = self.defect_q0_NE )
        
        self.host_fermi = get_fermi_siesta_from_fdf( path_dir = self.host_path,
                                                     fdf_name = self.host_fdf_name)
        self.defect_q0_fermi = get_fermi_siesta_from_fdf( path_dir = self.neutralize_defect_path,
                                                          fdf_name = self.neutralize_defect_fdf_name)
        self.defect_q_fermi = get_fermi_siesta_from_fdf( path_dir = self.charge_defect_path,
                                                         fdf_name = self.charge_defect_fdf_name)


        print("======================================= SIESTA DATA =====================================") 
        print("Host Energy    : {} with Total Numeber of Electrons : {}".format(self.host_Energy,self.host_NE))
        if self.neutral_defect_path.exists():
            print("Defect Neutral : {} with Total Numeber of Electrons : {}".format(self.defect_0_Energy,self.defect_0_NE))
        print("Defect Neutralize : {} with Total Numeber of Electrons : {}".format(self.defect_q0_Energy,self.defect_q0_NE))
        print("Defect Charged : {} with Total Numeber of Electrons : {}".format(self.defect_q_Energy, self.defect_q_NE))
        print(".........................................................................................")
        print("Host           VBM {} ".format(self.host_vbm))
        #print("Neutral Defect VBM {} ".format(self.defect_q0_vbm))
        #print("Charged Defect VBM {} ".format(self.defect_q_vbm))
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

    #def calculate_total_alignment_classic(self):
    #    """
    #    """
    #    self.total_alignment = 

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
    
    def calculate_total_correction(self,model=None):
        """

        """
        if model is not None:
            if model:
                print("Adding the Alignment (for models)")
                self.total_correction  = self.Input_Gaussian.model_correction_energies[1] + (self.align_h_q_model * self.defect_charge)
            else:
                print("Subtract the Alignment (for rho)")
                self.total_correction  = self.Input_Gaussian.model_correction_energies[1] - (self.align_h_q_model * self.defect_charge)

        else:
            if self.potential_align_scheme == "Classic" and self.correction_scheme == "rho":
                print("DEBUG:Classic,rho (subtarct)")
                self.total_correction  = self.Input_Gaussian.model_correction_energies[1] - (self.align_h_q_model * self.defect_charge)
            if self.potential_align_scheme == "Classic" and self.correction_scheme == "gaussian-model-tail":
                print("DEBUG:Classic,gaussian-model-tail (add)")
                self.total_correction  = self.Input_Gaussian.model_correction_energies[1] + (self.align_h_q_model * self.defect_charge)
            if self.potential_align_scheme == "Classic" and self.correction_scheme == "gaussian-model":
                print("DEBUG:Classic,gaussian-model (add)")
                self.total_correction  = self.Input_Gaussian.model_correction_energies[1] + (self.align_h_q_model * self.defect_charge)
            if self.potential_align_scheme == "Classic" and self.correction_scheme == "gaussian-model-general":
                print("DEBUG:Classic,gaussian-model-general (add)")
                self.total_correction  = self.Input_Gaussian.model_correction_energies[1] + (self.align_h_q_model * self.defect_charge)



        print ("Total Correction Energy (includeing Alignment) is {}".format(self.total_correction))

        #return self.total_correction

    def calculate_total_correction_old(self):
        """
        """
        #from utils_alignment import get_total_correction
        
        #self.total_correction = get_total_correction(self.iso_energy,self.total_alignment )
 
        self.total_correction = get_total_correction( self.Input_Gaussian.model_correction_energies[1] , self.total_alignment )
        print ("Total Correction Energy (includeing Alignment) is {}".format(self.total_correction))

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
        print(f"DEBUG: Fermi level set to VBM:{self.valence_band_maximum} , fermi level :{self.fermi_level} ")
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


    def calculate_formation_energy_neutral(self):
        """
        """

        self.neutral_fe = get_formation_energy_neutral(defect_energy = self.defect_0_Energy,
                                                   host_energy = self.host_Energy,
                                                   add_or_remove = self.add_or_remove,
                                                   chemical_potential = self.chemical_potential)

        print ("Formation Energy Neutral: {} ".format(self.neutral_fe))

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

    def Plot_V(self,
            hq=True,
            hqq0=False,
            model=True,
            hqmodel=True,
            hq0=False,
            qq0=True,
            qq0model=False,
            avg_plane=None,
            title=r"$Title $",
            save_plot=False,
            dpi_size=800,
            file_name=None,#"potential",
            rotate_angle=None,
            rotate_axes_plane='xy',
            save_grid = False,
            defect_site_location=False,
            ):
        """

            title=r"$2\times2\times2\ unrelax\ V_O^{\bullet\bullet} $"
        """
        if avg_plane is None:
            print("Using user avg plane...")
            avg_plane = self.avg_plane
        if avg_plane =='xy':
            plane = (0,1) 
        if avg_plane =='xz':
            plane = (0,2) 
        if avg_plane =='yz':
            plane = (1,2) 
        
        if file_name is None:
            if self.defect_charge <0:
                q="e"
            if self.defect_charge >0:
                q="p"
            if rotate_angle is not None:
                file_name = f"{self.correction_scheme}-{q}{self.defect_charge}-{rotate_angle}-{rotate_axes_plane}"
            else:
                file_name = f"{self.correction_scheme}-{q}{self.defect_charge}"
 


        print(f"Averaging Plane is {avg_plane}")
        self._V_hq= (self.Input_Gaussian.v_host_array - self.Input_Gaussian.v_defect_q_array)*2 # 2 bcz of VH
        self._V_hq.set_geometry(self.defect_structure_charge)
        self._V_qq0= (self.Input_Gaussian.v_defect_q0_array - self.Input_Gaussian.v_defect_q_array)*2 # 2 bcz of VH *4.0
        self._V_q0h= self.Input_Gaussian.v_defect_q0_array - self.Input_Gaussian.v_host_array
        self._V_qq0_q0h = self._V_qq0+self._V_q0h
        V_model_fix = get_interpolation_sisl_from_array(input_array=self.Input_Gaussian.v_model[1],
                target_shape=self._V_hq.shape)
        self._V_model = sisl.Grid(shape=V_model_fix.shape,
                geometry=self.Input_Gaussian.host_structure)
        self._V_model.grid = V_model_fix.real /unit_convert("Bohr","Ang")**3
        
        if save_grid:
            self._V_hq.write(f"{file_name}-V_hq.XSF")

        if rotate_angle is not None:
            if rotate_axes_plane =='xy':
                r_plane = (0,1) 
            if rotate_axes_plane =='xz':
                r_plane = (0,2) 
            if rotate_axes_plane =='yz':
                r_plane = (1,2) 

            print(f"rotating grids with angle {rotate_angle} ")
            print(f"rotating grids with axes {rotate_axes_plane} = {r_plane} ")
            
            self._V_hq = get_rotate_grid_sisl(grid=self._V_hq,
                                            angle=rotate_angle,
                                            axes = r_plane,
                                            geometry = self.defect_structure_charge,
                                            f_name="V_hq_rotate",
                                            save=save_grid)
            self._V_qq0 = get_rotate_grid_sisl(grid=self._V_qq0,
                                            angle=rotate_angle,
                                            axes = r_plane,
                                            geometry = self.defect_structure_charge,
                                            f_name="V_qq0_rotate",
                                            save=save_grid)
            self._V_q0h = get_rotate_grid_sisl(grid=self._V_q0h,
                                            angle=rotate_angle,
                                            axes = r_plane,
                                            geometry = self.defect_structure_charge,
                                            f_name="V_q0h_rotate",
                                            save=save_grid)

            self._V_qq0_q0h = get_rotate_grid_sisl(grid=self._V_qq0_q0h,
                                            angle=rotate_angle,
                                            axes = r_plane,
                                            geometry = self.defect_structure_charge,
                                            f_name="V_qq0_q0h_rotate",
                                            save=save_grid)

            self._V_model = get_rotate_grid_sisl(grid=self._V_model,
                                            angle=rotate_angle,
                                            axes = r_plane,
                                            geometry = self.defect_structure_charge,
                                            f_name="V_model_rotate",
                                            save=save_grid)



        #===================================
        # Taking Planer AVG
        #===================================
        self._V_hq_avg = np.average(self._V_hq.grid,plane)
        self._V_q0h_avg = np.average(self._V_q0h.grid,plane)
        self._V_qq0_avg = np.average(self._V_qq0.grid,plane)
        self._V_model_avg = np.average(self._V_model.grid.real,plane)
   
        print(f"Alignment {(self._V_hq_avg - self._V_model_avg )[int(self._V_hq_avg.shape[0]/2)]}")

        if avg_plane =='xy':
            x = np.linspace(0,self._V_hq.dcell[2][2]*self._V_hq_avg.shape[0],self._V_hq_avg.shape[0])
        if avg_plane =='xz':
            x = np.linspace(0,self._V_hq.dcell[1][1]*self._V_hq_avg.shape[0],self._V_hq_avg.shape[0])
        if avg_plane =='yz':
            x = np.linspace(0,self._V_hq.dcell[0][0]*self._V_hq_avg.shape[0],self._V_hq_avg.shape[0])


        #===================================
        # Ploting
        #===================================
        fig = plt.figure()
        ax = plt.subplot(111)
        epsilon = r"$\epsilon_{\infty}=$"+f"${self.epsilon}$"
        plt.title(title)
        #================================================
        if hq:
            ax.plot(x,self._V_hq_avg,label=r"$V_{pristine,q}$")
        if qq0:
            ax.plot(x,self._V_qq0_avg,label=r"$V_{q,q0}$")
        if model:
            ax.plot(x,self._V_model_avg,label=r"$V_{\rho}^{model\ DFT}$")
        if hq0:
            ax.plot(x,self._V_q0h_avg,label=r"$V_{q0,pristine}$")
        if hqmodel:
            ax.plot(x,self._V_hq_avg - self._V_model_avg,label=r"$(V_{pristine,q}-V_{\rho}^{model\ DFT})$")
        if hqq0:
            ax.plot(x,self._V_hq_avg-self._V_qq0_avg,label=r"$(V_{pristine,q}-V_{\rho}^{q,q0})$")
        #ax.plot(V_hqr_avg,label=r"$V_{pristine,q}^{relax}$")
        #ax.plot(V_model_s_avg,label=r"$V_{\rho}^{model\ DFT stats}$")
        #ax.plot(V_hqr_avg-V_model_avg,label=r"$(V_{pristine,q}^{relax}-V_{\rho}^{model\ DFT})$")
        #ax.plot(V_model_avg-V_qq0_avg,label=r"$(V_{model}-V_{\rho}^{q,q0})$")
        if qq0model:
            ax.plot(x,self._V_qq0_avg-self._V_model_avg,label=r"$(V_{\rho}^{q,q0}-V_{model})$")
        ax.axhline(linestyle="--",color="black")
        ax.set_ylabel(r"$Energy\ [eV]$")
        ax.set_xlabel(r"$Distance\ [\mathring{A}]$")


        if defect_site_location and rotate_angle is None:
            print("bla")
            s=shift_prepare(self.defect_site,self._V_hq)
            if avg_plane=="xy":
                mid_point = self._V_hq.shape[2] - s[2]
            if avg_plane=="xz":
                mid_point = self._V_hq.shape[1] - s[1]
            if avg_plane=="yz":
                mid_point = self._V_hq.shape[0] - s[0]
            self.align_h_q_model =  (self._V_hq_avg-self._V_model_avg)[mid_point]
            print(f"DEBUG: Alignment : {self.align_h_q_model} eV")
            ax.annotate(f"Alignment : {np.round((self.align_h_q_model),decimals=4)} eV",
                    xy=(x[mid_point],self.align_h_q_model ),
                    xytext=(x[mid_point], self._V_hq_avg.min()),
                    arrowprops=dict(arrowstyle="->"))

        else:
            self.align_h_q_model = (self._V_hq_avg-self._V_model_avg)[int(self._V_model_avg.shape[0]/2)]

            print(f"DEBUG: Alignment : {self.align_h_q_model} eV")
            ax.annotate(f"Alignment : {np.round((self.align_h_q_model),decimals=4)} eV", 
                    xy=(x[int(x.shape[0]/2)],self.align_h_q_model ), 
                    xytext=(x[int(x.shape[0]/2)], self._V_hq_avg.min()),
                    arrowprops=dict(arrowstyle="->"))
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0,box.width * 0.68, box.height])
        #ax.set(ylim=(-4.0,1.0))
        # Put a legend to the right of the current axis

        ax.legend(title=epsilon,loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Defining Name 
        if file_name is None:
            if self.defect_charge <0:
                q="e"
            if self.defect_charge >0:
                q="p"
            if rotate_angle is not None:
                file_name = f"V-avg-{self.correction_scheme}-{q}{self.defect_charge}-{avg_plane}-{rotate_angle}-{rotate_axes_plane}"
            else:
                file_name = f"V-avg-{self.correction_scheme}-{q}{self.defect_charge}-{avg_plane}"
 
        if save_plot:
            plt.savefig(f'{file_name}.png', dpi=dpi_size, bbox_inches='tight' )
            plt.savefig(f'{file_name}.jpeg', dpi=dpi_size, bbox_inches='tight')

        plt.show()

    def write_charge(self,file_name="defect_rho"):
        """
        """
        from toolbox.siesta.defects.Corrections.Gaussian.utils_gaussian_rho import shift_prepare
        from toolbox.siesta.defects.Corrections.Gaussian.utils_gaussian_rho import get_shift_initial_rho_model
        
        if self.defect_charge > 0:
            c = int(self.defect_charge)
            name = f"{file_name}-p{c}.XSF"
            name_s = f"{file_name}-p{c}-shift.XSF"
        
        if self.defect_charge < 0:
            c = int(self.defect_charge) 
            name = f"{file_name}-e{c}.XSF"
            name_s = f"{file_name}-e{c}-shift.XSF"


        self.Input_Gaussian.rho_defect_q_q0_array.set_geometry(self.defect_structure_charge)
        self.Input_Gaussian.rho_defect_q_q0_array.write(name)

        s = shift_prepare(self.defect_site,self.Input_Gaussian.rho_defect_q_q0_array)
        rho_s = get_shift_initial_rho_model(self.Input_Gaussian.rho_defect_q_q0_array.grid,s,self.defect_structure_charge)
        rho_s.write(name_s)

    @staticmethod
    def Plot_Phase_Diagram(PhaseData_name,
            label,
            title_name=r"$Phase\ Diagram$",
            file_name="phase_diagram",
            dpi_size=300,
            save=False,
            neutral_formation=None,
            neutral_label=r"$Neutral$",
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None ):
        """
        Static Method for Ploting Phase Diagram
        PhasData_name : Dictionay
        label : Dictionay
        title_name : Dictionay
        EX : 
           PhaseData_name = {'p1':"Oxygen-remove-p1.0-PhaseData.npy",
                             'p2':"Oxygen-remove-p2.0-PhaseData.npy",}        
           label = [r"$V_{O}^{+1}$",r"$V_{O}^{+2}$"]
        """
        import matplotlib.pyplot as plt
        import numpy as np
    
        ind= []
        fe = np.array([])
        for k,v in PhaseData_name.items():
            print(k)
            ind.append(k)
            fe=np.append(fe,np.load(v,allow_pickle=True))
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.title(f"{title_name}")
        count = 0
        for k,v in PhaseData_name.items():
            #print(k)
            fe=np.append(fe,np.load(v,allow_pickle=True))
            ax.plot(fe[count]['Energy'],fe[count]['Formation'],label=label[count])
            count = count+1
        if neutral_formation is not None :
            ax.axhline(neutral_formation,color='black',label=neutral_label)
        # Shrink current axis by 20%
        ax.set_xlabel(xlabel=r'$\mu_e\ [eV]$')
        ax.set_ylabel(ylabel= r'$Formation\ Energy\ \ [eV]$')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #ax.set(ylim=(3,7.0))
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if y_min is not None and y_max is not None:
            ax.set(ylim=(y_min,y_max))
        if x_min is not None and x_max is not None:
            ax.set(xlim=(x_min,x_max)) 
        #f.tight_layout()
        if save:
            plt.savefig(f'{file_name}.png', dpi=dpi_size, bbox_inches='tight' )
            plt.savefig(f'{file_name}.jpeg', dpi=dpi_size, bbox_inches='tight')

        plt.show()


    def Plot_Charge(self,
            plot_save = False,
            plot_qmodel = False,
            plot_q_q0 = True,
            plot_host_q = False,
            plot_host_q0 = False,
            plot_eta = False,
            **kwargs):
        """
        """
        import matplotlib.pyplot as plt
        epsilon = self.epsilon
        eta = self.eta
        dpi_size=600
        name_graph="eta-charge"
        avg_plane = "xy"
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
            if k == 'Charge':
                Charge_q_q0 = v
            if k == "avg_plane":
                avg_plane = v


        if avg_plane =="xy":
            avg_p = (0,1)
        if avg_plane =="xz":
            avg_p = (0,2)
        if avg_plane =="yz":
            avg_p = (1,2)
        print(f"avg_plane is : {avg_plane},{avg_p}")

        if plot_eta:
            charge_eta = (1-eta*(1/epsilon))*self.defect_charge
            charge_epsilon = (1-1*(1/epsilon))*self.defect_charge
            print(f" charge eta:{charge_eta}\n charge epsilon {charge_epsilon}")

        if self.defect_charge >0 :
            q = "p"
            fname = f"{name_graph}-{q}-{self.defect_charge}"
        if self.defect_charge < 0 :
            q= "e"
            fname = f"{name_graph}-{q}-{self.defect_charge}"
        

        if plot_q_q0:
            c_avg = np.average(self.Input_Gaussian.rho_defect_q_q0_array.grid*self.Input_Gaussian.host_structure.volume,avg_p)
            fig = plt.figure()
            plt.title(r"$q-q_0$")
            ax = plt.subplot(111)
            ax.plot(c_avg,label=r'$\rho_{DFT}$='+f'{self.defect_charge}{q}',linestyle="-",alpha=1)
            if plot_eta:
                ax.axhline(charge_eta,label=rf"$(1-\eta/\epsilon)q, \epsilon = {epsilon},\eta = {eta}$",linestyle="-.",color='red')
                ax.axhline(charge_epsilon,label=rf"$(1-1/\epsilon)q, \epsilon = {epsilon}, \eta = 1 $",linestyle="--",color='green')
            if plot_qmodel:
                inter_q = get_interpolation_sisl_from_array(self.Input_Gaussian.model_charges[1],self.Input_Gaussian.rho_defect_q_q0_array.shape)
                q_avg = np.average(inter_q*self.Input_Gaussian.host_structure.volume,avg_p)
                #q_avg = np.average(self.Input_Gaussian.model_charges[1]*self.Input_Gaussian.host_structure.volume,avg_p)
                #q_avg = np.average(self.Input_Gaussian.model_charges[1],avg_p)
                ax.plot(q_avg,label = r"$\rho_{model}$",linestyle="-.",alpha=1)
            ax.set_xlabel(r"$Postition\ z\ [grid]$")
            ax.set_ylabel(r"$Charge\ Density\ [e\ / Volume]$")
            ax.legend()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            #ax.set(ylim=(-0.25,0.25))
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if plot_save:
                print("Saving the plot")
              
                plt.savefig(f"{fname}-q-q0.jpeg", dpi=dpi_size, bbox_inches='tight')
                plt.savefig(f"{fname}-q-q0.png", dpi=dpi_size, bbox_inches='tight')

            plt.show()
        if plot_host_q:
            self.rho_host_q_array = self.Input_Gaussian.rho_host_array - self.Input_Gaussian.rho_defect_q_array
            c_avg = np.average(self.rho_host_q_array.grid*self.Input_Gaussian.host_structure.volume,avg_p)
            #c_avg = np.average(self.rho_defect_q_q0_array.grid*self.host_structure.volume,(0,1))
            fig = plt.figure()
            plt.title(r"$host-q$")
            #plt.title(title_name_graph)
            ax = plt.subplot(111)
            ax.plot(c_avg,label=r'$\rho_{DFT}$='+f'{self.defect_charge}{q}',linestyle="-",alpha=1)
            if plot_eta:
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
            if plot_save:
                print("Saving the plot")
                plt.savefig(f"{fname}-h-q.jpeg", dpi=dpi_size, bbox_inches='tight')
                plt.savefig(f"{fname}-h-q.png", dpi=dpi_size, bbox_inches='tight')

            plt.show()
        if plot_host_q0:
            self.rho_host_q0_array = self.Input_Gaussian.rho_host_array - self.Input_Gaussian.rho_defect_q0_array
            c_avg = np.average(self.rho_host_q0_array.grid*self.host_structure.volume,avg_p)
            #c_avg = np.average(self.rho_defect_q_q0_array.grid*self.host_structure.volume,(0,1))
            fig = plt.figure()
            plt.title(r"$host-q_0$")
            #plt.title(title_name_graph)
            ax = plt.subplot(111)
            ax.plot(c_avg,label=r'$\rho_{DFT}$='+f'{self.defect_charge}{q}',linestyle="-",alpha=1)
            if plot_eta:
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
            if plot_save:
                print("Saving the plot")
                plt.savefig(f"{fname}-h-q0.jpeg", dpi=dpi_size, bbox_inches='tight')
                plt.savefig(f"{fname}-h-q0.png", dpi=dpi_size, bbox_inches='tight')

            plt.show()

                          
