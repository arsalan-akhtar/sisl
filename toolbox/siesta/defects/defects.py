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
from .Corrections.Alignment.utils_alignment import get_total_alignment 
from .Corrections.Alignment.utils_alignment import get_total_correction

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
                 sigma = 0.5,
                 host_fdf_name  = None,
                 neutral_defect_fdf_name = None,
                 charge_defect_fdf_name = None,
                 host_output_name = None,
                 neutral_defect_output_name = None,
                 charge_defect_output_name = None,
                 cutoff = None,
                 ):

        self.defect_site = defect_site 
        self.defect_charge = defect_charge
        self.chemical_potential = chemical_potential
        self.fermi_level = fermi_level
        self.epsilon = epsilon
        self.correction_scheme = correction_scheme 
        self.sigma = sigma
        self.cutoff = cutoff


        self.host_path = pathlib.Path(host_path)
        self.neutral_defect_path = pathlib.Path(neutral_defect_path)
        self.charge_defect_path = pathlib.Path(charge_defect_path)
        
        self.host_fdf_name = host_fdf_name
        self.neutral_defect_fdf_name = neutral_defect_fdf_name 
        self.charge_defect_fdf_name = charge_defect_fdf_name
        
        self.host_output_name  = host_output_name
        self.neutral_defect_output_name = neutral_defect_output_name 
        self.charge_defect_output_name = charge_defect_output_name 
 

        # Structures
        print("DEBUG: initializing structures")
        self.host_structure = sisl.get_sile (self.host_path / host_fdf_name).read_geometry()
        self.host_structure_fdf = sisl.get_sile (self.host_path / host_fdf_name)
        self.host_systemlabel = self.host_structure_fdf.get("SystemLabel")
        print(f"DEBUG: System Label for host {self.host_systemlabel}")
        self.defect_structure_neutral = sisl.get_sile (self.neutral_defect_path / neutral_defect_fdf_name).read_geometry()
        self.defect_structure_neutral_fdf = sisl.get_sile (self.neutral_defect_path / neutral_defect_fdf_name)
        self.defect_neutral_systemlabel  = self.defect_structure_neutral_fdf.get("SystemLabel")
        print(f"DEBUG: System Label for host {self.defect_neutral_systemlabel}")
        self.defect_structure_charge = sisl.get_sile(self.charge_defect_path / charge_defect_fdf_name).read_geometry()
        self.defect_structure_charge_fdf = sisl.get_sile(self.charge_defect_path / charge_defect_fdf_name)
        self.defect_charge_sytemlabel = self.defect_structure_charge_fdf.get("SystemLabel") 
        print(f"DEBUG: System Label for charge defect {self.defect_charge_sytemlabel}")
        self.add_or_remove = add_or_remove

        
        # VTs
        print("DEBUG: initializing VTs for host")
        self.host_VT =  sisl.get_sile(self.initialize_potential ( self.host_path , self.host_systemlabel ))
        print("DEBUG: initializing VTs for neutral defect")
        self.defect_q0_VT = sisl.get_sile( self.initialize_potential ( self.neutral_defect_path , self.defect_neutral_systemlabel ))
        print("DEBUG: initializing VTs for charge defect")
        self.defect_q_VT = sisl.get_sile(self.initialize_potential (self.charge_defect_path , self.defect_charge_sytemlabel) )
       
        if self.correction_scheme == 'gaussian-rho' or self.correction_scheme == 'rho':
            print("DEBUG: initializing Rhos ...")
            self.rho_host = sisl.get_sile(self.initialize_rho( self.host_path, self.host_systemlabel))
            self.rho_defect_q0 = sisl.get_sile(self.initialize_rho ( self.neutral_defect_path , self.defect_neutral_systemlabel ))
            self.rho_defect_q = sisl.get_sile( self.initialize_rho (self.charge_defect_path , self.defect_charge_sytemlabel))

    def initialize_potential(self,path,label):
        """
        """
        lVT = label+'.VT'
        lnc = label+'.nc'
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
                 sigma = 0.5,
                 host_fdf_name ='input.fdf',
                 neutral_defect_fdf_name = 'input.fdf',
                 charge_defect_fdf_name = 'input.fdf',
                 host_output_name  = 'output.out',
                 neutral_defect_output_name = 'output.out',
                 charge_defect_output_name = 'output.out',
                 cutoff = None,
                 fit_params = None
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
                 host_fdf_name = host_fdf_name,
                 neutral_defect_fdf_name = neutral_defect_fdf_name ,
                 charge_defect_fdf_name = charge_defect_fdf_name ,
                 host_output_name  = host_output_name,
                 neutral_defect_output_name = neutral_defect_output_name,
                 charge_defect_output_name = neutral_defect_output_name,
                 cutoff = cutoff,
                 )
                  
        self.fit_params = fit_params
        self.check_cutoff()


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
            #self.Input_Gaussian = GaussianCounterChargeWorkchain(
            self.Input_Gaussian = GaussianCharge(
                    v_host = self.host_VT,
                    v_defect_q0 =  self.defect_q0_VT,
                    v_defect_q  = self.defect_q_VT,
                    defect_charge =  self.defect_charge,
                    defect_site = self.defect_site,
                    host_structure = self.host_structure,
                    scheme = 'gaussian-model',
                    sigma = self.sigma, 
                    epsilon = self.epsilon,
                    use_siesta_mesh_cutoff = self.use_siesta_mesh_cutoff,
                    cutoff = self.cutoff
                    )
            
            self.Input_Gaussian.run()
            
            print("Calculating Isolated Defect Energy ...") 
            
            self.iso_energy = self.Input_Gaussian.get_isolated_energy()
            
            print("-------------------------") 
            print("...Starting Alignments...") 
            print("-------------------------") 
            
            self.Input_Gaussian.compute_dft_difference_potential_q_q0()
            self.align_host_q0 = self.Input_Gaussian.compute_alignment_host_q0()
            self.align_model_q_q0 = self.Input_Gaussian.compute_alignment_model_q_q0()
            

        elif DefectsFormationEnergyBase.setup(self)=="gaussian-rho":
            print("Starting Gaussian Rho Correction ..")          
            self.Reading_inputs()
            #self.Input_Gaussian = GaussianCounterChargeWorkchain(
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
                    model_iterations_required = 3, #self.model_iterations_required
                    cutoff = self.cutoff,
                    rho_host = self.rho_host ,
                    rho_defect_q0 = self.rho_defect_q0 , #self.rho_host, 
                    rho_defect_q = self.rho_defect_q ,
                    fit_params = self.fit_params,
                    )
            self.Input_Gaussian.run()

            self.Input_Gaussian.compute_dft_difference_potential_q_q0()
            
            self.iso_energy = self.Input_Gaussian.get_isolated_energy()
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
                    model_iterations_required = 3,
                    use_siesta_mesh_cutoff = self.use_siesta_mesh_cutoff,
                    rho_host= self.rho_host,
                    rho_defect_q0 = self. rho_defect_q0, 
                    rho_defect_q= self.rho_defect_q ,

                    )
            self.Input_Gaussian.run()
            
            self.Input_Gaussian.compute_model_potential_for_alignment()
            self.Input_Gaussian.compute_dft_difference_potential_q_q0()
            
            self.iso_energy = self.Input_Gaussian.get_isolated_energy()
            self.align_host_q0 = self.Input_Gaussian.compute_alignment_host_q0()
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


    def calculate_total_alignment(self):
        """

        """
        self.total_alignment = get_total_alignment(self.align_model_q_q0,
                                                   self.align_host_q0,
                                                   self.defect_charge)
 
        print ("Total Alignment Energy is {}".format(self.total_alignment))

        #return self.total_alignment
 

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

        self.uncorrected_fe = get_raw_formation_energy( defect_energy = self.defect_q_Energy, 
                                                   host_energy = self.host_Energy, 
                                                   add_or_remove = self.add_or_remove, 
                                                   chemical_potential = self.chemical_potential,
                                                   charge = self.defect_charge, 
                                                   fermi_energy = self.fermi_level, 
                                                   valence_band_maximum = self.defect_q_vbm
                                                   )
        print ("Uncorrected Formation Energy : {} ".format(self.uncorrected_fe))


    def calculate_corrected_formation_energy(self):
        """
        Calulating Corrected formation_energy
        """
        self.corrected_fe = self.total_correction + self.uncorrected_fe
        
        print ("Corrected Formation Energy : {}".format(self.corrected_fe))
