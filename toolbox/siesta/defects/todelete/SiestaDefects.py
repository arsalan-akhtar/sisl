# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from .Corrections.gaussian_countercharge_model import GaussianCounterChargeWorkchain
from .SiestaDefectsBase import  FormationEnergyWorkchainBase as FEBase

#import utils
from .Utils import utils_defects

#from utils_defects.utils_defects import (
#    get_raw_formation_energy,
#    get_corrected_formation_energy,
#    get_corrected_aligned_formation_energy,
#)

from .Utils.utils_alignment import get_total_alignment 
from .Utils.utils_alignment import get_total_correction

__author__ = "Arsalan Akhatar"
__copyright__ = "Copyright 2021, SIESTA Group"
__version__ = "0.1"
__maintainer__ = "Arsalan Akhtar"
__email__ = "arsalan_akhtar@outlook.com," + \
        " miguel.pruneda@icn2.cat "
__status__ = "Development"
__date__ = "Janurary 30, 2021"


class FormationEnergyWorkchainSIESTA(FEBase):
    """
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
                 sigma,
                 host_fdf_name ='input.fdf',
                 rho_host = None,
                 rho_defect_q = None,
                 neutral_defect_fdf_name = None,
                 charge_defect_fdf_name = None,
                 use_siesta_mesh_cutoff = None,
                 cutoff = None,
                 fit_params = None
                 ):
    
        self.cutoff = cutoff
        self.use_siesta_mesh_cutoff = use_siesta_mesh_cutoff
        self.fit_params = fit_params
        FEBase.__init__(self,
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
                 sigma,
                 rho_host,
                 rho_defect_q,
                 host_fdf_name ,
                 neutral_defect_fdf_name ,
                 charge_defect_fdf_name ,
                 use_siesta_mesh_cutoff,
                 cutoff
                 )
                  
        self.check_cutoff()


    def run(self):
        print ("The Defect Correction Package for SIESTA...")
        print ("Starting ...")

        if  FEBase.setup(self) == "none":
            self.Reading_inputs()
            print("There is No Correction Asked I will do Raw Formation Energy..")            
            
            self.Reading_SIESTA_Data()
            self.calculate_uncorrected_formation_energy()
 
            #self.Check()
        elif FEBase.setup(self)=="gaussian-model":
            print("Starting Gaussian Model Correction ..")          
            self.Reading_inputs()
            self.Input_Gaussian = GaussianCounterChargeWorkchain(
                    self.host_VT,
                    self.defect_q0_VT,
                    self.defect_q_VT,
                    self.defect_charge,
                    self.defect_site,
                    self.host_structure,
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
            

        elif FEBase.setup(self)=="gaussian-rho":
            print("Starting Gaussian Rho Correction ..")          
            self.Reading_inputs()
            self.Input_Gaussian = GaussianCounterChargeWorkchain(
                    self.host_VT,
                    self.defect_q0_VT,
                    self.defect_q_VT,
                    self.defect_charge,
                    self.defect_site,
                    self.host_structure,
                    scheme = 'gaussian-rho',
                    rho_host= self.rho_host,
                    rho_defect_q= self.rho_defect_q ,
                    epsilon = self.epsilon,
                    fit_params = self.fit_params,
                    )
            self.Input_Gaussian.run()

            self.Input_Gaussian.compute_dft_difference_potential_q_q0()
            
            self.iso_energy = self.Input_Gaussian.get_isolated_energy()
            self.align_host_q0 = self.Input_Gaussian.compute_alignment_host_q0()
            self.align_model_q_q0 = self.Input_Gaussian.compute_alignment_model_q_q0()
          
        elif FEBase.setup(self)=="rho":
            print("Starting Rho Correction ..")          
            self.Reading_inputs()
            self.Input_Gaussian = GaussianCounterChargeWorkchain(
                    self.host_VT,
                    self.defect_q0_VT,
                    self.defect_q_VT,
                    self.defect_charge,
                    self.defect_site,
                    self.host_structure,
                    scheme = 'rho',
                    rho_host= self.rho_host,
                    rho_defect_q= self.rho_defect_q ,
                    epsilon = self.epsilon,
                    )
            self.Input_Gaussian.run()

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
        #import utils_defects.utils_defects 
        #self.host_Energy = utils.output_energy_manual(self.host_path)
        #self.defect_q0_Energy = utils.output_energy_manual(self.neutral_defect_path) 
        #self.defect_q_Energy = utils.output_energy_manual(self.charge_defect_path)

        #self.host_NE = utils.output_total_electrons_manual(self.host_path)
        #self.defect_q0_NE = utils.output_total_electrons_manual(self.neutral_defect_path)
        #self.defect_q_NE = utils.output_total_electrons_manual(self.charge_defect_path)

        #self.host_vbm = utils.get_vbm_siesta_manual_bands(self.host_path,self.host_NE )
        #self.defect_q0_vbm = utils.get_vbm_siesta_manual_bands(self.neutral_defect_path,self.defect_q0_NE )
        #self.defect_q_vbm = utils.get_vbm_siesta_manual_bands(self.charge_defect_path,self.defect_q0_NE )
        
        #self.host_fermi = utils.get_fermi_siesta_from_fdf(self.host_path,self.host_fdf_name)
        #self.defect_q0_fermi = utils.get_fermi_siesta_from_fdf(self.neutral_defect_path,self.neutral_defect_fdf_name)
        #self.defect_q_fermi = utils.get_fermi_siesta_from_fdf(self.charge_defect_path,self.charge_defect_fdf_name)
        
        self.host_Energy = utils_defects.output_energy_manual(self.host_path)
        self.defect_q0_Energy = utils_defects.output_energy_manual(self.neutral_defect_path) 
        self.defect_q_Energy = utils_defects.output_energy_manual(self.charge_defect_path)

        self.host_NE = utils_defects.output_total_electrons_manual(self.host_path)
        self.defect_q0_NE = utils_defects.output_total_electrons_manual(self.neutral_defect_path)
        self.defect_q_NE = utils_defects.output_total_electrons_manual(self.charge_defect_path)

        self.host_vbm = utils_defects.get_vbm_siesta_manual_bands(self.host_path,self.host_NE )
        self.defect_q0_vbm = utils_defects.get_vbm_siesta_manual_bands(self.neutral_defect_path,self.defect_q0_NE )
        self.defect_q_vbm = utils_defects.get_vbm_siesta_manual_bands(self.charge_defect_path,self.defect_q0_NE )
        
        self.host_fermi = utils_defects.get_fermi_siesta_from_fdf(self.host_path,self.host_fdf_name)
        self.defect_q0_fermi = utils_defects.get_fermi_siesta_from_fdf(self.neutral_defect_path,self.neutral_defect_fdf_name)
        self.defect_q_fermi = utils_defects.get_fermi_siesta_from_fdf(self.charge_defect_path,self.charge_defect_fdf_name)


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

        self.uncorrected_fe = utils_defects.get_raw_formation_energy( defect_energy = self.defect_q_Energy, 
                                                   host_energy = self.host_Energy, 
                                                   add_or_remove = self.add_or_remove, 
                                                   chemical_potential = self.chemical_potential,
                                                   charge = self.defect_charge, 
                                                   fermi_energy = self.fermi_level, 
                                                   #valence_band_maximum = self.host_vbm
                                                   valence_band_maximum = self.defect_q_vbm
                                                   )
        print ("Uncorrected Formation Energy : {} ".format(self.uncorrected_fe))


    def calculate_corrected_formation_energy(self):
        """
        Calulating Corrected formation_energy
        """
        self.corrected_fe = self.total_correction + self.uncorrected_fe
        
        print ("Corrected Formation Energy : {}".format(self.corrected_fe))
