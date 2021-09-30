# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np

def get_fermi_siesta_from_fdf(path_dir,fdf_name):
    """
    Getting fermi level from *.EIG file"
    """
    import sisl

    fdf = sisl.get_sile(path_dir+"/"+fdf_name)
    fermi = fdf.read_fermi_level()
    
    return fermi
 
def get_fermi_siesta_from_eig(path_dir,host_label):
    """
    Getting fermi level from *.EIG file"
    """
    import sisl

    EIG = sisl.get_sile(path_dir+"/"+host_label+".EIG")
    fermi = EIG.read_fermi_level()
    
    return fermi
    

def get_vbm_siesta_manual_bands(path_dir,NE):
    """
    Calculating valence band maximum from siesta calculations"
    """
    import sisl
    
    fdf =  sisl.get_sile(path_dir +"input.fdf")
    host_label = fdf.get("SystemLabel")
    BANDS = sisl.get_sile(path_dir+"/"+host_label+".bands")

    #N_electron = int(BANDS.file.read_text().split()[3])
    N_electron = NE
    #vb_index = int(N_electron/2)-1  May be A Bug!
    vb_index = int(N_electron/2)-2

    eig_gamma=BANDS.read_data()

    vbm = np.amax(eig_gamma[2][0][0][vb_index])
    #print(vbm)
    return vbm


def get_raw_formation_energy(defect_energy, host_energy, add_or_remove, chemical_potential,
                             charge, fermi_energy, valence_band_maximum):
    """
    Compute the formation energy without correction
    """
    # adding none
    sign_of_mu = {'add': +1.0, 'remove': -1.0, 'none' : 0.0}
    e_f_uncorrected = defect_energy - host_energy - sign_of_mu[add_or_remove]*chemical_potential + (
        charge * (valence_band_maximum + fermi_energy))
    #e_f_uncorrected = defect_energy - host_energy #- sign_of_mu[add_or_remove]
    #    charge * (valence_band_maximum + fermi_energy))

    return e_f_uncorrected


def get_corrected_formation_energy(e_f_uncorrected, correction):
    """
    Compute the formation energy with correction
    """
    e_f_corrected = e_f_uncorrected + correction
    return e_f_corrected


def get_corrected_aligned_formation_energy(e_f_corrected, alignment):
    """
    Compute the formation energy with correction and aligned
    """
    e_f_corrected_aligned = e_f_corrected + alignment
    return e_f_corrected_aligned



def output_energy_manual(path_dir):
    """
    Returns Energy from output.out file
    """
    import sisl
    out = sisl.io.outSileSiesta(path_dir+"/output.out")
    f=open(out.file)
    a=f.readlines()
    for line in range(len(a)):
        if 'siesta:         Total =' in a[line]:
            energy = a[line].split()[3]
            #print 
    return float(energy)


def output_total_electrons_manual(path_dir):
    """
    Return Number of Electrons in System from output.out file 
    """
    import sisl
    out = sisl.io.outSileSiesta(path_dir+"/output.out")
    f=open(out.file)
    a=f.readlines()
    for line in range(len(a)):
        if 'Total number of electrons:' in a[line]:
            number_of_electrons = int(float(a[line].split()[4]))
            #print  (a[line].split())
    return number_of_electrons

def Read_FDF_File(path_dir,fdf_name):
    """
    Read The FDF File
    """
    import sisl
    fdf = sisl.get_sile(path_dir+fdf_name+'.fdf')

    return fdf

def Read_VT_File(path_dir,VT_name):
    """
    Read The VT File
    """
    import sisl
    VT = sisl.get_sile(path_dir+VT_name+'.VT')

    return VT

def Read_Rho_File(path_dir,Rho_name):
    """
    Read The VT File
    """
    import sisl
    Rho = sisl.get_sile(path_dir+Rho_name+'.RHO')

    return Rho



def ASE2Siesta(A):
    """
    Getting ASE Atom Object and Converted to sisl
    """
    import sisl
    geo=sisl.Geometry.fromASE(A)
    return geo

#----------------------------------------------------------------------------------------
#def get_vbm_siesta(calc_node):
#    """
#    Calculating valence band maximum from siesta calculations"
#    """
#    import sisl
#    
#    EIG = sisl.get_sile(calc_node.outputs.remote_folder.get_remote_path()+"/aiida.EIG") 
#    
#    N_electron = int(EIG.file.read_text().split()[3]) 
#    vb_index = int(N_electron/2)-1
#
#    eig_gamma=EIG.read_data()[0][0]
#    
#    vbm = np.amax(eig_gamma[vb_index])
#
#    return vbm
#
#
#def get_vbm_siesta_manual(remote_node,host_label):
#    """
#    Calculating valence band maximum from siesta calculations"
#    """
#    import sisl
#
#    EIG = sisl.get_sile(remote_node.get_remote_path()+"/"+host_label+".EIG")
#
#    N_electron = int(EIG.file.read_text().split()[3])
#    vb_index = int(N_electron/2)-1
#
#    eig_gamma=EIG.read_data()[0][0]
#
#    vbm = np.amax(eig_gamma[vb_index])
#
#    return vbm

