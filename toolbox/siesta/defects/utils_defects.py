# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################

import numpy as np


def get_output_energy_manual(path_dir,output_name):
    """
    Returns Energy from provided output file
    """
    import sisl
    out = sisl.get_sile(path_dir / output_name )
    energy = out.read_energy()['total']
    return float(energy)

def get_output_total_electrons_manual(path_dir,output_name):
    """
    Return Number of Electrons in System from output.out file 
    """
    import sisl
    out = sisl.io.outSileSiesta(path_dir/output_name)
    f=open(out.file)
    a=f.readlines()
    for line in range(len(a)):
        if 'Total number of electrons:' in a[line]:
            number_of_electrons = int(float(a[line].split()[4]))
            #print  (a[line].split())
    return number_of_electrons

def get_vbm_siesta_manual_bands(path_dir,fdf_name, NE):
    """
    Calculating valence band maximum from siesta calculations"
    """
    import sisl
    
    fdf =  sisl.get_sile(path_dir /fdf_name)
    host_label = fdf.get("SystemLabel")
    band_file_name = host_label+".bands"
    BANDS = sisl.get_sile(path_dir/ band_file_name )

    #N_electron = int(BANDS.file.read_text().split()[3])
    N_electron = NE
    #vb_index = int(N_electron/2) #-1  #May be A Bug!
    vb_index = int(N_electron/2)-2 # Correct

    eig_gamma=BANDS.read_data()

    vbm = np.amax(eig_gamma[2][0][0][vb_index])
    #print(vbm)
    return vbm

def get_fermi_siesta_from_fdf(path_dir,fdf_name):
    """
    Getting fermi level from *.EIG file"
    """
    import sisl

    fdf = sisl.get_sile(path_dir / fdf_name)
    fermi = fdf.read_fermi_level()
    
    return fermi


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


