# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np

"""
Utility functions for the potential alignment workchain
"""

def get_alignment(potential_difference, charge_density, tolerance):
    """
    Compute the density-weighted potential alignment
    """
    # Unpack
    
    #charge_density = charge_density.get_array(
    #    charge_density.get_arraynames()[0])

    # Get array mask based on elements' charge exceeding the tolerance.
    mask = np.ma.greater(np.abs(charge_density), tolerance)

    # Apply this mask to the diff array
    v_diff_masked = np.ma.masked_array(potential_difference , mask=mask)

    # Check if any values are left after masking
    if v_diff_masked.count() == 0:
        print ("SOMETHING IS WROOOOOOOOOOOONGGGGGGGG")
        #raise AllValuesMaskedError

    # Compute average alignment
    alignment = np.average(np.abs(v_diff_masked))

    return alignment
