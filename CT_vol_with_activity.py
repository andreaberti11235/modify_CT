import os
import glob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import nibabel as nib
import pydicom
import scipy.ndimage
from nilearn import plotting
import nilearn

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Calculate the CT volume in ml")
    parse.add_argument('-m', '--ct_modified', help='Path to the modified CT in .nii (or .nii.gz)')
    parse.add_argument('-p', '--pet', help='Path to the PET (or activity map) in .nii (or .nii.gz)')
    parse.add_argument('-t', '--threshold', type=float, help='Threshold for the activity (default 30)')
    args = parse.parse_args()
    if args.ct_modified == None:
        ct_modified = '/Users/andreaberti/Desktop/tesi/pz_per_Aafke/modified_CT_nii/shift_x_1/shift_x_1.nii.gz'
    else:
        ct_modified = args.ct_modified

    if args.pet == None:
        pet = nib.load('/Users/andreaberti/Desktop/tesi/pet_original_patient/pet_median_large.nii.gz')
    else:
        pet = nib.load(args.pet)
    
    if args.threshold == None:
        threshold = 30
    else:
        threshold = args.threshold

    ct_name = ct_modified.split('/')[-1]
    ct_modified = nib.load(ct_modified)
    ct_original = nib.load('/Users/andreaberti/Desktop/tesi/pz_per_Aafke/RT_nii/image.nii.gz')

    # matrix of the modified region between the two CTs
    tumor_data = np.array(ct_original.get_fdata()) - np.array(ct_modified.get_fdata())
    # consider only the region that has a certain activity
    tumor_data[pet.get_fdata()<threshold] = 0
    volume = np.count_nonzero(tumor_data) * 0.9766 * 0.9766 * 2 / 1000
    volume = round(volume,1)

    print('{},{}'.format(ct_name, volume))

    pass