import os
import glob

import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

parse = argparse.ArgumentParser(description="Volumes of the PET")
parse.add_argument('-d', '--mpet_dir', help='Directory of the mPETs')
args = parse.parse_args()
mpet_dir = args.mpet_dir

# Import all PET images in an array
dose_profiles = glob.glob('/Users/andreaberti/Desktop/tesi/nifti_simulated/test_patient_*[!.median].nii.gz')
dose_profiles = np.array([nib.load(fname).get_fdata() for fname in dose_profiles])

# Matrix of the means of the histograms
dose_means = np.zeros((dose_profiles.shape[1], dose_profiles.shape[2], dose_profiles.shape[3]))
# The mean is performed on the first dimension of dose_profiles, which means that it is the mean over the same voxel in different PETs
dose_means = dose_profiles.mean(0)

for modified_PET in glob.glob(mpet_dir+'/*'):
    # Import the PET from modified CT and take only the central slice
    name = modified_PET.split('/')[-1]
    #modified_PET = '/Users/andreaberti/Pydio/inside/reconstructed_fort21/shift_x_1/LORfile_interspill.median.nii'
    modified_PET = np.array(nib.load(modified_PET).get_fdata())
    
    # Preallocate memory for p_values matrix
    p_values = np.zeros((modified_PET.shape[0], modified_PET.shape[1], modified_PET.shape[2]))
    
    modified_sign = np.sign(modified_PET - dose_means)

    max_value = np.ceil(np.amax(dose_profiles))
    bin_step = 1
    bin_edges_original = np.arange(-0.5, max_value+bin_step+0.5, bin_step)

    # Calculate the p_values: if the value of the modified_PET is less the dose_means, sum the bins on the left, otherwise sum the bins on the right
    for i in range(dose_profiles.shape[1]):
        for j in range(dose_profiles.shape[2]):
            for k in range(dose_profiles.shape[3]):
                idx = np.searchsorted(bin_edges_original, modified_PET[i, j, k], side='left')
                if modified_PET[i, j, k] > dose_means[i, j, k]:
                  idx -= 1
                  hist_values, _ = np.histogram(dose_profiles[:, i, j, k], bins=bin_edges_original)
                  hist_values = hist_values/(dose_profiles.shape[0]+1)
                  p_values[i, j, k] = np.sum(hist_values[idx:])
                  p_values[i, j, k] += 1/(dose_profiles.shape[0]+1)
                elif modified_PET[i, j, k] < dose_means[i, j, k]:
                  hist_values, _ = np.histogram(dose_profiles[:, i, j, k], bins=bin_edges_original)
                  hist_values = hist_values/(dose_profiles.shape[0]+1)
                  p_values[i, j, k] = np.sum(hist_values[:idx+1])
                  p_values[i, j, k] += 1/(dose_profiles.shape[0]+1)
                else:
                  p_values[i, j, k] = 1/2

    #statistic_threshold = 1/(dose_profiles.shape[0]+1)
    statistic_threshold = 0.025
    offset = 1

    p_values[p_values>statistic_threshold] = offset
    p_values_negatives = p_values.copy()
    p_values_negatives[modified_sign!=-1] = offset
    p_values_negatives_reverse = 1-p_values_negatives
    voxels = np.count_nonzero(p_values_negatives_reverse[59:84,30:48,75:95])
    volume = voxels * 1.6 * 1.6 * 1.6
    print('{},{}'.format(name,volume))