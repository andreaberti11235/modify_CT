import os
import glob

import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib



# Import all PET images in an array
dose_profiles = glob.glob('/Users/andreaberti/Desktop/tesi/activity_images/activityMap_0_360_test_patient_*.nii.gz')
dose_profiles = np.array([nib.load(fname).get_fdata() for fname in dose_profiles])

# Matrix of the means of the histograms
dose_means = np.zeros((dose_profiles.shape[1], dose_profiles.shape[2], dose_profiles.shape[3]))
# The mean is performed on the first dimension of dose_profiles, which means that it is the mean over the same voxel in different PETs
dose_means = dose_profiles.mean(0)

# Matrix of |xi-mu|
dose_diff = np.zeros((dose_profiles.shape[0], dose_profiles.shape[1], dose_profiles.shape[2], dose_profiles.shape[3]))
dose_diff = np.absolute(dose_profiles - dose_means)
dose_sign = np.zeros((dose_profiles.shape[0], dose_profiles.shape[1], dose_profiles.shape[2], dose_profiles.shape[3]))
dose_sign = np.sign(dose_profiles - dose_means)

# Take all the slices of the PET, choose unitary bins going from -0.5 to max+0.5
max_value = np.ceil(np.amax(dose_diff))
bin_edges_original = np.arange(-0.5, max_value+1.5)
n_bins = bin_edges_original.shape[0]-1

# Preallocate memory for the matrix of hist_values
hist_values = np.zeros((dose_profiles.shape[1], n_bins, dose_profiles.shape[2], dose_profiles.shape[3]))

# Calculate the histograms along the first axis of dose_profiles, save the  values of the histogram bins in hist_values matrix
for i in range(dose_diff.shape[1]):
    for j in range(dose_diff.shape[2]):
        for k in range(dose_diff.shape[3]):
            hist_values[i, :, j, k], _ = np.histogram(dose_diff[:, i, j, k], bins=bin_edges_original)

hist_values = hist_values/dose_diff.shape[0]

# Import the PET from modified CT and take only the central slice

modified_PET = '/Users/andreaberti/Desktop/tesi/activity_images/activityMap_0_360_shift_x_1.nii.gz'
modified_PET = np.array(nib.load(modified_PET).get_fdata())
name = 'shift_x_1'

# Preallocate memory for p_values matrix
p_values = np.zeros((modified_PET.shape[0], modified_PET.shape[1], modified_PET.shape[2]))
modified_diff = np.zeros(modified_PET.shape)
modified_diff = np.abs(modified_PET - dose_means)
modified_sign = np.sign(modified_PET - dose_means)

# Calculate the p_values: if the value of the modified_PET is less the dose_means, sum the bins on the left, otherwise sum the bins on the right
for i in range(dose_profiles.shape[1]):
    for j in range(dose_profiles.shape[2]):
        for k in range(dose_profiles.shape[3]):
            idx = np.searchsorted(bin_edges_original, modified_diff[i, j, k], side='left')
            idx -= 1
            p_values[i, j, k] = np.sum(hist_values[i, idx:, j, k])

p_values[p_values>0.05] = 1
p_values_negatives = p_values.copy()
p_values_negatives[modified_sign!=-1] = 1
p_values_negatives_reverse = 1-p_values_negatives
voxels = np.count_nonzero(p_values_negatives_reverse[59:84,30:48,75:95])
volume = voxels * 1.6 * 1.6 * 1.6
print('{},{}'.format(name,volume))