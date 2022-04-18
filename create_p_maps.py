import os
import glob

import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


if __name__ == "__main__":
    """Script to generate the postitive and the negative p-maps of the analysis on a PET/IDM image. The 
    posìtive p-map will be saved as "positive.nii.gz" and the negative one as "negative.nii.gz"
    """
    parse = argparse.ArgumentParser(description="Create p-maps for a PET/IDM")
    parse.add_argument('image', help='Absolute path of the input image (modified PET/IDM in RAI orientation)')
    parse.add_argument('-I', '--idm', help='Analysis on an IDM image', action='store_true')
    parse.add_argument('out', help='Output directory for "positive.nii.gz" and "negative.nii.gz"')
    args = parse.parse_args()

    if args.idm:
        dose_profiles = glob.glob('/Users/andreaberti/Desktop/tesi/activity_images/activity_images_median2/*test_patient*.nii.gz')
        dose_profiles = np.array([nib.load(fname).get_fdata() for fname in dose_profiles])
        first_pet = nib.load('/Users/andreaberti/Desktop/tesi/activity_images/activity_images_median2/null/activityMap_0_360_test_patient_1_RAI.median2.nii.gz')
    else:
        dose_profiles = glob.glob('/Users/andreaberti/Desktop/tesi/nifti_simulated/median_2/median_2_RAI/*.nii.gz')
        dose_profiles = np.array([nib.load(fname).get_fdata() for fname in dose_profiles])
        first_pet = nib.load('/Users/andreaberti/Desktop/tesi/nifti_simulated/median_2/median_2_RAI/null/test_patient_1_2.median_RAI.nii.gz')

    
    # Matrix of the means of the histograms
    dose_means = np.zeros((dose_profiles.shape[1], dose_profiles.shape[2], dose_profiles.shape[3]))
    # The mean is performed on the first dimension of dose_profiles, which means that it is the mean over the same voxel in different PETs
    dose_means = dose_profiles.mean(0)

    # Matrix of the sign of |xi-mu|
    dose_sign = np.zeros((dose_profiles.shape[0], dose_profiles.shape[1], dose_profiles.shape[2], dose_profiles.shape[3]))
    dose_sign = np.sign(dose_profiles - dose_means)

    # Take all the slices of the PET, choose unitary bins going from -0.5 to max+0.5
    max_value = np.ceil(np.amax(dose_profiles))
    bin_edges_original = np.arange(-0.5, max_value+1.5)
    n_bins = bin_edges_original.shape[0]-1

    # Preallocate memory for the matrix of hist_values
    hist_values = np.zeros((dose_profiles.shape[1], n_bins, dose_profiles.shape[2], dose_profiles.shape[3]))

    # Calculate the histograms along the first axis of dose_profiles, save the  values of the histogram bins in hist_values matrix
    for i in range(dose_profiles.shape[1]):
        for j in range(dose_profiles.shape[2]):
            for k in range(dose_profiles.shape[3]):
                hist_values[i, :, j, k], _ = np.histogram(dose_profiles[:, i, j, k], bins=bin_edges_original)

    hist_values = hist_values/(dose_profiles.shape[0]+1)

    # Import the modified PET/IDM
    modified_PET = np.array(nib.load(args.image).get_fdata())
    
    # Preallocate memory for p_values matrix
    p_values = np.zeros((modified_PET.shape[0], modified_PET.shape[1], modified_PET.shape[2]))
    modified_sign = np.sign(modified_PET - dose_means)

    # Calculate the p_values: if the value of the modified_PET is less the dose_means, sum the bins on the left, otherwise sum the bins on the right
    for i in range(dose_profiles.shape[1]):
        for j in range(dose_profiles.shape[2]):
            for k in range(dose_profiles.shape[3]):
                idx = np.searchsorted(bin_edges_original, modified_PET[i, j, k], side='left') # 'left' serve per decidere da che lato della disuguaglianza mettere il <= (vedi documentazione)
                if modified_PET[i, j, k] > dose_means[i, j, k]:
                  idx -= 1
                  p_values[i, j, k] = np.sum(hist_values[i, idx:, j, k])
                  p_values[i, j, k] += 1/(dose_profiles.shape[0]+1)
                elif modified_PET[i, j, k] < dose_means[i, j, k]:
                  p_values[i, j, k] = np.sum(hist_values[i, :idx+1, j, k])
                  p_values[i, j, k] += 1/(dose_profiles.shape[0]+1)
                else:
                  p_values[i, j, k] = 1/2

    #statistic_threshold = 1/(dose_profiles.shape[0]+1)
    statistic_threshold = 0.025
    offset = 1

    p_values[p_values>statistic_threshold] = offset

    # Create the positive p-map (and reverse the scale)
    p_values_positives = p_values.copy()
    p_values_positives[modified_sign!=1] = offset
    p_values_positives_reverse = 1-p_values_positives

    # Create the negative p-map (and reverse the scale)
    p_values_negatives = p_values.copy()
    p_values_negatives[modified_sign!=-1] = offset
    p_values_negatives_reverse = 1-p_values_negatives

    # Save the nifti p-maps images
    x_r_pos = nib.Nifti1Image(p_values_positives_reverse, first_pet.affine, header=first_pet.header)
    x_r_pos.to_filename(args.out + '/positive.nii.gz')
    x_r_neg = nib.Nifti1Image(p_values_negatives_reverse, first_pet.affine, header=first_pet.header)
    x_r_neg.to_filename(args.out + '/negative.nii.gz')
    pass
