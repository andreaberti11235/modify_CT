import pydicom
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pydicom.filereader import read_dicomdir
import scipy.io as sio
import scipy.ndimage

def read_dicom(dicom_dir):
    """Reads a dicom folder and sorts the slices by means of their slice location. Returns the slices in a list.
    Arguments:
    "dicom_dir"   the path of the folder containing the dicom files
    """
    dicom_files = glob.glob(os.path.join(dicom_dir, '*.dcm'))
    dicom_slices = [pydicom.read_file(fname) for fname in dicom_files]
    # selecting only the slices that have the "SliceLocation" attribute
    slices = [dcm_slice for dcm_slice in dicom_slices if hasattr(dcm_slice, 'SliceLocation')]
    slices.sort(key = lambda x: x.SliceLocation)
    return slices

def dicom_to_array(slices):
    """Creates a Numpy array from dicom slices. Returns the CT as a Numpy array.
    Arguments:
    "slices"  the list of dicom slices
    """
    shape = slices[0].Rows, slices[0].Columns, len(slices)
    # preallocate memory for the array
    CT_array = np.zeros(shape, dtype=slices[0].pixel_array.dtype)
    # associating dcm slices to z coordinates of the array
    for i, dcm in enumerate(slices):
        CT_array[:, :, i] = dcm.pixel_array
    return CT_array

def shift_mask(mask, dx, ord=3):
    """Shifts the mask in x direction. Returns the shifted mask as a Numpy array.
    Arguments:
    "mask"      Numpy array
    "dx"        the absolute value of the shift
    "ord"       the order of the spline (default 3)
    """
    mask_shifted = np.copy(mask)
    scipy.ndimage.shift(mask, [0, -dx, 0], output=mask_shifted, order=ord, mode='constant', cval=0.0, prefilter=True)
    return mask_shifted

def mask_intersection(mask, mask_shifted):
    """Returns the intersection of the original mask and the shifted mask as a Numpy array.
    Arguments:
    "mask"          the original mask (np array)
    "mask_shifted"  the shifted mask (np array)
    """
    mask_new = np.copy(mask)
    mask_new *= mask_shifted
    mask_new[mask_new>0]=1
    return mask_new


def modify_CT(CT_array, mask_new):
    """Returns the np array of the modified CT, where the elements corresponding to the mask are set to zero.
    Arguments:
    "CT_array"  the original CT (np array) to be modified
    "mask_new"  the mask (np array)
    """
    CT_modified = np.copy(CT_array)
    CT_modified[mask_new>0] = 0
    return CT_modified

def save_as_dicom(CT_modified, slices, out_path, dx):
    """Saves the CT as a dicom series.
    Arguments:
    "CT_modified"   the CT to save (np array)
    "slices"        the dicom slices
    """
    out_path = os.path.join(out_path, 'shift_x_{}'.format(dx))
    os.mkdir(out_path)
    for i, dcm in enumerate(slices):
        dcm.PixelData = CT_modified[:, :, i].tobytes()
        dcm.save_as(os.path.join(out_path, 'CT_modified{}.dcm'.format(i)))




if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Modify a CT from a mask")
    parse.add_argument('-d', '--dicom_dir', help='Directory of the original DICOM files')
    parse.add_argument('-m', '--mask_mat', help='Path to the mask in MATLAB format')
    parse.add_argument('-o', '--out_path', help='Path to the output directory')
    args = parse.parse_args()
    if args.dicom_dir == None:
        dicom_dir = '/Users/andreaberti/Desktop/tesi/pz_per_Aafke/CT_16-03-18'
    else:
        dicom_dir = args.dicom_dir
    
    if args.mask_mat == None:
        mask_mat = '/Users/andreaberti/Desktop/tesi/modify_CT/maschera.mat'
    else:
        mask_mat = args.mask_mat

    if args.out_path == None:
        out_path = '/Users/andreaberti/Desktop/tesi/pz_per_Aafke/CT_modified'
    else:
        out_path = args.out_path

    slices = read_dicom(dicom_dir)
    CT_array = dicom_to_array(slices)
    maschera = sio.loadmat(mask_mat)
    mask = maschera['Vnew']
    mask[mask>0] = 1
    hu_mean = np.sum(CT_array[mask==1])
    hu_mean /= np.count_nonzero(mask)
    check = True
    dx = 1
    while check == True:
        mask_shifted = shift_mask(mask, dx)
        mask_new = mask_intersection(mask, mask_shifted)
        if 1 not in np.unique(mask_new):
            break
        else:
            volume = np.count_nonzero(mask_new) * 0.9766 * 0.9766 * 2
            hu_sum = np.sum(CT_array[mask_new!=0])
            weighted_volume = 0.9766 * 0.9766 * 2 * hu_sum / hu_mean
            print('shift_x_{},{}'.format(dx,weighted_volume))
            dx += 1
    pass
