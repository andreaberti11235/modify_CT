import os
import glob

import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

parse = argparse.ArgumentParser(description="Mean value and STD of the p-maps")
parse.add_argument('-f', '--volumes_file', help='Paresed file of the volumes measured with the p-maps')
# parse.add_argument('-t', '--threshold', type=float, help='Threshold for the activity (default 30)')
args = parse.parse_args()

file = args.volumes_file

with open(file) as f:
    vol = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
vol = [x.strip() for x in vol] 

vol = [float(i) for i in vol]

name = file.split('/')[-1]

mean = np.mean(vol)
mean = round(mean,1)
std = np.std(vol)
std = round(std,1)

print('{},{},{}'.format(name,mean,std))