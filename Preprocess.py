# -*- coding: utf-8 -*-

'''
Pre-processing program for the preparation of X-ray projections

'''

from __future__ import print_function
from __future__ import division
import os
from os import makedirs
from os.path import join, exists
import numpy as np
from imageio import imread, get_writer

#%% Prerequisites

num_projections = 360
margin = 10
 
# You need to create four folders in your current directory as follows:
raw_dir = 'Raw_dir'
preproc_dir = 'preprocessed'
proj_dir = 'projections'
reco_dir = 'reconstruction'

#%% Functions
 
def save_tiff(name, im_in):
    # Expects image as floating point in range [0, 1].
    im = im_in.copy()
    im = np.round(im * 65535).astype(np.uint16)
    with get_writer(name) as writer:
        writer.append_data(im, {'compress': 0})
 
    
if not os.path.exists(preproc_dir):
   os.makedirs(preproc_dir)
if not os.path.exists(proj_dir):
   os.makedirs(proj_dir)

#%% Flat-field (We can perform once/for higher precision, you may take dark/white before and after image acquisition)
# Dark frame.
dark_frame = \
  imread(join(raw_dir, 'Dark.tif')).astype(float)
 
# Flat fields.
pre_flat_field = \
  imread(join(raw_dir, 'White.tif')).astype(float)
pre_flat_field -= dark_frame
pre_flat_field[pre_flat_field <=0] = 50000 # A few pixels are zero as they have been off, change to a high value

num_proj = num_projections

#%% An extra step, just to be sure that everything goes well. This is because some images have undefined pixels
# Determine maximum of projections.
M = -np.inf  # max
for proj in range(num_proj):
    print('Preprocessing image {} (step 1)'.format(proj))
    im = imread(join(raw_dir, 'frame_{:01d}.0.tif'.format(proj))).astype(float)
    im[im==0] = 60200 # Dead pixels which are zero should be replaced, if not, they produce errors
    im -= dark_frame
    im /= pre_flat_field # Compute flatfield image

    I0 = np.mean(pre_flat_field)
    # Values above I0 are due to noise and will produce negative densities later.
    im[im > I0] = I0
    im[im <=0] = 0.001
    im = -np.log(im / I0)
    if np.max(im) > M:
        M = np.max(im)

#%% Convert raw images to projections

for proj in range(num_proj):
    print('Preprocessing image {} (step 2)'.format(proj))
    im = imread(join(raw_dir, 'frame_{:01d}.0.tif'.format(proj))).astype(float)
    im[np.isnan(im)] = 0 # Replace nan pixels with 0
    im -= dark_frame
    im /= pre_flat_field

    I0 = np.mean(pre_flat_field)
    # Values above I0 are due to noise and wil produce negative densities later.
    im[im > I0] = I0
    save_tiff(join(preproc_dir, 'prep{:04d}.tif'.format(proj)), im)
    im = -np.log(im / I0) # projection
    im /= M
    save_tiff(join(proj_dir, 'proj{:04d}.tif'.format(proj)), im)

#%% -------------------------------------------------



