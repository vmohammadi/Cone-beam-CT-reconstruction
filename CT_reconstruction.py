# -*- coding: utf-8 -*-
"""
Program for CT reconstruction using ASTRA toolbax

"""

from __future__ import print_function
from __future__ import division
import os
from os import makedirs
from os.path import join, exists
import numpy as np
from imageio import imread, get_writer
from matplotlib import pyplot as plt

import astra

'''
You may use the follwing syntax to be sure that the Astra has been installed 
well and the GPU has been detected

astra.test()
'''

#%% Prerequisites

distance_source_origin = 225.350  # [mm]
distance_origin_detector = 294.6  # [mm]
detector_pixel_size = 0.06836  # [mm]
num_projections = 360
angles = np.linspace(0, 2 * np.pi, num_projections, endpoint=False)
 
horizontal_shift = -(-23) # We change the sign for numpy to roll the image in the correct direction
 
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


if not os.path.exists('reco_dir'):
   os.makedirs('reco_dir')

#%% CT reconstruction

detector_pixel_size_in_origin = \
  detector_pixel_size * distance_source_origin / \
  (distance_source_origin + distance_origin_detector)
 
# Determine dimensions of projection images
im = imread(join(proj_dir, 'proj0000.tif'))
dims = im.shape
# dims[0]: Number of rows in the projection image, i.e., the height of the
#          detector. This is Y in the Cartesian coordinate system.
# dims[1]: Number of columns in the projection image, i.e., the width of the
#          detector. This is X in the Cartesian coordinate system.
detector_rows = dims[0]
detector_columns = dims[1]
 
# Load projection images
projections = np.zeros((detector_rows, num_projections, detector_columns))
for proj in range(num_projections):
    im = imread(join(proj_dir, 'proj{:04d}.tif'.format(proj))).astype(float)
    im /= 65535
    im = np.roll(im, horizontal_shift, axis=1)
    projections[:, proj, :] = im
 
# Copy projection images into ASTRA Toolbox
proj_geom = astra.create_proj_geom('cone', 1, 1, detector_rows,
                                   detector_columns, angles,
                                   distance_source_origin /
                                   detector_pixel_size_in_origin, 0)
projections_id = astra.data3d.create('-sino', proj_geom, projections)
 
# Perform reconstruction
vol_geom = astra.creators.create_vol_geom(detector_columns, detector_columns,
                                          detector_rows)
reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
alg_cfg = astra.astra_dict('FDK_CUDA')
alg_cfg['ProjectionDataId'] = projections_id
alg_cfg['ReconstructionDataId'] = reconstruction_id
algorithm_id = astra.algorithm.create(alg_cfg)
print('Reconstructing... ', end='')
astra.algorithm.run(algorithm_id)
print('done')
reconstruction = astra.data3d.get(reconstruction_id)
 
# Limit and scale reconstruction.
reconstruction[reconstruction < 0] = 0
middle = detector_rows // 2
M = np.max(reconstruction[middle - 100 : middle + 100, :, :])
reconstruction /= M
reconstruction[reconstruction > 1] = 1
 
# Save reconstruction.
for proj in range(detector_rows):
    print('Saving slice %d' % proj)
    slice = reconstruction[proj, :, :]
    save_tiff(join(reco_dir, 'reco%04d.tif' % proj), slice)
 
# Cleanup
astra.algorithm.delete(algorithm_id)
astra.data3d.delete(reconstruction_id)
astra.data3d.delete(projections_id)

#%% Show the reconstruction slices (may take time!)

for proj in range(num_projections):
    im = imread(join(reco_dir, 'reco{:04d}.tif'.format(proj))).astype(float)
    plt.figure()
    plt.imshow(im,'gray')

#%%



