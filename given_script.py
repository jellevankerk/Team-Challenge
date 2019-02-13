# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:50:58 2019

@author: s131234
"""

import SimpleITK as sitk # First we import SimpleITK
import numpy as np
import matplotlib.pyplot as plt # matplotlib is a nice library for plotting


# Directory of ACDC dataset! 
datapath = r'D:\Documents\BMT Master Medical Imaging\vakken\Team challenge\part 2\data\Data\patient077'

im = sitk.ReadImage(r'{}\patient077_4d.nii.gz'.format(datapath))

# Print the size of the image
print('Size of the image is {}'.format(im.GetSize()))

# Print the spacing of the image
spacing = im.GetSpacing()
print('Spacing of the image is {}'.format(spacing))

im_np = sitk.GetArrayFromImage(im)
print('Size of the image in Numpy is {}'.format(im_np.shape))

# plot 2D slices over all frames
plt.rcParams["figure.figsize"] = (22, 26) # (w, h)
z_index = 4
n_rows = np.ceil(np.sqrt(im_np.shape[0]))
n_cols = np.ceil(np.sqrt(im_np.shape[0]))

for t in range(im_np.shape[0]): 
  plt.subplot(n_rows, n_cols, 1 + t)
  plt.imshow(im_np[t, z_index, :, :], clim=(0, 150), cmap='gray')
  plt.title('Frame {}'.format(t + 1))
plt.show()

# Load reference segmentations and show them along with the image at end-diastole.
im_ED = sitk.ReadImage(r'{}\patient077_frame01.nii.gz'.format(datapath))
im_ED_np = sitk.GetArrayFromImage(im_ED)
gt_ED = sitk.ReadImage(r'{}\patient077_frame01_gt.nii.gz'.format(datapath))
gt_ED_np = sitk.GetArrayFromImage(gt_ED)

n_rows = np.ceil(np.sqrt(im_ED_np.shape[0]))
n_cols = np.ceil(np.sqrt(im_ED_np.shape[0]))

for z in range(im_ED_np.shape[0]): 
  plt.subplot(n_rows, n_cols, 1 + z)
  plt.imshow(im_ED_np[z, :, :], clim=(0, 150), cmap='gray')
  plt.imshow(np.ma.masked_where(gt_ED_np[z, :, :]!=2, gt_ED_np[z, :, :]==2), alpha=0.6, cmap='Blues', clim=(0, 1))  
  plt.imshow(np.ma.masked_where(gt_ED_np[z, :, :]!=3, gt_ED_np[z, :, :]==3), alpha=0.6, cmap='Reds', clim=(0, 1))
  plt.title('Slice {}'.format(z + 1))
plt.show()

# Load reference segmentations and show them along with the image at end-systole
im_ES = sitk.ReadImage(r'{}\patient077_frame09.nii.gz'.format(datapath))
im_ES_np = sitk.GetArrayFromImage(im_ES)
gt_ES = sitk.ReadImage(r'{}\patient077_frame09_gt.nii.gz'.format(datapath))
gt_ES_np = sitk.GetArrayFromImage(gt_ES)

n_rows = np.ceil(np.sqrt(im_ES_np.shape[0]))
n_cols = np.ceil(np.sqrt(im_ES_np.shape[0]))

for z in range(im_ES_np.shape[0]): 
  plt.subplot(n_rows, n_cols, 1 + z)
  plt.imshow(im_ES_np[z, :, :], clim=(0, 150), cmap='gray')
  plt.imshow(np.ma.masked_where(gt_ES_np[z, :, :]!=2, gt_ES_np[z, :, :]==2), alpha=0.6, cmap='Blues', clim=(0, 1))  
  plt.imshow(np.ma.masked_where(gt_ES_np[z, :, :]!=3, gt_ES_np[z, :, :]==3), alpha=0.6, cmap='Reds', clim=(0, 1))
  plt.title('Slice {}'.format(z + 1))
plt.show()

# Compute the Ejection Fraction
voxelvolume = spacing[0]*spacing[1]*spacing[2]
ED_volume = np.sum(gt_ED_np==3)*voxelvolume
ES_volume = np.sum(gt_ES_np==3)*voxelvolume

strokevolume = ED_volume - ES_volume
LV_EF = (strokevolume/ED_volume)*100

# print the EF
print('LV stroke volume is {:.2f} ml and ejection fraction is {:.2f}%'.format(strokevolume*0.001, LV_EF))


