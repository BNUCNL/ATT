# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:1

import os
import pandas as pd
from atlas import *
import nibabel as nib
import cPickle

# config  basic info
roi_name = ['rV3', 'lV3', 'rMT', 'lMT']
roi_id = [1, 2, 3, 4]
task = 'motion'
contrast = 'motion-fix'
threshold = 5.0

# load gender and behavior data
data_path = './data'
subj = pd.read_csv(os.path.join(data_path, 'subj_info.csv'))
subj_gender = subj['gender'].tolist()
subj_id = subj['NSPID'].tolist()

# construct an Atlas object
mask_img_file = os.path.join(data_path, 'mt.nii.gz')
mt_atlas = Atlas(mask_img_file, roi_id, roi_name, task, contrast, threshold, subj_id, subj_gender)

# calculate probabilistic map and save it
pm = mt_atlas.make_pm(meth='part')
pm_img = nib.Nifti1Image(pm, None, mt_atlas.atlas_img.header)
nib.save(pm_img, os.path.join(data_path, 'pm.nii.gz'))

# save maximum probabilistic map and save it
mpm = mt_atlas.make_mpm(0.1)
mpm_img = nib.Nifti1Image(mpm, None, mt_atlas.atlas_img.header)
nib.save(mpm_img, os.path.join(data_path, 'mpm.nii.gz'))

# extract meas
meas_name = ['zstat.nii.gz', 'falff.nii.gz']
meas_mean = np.array([]).reshape((len(subj_id), 0)) # store the data as 2d array
meas_peak_coords = np.array([]).reshape((len(subj_id), 0))
for m in meas_name:
    meas_img_path = os.path.join(data_path, m)
    meas_img = nib.load(meas_img_path)

    mean_value = mt_atlas.collect_scalar_meas(meas_img, 'mean')
    meas_mean = np.hstack((meas_mean, mean_value))

    peak_coords = mt_atlas.collect_geometry_meas(meas_img, 'peak')
    meas_peak_coords = np.hstack((meas_peak_coords, peak_coords))

# reorganize data as a dict
data = dict.fromkeys(['meas_mean', 'meas_name', 'meas_peak_coords', 'roi_name', 'subj_id', 'subj_gender'], None)
data['meas_mean'] = meas_mean
data['meas_peak_coords'] = meas_peak_coords
data['subj_id'] = subj_id
data['roi_name'] = roi_name
data['subj_gender'] = subj_gender
data['meas_name'] = ['act-mean', 'fallf-mean']

# split data in two half and save it
sph_data = split_half_data(data, ['meas_mean', 'meas_peak_coords'])
file_name = 'mt-zstat-falff'
for f in range(2):
    with open(os.path.join(data_path, file_name+'-sph%d.pkl' % f), 'wb') as out_file:
        cPickle.dump(sph_data[f], out_file, -1)