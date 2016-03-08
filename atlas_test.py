# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:1


import os
import pandas as pd
from atlas import *
import nibabel as nib
import cPickle

roi_name = ['rV3', 'lV3', 'rMT', 'lMT']
roi_id = [1, 2, 3, 4]
task = 'motion'
contrast = 'motion-fix'
threshold = 5.0


data_path = './data'

subj_gender = pd.read_csv(os.path.join(data_path, 'act_sex.csv'))['gender'].tolist()
subj_id = open(os.path.join(data_path, 'actID'), 'rU').read().splitlines()


mask_img_file = os.path.join(data_path, 'mt.nii.gz')
mt_atlas = Atlas(mask_img_file, roi_id, roi_name, task, contrast, threshold, subj_id, subj_gender)


meas_name = ['zstat.nii.gz', 'falff.nii.gz']
meas_mean = np.array([]).reshape(len(subj_id), len(roi_id))
meas_peak_coords = np.array([]).reshape(len(subj_id), len(roi_id), 3)
for m in meas_name:
    meas_img_path = os.path.join(data_path, m)
    meas_img = nib.load(meas_img_path)

    mean_value = mt_atlas.collect_scalar_meas(meas_img, 'mean')
    meas_mean = np.hstack((meas_mean, mean_value))

    peak_coords = mt_atlas.collect_geometry_meas(meas_img, 'peak')
    meas_peak_coords = np.hstack((meas_peak_coords, peak_coords))
    print m

data = dict.fromkeys(['meas', 'subj_id', 'roi_name', 'feat_name'], None)
data['meas'] = meas_mean
data['subj_id'] = subj_id
data['roi_name'] = roi_id
data['feat_name'] = ['act-mean', 'fallf-mean']


with open(os.path.join(data_path, 'zstat-falff'), 'wb') as out_file:
        cPickle.dump(data, out_file, -1)






