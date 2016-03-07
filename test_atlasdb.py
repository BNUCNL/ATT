# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:1


import os
import pandas as pd
from atlasdb import *
import nibabel as nib


data_path = './data'

zstat_img_path = os.path.join(data_path, 'zstat.nii.gz')
zstat_img = nib.load(zstat_img_path)
falff_img_path = os.path.join(data_path, 'falff.nii.gz')
falff_img = nib.load(falff_img_path)


mask_img_file = os.path.join(data_path, 'mt.nii.gz')

roi_name = ['rV3', 'lV3', 'rMT', 'lMT']
roi_id = [1, 2, 3, 4]
task = 'motion'
contrast = 'motion-fix'
threshold = 5.0


subj_gender = pd.read_csv( os.path.join(data_path, 'act_sex.csv'))['gender'].tolist()
subj_id = open( os.path.join(data_path,'actID'),'rU').read().splitlines()

mt_atlas = Atlas(mask_img_file, roi_id, roi_name, task, contrast, threshold, subj_id, subj_gender)

mt_atlas.collect_scalar_meas(zstat_img, 'mean')
mt_atlas.collect_geometry_meas(zstat_img, 'peak')

# atlas.save_to_pkl(outpath, 'data_v3.pkl')
# atlas.save_to_mat(outpath, 'data_v3.mat')







