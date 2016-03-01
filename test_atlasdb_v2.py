# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:
import os
import numpy as np
import pandas as pd
from atlasdb_v2 import *
import nibabel as nib

fpath = os.path.join

rawdatapath = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/mergedata/mergedata_mt'

zstat_img_path = fpath(rawdatapath, 'zstat_combined.nii.gz')
zstat_img = nib.load(zstat_img_path)
psc_img_path = fpath(rawdatapath, 'psc_combined.nii.gz')
psc_img = nib.load(psc_img_path)
alff_img_path = fpath(rawdatapath, 'alff_combined.nii.gz')
alff_img = nib.load(alff_img_path)
falff_img_path = fpath(rawdatapath, 'falff_combined.nii.gz')
falff_img = nib.load(falff_img_path)
reho_img_path = fpath(rawdatapath, 'reho_combined.nii.gz')
reho_img = nib.load(reho_img_path)

mask_img_file = fpath(rawdatapath, 'mt_z5.0.nii.gz')


outpath = '/nfs/j3/userhome/huangtaicheng/workingdir/program/ATT/data'

areaname = ['rV3', 'lV3', 'rMT', 'lMT']
areanum = [1,2,3,4]
taskname = 'motion'
contrast = 'motion-fix'
threshold = 5.0

pathsex = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/doc/dfsf/modeID'
gender = pd.read_csv(fpath(pathsex, 'act_sex.csv'))['gender'].tolist()

sessid = open('/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/doc/dfsf/modeID/actID','rU').read().splitlines()


atlas = Atlas(mask_img_file, areanum, areaname, gender, sessid, taskname, contrast, threshold)
metric = ['mean', 'max', 'min', 'std', 'median', 'cv', 'skewness', 'kurtosis']
# index = ['zstat', 'psc', 'alff', 'falff', 'reho']

for met in metric:
    atlas.mask_index(zstat_img, 'zstat', met)
    atlas.mask_index(psc_img, 'psc', met)
    atlas.mask_index(alff_img, 'alff', met)
    atlas.mask_index(falff_img, 'falff', met)
    atlas.mask_index(reho_img, 'reho', met)
atlas.peakcoord_index(zstat_img, 'zstat')
atlas.peakcoord_index(psc_img, 'psc')
atlas.peakcoord_index(alff_img, 'alff')
atlas.peakcoord_index(falff_img, 'falff')
atlas.peakcoord_index(reho_img, 'reho')

atlasout = AtlasDB(atlas.data)
atlasout.save_to_pkl(outpath, 'data_v2.pkl')
atlasout.save_to_mat(outpath, 'data_v2.mat')







