# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:1


import os
import pandas as pd
from atlasdb import *
import nibabel as nib


data_path = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/mergedata/mergedata_mt/'

info_path = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/doc/dfsf/modeID'

zstat_img_path = os.path.join(data_path, 'zstat_combined.nii.gz')
zstat_img = nib.load(zstat_img_path)
psc_img_path = os.path.join(data_path, 'psc_combined.nii.gz')
psc_img = nib.load(psc_img_path)
alff_img_path = os.path.join(data_path, 'alff_combined.nii.gz')
alff_img = nib.load(alff_img_path)
falff_img_path = os.path.join(data_path, 'falff_combined.nii.gz')
falff_img = nib.load(falff_img_path)
reho_img_path = os.path.join(data_path, 'reho_combined.nii.gz')
reho_img = nib.load(reho_img_path)


mask_img_file =  os.path.join(data_path, 'mt_z5.0.nii.gz')

roi_name = ['rV3', 'lV3', 'rMT', 'lMT']
roi_id = [1,2,3,4]
task = 'motion'
contrast = 'motion-fix'
threshold = 5.0


subj_gender = pd.read_csv(os.path.join(info_path, 'act_sex.csv'))['gender'].tolist()
subj_id = open(os.path.join(info_path,'actID'),'rU').read().splitlines()

dataDB = {}

mt_atlas = Atlas(mask_img_file,roi_id, roi_name, task, contrast, threshold, subj_id,subj_gender)

dataDB['basic'] = {}
dataDB['basic']['task'] = mt_atlas.task
dataDB['basic']['subjid'] = mt_atlas.subj_id
dataDB['basic']['threshold'] = mt_atlas.threshold
dataDB['basic']['gender'] = mt_atlas.subj_gender
dataDB['basic']['roiname'] = mt_atlas.roi_name
dataDB['basic']['roiid'] = mt_atlas.roi_id
dataDB['basic']['contrast'] = mt_atlas.contrast

dataDB['geo'] = {}
dataDB['geo']['volume'] = mt_atlas.volume

metric_sca = ['mean', 'max', 'min', 'std', 'median', 'skewness', 'kurtosis']


for met in metric_sca:        
    if not dataDB.has_key('act'):
        dataDB['act'] = {}
    if not dataDB['act'].has_key('zstat'):
        dataDB['act']['zstat'] = {}
    dataDB['act']['zstat'][met + '_zstat'] = mt_atlas.collect_meas(zstat_img, met)
    if not dataDB['act'].has_key('psc'):
        dataDB['act']['psc'] = {}
    dataDB['act']['psc'][met + '_psc'] = mt_atlas.collect_meas(psc_img, met)
    
    if not dataDB.has_key('rest'):
        dataDB['rest'] = {}
    if not dataDB['rest'].has_key('alff'):
        dataDB['rest']['alff'] = {}
    dataDB['rest']['alff'][met + '_alff'] = mt_atlas.collect_meas(alff_img, met)
    if not dataDB['rest'].has_key('falff'):
        dataDB['rest']['falff'] = {}
    dataDB['rest']['falff'][met + '_falff'] = mt_atlas.collect_meas(falff_img, met)
    if not dataDB['rest'].has_key('reho'):
        dataDB['rest']['reho'] = {}
    dataDB['rest']['reho'][met + '_reho'] = mt_atlas.collect_meas(reho_img, met)

metric_mtr = ['peak', 'center']
for met in metric_mtr:
    if not dataDB['geo'].has_key(met + 'coor'):
        dataDB['geo'][met+'coor'] = {}
    dataDB['geo'][met+'coor']['zstat_'+met+'coor'] = mt_atlas.collect_meas(zstat_img, met)
    dataDB['geo'][met+'coor']['psc_'+met+'coor'] = mt_atlas.collect_meas(psc_img, met)
    dataDB['geo'][met+'coor']['alff_'+met+'coor'] = mt_atlas.collect_meas(alff_img, met)
    dataDB['geo'][met+'coor']['falff_'+met+'coor'] = mt_atlas.collect_meas(falff_img, met)
    dataDB['geo'][met+'coor']['reho_'+met+'coor'] = mt_atlas.collect_meas(reho_img, met)







# atlas.save_to_pkl(outpath, 'data_v3.pkl')
# atlas.save_to_mat(outpath, 'data_v3.mat')







