# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import nibabel as nib
import os
import numpy as np
import pandas as pd
from ATT.corefunc import atlasbase, analysebase
from ATT.iofunc import iofiles

pjoin = os.path.join

out_parpath = '/nfs/j3/userhome/huangtaicheng/hworkingshop/parcellation_MT/Summary_result/program/mttask/data'
nifti_parpath = '/nfs/j3/userhome/huangtaicheng/hworkingshop/parcellation_MT/BAA/mergedata/mergedata_mt/modecomb_beh_noout'

## ---------------------------------------------
# subject specific atlas
img_atlas_spec = nib.load(pjoin(nifti_parpath, 'mt_z5.0_ff.nii.gz'))
header = img_atlas_spec.get_header()
atlas_spec = img_atlas_spec.get_data()
region_spec = ['rV3', 'lV3', 'rMT', 'lMT']
# signal nifti file
actvalues = nib.load(pjoin(nifti_parpath, 'zstat1.nii.gz')).get_data()

signal_roi_spec = atlasbase.ExtractSignals(atlas_spec, region_spec)
signals_spec = signal_roi_spec.getsignals(actvalues)

iofactory = iofiles.IOFactory()
factory = iofactory.createfactory('.', 'zstatavg.csv')
factory.nparray2csv(signals_spec, ['rV3','lV3', 'rMT', 'lMT'])
# coordinate_spec = signal_roi_spec.getcoordinate(actvalues)
# dist_point_spec = signal_roi_spec.getdistance_array2point(actvalues, [[34, -88, -4], [-32, -90, -4], [46, -66, 4], [-46, -72, 4]], distmeth = 'minkowski')
# dist_point_spec = signal_roi_spec.getdistance_array2point(actvalues, [[0,0,0]], distmeth = 'minkowski')

# makemask_spec = atlasbase.MakeMasks(header, issave = True, savepath = 'data')
# makemask_spec.makepm(atlas_spec)
# makemask_spec.makempm(0.1, 'mpm_0.1.nii.gz')
# makemask_spec.makempm(0.2, 'mpm_0.2.nii.gz')
# ----------------------------------------------

# -----------------------------------------------
# img_atlas_rg = nib.load('/nfs/h1/workingshop/huangtaicheng/parcellation_MT/Summary_result/rawdata/masks/heatmap_mask/face_mt_300.nii.gz')
# header = img_atlas_rg.get_header()
# atlas_rg = img_atlas_rg.get_data()
# region_rg = ['rOFA', 'lOFA', 'rpFus', 'lpFus', 'rpSTS', 'lpSTS', 'rMT', 'lMT']

# sessid = pd.read_csv('/nfs/h1/workingshop/huangtaicheng/parcellation_MT/Summary_result/rawdata/subjinfo/subj_info.csv').values[:,0].tolist()
# restdata_parpath = '/nfs/t1/nsppara/resting'
# stem_rest = 'res4dstandard'
# restfile = 'res4dmni.nii.gz'
# restpath = [pjoin(restdata_parpath, sid, stem_rest, restfile) for sid in sessid]
#
# signal_rg = atlasbase.ExtractSignals(atlas_rg, region_rg)
#
# rest_signals_rg = np.empty((236, len(region_rg), len(sessid)))
# for i,rp in enumerate(restpath):
#    print('%s' % sessid[i])
#    restdata = nib.load(rp).get_data()
#     rest_signals_rg[...,i] = signal_rg.getsignals(restdata)
     
# actdata = nib.load('/nfs/h1/workingshop/huangtaicheng/parcellation_MT/BAA/mergedata/mergedata_mt/modecomb_beh_noout/zstat1.nii.gz').get_data()
# act_signals_rg = signal_rg.getsignals(actdata)

# --dice----------------------------------------------------------------
# outstem = 'data_spec/dice'
# htc_data = nib.load(pjoin(nifti_parpath, 'htc_mt_z5.0.nii.gz')).get_data()
# lzg_data = nib.load(pjoin(nifti_parpath, 'lzg_mt_z5.0.nii.gz')).get_data()
# zgf_data = nib.load(pjoin(nifti_parpath, 'zgf_mt_z5.0.nii.gz')).get_data()
# evaluate_class = analysebase.EvaluateMap(issave = True, savepath = pjoin(out_parpath, outstem))
# dice_lzg_htc = evaluate_class.dice_evaluate(lzg_data, htc_data, 'dice_lzg_htc.pkl')
# dice_lzg_zgf = evaluate_class.dice_evaluate(lzg_data, zgf_data, 'dice_lzg_zgf.pkl')
# dice_zgf_htc = evaluate_class.dice_evaluate(zgf_data, htc_data, 'dice_zgf_htc.pkl')




