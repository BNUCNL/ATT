# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:1

"Test for atlas extraction and analysis"

from atlasdb_htc import *
import pandas as pd
import random
import time


def random_split_list(raw_list):
    length_list = len(raw_list)
    split_list = random.sample(raw_list,length_list)
    pre_splitlist = split_list[0:length_list/2]
    post_splitlist = split_list[length_list/2:]
    return pre_splitlist,post_splitlist

time0 = time.time()
# Index files
zstat_img_file = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/results/yang_test/sub/sub_split/mergedata/zstat_combined.nii.gz'
mask_img_file = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/results/yang_test/sub/sub_split/mergedata/mt_z5.0.nii.gz'
psc_img_file = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/results/yang_test/sub/sub_split/mergedata/psc_combined.nii.gz'
alff_img_file = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/results/yang_test/sub/sub_split/mergedata/alff_combined.nii.gz'
falff_img_file = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/results/yang_test/sub/sub_split/mergedata/falff_combined.nii.gz'
reho_img_file = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/results/yang_test/sub/sub_split/mergedata/reho_combined.nii.gz'

# roi files
zgf_img_file = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/results/yang_test/sub/sub_split/mergedata/zgf_z5.0.nii.gz'
lzg_img_file = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/results/yang_test/sub/sub_split/mergedata/lzg_z5.0.nii.gz'
htc_img_file = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/results/yang_test/sub/sub_split/mergedata/htc_z5.0.nii.gz'


areaname = ['rV3','lV3','rMT','lMT']
areanum = [1,2,3,4]
taskname = 'motion'
contrast = 'motion-fix'
threshold = 5.0

pathsex = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/doc/dfsf/modeID'
gender = pd.read_csv(os.path.join(pathsex, 'act_sex.csv'))['gender'].tolist()
sessid = open('/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/doc/dfsf/modeID/actID','rU').read().splitlines()

time1 = time.time()
print 'time of initial directory : %d' % (time1 - time0)
#-----------------------------------------------------------------------------#
# Split sessid into two halves
# pre_enum,post_enum = random_split_list(range(len(list(sessid))))
# pre_sessid = list(np.array(sessid)[pre_enum])
# post_sessid = list(np.array(sessid)[post_enum])
# pre_gender = list(np.array(gender)[pre_enum])
# post_gender = list(np.array(gender)[post_enum])
#-----------------------------------------------------------------------------#
# Prepare data
sessn = range(len(sessid))
# zstat
zstat_rawdata = Dataset(zstat_img_file, mask_img_file, areaname, areanum, gender, sessid, taskname, contrast)
zstat_rawdata.loadfile()
# psc
psc_rawdata = Dataset(psc_img_file, mask_img_file, areaname, areanum, gender, sessid, taskname, contrast)
psc_rawdata.loadfile()
# alff
# alff_rawdata = Dataset(alff_img_file, mask_img_file, areaname, areanum, gender, sessid, taskname, contrast)
# alff_rawdata.loadfile()
# falff
# falff_rawdata = Dataset(falff_img_file, mask_img_file, areaname, areanum, gender, sessid, taskname, contrast)
# falff_rawdata.loadfile()
# reho
# reho_rawdata = Dataset(reho_img_file, mask_img_file, areaname, areanum, gender, sessid, taskname, contrast)
# reho_rawdata.loadfile()

time2 = time.time()
print 'time of loadfile : %d' % (time2 - time1)

#---------------------------calculate index for whole data---------------------#
zstat_index = cal_index(zstat_rawdata, sessid, sessn, gender)
zstat_index.volume_index()
zstat_index.mask_index('zstat', 'mean')
zstat_index.peakcoord_index()

psc_index = cal_index(psc_rawdata, sessid, sessn, gender)
psc_index.psc_index('mean')
# psc_index.peakcoord_index()

# alff_index = cal_index(alff_rawdata, sessid, sessn, gender)
# alff_index.mask_index('alff')
# alff_index.peakcoord_index()

# falff_index = cal_index(falff_rawdata, sessid, sessn, gender)
# falff_index.mask_index('falff')
# falff_index.peakcoord_index()

# reho_index = cal_index(reho_rawdata, sessid, sessn, gender)
# reho_index.mask_index('reho')
# reho_index.peakcoord_index()

time3 = time.time()
print 'time of calculate index : %d' % (time3 - time2)

#---------------------------calculate PM and MPM------------------------------#
# getprob = make_atlas(zstat_rawdata, sessid, sessn)
# getprob.probatlas()
# getprob.MPM(0.2)


time4 = time.time()
print 'time of calculate probdata and mpmdata : %d' % (time4 - time3)
# --------------------------calculate reliability-----------------------------#
# reliab_hz = reliability(areanum)
# reliab_hz.loadfile(htc_img_file, zgf_img_file)
# reliab_hz.cal_dice()
# reliab_hl = reliability(areanum)
# reliab_hl.loadfile(htc_img_file, lzg_img_file)
# reliab_hl.cal_dice()
# reliab_zl = reliability(areanum)
# reliab_zl.loadfile(zgf_img_file, lzg_img_file)
# reliab_zl.cal_dice()

time5 = time.time()
print 'time of reliability : %d' % (time5 - time4)



print 'whole calculate time : %d' % (time7 - time0)






