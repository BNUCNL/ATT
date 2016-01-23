import numpy as np
import nibabel as nib
from scipy import ndimage as nd

import pandas as pd
import time
import atlas.atlas as atlas
import atlas.atlastk as atlastk

t0 = time.time()

infodir = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/doc/dfsf/sub/'
datadir_roi = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/results/yang_test/sub/'
datadir_act = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/results/yang_test/sub/activation/'

probmapdir = datadir_roi+'data/probmaps/'

subjectid_file = infodir+'subjID'
subjectsex_file = infodir+'sex.csv'
label_file = datadir_roi+'mt_z5.0.nii.gz'
########################################################################
label = nib.load(label_file)
label_data = label.get_data()
img_affine = label.get_affine()
img_resolution = label.get_header().get_zooms()[:3]
MT_lbs = ['rV3','lV3','rMT','lMT']
MT_lbid = range(1,5)

subjs = atlas.get_subjects(subjectid_file)
subj_sex = pd.read_csv(subjectsex_file)['gender'].tolist()
######################################################################
atlas_deposite = atlas.Atlas(label_data,'MT',MT_lbs,MT_lbid,img_affine,img_resolution,subjs,subj_sex)
atlas_basestat = atlas_deposite.get_basestatset()
atlas_stattk = atlastk.AtlasStatTk(atlas_basestat)
atlas_threshedstat = atlas_stattk.size_thresh(0)
atlas_stattk = atlastk.AtlasStatTk(atlas_threshedstat)

t1 = time.time() - t0
print 'atlas object construction done: %f sec' % t1
#########################################################################
MT_con = ['motion-fix']
MT_con_key = ['mtfix']
sigtype = ['zstat','tstat','cope','varcope']
valtype = ['Z','T','B','V','E']

def gen_signals(sigdir, cons, types):
    signal = [[sigdir+type+'-'+con+'.nii.gz' for con in cons] for type in types]
    return signal

def gen_sigkeys(cons, types):
    keys = [[type+con for con in cons] for type in types]
    return keys

def apply_loadimg(dirs):
    return map(lambda x: nib.load(x).get_data(), dirs)

con = MT_con
con_key = MT_con_key
signals = gen_signals(datadir_act, con, sigtype)
sigkeys = gen_sigkeys(con_key, valtype)

# z statistics
zsignals = apply_loadimg(signals[0])
zsigkeys = sigkeys[0]

# t statistics
tsignals = apply_loadimg(signals[1])
tsigkeys = sigkeys[1]

# beta of cope 
bsignals = apply_loadimg(signals[2])
bsigkeys = sigkeys[2]

# var statistics
vsignals = apply_loadimg(signals[3])
vsigkeys = sigkeys[3]

esigkeys = sigkeys[4]

ztbvsignals = zsignals+tsignals+bsignals+vsignals
ztbvsigkeys = zsigkeys+tsigkeys+bsigkeys+vsigkeys

t2 = time.time() - t1
print 'atlas object construction done: %f sec' % t2
#########################################################################
pk_center_tk = atlas_stattk.cal_point_corrdinates(zsignals[0])
atlas_stattk.extract_peak_vals(ztbvsignals, ztbvsigkeys)
peak_vkeys = ['peak_'+ x for x in vsigkeys]
peak_ekeys = ['peak_'+ x for x in esigkeys]
atlas_stattk.update_cached_stats(np.sqrt, peak_vkeys, peak_ekeys)
#########################################################################
meantk = atlastk.SignalTk([nd.mean], ['mean'])
stdtk = atlastk.SignalTk([nd.standard_deviation], ['std'])
atlas_stattk.extract_signals_stats(zsignals+tsignals+bsignals, meantk, zsigkeys+tsigkeys+bsigkeys)
atlas_stattk.extract_signals_stats(zsignals+tsignals+bsignals, stdtk, zsigkeys+tsigkeys+bsigkeys)
#########################################################################
mean_bkeys = ['mean_'+x for x in bsigkeys]
size_keys = ['size'] * len(vsigkeys)
roi_bkeys = ['roi_'+x for x in bsigkeys]
roi_tkeys = ['roi_'+x for x in tsigkeys]
roi_zkeys = ['roi_'+x for x in zsigkeys]
roi_vkeys = ['roi_'+x for x in vsigkeys]
roi_ekeys = ['roi_E'+x for x in MT_con_key]
mean_vkeys = ['mean_'+x for x in vsigkeys]
#########################################################################
atlas_stattk.update_cached_stats(lambda x:x, mean_bkeys, roi_bkeys)
pooled_fixedvar_tk = meantk
atlas_stattk.extract_signals_stats(vsignals, pooled_fixedvar_tk, vsigkeys)
atlas_stattk.update_cached_stats(lambda x, y: x/y, zip(mean_vkeys, size_keys), roi_vkeys) # pooled fixed variance == mean region variance / region size == sum of variance / region size**2  AND DoF = sum of each voxel's DoF
atlas_stattk.update_cached_stats(np.sqrt, roi_vkeys, roi_ekeys)
atlas_stattk.update_cached_stats(lambda x, y: x/y, zip(roi_bkeys, roi_ekeys), roi_tkeys)
dof = 160
atlas_stattk.update_cached_stats(lambda x, y: atlastk.t_ztransform(x, dof*y - 1), zip(roi_tkeys, size_keys), roi_zkeys)

t3 = time.time() - t2
print 'Signal Value and Statistics Done: %f sec' % t3
########################################################################
anat_img = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/results/yang_test/template/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz'
anat_img = nib.load(anat_img).get_data()

atlas_stattk.cal_anat_relations('overlap', atlastk.multi_left_overlap, 'HOcort_thr0', anat_img)
#########################################################################
atlas_stattk.cal_between_subj_relation('dice', atlastk.between_dice)
#########################################################################
dist_func = lambda x, y:atlastk.dist(x, y, atlas_deposite.img_resolution)
atlas_stattk.cal_multi_roi_relation('peak_distance', dist_func, atlastk.RoiStatTk.get_subject_peak)
hausdorff_func = lambda x, y: atlastk.hausdorff(x, y, atlas_deposite.img_resolution)
atlas_stattk.cal_multi_roi_relation('hausdorff_distance', hausdorff_func, atlastk.RoiStatTk.get_subject_label)

t4 = time.time() - t3
print 'atlas object construction done: %f sec' % t4
#########################################################################
atlas_stattk.cal_prob_maps()
atlas_stattk.save_prob_maps(probmapdir)
atlas_stattk.cal_max_prob_map(0, mapno=0, save=True, filename=probmapdir+'maxprob_thr0.nii.gz')
#########################################################################
mat_dir = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/results/yang_test/sub/data'
atlastk.output_mats(atlas_stattk, mat_dir, 'MT', MT_con, MT_con_key,'motion-fix' , 5.0, 0, 160)

















