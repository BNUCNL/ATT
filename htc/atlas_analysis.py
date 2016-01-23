# coding: utf-8
import numpy as np
import nibabel as nib
from scipy import ndimage as nd

import atlas.atlas as atlas
import atlas.atlastk as atlastk
import pandas as pd
import time

t0 = time.time()
#-----------------------------------Atlas, AtlasStat and AtlasStatTk object construction------------------------

##------------------------------------------- dir and file preparation -----------------------------------------
infodir = 'info/'
datadir = 'data/'
sigdir = '/nfs/t3/workingshop/yangzetian/Atlas/activation_img/2006_activation/'
probmapdir = datadir+'probmaps/new_atlas_lib/'

subjectid_file = infodir+'2006subID'
subjectsex_file = infodir+'2006subSex.csv'
label_file = datadir+'label_res/Round5/face_z2.3.nii.gz'
label_data = nib.load(label_file).get_data()

### atlas datadeposite construction

img_affine = nib.load(label_file).get_affine()
img_resolution = nib.load(label_file).get_header().get_zooms()[:3]
face_lbs = ['rOFA', 'lOFA', 'rpFus', 'lpFus', 'raFus', 'laFus', 'rpcSTS', 'lpcSTS', 'rpSTS', 'lpSTS', 'raSTS', 'laSTS']
face_lbid = range(1, 13)
#obj_lbs = ['rpFs', 'lpFs', 'rLO', 'lLO']
#scene_lbs = ['rPPA', 'lPPA', 'rHipp', 'lHipp', 'rRSC', 'lRSC', 'rTOS', 'lTOS']

subjs = atlas.get_subjects(subjectid_file)
subj_sex = pd.read_csv(subjectsex_file)['sex'].tolist()

##------------------------------------------Object initialization and size threshold-----------------------------
atlas_deposite = atlas.Atlas(label_data, 'face', face_lbs, face_lbid, img_affine, img_resolution, subjs, subj_sex)
atlas_basestat = atlas_deposite.get_basestatset()
atlas_stattk = atlastk.AtlasStatTk(atlas_basestat)
atlas_threshedstat = atlas_stattk.size_thresh(0)
atlas_stattk = atlastk.AtlasStatTk(atlas_threshedstat)

t1 = time.time() - t0
print 'atlas object construction done: %f sec' % t1

#------------------------------------------- Signal Value/Statistics Extraction-----------------------------------

##---------------------------------------------Signal Loading and Signal Keys Specification------------------------
face_con = ['face-object', 'face-fix', 'object-fix', 'scene-fix', 'scram-fix']
face_con_key = ['faceobj', 'facefix', 'objfix', 'scenefix', 'scramfix']
sigtype = ['zstat', 'tstat', 'cope', 'varcope']   # string for signal loading 

valtype = ['Z', 'T', 'B', 'V', 'E']                    # z-stat, t-stat, beta, variance of beta, standard error

def gen_signals(sigdir, cons, types):
    signal = [[sigdir+type+'-'+con+'.nii.gz' for con in cons] for type in types]
    return signal

def gen_sigkeys(cons, types):
    keys = [[type+con for con in cons] for type in types]
    return keys

def apply_loadimg(dirs):
    return map(lambda x: nib.load(x).get_data(), dirs)

con = face_con
con_key = face_con_key
signals = gen_signals(sigdir, con, sigtype)
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
print "Signal loading and signal value key specification done: %f sec" % t2


##--------------------------------------- Value Extraction and Calculation from Signals -------------------------

###--------------------------Peak/Center/Gravity Center coordinates & Peak Statistics-------------------------------
pk_center_tk = atlas_stattk.cal_point_corrdinates(zsignals[0])
atlas_stattk.extract_peak_vals(ztbvsignals, ztbvsigkeys)
peak_vkeys = ['peak_'+ x for x in vsigkeys]
peak_ekeys = ['peak_'+ x for x in esigkeys]
atlas_stattk.update_cached_stats(np.sqrt, peak_vkeys, peak_ekeys)

###------------------------------------- Spatial mean/std Statistics -----------------------------------------------
meantk = atlastk.SignalTk([nd.mean], ['mean'])
stdtk = atlastk.SignalTk([nd.standard_deviation], ['std'])
atlas_stattk.extract_signals_stats(zsignals+tsignals+bsignals, meantk, zsigkeys+tsigkeys+bsigkeys)
atlas_stattk.extract_signals_stats(zsignals+tsignals+bsignals, stdtk, zsigkeys+tsigkeys+bsigkeys)
#
# The last 4 lines Could also be accomplished in a more compact fashion:
#     mean_std_tk = atlastk.SignalTk([nd.mean, nd.standard_deviation], ['mean', 'std'])
#     atlas_stattk.extract_signals_stats(zsignals+tsignals+bsignals, mean_std_tk, zsigkeys+tsigkeys+bsigkeys)

###-------------------- ROI statistics (view ROI as one entity): beta, t, z, var, se ------------------------------
mean_bkeys = ['mean_'+x for x in bsigkeys]
size_keys = ['size'] * len(vsigkeys)
roi_bkeys = ['roi_'+x for x in bsigkeys]
roi_tkeys = ['roi_'+x for x in tsigkeys]
roi_zkeys = ['roi_'+x for x in zsigkeys]
roi_vkeys = ['roi_'+x for x in vsigkeys]
roi_ekeys = ['roi_E'+x for x in face_con_key]
mean_vkeys = ['mean_'+x for x in vsigkeys]
# ROI statistics
atlas_stattk.update_cached_stats(lambda x:x, mean_bkeys, roi_bkeys)
pooled_fixedvar_tk = meantk
atlas_stattk.extract_signals_stats(vsignals, pooled_fixedvar_tk, vsigkeys)
atlas_stattk.update_cached_stats(lambda x, y: x/y, zip(mean_vkeys, size_keys), roi_vkeys) # pooled fixed variance == mean region variance / region size == sum of variance / region size**2  AND DoF = sum of each voxel's DoF
atlas_stattk.update_cached_stats(np.sqrt, roi_vkeys, roi_ekeys)
atlas_stattk.update_cached_stats(lambda x, y: x/y, zip(roi_bkeys, roi_ekeys), roi_tkeys)
dof = 169
atlas_stattk.update_cached_stats(lambda x, y: atlastk.t_ztransform(x, dof*y - 1), zip(roi_tkeys, size_keys), roi_zkeys)

t3 = time.time() - t2
print 'Signal Value and Statistics Done: %f sec' % t3

#-------------------------------------------------- ROI Relationship ---------------------------------------------------

##--------------------------- Relationship between anatomical labels (for each ROI and each subject) -------------------
anat_img = '/nfs/t3/workingshop/yangzetian/Atlas/2006ObjectAtlas/info/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz'
anat_img = nib.load(anat_img).get_data()

atlas_stattk.cal_anat_relations('overlap', atlastk.multi_left_overlap, 'HOcort_thr0', anat_img)
#atlas_stattk.cal_anat_relations('jaccard', atlastk.multi_jaccard, 'HOcombo', anat_img)
#atlas_stattk.cal_anat_relations('dice', atlastk.multi_dice, 'HOcombo', anat_img)

#multi_hausdorff_func = lambda label, anat, aids, nid: atlastk.multi_hausdorff(label, anat, aids, nid, atlas_deposite.img_resolution)
#atlas_stattk.cal_anat_relations('dice', multi_hausdorff_func, 'HOcombo', anat_img)

##--------------------------------- Relationship between different subject (for each ROI) ------------------------------
atlas_stattk.cal_between_subj_relation('dice', atlastk.between_dice)

##--------------------------------- Relationship between different ROIs (for each subject)------------------------------
dist_func = lambda x, y:atlastk.dist(x, y, atlas_deposite.img_resolution)
atlas_stattk.cal_multi_roi_relation('peak_distance', dist_func, atlastk.RoiStatTk.get_subject_peak)
hausdorff_func = lambda x, y: atlastk.hausdorff(x, y, atlas_deposite.img_resolution)
atlas_stattk.cal_multi_roi_relation('hausdorff_distance', hausdorff_func, atlastk.RoiStatTk.get_subject_label)

t4 = time.time() - t3
print  'ROI relationship done: %f sec' % t4

#--------------------------------- Probmaps and MaxProbMap -------------------------------------------------------
atlas_stattk.cal_prob_maps()
atlas_stattk.save_prob_maps(probmapdir)
atlas_stattk.cal_max_prob_map(0, mapno=0, save=True, filename=probmapdir+'maxprob_thr0.nii.gz')

#---------------------------------- Output -------------------------------------------------------
mat_dir = '/nfs/t3/workingshop/yangzetian/Atlas/2006ObjectAtlas/data/mats'
atlastk.output_mats(atlas_stattk, mat_dir)

rint 'Total time: %f sec' % (time.tim:e()-t0)
