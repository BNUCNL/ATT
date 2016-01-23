import yangtk as ytk
import atlas.atlas as atlas
import atlas.atlastk as atlastk
from scipy import io as sio
import numpy as np


def array_dice(fsizes, ssizes, osizes, usizes):
    return 2. * np.sum(osizes) / np.sum(fsizes+ssizes)

def array_jaccard(fsizes, ssizes, osizes, usizes):
    return np.sum(osizes) / float(np.sum(usizes))

def array_dice_each_item(fsizes, ssizes, osizes, usizes):
    dice =  2. * osizes / (fsizes+ssizes)
    #dice = dice[~np.isnan(dice)]
    return dice

def array_jaccard_each_item(fsizes, ssizes, osizes, usizes):
    jaccard = osizes.astype(np.float) / usizes
    #jaccard = jaccard[~np.isnan(jaccard)]
    return jaccard

atlasname='face'
labelnames = ['rOFA', 'lOFA', 'rpFus', 'lpFus', 'raFus', 'laFus', 'rpcSTS', 'lpcSTS', 'rpSTS', 'lpSTS', 'raSTS', 'laSTS']
labelids = range(1, 13)
labelers = ['dxb', 'hlj', 'yzt', 'wx', 'hyy', 'kxz', 'zzl']

label_dir = '/nfs/t3/workingshop/yangzetian/Atlas/2006ObjectAtlas/data/label_res/'

round1_images = [label_dir+'Round1/'+file+'face_z2.3.nii.gz' for file in ['', 
    'G1_dxb_', 'G2_hlj_', 'G3_yzt_', 'G4_wx_', 'G5_hyy_', 'G6_kxz_', 'G7_zzl_']]
round2_images = [label_dir+'Round2/'+file+'face_z2.3.nii.gz' for file in ['', 
    'G1_zzl_', 'G2_dxb_', 'G3_hlj_', 'G4_yzt_', 'G5_wx_', 'G6_hyy_', 'G7_kxz_']]
round3_images = [label_dir+'Round3/'+file+'face_z2.3.nii.gz' for file in ['', 
    'G1_dxb_', 'G2_hlj_', 'G3_yzt_', 'G4_wx_', 'G5_hyy_', 'G6_kxz_', 'G7_zzl_']]

round1_images = ytk.get_imgs_data(round1_images)
round2_images = ytk.get_imgs_data(round2_images)
round3_images = ytk.get_imgs_data(round3_images)

atlas_reliability = atlas.AtlasLabelingReliability(atlasname, labelnames, labelids, labelers)
reliabilitytk = atlastk.AtlasLabelingReliabilityTk(atlas_reliability)
reliabilitytk.cal_inter_label_size_stats(round1_images, round2_images)
reliabilitytk.cal_inter_reliability('dice', array_dice)
reliabilitytk.cal_inter_reliability('jaccard', array_jaccard)
reliabilitytk.cal_inter_reliability_each_labeler('dice', array_dice_each_item)
reliabilitytk.cal_inter_reliability_each_labeler('jaccard', array_jaccard_each_item)

reliabilitytk.cal_intra_label_size_stats(round1_images, round3_images)
reliabilitytk.cal_intra_reliability('dice', array_dice)
reliabilitytk.cal_intra_reliability('jaccard', array_jaccard)
reliabilitytk.cal_intra_reliability_each_labeler('dice', array_dice_each_item)
reliabilitytk.cal_intra_reliability_each_labeler('jaccard', array_jaccard_each_item)


##################################### save to mat ########################

matfile = '/nfs/t3/workingshop/yangzetian/Atlas/2006ObjectAtlas/data/mats/reliability.mat'
inter_dice = atlas_reliability.inter_reliability_measures_each_labeler['dice'].to_dict()
intra_dice = atlas_reliability.intra_reliability_measures_each_labeler['dice'].to_dict()

def mod_reliab_dict(reliab_dict):
    reliab = np.zeros((13,202))
    stats = np.zeros((13, 4)) # num of not nan, mean, std, se
    all_labler_mean = np.zeros((13, 8))
    for i, label in enumerate(['atlas',]+labelnames):
        roi_reliab = reliab_dict['all_labelers'][label]
        reliab[i] = roi_reliab
        mask = ~np.isnan(roi_reliab)
        stats[i, 0] = mask.sum()
        stats[i, 1] = np.mean(roi_reliab[mask])
        stats[i, 2] = np.std(roi_reliab[mask])
        stats[i, 3] = stats[i,2] / stats[i,0] ** 0.5
        for j, lbler in enumerate(['all_labelers',]+labelers):
            data = reliab_dict[lbler][label]
            all_labler_mean[i, j] = np.mean(data[~np.isnan(data)])
    return {'reliability': reliab, 'reliability_stats':stats, 'all_labeler_mean':all_labler_mean}

dice = {'inter_dice':mod_reliab_dict(inter_dice), 'intra_dice':mod_reliab_dict(intra_dice)}
sio.savemat(matfile, dice)
