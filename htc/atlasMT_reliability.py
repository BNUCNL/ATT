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

atlasname = 'MT'
labelnames = ['rV3','lV3','rMT','lMT']
labelids = range(1,5)
labelers = ['lzg','zgf']

label_dir = '/nfs/j3/userhome/huangtaicheng/workingdir/parcellation_MT/BAA/results/yang_test/'


round1_images = [label_dir+'lzg_mt_z5.0.nii.gz']
round2_images = [label_dir+'zgf_mt_z5.0.nii.gz']

round1_images = atlastk.get_imgs_data(round1_images)
round2_images = atlastk.get_imgs_data(round2_images)

atlas_reliability = atlas.AtlasLabelingReliability(atlasname, labelnames, labelids, labelers)
reliabilitytk = atlastk.AtlasLabelingReliabilityTk(atlas_reliability)
reliabilitytk.cal_inter_label_size_stats(round1_images,round2_images)
reliabilitytk.cal_inter_reliability('dice', array_dice)
reliabilitytk.cal_inter_reliability('jaccard', array_jaccard)
reliabilitytk.cal_inter_reliability_each_labeler('dice', array_dice_each_item)
reliabilitytk.cal_inter_reliability_each_labeler('jaccard', array_jaccard_each_item)








