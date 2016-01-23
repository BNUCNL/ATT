# coding: utf-8 
import os
import subprocess
import numpy as np
import nibabel as nib
from pandas import DataFrame

from datadeposite import DataDeposite, Struct, update

# In this file,it defines lots of different class for specific operation
# Such class mainly includes for two aspects:
# 1) labels - label check,label define, etc. 
# 2) rois - roi's attributes,in this file just init it

# instead os.path.join into pjoin
pjoin = os.path.join

# Merge label data
# What I want to mention here:
# In this stream,different people in charge of different part of job
# Each people need to finish roi-defined work for several areas and labels.
# Everyone labelled several subjects but not all.
# So it's meaningful to consider how to merge label data together
class LabelDataMerger(object):

# init attributes,variables include file path and output path
    def __init__(self, datadir, contrast, atlasname, thrstr, sep='_', imgsuf='.nii.gz', outdir='.'):
        update(self, datadir=datadir, contrast=contrast, atlasname=atlasname, 
                     thrstr=thrstr, sep=sep, middle_str=atlasname+sep+thrstr, 
                     imgsuf=imgsuf, outdir=outdir)
# Check whether label file is exist
    def check_lbintegrity(self, gflist, labelers, rsuf=''):
        for gsubjf, labeler in zip(gflist, labelers):
            for f in self._lb_filelist(gsubjf, labeler, rsuf):
                if not os.path.exists(f):
                    print 'Not Exist:', f
# merge labels signed by same persons together
# if merge_group is True,merge groups together
    def mergelb(self, gflist, labelers, outpres, rsuf='', merge_group=True):
        make_dir(self.outdir)
        outs = []
        for gsubjf, labeler, outpre in zip(gflist, labelers, outpres):
            output = pjoin(self.outdir, 
                           outpre+self.sep+labeler+self.sep+self.middle_str)
            subprocess.call(['fslmerge', '-t', output] + 
                            self._lb_filelist(gsubjf, labeler, rsuf))
            outs.append(output)
        if merge_group:
            all_merge = pjoin(self.outdir, self.middle_str)
            subprocess.call(['fslmerge', '-t', all_merge] + outs)
                    
    def _lb_filelist(self, gsubjf, labeler, rsuf=''):
        """Get the label file list according to group subject file and labler name
        """
        subjects = get_subjects(gsubjf)
        all_suf = self.sep+self.middle_str+rsuf+self.imgsuf
        filelist = [pjoin(self.datadir, s, self.contrast, labeler+all_suf) 
                    for s in subjects]
        return filelist


class LabelQualityChecker(object):
# Do quality check
# 1) check whether anyone who labelled as referece by mistakes
# 2) check whether anyone who labelled in wrong direction

# Init class.Import data from label,labelref,labelid,etc.
    def __init__(self, label, labelref, labelids, isright_roi=None, subjsf=None, gflist=None):
        self.label = get_img_data(label) 
        self.labelref = get_img_data(labelref)
        self.labelids = labelids
        self.isright_roi = self._right_lb if (isright_roi is None) else None
        self.subjs = get_subjects(subjsf) if (subjsf is not None) else None
        self.subjgrps = get_subjects_list(gflist) if (gflist is not None) else None
# label checker main function.
# Call methods of same_as_ref for wrong determined labels
# Call method of lr_misplaced for wrong label direction
    def check_lbquality(self):
        self.same_as_ref()
        self.lr_misplaced()
# Method of same_as_ref
    def same_as_ref(self):
        """
        If provide group, return the name of the subject.
        If provide sub_groups, return the specific sub_group id and the name of the subject.
        """
        ref = self.labelref
        label = self.label
        if label.ndim == 3:
            label = label[...,np.newaxis]    # make it useful for 3D label either
        for sub in range(label.shape[3]):
            for l in self.label_ids:
                mask2 = label[...,sub] == l
                if np.any(mask2) and np.all(ref[mask2]) and (ref[mask2].max() == ref[mask2].min()):
                    mask1 = (ref == ref[mask2].min())
                    if np.all(mask1 == mask2):
                        nz = mask1.nonzero()
                        pos = nz[0][0], nz[1][0], nz[2][0]
                        self._sublb_err('Label Warning: same as ref', sub+1, l, pos)
# Function of lr_misplaced
    def lr_misplaced(self):
        label = self.label
        median = cal_median(label.shape[0])
        for l in self.label_ids:
            for sub in range(label.shape[3]):
                nzero = (label[...,sub]==l).nonzero()
                if not self.isright_roi(l):
                    # left hemisphere
                    if len(nzero[0]) > 0:
                        x_min = nzero[0].min()
                        pos = nzero[0].argmin()
                        if x_min < median:
                            self._sublb_err('Left Right Misplace!', sub+1, l, 
                                            (nzero[0][pos],nzero[1][pos], nzero[2][pos]))
                else:
                    # right hemisphers
                    if len(nzero[0]) > 0:
                        x_max = nzero[0].max()
                        pos = nzero[0].argmax()
                        if x_max > median:
                            self._sublb_err('Left Right Misplace!', sub+1, l, 
                                            (nzero[0][pos],nzero[1][pos], nzero[2][pos]))
# Actually,even label id means left,odd label id means right
    def _right_lb(self, lbid):
        return lbid % 2
# Return error information
    def _sublb_err(self, errstr, subnum, l, ijk):
        if self.subjgrps is not None:
            try:
                (sg, sname) = self._get_subjgroup(subnum) 
            except TypeError:
                raise ValueError, 'No subject found in the subject group file list'
        elif self.subjs is not None:
            sname = self._get_subjname(subnum)
            sg = '?'
        else:
            sg, sname = '?', '?'
        print (errstr, 'Sub_num:', subnum, 'Subject:', sname, 'Sub_group:', 
               'Group'+str(sg), 'ROI:', l , 'at', ijk[0], ijk[1], ijk[2])

    def _get_subjname(self, num):
        """Get the name of a specific numbered subject.
        """
        return self.subjs[num-1]

    def _get_subjgroup(self, num):
        """Get the group of a specific numbered subject.
        """
        sb_total = 0
        for idx, g in enumerate(self.subjgrps):
            g_len = len(g)
            if sb_total+g_len >= num:
                return (idx+1, g[num-sb_total-1])
            sb_total += g_len


class PairLabelSizeStats(DataDeposite):
# this class save sizes of different labels
# Everyone needs to label different areas twice
# Therefor there's two label size
# And their intersection and union

# By call the class of DataDeposite,add attributes of _attrs into __dict__
    _attrs = ['first_label_sizes',
              'second_label_sizes',
              'overlap_label_sizes',
              'union_label_sizes']
# Init this class,assign values to new-added attributes
    def __init__(self, fsizes, ssizes, osizes, usizes):
        super(PairLabelSizeStats, self).__init__(
                first_label_sizes=fsizes, second_label_sizes=ssizes,
                overlap_label_sizes=osizes, union_label_sizes=usizes)


class AtlasLabelingReliability(DataDeposite):
# Init Label reliability
# All of reliability includes inter- and intra- reliability
    _attrs = ['atlasname',                
              'labelnames',
              'labelids',
              'labelers',

              'inter_reliability_measures',
              'intra_reliability_measures',

              'inter_reliability_measures_each_labeler',
              'intra_reliability_measures_each_labeler',

              'inter_label_size_stats',
              'intra_label_size_stats'
             ]
# Init attributes
    def __init__(self, atlasname, labelnames, labelids, labelers):
        super(AtlasLabelingReliability, self).__init__(
                atlasname=atlasname, labelnames=labelnames,
                labelids=labelids, labelers=labelers)
        self.inter_reliability_measures = {}
        self.intra_reliability_measures = {}
        self.inter_reliability_measures_each_labeler = {}
        self.intra_reliability_measures_each_labeler = {}


class AtlasStat(DataDeposite):
# This class record relationship between atlas,such as multi_roi_relations,MPM,etc.

    _attrs = ['name', 
              'roistatset',
              'sthr',
              'multi_roi_relations',
              'max_prob_map',

              'atlas'
             ]
    #_nodump_attrs = ['atlas']

# Init some of attributes and values.
# But I'm curious that why not init max_prob_map
    def __init__(self, name, roistatset, sthr, atlas):
        super(AtlasStat, self).__init__(name=name, roistatset=roistatset, sthr=sthr, atlas=atlas)
        self.multi_roi_relations = {}


class Atlas(DataDeposite):
# This class is important,it mainly inits some necessary attributes
# Attention that roiset will be init by method init_roiset,which contains 
# values in class ROI 
    _attrs = ['name',
              'labelnames',
              'labelids',
              'roiset',
              'nsubjs',
              'subjs',
              'sex',
              'img_affine',                 # affine of the image data
              'img_resolution',             # resolution of image data
             ]

    _nodump_attrs = ['label'                # the labeled 4d image
                    ]

# init attributes
    def __init__(self, label, atlasname, labelnames, labelids, img_affine, img_resolution, subjs=None, sex=None):
        self.name = atlasname
        self.label = get_img_data(label)
        self.nsubjs = self.label.shape[3]
        self.labelnames = labelnames
        self.labelids = labelids
        self.img_affine = img_affine
        self.img_resolution = img_resolution
        self.subjs = subjs
        self.sex = sex
        self.init_roiset()
# roiset is a list,which contains area name,label data,labels,subj,nsubjs,resolution,gender and some attributes defined by RoiStat,such as peak maps,prob maps,anatomical relations,etc.
    def init_roiset(self):
        label = self.label
        roiset = []
        for lname,lid in zip(self.labelnames, self.labelids):
            roiset.append(Roi(label==lid, lname, self.img_resolution, self.subjs, self.sex))
        self.roiset = roiset
# get attributes in roiset to class of AtlasStat
    def get_basestatset(self):
        roistatset = [getattr(roi, 'basestat') for roi in self.roiset]
        return AtlasStat(name=self.name, roistatset=roistatset, sthr=None, atlas=self)

class MultiRoiRelation(DataDeposite):
# Defined relation matrix,relation name and size threshold
    _attrs = ['relation_matrix',        # the data matrix
              'relation_name',          # the name of the relationship
              'size_thresh'             # which size threshold this relation object is calculated based on
             ]


class RoiProbMap(DataDeposite):
    _attrs = ['probmap', # 3D probmap
              'allsubj'  # bool, is allsub
             ]

class PeakKernelMap(DataDeposite):
    _attrs = ['kernelmap', 'sigma']

class AnatRelation(DataDeposite):
    _attrs = ['relation_matrix', 'relation_name']

class BetweenSubjRelation(DataDeposite):
    _attrs = ['relation_matrix', 'relation_name']

class RankOrderMap(DataDeposite):
    _attrs = ['rankmap', 'allsubj']

# The attribute names used in the dataframe object in each RoiStat object, i.e. the dfkey defines the structure of the dataframe object.
dfkey = Struct(idx     = 'dataidx',       # this dataframe's statistical elements' indices in the original Roi object's ``label`` array
               size    = 'size',
               sex     = 'sex',
               peak    = 'peak',
               center  = 'center',
               gcenter = 'gcenter',
               peakval = 'peakval',
              )


class RoiStat(DataDeposite):
# Followed with the former class(Roi,in init_basestat method).
# It contains several roi state attributes,such as probmaps(may call the class of RoiProbMap)
    _attrs = ['name',                     # name of this ROI
              'sthr',                     # size threshold
              'df',                       # dataframe
              'subj_flag',                # the flag which shows whether one subject remains in this stat object
              'nsubj',                    # number of subjects in this stat
              
              'any_map',                  # map of voxel where exists at least a subject's activation
              'count_map',                # count map
              'prob_maps',                # probability map set, list
              'peak_kernel_maps',         # kernel peak map set, list
              'rank_order_maps',          # rank order map set, list

              'anat_relations',           # anatomy relation map set, dict
              'between_subj_relations',   # between subject relations, dict

              'roi'
             ]
    #_nodump_attrs = ['roi']

    def __init__(self, name, sthr, dataframe, roi):
        self.name = name
        self.sthr = sthr
        self.df = dataframe
        subj_flag = np.zeros(roi.nsubj)
        subj_flag[dataframe[dfkey.idx].values] = 1
        self.subj_flag = subj_flag
        self.nsubj = len(dataframe)
        self.roi = roi
        self.any_map = None
        self.count_map = None
        self.prob_maps = []
        self.peak_kernel_maps = []
        self.rank_order_maps = []
        self.anat_relations = {}
        self.between_subj_relations = {}


class Roi(DataDeposite):
# Defined some roi attributes
    _attrs = ['name', 'label', 'subjs', 'nsubj', 'sex', 'basestat']
    
    def __init__(self, label, name, resolution, subjs=None, sex=None):
        self.name = name
        label = get_img_data(label)
        self.label = label.astype(np.bool)
        self.subjs = subjs
        self.nsubj = label.shape[3]
        self.resolution = resolution
        self.sex = sex
        self.init_basestat()

    def init_basestat(self):
# Basestat came from Roistat,df contains index of subjects(number),gender,volumes and sid of subjs
        size = np.sum(self.label, (0,1,2))
        df = DataFrame({dfkey.idx: np.arange(self.nsubj), 
                        dfkey.sex: self.sex, 
                        dfkey.size: size}, 
                       index=self.subjs)
        basestat = RoiStat(self.name, None, df, self)
        self.basestat = basestat

# make directory
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.path.isdir(path):
        raise ValueError, "%s has already exists and not a directory!" % path
# get subjects sid
def get_subjects(filename):
    with open(filename, 'r') as f:
        subs = f.read().split()
    return subs

def get_subjects_list(filenames):
    return map(get_subjects, filenames)
# get img data
def get_img_data(img):
    if isinstance(img, np.ndarray):
        return img
    elif isinstance(img, str):
        return nib.load(img).get_data()
    else:
        raise ValueError, 'Unknown Img Type: not a string and not a array'
# For judging left or right hemisphere,x < median,left hemisphere;x > median,right hemisphere
def cal_median(len):
    if len % 2:
        return int(len)/2
    else:
        return len/2.
