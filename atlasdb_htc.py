# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
import copy
import cPickle
import scipy.io as si
from scipy import stats


class user_defined_exception(Exception):
    def __init__(self, str):
        Exception.__init__(self)
        self._str = str



class Dataset(object):
    def __init__(self, targ_img_file, mask_img_file, areaname, areanum, gender, sessid, taskname, contrast, pictype = 'zstat'):
        self.ftarg_img = targ_img_file
        self.fmask_img = mask_img_file
        self.areaname = areaname
        self.areanum = areanum
        self.gender = gender
        self.taskname = taskname
        self.contrast = contrast
        self.affine = []
        self.targ_data = []
        self.mask_data = []
        self.shape = []
        self.header = []
        self.sessid = sessid
        self.narea = len(areanum)
        self.pictype = pictype
    def loadfile(self):
    # load targ_img_file and mask_img_file
        targ_img = nib.load(self.ftarg_img)
        if len(targ_img.get_shape()) != 4:
             raise user_defined_exception('targ_img is not a Nifti 4D image!')
        targ_data = targ_img.get_data()
        self.affine = targ_img.get_affine()
        self.header = targ_img.get_header()
        self.shape = targ_img.get_shape()

    # import mask files.Pay attention if masktype is 'subject',mask is a 4D image
    # if masktype is 'mpm',mask is a 3D image for each subjects
        mask_img = nib.load(self.fmask_img)
        mask_data = mask_img.get_data()

        self.targ_data = targ_data
        self.mask_data = mask_data

class cal_index(object):
    def __init__(self, ds):
    # nsubj is the number of subjects
    # narea is the number of areas
        self.targ_data = ds.targ_data
        self.mask_data = ds.mask_data
        self.areanum = ds.areanum
        self.affine = ds.affine
        self.sessid = ds.sessid
        self.sessn = range(len(ds.sessid))
        self.gender = ds.gender

        self.index = {}

    def volume_index(self, res=[2,2,2]):
        act_volume = []
        if len(self.mask_data.shape) == 4:
            for i in self.sessn:
                act_volume.append(cal_volume(self.mask_data[:,:,:,i], self.areanum, res))
        elif len(self.mask_data.shape) == 3:
            for i in self.sessn:
                act_volume.append(cal_volume(self.mask_data, self.areanum, res))
        else:
            raise user_defined_exception('mask_data need to be 3D or 4D volume!')
        self.index['act_volume'] = act_volume

    def mask_index(self, index, metric):
# for mean and max value of z-values,falff,alff,reho,etc.
        mask_value = []

        if metric == 'mean':
            calfunc = np.nanmean
        elif metric == 'max':
            calfunc = np.max
        elif metric == 'min':
            calfunc = np.min
        elif metric == 'std':
            calfunc = np.std
        elif metric == 'skewness':
            calfunc = stats.skew
        elif metric == 'kurtosis':
            calfunc = stats.kurtosis


        if index == 'psc':
            cal_function = cal_psc
        else:
            cal_function = cal_mask

        if len(self.mask_data.shape) == 4:
            for i in self.sessn:
                m_value = cal_function(self.targ_data[:,:,:,i], self.mask_data[:,:,:,i], self.areanum, calfunc)
                mask_value.append(m_value)
        elif len(self.mask_data.shape) == 3:
            for i in self.sessn:
                m_value = cal_function(self.targ_data[:,:,:,i], self.mask_data, self.areanum, calfunc)
                mask_value.append(m_value)
        else:
            raise user_defined_exception('mask_data need to be 3D or 4D volume!')


        key_index = metric + '_' + index
        self.index[key_index] = mask_value



    def peakcoord_index(self, index):
        peak_coordin = []
        if len(self.mask_data.shape) == 4:
            for i in self.sessn:
                pcor = cal_coordin(self.targ_data[:,:,:,i], self.mask_data[:,:,:,i], self.areanum, self.affine)
                peak_coordin.append(pcor)
        elif len(self.mask_data.shape) == 3:
            for i in self.sessn:
                pcor = cal_coordin(self.targ_data[:,:,:,i], self.mask_data, self.areanum, self.affine)
                peak_coordin.append(pcor)
        else:
            raise user_defined_exception('mask_data need to be 3D or 4D volume!')

        key_index = index + '_' + 'peak_coordinate'
        self.index[key_index] = peak_coordin

class make_atlas(object):
    def __init__(self, ds, sessid, sessn):
        self.mask_data = ds.mask_data
        self.areaname = ds.areaname
        self.areanum = ds.areanum
        self.header = ds.header
        self.sessid = sessid
        self.sessn = sessn
        self.probdata = []
        self.mpmdata = []

    def probatlas(self):
        probdata = np.zeros([91,109,91,len(self.areanum)])
        for arean in self.areanum:
            for i in self.sessn:
                probdata[:,:,:,arean-1][self.mask_data[:,:,:,i] == (arean)] += 1
        probdata = probdata/len(self.sessid)
        self.probdata = probdata

    def MPM(self, thr):
        probdata_new = np.zeros([91,109,91,len(self.areanum)+1])
        self.probdata[self.probdata<thr] = 0
        probdata_new[:,:,:,1:len(self.areanum)+1] = self.probdata
        self.mpmdata = probdata_new.argmax(axis = 3)

class reliability(object):
    def __init__(self, areanum):
        self.maska = []
        self.maskb = []
        self.areanum = areanum
        self.dice = []

    def loadfile(self, maska_imag, maskb_imag):
        maska = nib.load(maska_imag).get_data()
        maskb = nib.load(maskb_imag).get_data()
        self.maska = maska
        self.maskb = maskb

    def cal_dice(self):
    # dice value output like [[brain areas],[],...]
        dice = []
        sessnum = self.maska.shape[3]
        for sessi in range(sessnum):
            temp = []
            for areai in self.areanum:
                dice_value = do_dice(self.maska[:,:,:,sessi], self.maskb[:,:,:,sessi], areai)
                temp.append(dice_value)
            dice.append(temp)
        self.dice = dice





class AtlasInfo(object):
    def __init__(self,task, contrast, threshold, roi_name, subj_id, subj_gender):
        """
        Parameters
        ----------
        task : task name, str
        contrast : contrast name, str
        threshold : threshold to define atlas
        roi_name  :  roi name and id, a dict
        subj_id : subject id, a list
        """

        self.task = task
        self.contrast = contrast
        self.threshold = threshold
        self.roiname = roi_name
        self.subjid = subj_id
        self.gender = subj_gender

    def set_attr(self,name, value):
        """
        Parameters
        ----------
        name : attribute name
        value : attribute value

        -------

        """
        if name == 'task':
            self.task = value
        elif name == 'contrast':
            self.contrast = value
        elif name == 'threshold':
            self.threshold = value
        elif name == 'roi':
            self.roi = value
        elif name == 'subj_id':
            self.subj_id = value


    def get_attr(self, name):

        """

        Parameters
        ----------
        name : attribute name

        Returns
        -------
        value : attribute value
        """

        if name == 'task':
            value = self.task
        elif name == 'contrast':
            value = self.contrast
        elif name == 'threshold':
            value = self.threhold
        elif name == 'roi':
            value = self.roi
        elif name == 'subj_id':
            value = self.subj_id
        else:
            raise user_defined_exception('Wrong attribute name!')
        return value

class AtlasDB(object):
    def __init__(self):
        self.data = {}

    def import_data(self, ds, modal = 'geo', param = 'volume'):
        do_index = cal_index(ds)

        modal_all = ['geo','act','rest','morp','fiber']

        param_all = [['volume', 'peakcoor'], ['zstat', 'psc'], ['alff', 'falff', 'reho'], ['vbm'], ['fa']]

        matric = ['mean', 'max', 'min', 'std', 'skewness', 'kurtosis']


        if modal in modal_all:
            item_index = modal_all.index(modal)
            if not self.data.has_key(modal):
                self.data[modal] = {}
            if param in param_all[item_index]:
                if not self.data[modal].has_key(param):
                    self.data[modal][param] = {}

                if param == 'volume':
                    do_index.volume_index()
                elif param == 'peakcoor':
                    do_index.peakcoord_index(ds.pictype)
                else:
                    for matr in matric:
                        do_index.mask_index(param, matr)

        self.data[modal][param].update(do_index.index)







#-------------------functions-----------------------#

#--------------functions for index------------------#
#-------------------volume--------------------------#
def listinmul(mul_list):
    outnum = reduce(lambda x,y:x*y,mul_list)
    return outnum

def cal_volume(mask_data, areanum, resolu):
    volume = []
    for areai in areanum:
        volume.append(np.sum(mask_data == (areai))*listinmul(resolu))
    return volume
#-------------------z-value--------------------------#
def cal_mask(targ_data, mask_data, areanum, calfunc):
    mask_ind = []
    for areai in areanum:
        if len(targ_data[mask_data == areai])!=0:
            mask_ind.append(calfunc(targ_data[mask_data == areai]))
        else:
            mask_ind.append(np.nan)
    return mask_ind
#-------------------psc-value-------------------------#
def cal_psc(targ_data, mask_data, areanum, calfunc):
    psc_ind = []
    for areai in areanum:
        if len(targ_data[mask_data == areai])!=0:
            psc_ind.append(calfunc(targ_data[mask_data == areai])/100)
        else:
            psc_ind.append(np.nan)
    return psc_ind
#--------------------MNI coordinate------------------#
def vox2MNI(vox, affine):
    vox_new = np.ones([4,1])
    vox_new[0:-1,0] = vox[:]
    MNI = affine.dot(vox_new)
    MNI_new = MNI[0:-1].tolist()    # transform array into list
    return sum(MNI_new,[])  # extend multiple list

def cal_coordin(targ_data, mask_data, areanum, affine):
# The final data format will be like this:[[x,y,z],[x,y,z],etc] for each subject
    co_area = []
    for areai in areanum:
        if len(targ_data[mask_data == areai])!=0:
            temp = np.zeros([91,109,91])
            temp[mask_data == areai] = targ_data[mask_data == areai]
            peakcor_vox = np.unravel_index(temp.argmax(), temp.shape)
            peakv = temp.argmax()
            peakcor_mni = list(vox2MNI(peakcor_vox ,affine))
            co_area.append(peakcor_mni)
            peakcor_mni = []
        else:
            co_area.append(np.nan)
    return co_area

def do_dice(maska, maskb, value):
# value here is aim to filter areas so that we can get dice in every brain areas
    maska_bin = copy.deepcopy(maska)
    maskb_bin = copy.deepcopy(maskb)
    maska_bin[maska_bin!=value] = 0
    maskb_bin[maskb_bin!=value] = 0
    maska_bin[maska_bin==value] = 1
    maskb_bin[maskb_bin==value] = 1
    dice_value = 2*((maska_bin*maskb_bin).sum()/(maska_bin.sum()+maskb_bin.sum()))
    return dice_value


#----------------functions for others-----------------------------------------#
def get_attrname(instan):
# Get all attribute name of an instance
    return instan.__dict__.keys()

def get_attrvalue(instan, attrname):
# Get value of attrname in an instance
    return instan.__getattribute__(attrname)

def finditem(rawlist, keywords):
# Return items containing keywords of a list
    return [val for val in rawlist if keywords in val]

def todict(instan):
# transform all attributes from an instance into dict
    atlas = {}
    attr_list = get_attrname(instan)
    for attr in attr_list:
        atlas[attr] = get_attrvalue(instan, attr)
    return atlas




