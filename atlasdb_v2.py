# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:
from __future__ import division
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






class Atlas(object):
    def __init__(self, atlas_img, areanum, areaname, gender, sessid, taskname, contrast, threshold):
        self.atlas_img = loadimg(atlas_img)
        self.areanum = areanum
        self.areaname = areaname
        self.gender = gender
        self.sessid = sessid
        self.taskname = taskname
        self.contrast = contrast

        self.data= {}

        self.data['basic'] = {}
        self.data['basic']['contrast'] = contrast
        self.data['basic']['threshold'] = threshold
        self.data['basic']['roi'] = dict(zip(areaname, areanum))
        self.data['basic']['subjid'] = sessid
        self.data['basic']['gender'] = gender

        self.data['geo'] = {}
        self.data['geo']['volume'] = volume_index(self.atlas_img, range(len(sessid)), areanum, [2,2,2])



#--------------------cal_index-----------------------------------------------#
#-------------calculate activation index--------------------#
    def mask_index(self, targ_img, index, metric):
# for mean and max value of z-values,falff,alff,reho,etc.
        mask_value = []
        targ_img = loadimg(targ_img)
        targ_data = targ_img.get_data()
        atlas_data = self.atlas_img.get_data()
        temp_data = {}

        if metric == 'mean':
            calfunc = np.mean
        elif metric == 'max':
            calfunc = np.max
        elif metric == 'min':
            calfunc = np.min
        elif metric == 'std':
            calfunc = np.std
        elif metric == 'median':
            calfunc = np.median
        elif metric == 'cv':
            calfunc = calcv
        elif metric == 'skewness':
            calfunc = stats.skew
        elif metric == 'kurtosis':
            calfunc = stats.kurtosis

        if index == 'psc':
            cal_function = cal_psc
        else:
            cal_function = cal_mask

        if len(self.atlas_img.shape) == 4:
            for i in range(len(self.sessid)):
                m_value = cal_function(targ_data[:,:,:,i], atlas_data[:,:,:,i], self.areanum, calfunc)
                mask_value.append(m_value)
        elif len(self.atlas_img.shape) == 3:
            for i in range(len(self.sessid)):
                m_value = cal_function(targ_data[:,:,:,i], atlas_data, self.areanum, calfunc)
                mask_value.append(m_value)
        else:
            raise user_defined_exception('mask_data need to be 3D or 4D volume!')

        mask_value = np.array(mask_value)
        key_index = metric + '_' + index
        temp_data[key_index] = mask_value
        if (index == 'zstat') | (index == 'psc'):
            if not self.data.has_key('act'):
                self.data['act'] = {}
            if self.data['act'].has_key(index):
                self.data['act'][index].update(temp_data)
            else:
                self.data['act'][index] = {}
                self.data['act'][index].update(temp_data)
        elif (index == 'alff') | (index == 'falff') | (index == 'reho'):
            if not self.data.has_key('rest'):
                self.data['rest'] = {}
            if self.data['rest'].has_key(index):
                self.data['rest'][index].update(temp_data)
            else:
                self.data['rest'][index] = {}
                self.data['rest'][index].update(temp_data)
        else:
            raise user_defined_exception('Assignment of index is wrong!')

#------------------------------------------------------------------#


#-----------------calculate peak coordinate------------------------#
    def peakcoord_index(self, targ_img, index):
        peak_coordin = []
        temp_data = {}
        targ_img = loadimg(targ_img)
        targ_data = targ_img.get_data()
        atlas_data = self.atlas_img.get_data()
        affine = self.atlas_img.get_affine()

        if len(self.atlas_img.shape) == 4:
            for i in range(len(self.sessid)):
                pcor = cal_coordin(targ_data[:,:,:,i], atlas_data[:,:,:,i], self.areanum, affine)
                peak_coordin.append(pcor)
        elif len(self.mask_data.shape) == 3:
            for i in range(len(self.sessid)):
                pcor = cal_coordin(targ_data[:,:,:,i], self.mask_data, self.areanum, self.affine)
                peak_coordin.append(pcor)
        else:
            raise user_defined_exception('mask_data need to be 3D or 4D volume!')

        peak_coordin = np.array(peak_coordin)
        key_index = index + '_' + 'peakcoor'
        temp_data[key_index] = peak_coordin
        if self.data['geo'].has_key('peakcoor'):
            self.data['geo']['peakcoor'].update(temp_data)
        else:
            self.data['geo']['peakcoor'] = {}
            self.data['geo']['peakcoor'].update(temp_data)
#--------------------------------------------------------------------#
#------------------------------------------------------------------------------------#



class AtlasDB(object):
    def __init__(self, data):
        self.data = data

    def save_to_pkl(self, path, filename):
        """

        Parameters
        ----------
        save data in os.path.join(path, filename) with type of .pkl
        path    parent path
        filename    filename .pkl

        Returns
        A .pkl file,which can be loaded by cPickle
        -------

        """
        if hasattr(self, 'data'):
            with open(os.path.join(path, filename), 'wb') as output:
                cPickle.dump(self.data, output, -1)

    def save_to_mat(self, path, filename):
        """

        Parameters
        ----------
        save data in os.path.join(path, filename) with type of .mat
        path
        filename

        Returns
        -------
        A .mat file,which can be loaded by matlab
        """
        if hasattr(self, 'data'):
            si.savemat(os.path.join(path, filename), mdict = self.data)






















#-------------------functions-----------------------#

#--------------functions for index------------------#
#-------------------volume--------------------------#

def calcv(data):
    return np.std(data)/np.mean(data)

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
#-----------------------------------------------------------------------------#



#----------------functions for atlas calculation-----------------------------#
#------------volume index-----------------------#
def volume_index(atlas, sessn, areanum, res):
    act_volume = []
    atlas_data = atlas.get_data()
    if len(atlas.shape) == 4:
        for i in sessn:
            act_volume.append(cal_volume(atlas_data[:,:,:,i], areanum, res))
    elif len(atlas.shape) == 3:
        for i in sessn:
            act_volume.append(cal_volume(atlas_data, areanum, res))
    else:
        raise   user_defined_exception('mask_data need to be 3D or 4D volume!')

    act_volume = np.array(act_volume)
    return act_volume

#-----------------------------------------------------------------------------#




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

def getdata_areas(data, roi, roi_name):
    """

    Parameters
    ----------
    data  data you want to extract
    roi   areaname,such as 'lMT'
    roi_name   dict of areanames,such as {'rMT':1,'lMT':2},etc.

    Returns
    -------
    spec_data  data with specfic area
    """
    spec_data = data[:,roi_name[roi]-1]
    return spec_data

def loadimg(atlas):
    if type(atlas) is str:
        img = nib.load(atlas)
    else:
        img = atlas
    return img









