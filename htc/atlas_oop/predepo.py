# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
import copy

class user_defined_exception(Exception):
    def __init__(self, str):
        Exception.__init__(self)
        self._str = str



class Dataset(object):
    def __init__(self, targ_img_file, mask_img_file, areaname, areanum, gender, sessid, taskname, contrast):
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
    def __init__(self, ds, sessid, sessn, gender):
    # nsubj is the number of subjects
    # narea is the number of areas
        self.targ_data = ds.targ_data
        self.mask_data = ds.mask_data
        self.areanum = ds.areanum
        self.affine = ds.affine
        self.sessid = sessid
        self.sessn = sessn
        self.gender = gender

        self.act_volume = []
        self.mean_zstat = []
        self.peak_zstat = []
        self.std_zstat = []
        self.mean_psc = []
        self.peak_psc = []
        self.std_psc = []
        self.peak_coordin = []
        self.mean_alff = []
        self.peak_alff = []
        self.std_alff = []
        self.mean_falff = []
        self.peak_falff = []
        self.std_falff = []
        self.mean_reho = []
        self.peak_reho = []
        self.std_reho = []
        
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
        self.act_volume = act_volume
                               
    def mask_index(self, index):
# for mean and max value of z-values,falff,alff,reho,etc.
        mean_value = []
        peak_value = []
        std_value = []
        if len(self.mask_data.shape) == 4:
            for i in self.sessn:
                [mvalue,pvalue,stdvalue] = cal_mask(self.targ_data[:,:,:,i], self.mask_data[:,:,:,i], self.areanum)
                mean_value.append(mvalue)
                peak_value.append(pvalue)
                std_value.append(stdvalue)
        elif len(self.mask_data.shape) == 3:
            for i in self.sessn:
                [mvalue,pvalue,stdvalue] = cal_mask(self.targ_data[:,:,:,i], self.mask_data, self.areanum)
                mean_value.append(mvalue)
                peak_value.append(pvalue)
                std_value.append(stdvalue)
        else:
            raise user_defined_exception('mask_data need to be 3D or 4D volume!')

        if index == 'zstat':
            self.mean_zstat = mean_value
            self.peak_zstat = peak_value
            self.std_zstat = std_value
        elif index == 'alff':
            self.mean_alff = mean_value
            self.peak_alff = peak_value
            self.std_alff = std_value
        elif index == 'falff':
            self.mean_falff = mean_value
            self.peak_falff = peak_value
            self.std_falff = std_value
        elif index == 'reho':
            self.mean_reho = mean_value
            self.peak_reho = peak_value
            self.std_reho = std_value
        else:
            raise user_defined_exception("please input index as 'zstat' or 'alff' or 'falff' or 'reho'!")
        
    def psc_index(self):
        mean_psc = []
        peak_psc = []
        std_psc = []
        if len(self.mask_data.shape) == 4:
            for i in self.sessn:
                [mpsc,ppsc,stdpsc] = cal_psc(self.targ_data[:,:,:,i], self.mask_data[:,:,:,i], self.areanum)
                mean_psc.append(mpsc)
                peak_psc.append(ppsc)
                std_psc.append(stdpsc)
        elif len(self.mask_data.shape) == 3:
            for i in self.sessn:
                [mpsc,ppsc,stdpsc] = cal_psc(self.targ_data[:,:,:,i], self.mask_data, self.areanum)
                mean_psc.append(mpsc)
                peak_psc.append(ppsc)
                std_psc.append(stdpsc)
        else:
            raise user_defined_exception('mask_data need to be 3D or 4D volume!')
        self.mean_psc = mean_psc
        self.peak_psc = peak_psc
        self.std_psc = std_psc

    def peakcoord_index(self):
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
        self.peak_coordin = peak_coordin

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
         
class save_data(object):
    def __init__(self): 
        pass
    
    def combinattr(self, instan, attri, new_attri):
    # combined attributes from other instances into the instance of 'save_data'
        if hasattr(instan, attri): 
            if len(get_attrvalue(instan, attri))!=0:
                setattr(self, new_attri, get_attrvalue(instan, attri))
            else:
                print '%s is empty!' % attri 


    

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
def cal_mask(targ_data, mask_data, areanum):
    mzstat = []
    pzstat = []
    stdzstat = []
    for areai in areanum:
        if len(targ_data[mask_data == areai])!=0:
            mzstat.append(np.nanmean(targ_data[mask_data == areai]))
            pzstat.append(np.nanmax(targ_data[mask_data == areai]))
            stdzstat.append(np.std(targ_data[mask_data == areai]))
        else: 
            mzstat.append(0)
            pzstat.append(0)
            stdzstat.append(0)
    return mzstat,pzstat,stdzstat
#-------------------psc-value-------------------------#
def cal_psc(targ_data, mask_data, areanum):
    mpsc = []
    ppsc = []
    stdpsc = []
    for areai in areanum:
        if len(targ_data[mask_data == areai])!=0:
            mpsc.append(np.nanmean(targ_data[mask_data == areai])/100)
            ppsc.append(np.nanmax(targ_data[mask_data == areai])/100)
            stdpsc.append(np.std(targ_data[mask_data == areai])/100)
        else:
            mpsc.append(0)
            ppsc.append(0)
            stdpsc.append(0)
    return mpsc,ppsc,stdpsc
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
            co_area.append([])
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

