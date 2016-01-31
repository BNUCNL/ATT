# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
import copy
from scipy import stats




class UserDefinedException(Exception):
    def __init__(self, str):
        Exception.__init__(self)
        self._str = str


def load_img(fimg):
    """
    Load Nifti1Image
    Parameters
    ----------
    fimg : a file or a Nifti1Image object.
    Returns
    -------
    img : a Nifti1Image object
    """
    if isinstance(fimg, nib.Nifti1Image):
        img = fimg

    # load nifti image with nibabel
    elif os.path.isfile(fimg):
        img = nib.load(fimg)
    else:
        raise UserDefinedException('Wrong Image!')

    return img


def extract_param(targ_img, atlas_img, roi_id, metric):
    """
    Parameters
    ----------
    targ_img    : a file or a Nifti1Image object for target image
    atlas_img   : a file or a Nifti1Image object for atlas image
    roi_id  : a scalar to indicate roi id
    metric  :  metric used to summarize ROI
    Returns
    -------
    param   : 2-d np.array
    """
    targ = load_img(targ_img).get_data()
    atlas = load_img(atlas_img).get_data()

    if  atlas.shape != targ.shape or atlas.shape != targ.shape[:3]:
        raise UserDefinedException('Atlas image and target image are not match!')

    if targ.ndim == 3:
        targ = np.tile(targ, (1, 1))

    if atlas.ndim == 3:
        atlas = np.tile(atlas, (1, targ.shape[3]))

    NS = targ.shape[3] # number of subjects
    if metric == 'center' or metric == 'peak':
        param = np.empty((NS,3))
    else:
        param = np.empty(NS)
    param.fill(np.nan)

    if metric == 'peak':
        for s in range(NS):
            d = targ[:, :, :, s] * (atlas[:, :, :, s] == roi_id)
            param[s, :] = np.unravel_index(d.argmax(), d.shape)

    elif metric == 'center':
        for s in range(NS):
            d = targ[:,:,:,s] * (atlas[:,:,:,s] == roi_id)
            param[s, :]  = np.mean(np.transpose(np.nonzero(d)))

    else: # scalar metric
        if metric == 'sum' or metric == 'volume':
            cpu = np.nansum
        elif metric == 'max':
            cpu = np.max
        elif metric == 'min':
            cpu = np.min
        elif metric == 'std':
            cpu = np.nanstd
        elif metric == 'skewness':
            cpu = stats.skew
        elif metric == 'kurtosis':
            cpu = stats.kurtosis
        else:
            cpu = []

        for s in range(NS):
            d = targ[:,:,:,s]
            r =  atlas[:,:,:,s] == roi_id
            param[s] = cpu(d[r], axis=1)

    return param


class AtlasInfo(object):
    def __init__(self,task, contrast, threshold, roi, subj_id, subj_gender):
        """
        Parameters
        ----------
        task : task name, str
        contrast : contrast name, str
        threshold : threshold to define atlas
        roi  :  roi name and id, a dict
        subj_id : subject id, a list
        """
        self.task = task
        self.contrast = contrast
        self.threshold = threshold
        self.roi = roi
        self.subj_id = subj_id

    def set_attr(self, name, value):
        """
        Parameters
        ----------
        name : attribute name
        value : attribute value
        """
        if name == 'task':
            self.task = value
        elif name == 'contrast':
            self.contrast = value
        elif name  == 'threshold':
            self.threshold = 'threshold'
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
        elif name  == 'threshold':
           value = self.threshold
        elif name == 'roi':
            value = self.roi
        elif name == 'subj_id':
            value = self.subj_id
        else:
             raise UserDefinedException('Wrong attribute name!')

        return value

class AtlasDB(object):
    def __init__(self, atlas_img, basic, metric={}):
        """
        Parameters
        ----------
        atlas_img : Nifti1image file or Nifti1image object
        basic : basic info for atlas, AtlasInfo object
        data    : data for each roi, dict
        metric  : metric for each params, dict
        """

        self.atlas = load_img(atlas_img)
        self.basic = basic
        self.metric = metric
        self.data = {}

    def import_data(self, targ_img, roi=None, modal='geo', param='volume'):
        """
        Parameters
        ----------
        targ_img :  a file or a Nifti1Image object for target image
        roi : roi info, a (name, id) dict
        modal : modality, str
        param : parameter name, str
        """
        if roi is None:
            roi = self.basic.roi

        metric  = self.metric[param]
        if modal == 'geo':
            meas = np.empty((len(self.basic.subjid), len(metric)), 3)
        else:
             meas = np.empty((len(self.basic.subjid), len(metric)))
        meas.fill(np.nan)

        for key in roi:
            for m in range(metric.shape[0]):
               meas[:,m] =  extract_param(targ_img, self.atlas, roi[key], metric[m])

            self.data[key][modal][param] = meas

    def export_data(self, sessid=None, roi=None, modal='geo', param='volume'):
        """
        Parameters
        ----------
        sessid : sessid, list
        roi : roi info a (name, id) dict
        modal : modality
        param :  parameter name, str
        Returns
        -------
        meas : a list
        """

        meas = []
        for key in roi:
            meas.append(self.data[key][modal][param])

        return meas

    def set_attr(self,name, value):
        pass

    def get_attr(self, name, value):
        pass
