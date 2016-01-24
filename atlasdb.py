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
    fimg : a file or a Nifti1Image object

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
    targ = load_img(targ_img).get_data()
    atlas = load_img(atlas_img).get_data()

    if  atlas.shape != targ.shape or atlas.shape != targ.shape[:3]:
        raise UserDefinedException('Atlas image and target image are not match!')

    if targ.ndim == 3:
        targ = np.tile(targ, (1, 1))

    if atlas.ndim == 3:
        atlas = np.tile(atlas, (1, targ.shape[3]))

    NS = targ.shape[3]
    if metric == 'center' or metric == 'peak':
        param = np.empty((NS,3))
    else:
        param = np.empty(NS)
    param.fill(np.nan)

    if metric == 'peak':
        for s in range(NS):
           d = targ[:,:,:,s] * atlas[:,:,:,s] == roi_id
           param[s, :] = np.unravel_index(d.argmax(), d.shape)

    elif metric == 'center':
        for s in range(NS):
            d = targ[:,:,:,s] * atlas[:,:,:,s] == roi_id
            coords = np.transpose(np.nonzero(d))
            param[s, :]  = np.mean(coords)
    else:
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
    def __init__(self,task, contrast, threshold, roi_name, subj_id, subj_gender):
        self.task = task
        self.contrast = contrast
        self.threshold = threshold
        self.roiname = roi_name
        self.subjid = subj_id
        self.gender = subj_gender

    def set(self,attr_name, attr_value):
        pass

    def get(self, attr_name, attr_value):
        pass


class AtlasDB(object):
    def __init__(self, atlas_img, basic, metric={}):
        """

        Parameters
        ----------
        atlas_img : Nifti1image file or Nifti1image object
        basic : basic info for atlas
        data    : data for each roi


        Returns
        -------

        """

        self.atlas = load_img(atlas_img)
        self.basic = basic
        self.metric = metric
        self.data = {}

    def import_data(self, targ_img, roiname=None, modal='geo', param='volume'):
        if roiname is None:
            roiname = self.basic.roiname

        metric  = self.metric[param]
        if modal == 'geo':
            meas = np.empty((len(self.basic.subjid), len(metric)), 3)
        else:
             meas = np.empty((len(self.basic.subjid), len(metric)))
        meas.fill(np.nan)

        for key in roiname:
            for m in range(metric.shape[0]):
               meas[:,m] =  extract_param(targ_img, self.atlas, roiname[key], metric[m])

            self.data[key][modal][param] = meas


    def export_data(self, sessid=None, roiname=None, modal='geo', param='volume'):

        return self.data[roiname][modal][param]


    def set(self,attr_name, attr_value):
        pass

    def get(self, attr_name, attr_value):
        pass

