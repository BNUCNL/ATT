# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
from scipy import stats
import cPickle
import scipy.io as si


class UserDefinedException(Exception):
    def __init__(self, str):
        Exception.__init__(self)
        self._str = str


def load_img(fimg):
    """
    Load nifti image
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


class Atlas(object):
    def __init__(self, atlas_img, roi_id, roi_name, task, contrast, threshold, subj_id, subj_gender):
        self.atlas_img = load_img(atlas_img)
        self.roi_name = roi_name
        self.roi_id = roi_id
        self.task = task
        self.contrast = contrast
        self.threshold = threshold
        self.subj_id = subj_id
        self.subj_gender = subj_gender
        self.volume = self.__volume_meas()
    
    def collect_meas(self, targ_img, index = None, metric = 'mean', isnorm = False):
        """
        Collect measures for atlas

        Parameters
        ----------
        targ_img: target image
        metric: metric to summarize  ROI info

        Returns
        -------
        param:  nSubj x nRoi array

        """

        candidate_metric = ['mean', 'max', 'min', 'std', 'median', 'skewness', 'kurtosis', 'center', 'peak']
        if metric not in candidate_metric:
            raise UserDefinedException('Metric is not supported!')

        targ = load_img(targ_img).get_data()
        if index == 'psc':
            targ = targ/100
        if isnorm:
            targ = self.__norm_targdata(targ, 1e-5)
        mask = self.atlas_img.get_data()

        if mask.shape != targ.shape and mask.shape != targ.shape[:3]:
            raise UserDefinedException('Atlas image and target image are not match!')

        # reshape 3d volume to 4d
        if targ.ndim == 3:
            targ = np.tile(targ, (1, 1))

        if mask.ndim == 3:
            mask = np.tile(mask, (1, targ.shape[3]))

        nSubj = targ.shape[3] # number of subjects
        nRoi = len(self.roi_id)
        affine = self.atlas_img.get_affine()

        if metric == 'center' or metric == 'peak':
            param = np.empty((nSubj, nRoi, 3))
        else:
            param = np.empty((nSubj, nRoi))

        if metric == 'peak':
            for s in np.arange(nSubj):
                ijk = np.ones((nRoi, 4))
                for r in np.arange(nRoi):
                    d = targ[:, :, :, s] * (mask[:, :, :, s] == self.roi_id[r])
                    ijk[r, 0:3] = np.unravel_index(d.argmax(), d.shape)
                # ijk to coords
                mni = np.dot(affine, ijk.T)[0:3, :].T
                for r in np.arange(nRoi):
                    if ([90, -126, -72] == mni[r, :]).all():
                        mni[r, :] = np.nan
                param[s, :, :] = mni

        elif metric == 'center':
            for s in np.arange(nSubj):
                ijk = np.ones((nRoi, 4))
                for r in np.arange(nRoi):
                    d = targ[:, :, :, s] * (mask[:, :, :, s] == self.roi_id[r])
                    ijk[r, 0:3] = np.mean(np.transpose(np.nonzero(d)))
                # ijk to coords
                param[s, :, :]  = np.dot(affine, ijk.T)[0:3, :].T

        else: # scalar metric
            if metric == 'sum':
                meter = np.nansum
            elif metric == 'mean':
                meter = np.nanmean
            elif metric == 'max':
                meter = np.nanmax
            elif metric == 'min':
                meter = np.nanmin
            elif metric == 'std':
                meter = np.nanstd
            elif metric == 'median':
                meter = np.median
            elif metric == 'skewness':
                meter = stats.skew
            elif metric == 'kurtosis':
                meter = stats.kurtosis
            else:
                meter = []

            for s in np.arange(nSubj):
                for r in np.arange(nRoi):
                    d = targ[:, :, :, s]
                    m = mask[:, :, :, s] == self.roi_id[r]
                    try:
                        param[s, r] = meter(d[m])
                    except ValueError:
                        param[s, r] = np.nan
        return param

    def __volume_meas(self):
        mask = self.atlas_img.get_data()

        # extend 3d mask to 4d
        if mask.ndim == 3:
            mask = np.tile(mask, (1, 1))

        # number of subjects
        nSubj = mask.shape[3]
        nRoi = len(self.roi_id)

        vol = np.zeros((nSubj,nRoi))
        # iterate for subject and roi
        for s in np.arange(nSubj):
                for r in np.arange(nRoi):
                    vol[s, r] = np.sum(mask[:, :, :, s] == self.roi_id[r])

        res = self.atlas_img.header.get_zooms()
        return vol*np.prod(res)

    def __norm_targdata(self, data, thresh):
        normdata = np.zeros(data.shape)
        gnorm = lambda x:(x - x.mean())/x.std()
        subjnum = data.shape[3]
        for i in range(subjnum):
            normdata[:,:,:,i][abs(data[:,:,:,i]) - thresh>0] = gnorm(data[:,:,:,i][abs(data[:,:,:,i]) - thresh>0])
        return normdata

def save_to_pkl(data, path, filename):
    """
    save data with .pkl
    """
    with open(os.path.join(path, filename), 'wb') as output:
        cPickle.dump(data, output, -1)    

def save_to_mat(data, path, filename):
    """
    save data with .mat
    """    
    si.savemat(os.path.join(path, filename), mdict = data)



