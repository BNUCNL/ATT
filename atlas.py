# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
from scipy import stats


class UserDefinedException(Exception):
    def __init__(self, str):
        Exception.__init__(self)
        self._str = str


def load_img(target_img):
    """
    Load nifti image
    Parameters
    ----------
    target_img : target image to load, a str(file path) or a Nifti1Image object.
    Returns
    -------
    img : a Nifti1Image object
    """
    if isinstance(target_img, nib.Nifti1Image):
        img = target_img

    # load nifti image with nibabel
    elif os.path.isfile(target_img):
        img = nib.load(target_img)
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
        self.volume = self.volume_meas()

    def collect_scalar_meas(self, meas_img, metric='mean'):

        """
        Collect scalar measures for atlas

        Parameters
        ----------
        meas_img: measures image, str(nii file path) or a nii object
        metric: metric to summarize  ROI info, str

        Returns
        -------
        meas : collected scalar measures,  nSubj x nRoi np.array

        """

        scalar_metric = ['mean', 'max', 'min', 'std', 'median', 'skewness', 'kurtosis']
        if metric not in scalar_metric:
            raise UserDefinedException('Metric is not supported!')

        targ = load_img(meas_img).get_data()
        mask = self.atlas_img.get_data()

        if mask.shape != targ.shape and mask.shape != targ.shape[:3]:
            raise UserDefinedException('Atlas image and target image are not match!')

        # reshape 3d volume to 4d
        if targ.ndim == 3:
            targ = np.tile(targ, (1, 1))

        if mask.ndim == 3:
            mask = np.tile(mask, (1, targ.shape[3]))

        nSubj = targ.shape[3] # number of subjects
        nRoi = len(self.roi_id) # number of ROI
        meas = np.empty((nSubj, nRoi))
        meas.fill(np.nan)

        if metric == 'sum':
            meter = np.nansum
        elif metric == 'mean':
            meter = np.nanmean
        elif metric == 'max':
            meter = np.max
        elif metric == 'min':
            meter = np.min
        elif metric == 'std':
            meter = np.nanstd
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
                meas[s, r] = meter(d[m])
        meas[meas == 0] = np.nan
        return meas

    def collect_geometry_meas(self, meas_img, metric='mean'):
        """
        Collect geometry measures for atlas

        Parameters
        ----------
        meas_img: target measure image, str(a nii file path) or a nii object
        metric: metric to summarize ROI info

        Returns
        -------
        meas:  collected geometry measures, nSubj x nRoi x 3, np.array

        """

        geometry_metric = ['center', 'peak']
        if metric not in geometry_metric:
            raise UserDefinedException('Metric is not supported!')

        targ = load_img(meas_img).get_data()
        mask = self.atlas_img.get_data()

        if mask.shape != targ.shape and mask.shape != targ.shape[:3]:
            raise UserDefinedException('Atlas image and target image are not match!')

        # reshape 3d volume to 4d
        if targ.ndim == 3:
            targ = np.tile(targ, (1, 1))

        if mask.ndim == 3:
            mask = np.tile(mask, (1, targ.shape[3]))

        nSubj = targ.shape[3] # number of subjects
        nRoi = len(self.roi_id) # number of ROI
        affine = self.atlas_img.get_affine()
        meas = np.empty((nSubj, nRoi, 3))
        meas.fill(np.nan)

        if metric == 'peak':
            for s in np.arange(nSubj):
                ijk = np.ones((nRoi, 4))
                for r in np.arange(nRoi):
                    d = targ[:, :, :, s] * (mask[:, :, :, s] == self.roi_id[r])
                    ijk[r, 0:3] = np.unravel_index(d.argmax(), d.shape)

                # ijk to coordinates
                meas[s, :, :] = np.dot(affine, ijk.T)[0:3, :].T

        elif metric == 'center':
            for s in np.arange(nSubj):
                ijk = np.ones((nRoi, 4))
                for r in np.arange(nRoi):
                    d = targ[:, :, :, s] * (mask[:, :, :, s] == self.roi_id[r])
                    ijk[r, 0:3] = np.mean(np.transpose(np.nonzero(d)))

                # ijk to coordinates
                meas[s, :, :] = np.dot(affine, ijk.T)[0:3, :].T

        return meas

    def volume_meas(self):
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

