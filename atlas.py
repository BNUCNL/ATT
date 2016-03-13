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
        self.pm = None
        self.mpm = None

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

        n_subj = targ.shape[3] # number of subjects
        n_roi = len(self.roi_id) # number of ROI
        meas = np.empty((n_subj, n_roi))
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

        for s in np.arange(n_subj):
            for r in np.arange(n_roi):
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

        n_subj = targ.shape[3] # number of subjects
        n_roi = len(self.roi_id) # number of ROI
        affine = self.atlas_img.get_affine()
        meas = np.empty((n_subj, n_roi, 3))
        meas.fill(np.nan)

        if metric == 'peak':
            for s in np.arange(n_subj):
                ijk = np.ones((n_roi, 4))
                for r in np.arange(n_roi):
                    d = targ[:, :, :, s] * (mask[:, :, :, s] == self.roi_id[r])
                    ijk[r, 0:3] = np.unravel_index(d.argmax(), d.shape)

                # ijk to coordinates
                meas[s, :, :] = np.dot(affine, ijk.T)[0:3, :].T

        elif metric == 'center':
            for s in np.arange(n_subj):
                ijk = np.ones((n_roi, 4))
                for r in np.arange(n_roi):
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
        n_subj = mask.shape[3]
        n_roi = len(self.roi_id)

        vol = np.zeros((n_subj, n_roi))
        # iterate for subject and roi
        for s in np.arange(n_subj):
            for r in np.arange(n_roi):
                vol[s, r] = np.sum(mask[:, :, :, s] == self.roi_id[r])

        res = self.atlas_img.header.get_zooms()
        return vol*np.prod(res)

    def make_pm(self, meth='all'):
        """
        make proabilistic map
        Parameters
        ----------
        meth : 'all' or 'part'. all, all subjects are taken into account; part, only
        part of subjects who have roi are taken into account.

        Returns
        -------
        pm: array for pm
        """
        mask = self.atlas_img.get_data()
        n_roi = len(self.roi_id)
        pm = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], n_roi))
        if meth is 'all':
            for r in np.arange(n_roi):
                pm[:, :, :, r] = np.mean(mask == self.roi_id[r], axis=3)
        elif meth is 'part':
            for r in np.arange(n_roi):
                mask_r = mask == self.roi_id[r]
                subj = np.any(mask_r, axis=(0, 1, 2))
                pm[:, :, :, r] = np.mean(mask_r[:, :, :, subj], axis=3)
        else:
            raise UserDefinedException('meth is not supported!')

        self.pm = pm
        return self.pm

    def make_mpm(self, threshold):
        """

        Parameters
        ----------
        threshold

        Returns
        -------
        mpm: array for mpm

        """
        if self.pm is None:
            raise UserDefinedException('pm is empty! You should make pm first')
        pms = self.pm.shape
        pm = np.zeros((pms[0], pms[1], pms[2], pms[3]+1))
        pm[:, :, :, np.arange(1, pms[3]+1)] = self.pm
        pm[pm < threshold] = 0
        mpm = np.argmax(pm, axis=3)
        self.mpm = mpm

        return mpm
