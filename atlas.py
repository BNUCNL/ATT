# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
from scipy import stats
from sklearn.cross_validation import StratifiedKFold


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
    elif os.path.isfile(fimg):
        img = nib.load(fimg)
    else:
        raise UserDefinedException('Wrong Image!')

    return img


def split_half_data(data, keys, dl=None):
    """

    Parameters
    ----------
    data: a dict, the data in the dict to be split should be 1d or 2d nd.array,
    and all data should have the same number of rows(samples)
    keys: keys which will be spilt
    dl: dependent labels. The spilt will be even for each labels in each split

    Returns
    -------
    sph_data: a list to store the first and second half data(2x1)

    """
    n_sample = data[keys[0]].shape[0]
    if dl is None:
        dl = np.ones(n_sample)

    fold = 2
    skf = StratifiedKFold(dl, n_folds=fold, shuffle=True)
    index = []
    for train, test in skf:
        index.append(train)
        index.append(test)
        break

    sph_data = []
    for f in np.arange(fold):
        f_data = data.copy()
        for k in keys:
            if f_data[k].ndim == 1:
                f_data[k] = f_data[k][index[f]]
            else:
                f_data[k] = f_data[k][index[f], :]

        sph_data.append(f_data)
    return sph_data

class SubjIdOperator(object):
    def list_subjid(parpath, stem):
    """
        List sessid when data exists
        Parameters:
            parpath: parent path, can list raw sessid in this parent path
            stem: stem directory
        Output:
            sessid (list) 
    """
        sessid = []
        sid_raw = os.listdir(parpath)
        for i in sid_raw:
            if os.path.exists(os.path.join(parpath, i, stem)):
                sessid.append(i)
        return sessid 
       
    def pool_subjid(id_list_1, id_list_2, operator = 'int'):
    """
        Do operation for combining two list of id
        Parameters:
            id_list_1, id_list_2: two id list you want to combined with together
            operator: method. 'int' for intersection. 'union' for union. 
                      'sub' for substrate.
                      Default as intersection
        Output:
            sessid: combined sessid
    """
        if isinstance(id_list_1, str):
            id_list_1 = open(id_list_1, 'rb').read().splitlines()
        if isinstance(id_list_2, str):
            id_list_2 = open(id_list_2, 'rb').read().splitlines()
        if operator == 'int':
            sessid = [sid for sid in id_list_1 if sid in id_list_2]
        elif operator == 'union':
            sessid = list(set(id_list_1 + id_list_2))
        elif operator == 'sub':
            sessid = [sid for sid in id_list_1 if sid not in id_list_2]
        else:
            raise Exception('please note parameters as int, union or sub')
        return sessid

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
        self.vol = None
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
        meas : collected scalar measures,  n_subj x n_roi np.array
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
            targ = np.expand_dims(targ, axis=1)

        if mask.ndim == 3:
            mask = np.tile(mask[..., np.newaxis], (1, targ.shape[3]))

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

        # assign meas 0 as nan as no measure are zeros, besides out of mask
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
        meas:  collected geometry measures, n_subj x n_roi x 3, np.array
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
            targ = np.expand_dims(targ, axis=1)

        if mask.ndim == 3:
            mask = np.tile(mask[..., np.newaxis], (1, targ.shape[3]))

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
                    if np.any(d):
                        ijk[r, 0:3] = np.unravel_index(d.argmax(), d.shape)
                    else:
                        ijk[r, 0:3] = np.nan
                # ijk to coordinates
                meas[s, :, :] = np.dot(affine, ijk.T)[0:3, :].T

        elif metric == 'center':
            for s in np.arange(n_subj):
                ijk = np.ones((n_roi, 4))
                for r in np.arange(n_roi):
                    d = targ[:, :, :, s] * (mask[:, :, :, s] == self.roi_id[r])
                    if np.any(d):
                        ijk[r, 0:3] = np.mean(np.transpose(np.nonzero(d)))
                    else:
                        ijk[r, 0:3] = np.nan
                # ijk to coordinates
                meas[s, :, :] = np.dot(affine, ijk.T)[0:3, :].T

        return np.reshape(meas, (n_subj, n_roi*3))

    def volume(self):
        """

        Parameters
        ----------
        self

        Returns
        -------
        vol: volume of the rois

        """
        mask = self.atlas_img.get_data()
        # extend 3d mask to 4d
        if mask.ndim == 3:
            mask = np.expand_dims(mask, axis=1)

        # number of subjects
        n_subj = mask.shape[3]
        n_roi = len(self.roi_id)
        vol = np.zeros((n_subj, n_roi))

        # iterate for subject and roi
        for s in np.arange(n_subj):
            for r in np.arange(n_roi):
                vol[s, r] = np.sum(mask[:, :, :, s] == self.roi_id[r])

        res = self.atlas_img.header.get_zooms()
        vol = vol*np.prod(res)
        self.vol = vol

        return vol

    def make_pm(self, meth='all'):
        """
        make probabilistic map(pm)from 4D atlas image
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
        make maximum probabilistic map(mpm) from 4D probabilistic maps
        Parameters
        ----------
        threshold : threshold to mask probabilistic maps

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
