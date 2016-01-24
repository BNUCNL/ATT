# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
import copy



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


class UserDefinedException(Exception):
    def __init__(self, str):
        Exception.__init__(self)
        self._str = str


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
    def __init__(self, atlas_img, basic, data={}, geo={}, act={}, rest={}, morp={}, fiber={}):
        """

        Parameters
        ----------
        atlas_img : Nifti1image file or Nifti1image object
        basic : basic info for atlas
        geo   : Geo object
        act   : Act object
        rest  : Rest object
        morp  : morp object
        fiber :fiber object

        Returns
        -------

        """

        self.atlas = load_img(atlas_img)
        self.basic = basic
        self.data = data
        self.geo = geo
        self.act = act
        self.rest = rest
        self.morp = morp
        self.fiber = fiber

    def import_data(self, targ_img, roiname=None, modal='geo', meas='volume'):
        if roiname is None:
            roiname = self.basic.roiname

        if modal == 'geo':
            pass
        elif modal == 'act' or modal == 'rest' or modal == 'morp'  or modal == 'fiber':
            self.data[modal][meas] = _extract_scalar_meas(targ_img)


    def export_data(self, sessid=None, roiname=None, modal='geo', meas='volume'):
        if modal == 'geo':
            pass


        elif modal == 'act' or modal == 'rest' or modal == 'morp'  or modal == 'fiber':
            pass

    pass


def _extract_scalar_meas(self, targ_img):
    pass

    return 1


def _extract_geo_meas(self, targ_img):
    pass


