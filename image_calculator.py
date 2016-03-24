import numpy as np
import nibabel as nib
from atlas import UserDefinedException


class ImageCalculator(object):
    def __init__(self, ndim=4, type='temporal'):
        if ndim in (4, 3):
            self.ndim = ndim
        else:
            raise UserDefinedException('image dim should be 4 or 3.')

        if type in ('spatial', 'temporal'):
            self.type = type
        else:
            raise UserDefinedException('type is error: it should be spatial or temporal')

    def add(self, ia, ib, im=None):
        """

        Parameters
        ----------
        ia: image A, a nifiti object
        ib: image B, a nifiti object
        im: image mask, a nifiti object


        Returns
        -------
        ic: new image, a nifiti object

        """
        if ia.shape != ib.shape:
            raise UserDefinedException('image shapes are  not match!')

        hdr = ia.header
        ia = ia.get_data()
        ib = ib.get_data()

        if im is None:
            if self.ndim == 4:
                im = np.logical_and(np.prod(ia, axis=3), np.prod(ib, axis=3))
            else:
                im = np.logical_and(ia, ib)

        if self.ndim == 4:
            ia[im, :] = np.add(ia[im, :], ib[im, :])
            ia[np.logical_not(im), :] = np.nan
        else:
            ia[im] = np.add(ia[im, :], ib[im, :])
            ia[np.logical_not(im)] = np.nan

        return nib.Nifti1Image(ia, None, hdr)

    def subtract(self, ia, ib, im=None):
        if ia.shape != ib.shape:
            raise UserDefinedException('image shapes are not match!')

        hdr = ia.header
        ia = ia.get_data()
        ib = ib.get_data()

        if im is None:
            if self.ndim == 4:
                im = np.logical_and(np.prod(ia, axis=3), np.prod(ib, axis=3))
            else:
                im = np.logical_and(ia, ib)

        if self.ndim == 4:
            ia[im, :] = np.subtract(ia[im, :], ib[im, :])
            ia[np.logical_not(im), :] = np.nan
        else:
            ia[im] = np.subtract(ia[im], ib[im])
            ia[np.logical_not(im)] = np.nan

        return nib.Nifti1Image(ia, None, hdr)

    def multiply(self, ia, ib, im=None):

        if ia.shape != ib.shape:
            raise UserDefinedException('image shapes are not match!')

        hdr = ia.header
        ia = ia.get_data()
        ib = ib.get_data()

        if im is None:
            if self.ndim == 4:
                im = np.logical_and(np.prod(ia, axis=3), np.prod(ib, axis=3))
            else:
                im = np.logical_and(ia, ib)

        if self.ndim == 4:
            ia[im, :] = np.multiply(ia[im, :], ib[im, :])
            ia[np.logical_not(im), :] = np.nan
        else:
            ia[im] = np.multiply(ia[im], ib[im])
            ia[np.logical_not(im)] = np.nan

        return nib.Nifti1Image(ia, None, hdr)

    def divide(self, ia, ib, im=None):

        if ia.shape != ib.shape:
            raise UserDefinedException('image shapes are not match!')

        hdr = ia.header
        ia = ia.get_data()
        ib = ib.get_data()

        if im is None:
            if self.ndim == 4:
                im = np.logical_and(np.prod(ia, axis=3), np.prod(ib, axis=3))
            else:
                im = np.logical_and(ia, ib)

        if self.ndim == 4:
            ia[im, :] = np.divide(ia[im, :], ib[im, :])
            ia[np.logical_not(im), :] = np.nan
        else:
            ia[im] = np.divide(ia[im], ib[im])
            ia[np.logical_not(im)] = np.nan

        return nib.Nifti1Image(ia, None, hdr)

    def pearsonr(self, ia, ib, im=None):
        from scipy.stats import zscore

        if ia.shape != ib.shape:
            raise UserDefinedException('image shapes are not match!')

        hdr = ia.header
        ia = ia.get_data()
        ib = ib.get_data()

        if im is None:
            if self.ndim == 4:
                im = np.logical_and(np.prod(ia, axis=3), np.prod(ib, axis=3))
            else:
                im = np.logical_and(ia, ib)

        if self.ndim == 4:
            ic = ia[:, :, :, 0]
            ic[im] = np.sum(np.multiply(zscore(ia[im, :], axis=1), zscore(ib[im], axis=1)), axis=1)/ia.shape[3]
            ic[np.logical_not(im)] = 0
        else:
            raise UserDefinedException('pearson r only calculate for 4d image!')

        return nib.Nifti1Image(ic, None, hdr)


