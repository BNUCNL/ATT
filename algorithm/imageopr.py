# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import numpy as np


class FeatureScale(object):
    """
    A class for feature scaling
    -------------------------------
    Parameters:
        data: raw array data
    """
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise Exception("Data should be np.ndarray data")
        self._data = data
        self._shape = data.shape
    def rescaling(self):
        """
        Rescaling data
        x' = (x-min(x))/(max(x)-min(x))
        --------------------------
        Example:
            >>> featCls = tools.FeatureScale(data)
            >>> outdata = featCls.rescaling()
        """
        flattendata = self._data.flatten(order='C').tolist()
        mindata = np.min(self._data[self._data!=0])
        maxdata = np.max(self._data[self._data!=0])
        delta = float(maxdata - mindata)
        flattenoutput = [(x-mindata)/delta if x!=0 else 0 for x in flattendata]
        return np.reshape(np.array(flattenoutput), self._shape, order='C')

    def standardization(self):
        """
        Feature standardization
        x' = (x-mean(x))/std(x)
        -------------------------
        Example:
            >>> featCls = tools.FeatureScale(data)
            >>> outdata = featCls.standardization()
        """
        flattendata = self._data.flatten(order='C').tolist()
        meandata = np.mean(self._data[self._data!=0])
        stddata = float(np.std(self._data[self._data!=0]))
        flattenoutput = [(x-meandata)/stddata if x!=0 else 0 for x in flattendata]
        return np.reshape(np.array(flattenoutput), self._shape, order='C')

    def scale_unit_length(self, para = 'L1'):
        """
        Scaling to unit length
        x' = x/||x||
        -----------------------------
        Parameters:
            para: 'L1', L1 norm. ||x|| = sum(abs(x))
                  'L2', L2 norm. ||x|| = sum(x**2)
        Example:
            >>> featCls = tools.FeatureScale(data)
            >>> outdata = featCls.scale_unit_length(para='L1')
        """
        flattendata = self._data.flatten(order='C').tolist()
	if para == 'L1':
            normdata = sum(map(lambda a: abs(a), flattendata))
        elif para == 'L2':
            import math
            normdata = sum(map(lambda a: a**2, flattendata))
            normdata = math.sqrt(normdata)
        flattenoutput = [x/normdata if x!=0 else 0 for x in flattendata]
        return np.reshape(np.array(flattenoutput), self._shape, order='C')

