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
    def rescaling(self, step = (0,1)):
        """
        Rescaling data
        x' = (x-min(x))/(max(x)-min(x))
        --------------------------
        Example:
            >>> featCls = tools.FeatureScale(data)
            >>> outdata = featCls.rescaling()
        """
        if len(step) != 2:
            raise Exception('step should be an 2 factor list')
        if step[1]<step[0]:
            step = list(step)
            step[1], step[0] = step[0], step[1]
        flattendata = self._data.flatten(order='C').tolist()
        mindata = np.min(self._data[self._data!=0])
        maxdata = np.max(self._data[self._data!=0])
        delta = float(maxdata - mindata)
        flattenoutput = [step[0]+(step[1]-step[0])*((x-mindata)/delta) if x!=0 else 0 for x in flattendata]
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

def calgradient3D(A, loc, oprx, opry, oprz):
    """
    Calculate gradient in location loc
    ----------------------------------------------------
    Parameters:
        A: raw image data
        loc: a 3D coordinate
        oprx: gradient operator, in x axis, a 3D matrix
        opry: gradient operator, in y axis, a 3D matrix
        oprz: gradient operator, in z axis, a 3D matrix
    Output:
        g: gradient magnitude values.
        vectorg: gradient vector. g = sum(abs(vectorg[i]))
    Example:
        >>> g, vectorg = calgradient3D(imgdata, [31,22,45], oprx, opry, oprz)
    """
    if isinstance(loc, list):
        loc = np.array(loc)
    neighbor = ((-1,-1,-1),\
                (-1,-1,0),\
                (-1,-1,1),\
                (-1,0,-1),\
                (-1,0,0),\
                (-1,0,1),\
                (-1,1,-1),\
                (-1,1,0),\
                (-1,1,1),\
                (0,-1,-1),\
                (0,-1,0),\
                (0,-1,1),\
                (0,0,-1),\
                (0,0,0),\
                (0,0,1),\
                (0,1,-1),\
                (0,1,0),\
                (0,1,1),\
                (1,-1,-1),\
                (1,-1,0),\
                (1,-1,1),\
                (1,0,-1),\
                (1,0,0),\
                (1,0,1),\
                (1,1,-1),\
                (1,1,0),\
                (1,1,1))
    cubeloc = [loc+np.array(i) for i in neighbor] 
    signal = [A[tuple(i)] for i in cubeloc]
    signal = np.reshape(signal, [3,3,3])
    gx = np.sum((oprx*signal).flatten())
    gy = np.sum((opry*signal).flatten())
    gz = np.sum((oprz*signal).flatten())
    g = np.abs(gx)+np.abs(gy)+np.abs(gz)
    vectorg = (gx, gy, gz)
    return g, vectorg
  
class GradientImg(object):
    """
    A class for computing gradient of a image.
    ---------------------------------------
    Parameters:
        method: 'sobel' or 'prewitt'
    Example:
        >>> gi = GradientImg()
        >>> graidentmap = gi.computegradientimg()
    """
    def __init__(self, method = 'sobel'):
        oprx = np.zeros([3,3,3])
        opry = np.zeros([3,3,3])
        oprz = np.zeros([3,3,3])
        if method == 'sobel':
            pattern = [[1,2,1], [2,2,2], [1,2,1]]
        elif method == 'prewitt':
            pattern = [[1,1,1], [1,1,1], [1,1,1]]
        else:
            raise Exception('Please input sobel or prewitt')
        oprx[2,:,:] = pattern
        oprx[0,:,:] = -1*np.array(pattern)
        opry[:,2,:] = pattern
        opry[:,0,:] = -1*np.array(pattern)
        oprz[:,:,2] = pattern
        oprz[:,:,0] = -1*np.array(pattern)
        self._oprx = oprx
        self._opry = opry
        self._oprz = oprz
    def computegradientimg(self, imgdata):
        """
        Compute graident image
        -----------------------------
        Parameters:
            imgdata: raw nifti data
        Output:
            gradientimg: gradient image
        """ 
        gradientimg = np.zeros_like(imgdata)
        for i in range(1,imgdata.shape[0]-1):
            for j in range(1,imgdata.shape[1]-1):
                for k in range(1,imgdata.shape[2]-1):
                    gradientimg[i,j,k], vectorg = calgradient3D(imgdata, [i,j,k], self._oprx, self._opry, self._oprz)
            print('{}'.format(i))
        return gradientimg




