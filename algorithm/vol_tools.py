# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import numpy as np

def vox2MNI(vox, affine):
    """
    Voxel coordinates transformed to MNI coordinates
    ------------------------------------------------
    Parameters:
        vox: voxel coordinates
        affine: affine matrix
    Return:
        mni
    """
    vox = np.append(vox, 1)
    mni = np.dot(affine, vox)
    return mni[:3]

def MNI2vox(mni, affine):
    """
    MNI coordintes transformed to voxel coordinates
    ----------------------------------------------
    Parameters:
        mni: mni coordinates
        affine: affine matrix
    Return:
        vox
    """
    mni = np.append(mni,1)
    vox = np.dot(mni, np.linalg.inv(affine.T))
    return vox[:3]

def caldice(data1, data2, label):
    """
    Compute dice coefficient
    ---------------------------------
    Parameters:
        data1, data2: matrix data with labels
                      data is 3 dimension
        label: class(region) labels
    Output:
        dice: dice values
    Example:
        >>> dice = caldice(data1, data2, [1,2,3,4])
    """
    if isinstance(label, list):
        label = np.array(label)
    dice = []
    for i in range(label.shape[0]):
        data_mul = (data1 == (i+1)) * (data2 == (i+1))
        data_sum = (data1 == (i+1)) + (data2 == (i+1))
        if not np.any(data_sum[data_sum!=0]):
            dice.append(np.nan)
        else:
            dice.append(2.0*np.sum(data_mul)/(np.sum(data1 == (i+1)) + np.sum(data2 == (i+1))))
    return dice

def caleta2(data1, data2, mask):
    """
    Compute eta2 between data1 and data2.
    eta2 = 1-sum((ai-mi)^2+(bi-mi)^2)/sum((ai-M)^2+(bi-M)^2)
    ai, value of each voxel in data1
    bi, value of each voxel in data2
    mi, 0.5*(ai+bi)
    M, average of sum(all signal voxels that have values)
    eta2 measures similarity between two images, note they need to comparable.
         higher eta2 means higher similarity between two images
    ------------------------------------------------------------------
    Parameters:
        data1: matrix data
        data2: matrix data
        mask: used to get inner-brain data
    Output:
        eta: eta2
    Example:
        >>> eta = caleta2(data1, data2, mask)
    """
    if isinstance(data1, list):
        data1 = np.array(data1)
    if isinstance(data2, list):
        data2 = np.array(data2)
    if data1.shape != data2.shape:
        raise Exception('data1 and data2 should have same shape!')
    data_avg1 = np.mean(data1[mask!=0])
    data_avg2 = np.mean(data2[mask!=0])
    M = 0.5*(data_avg1+data_avg2)
    sumwithin = 0
    sumtotal = 0
    flattendata1 = data1[mask!=0].flatten()
    flattendata2 = data2[mask!=0].flatten()
    for i in range(flattendata1.shape[0]):
        mi = (flattendata1[i] + flattendata2[i])/2
        sumwithin += (flattendata1[i]-mi)**2+(flattendata2[i]-mi)**2
        sumtotal += (flattendata1[i]-M)**2+(flattendata2[i]-M)**2
    eta = 1-float(sumwithin)/sumtotal
    return eta

def get_masksize(mask):
    """
    Compute mask size
    -------------------------------------
    Parameters:
        mask: mask.
    Return:
        masksize: mask size of each roi
    """
    labels = np.unique(mask)[1:]
    if mask.ndim == 3:
        mask = np.expand_dims(mask, axis = 3)
    masksize = np.empty((mask.shape[3], int(np.max(labels))))
    for i in range(mask.shape[3]):
        for j in range(int(np.max(labels))):
            if np.any(mask[...,i] == j+1):
                masksize[i, j] = np.sum(mask[...,i] == j+1)
            else:
                masksize[i, j] = np.nan
    return masksize

def get_signals(atlas, mask, method = 'mean', labelnum = None):
    """
    Extract roi signals of atlas
    --------------------------------------
    Parameters:
        atlas: atlas
        mask: masks. Different roi labelled differently
        method: 'mean', 'std', 'ste'(standard error), 'max', 'voxel', etc.
        labelnum: Mask's label numbers, by default is None. Add this parameters for group analysis
    Return:
        signals: nroi for activation data
                 resting signal x roi for resting data
    """
    labels = np.unique(mask)[1:]
    if labelnum is None:
        labelnum = int(np.max(labels))
    signals = []
    if method == 'mean':
        calfunc = np.nanmean
    elif method == 'std':
        calfunc = np.nanstd
    elif method == 'ste':
        calfunc = ste
    elif method == 'max':
        calfunc = np.max
    elif method == 'voxel':
        calfunc = np.array
    else:
        raise Exception('Method contains mean or std or peak')
    for i in range(labelnum):
        loc_raw = np.where(mask == (i+1))
        roiloc = zip(loc_raw[0], loc_raw[1], loc_raw[2])
        roisignal = [atlas[roiloc[i]] for i in range(len(roiloc))]
        if np.any(roisignal):
            signals.append(roisignal)
        else:
            signals.append([np.nan])
    # return signals    
    return [calfunc(sg) for sg in signals]

def get_coordinate(atlas, mask, size = [2,2,2], method = 'peak', labelnum = None):
    """
    Extract peak/center coordinate of rois
    --------------------------------------------
    Parameters:
        atlas: atlas
        mask: roi mask.
        size: voxel size
        method: 'peak' or 'center'
        labelnum: mask label numbers in total, by default is None, set parameters if you want to do group analysis
    Return:
        coordinates: nroi x 3 for activation data
                     Note that do not extract coordinate of resting data
    """
    labels = np.unique(mask)[1:]
    if labelnum is None:
        labelnum = np.max(labels)
    coordinate = np.empty((labelnum, 3))

    extractpeak = lambda x: np.unravel_index(x.argmax(), x.shape)
    extractcenter = lambda x: np.mean(np.transpose(np.nonzero(x)))

    if method == 'peak':
        calfunc = extractpeak
    elif method == 'center':
        calfunc = extractcenter
    else:
        raise Exception('Method contains peak or center')
    for i in range(labelnum):
        roisignal = atlas*(mask == (i+1))
        if np.any(roisignal):
            coordinate[i,:] = calfunc(roisignal)
            coordinate[i,:] = vox2MNI(coordinate[i,:], size)
        else:
            coordinate[i,:] = np.array([np.nan, np.nan, np.nan])
    return coordinate





