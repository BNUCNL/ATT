# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import numpy as np

def make_pm(mask, meth = 'all', labelnum = None):
    """
    Compute probabilistic map
    
    Parameters:
    -----------
    mask: merged mask
          note that should be 2/4 dimension data
    meth: 'all' or 'part'
          'all', all subjects are taken into account
          'part', part subjects are taken into account, except subject with no roi label in specific roi
    labelnum: label number, by default is None

    Return:
    -------
    pm: probablistic map

    Example:
    --------
    >>> pm = make_pm(mask, 'all')
    
    """
    if (mask.ndim != 2)&(mask.ndim != 4):
        raise Exception('masks should be a 2/4 dimension file to get pm')
    if mask.ndim == 4:
        mask = mask.reshape(mask.shape[0], mask.shape[3])
    if labelnum is None:
        labelnum = int(np.max(mask))
    labels = np.unique(mask)[1:]
    pm = np.zeros((mask.shape[0],labelnum))
    if meth == 'all':
        for i in range(labelnum):
            pm[...,i] = np.mean(mask == i+1, axis = 1)
    elif meth == 'part':
        for i in range(labelnum):
            mask_i = mask == i+1
            subj = np.any(mask_i, axis = 0)
            pm[...,i] = np.mean(mask_i[...,subj],axis = 1)
    else:
        raise Exception('Miss parameter meth')
    pm = pm.reshape((pm.shape[0], 1, 1, pm.shape[1]))
    return pm

def make_mpm(pm, threshold):
    """
    Make maximum probablistic map (mpm)
    
    Parameters:
    -----------
    pm: probabilistic map
    threshold: threshold to filter vertex with low probability
    
    Return:
    -------
    mpm: maximum probabilistic map
    
    Example:
    >>> mpm = make_mpm(pm, 0.2)
    """
    if (pm.ndim != 4)&(pm.ndim != 2):
        raise Exception('Probablistic map should be 2/4 dimension to get maximum probablistic map')
    if pm.ndim == 4:
        pm = pm.reshape((pm.shape[0], pm.shape[3]))
    pm_temp = np.empty((pm.shape[0], pm.shape[1]+1))
    pm_temp[..., range(1,pm.shape[1]+1)] = pm
    pm_temp[pm_temp<threshold] = 0
    mpm = np.argmax(pm_temp, axis=1)
    mpm = mpm.reshape((mpm.shape[0], 1, 1))
    return mpm
    
        









