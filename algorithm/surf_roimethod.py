# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import numpy as np
from surf_tools import caldice

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
        labels = range(int(np.max(mask)))
    else:
        labels = range(1, labelnum+1)
    pm = np.zeros((mask.shape[0],len(labels)))
    if meth == 'all':
        for i,e in enumerate(labels):
            pm[...,i] = np.mean(mask == e, axis = 1)
    elif meth == 'part':
        for i,e in enumerate(labels):
            mask_i = mask == e
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
        Note that pm.shape[3] should correspond to specific label of region
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
        pm = pm.reshape(pm.shape[0], pm.shape[3])
    pm[np.isnan(pm)] = 0
    pm_temp = np.zeros((pm.shape[0], pm.shape[1]+1))
    pm_temp[:, range(1,pm.shape[1]+1)] = pm
    pm_temp[pm_temp<threshold] = 0
    mpm = np.argmax(pm_temp, axis=1)
    mpm = mpm.reshape((mpm.shape[0], 1, 1))
    return mpm
    
def get_maximum_threshold(imgdata, labels, labelnum = None, prob_meth = 'part', n_fold=2, thr_range = [0,1,0.1], n_permutation=10):
    """
    Decide the maximum threshold from raw image data.
    Here using the cross validation method to decide threhold using for getting the maximum probabilistic map
    
    Parameters:
    -----------
    imgdata: An 2/4 dimensional data
    label: label number
    prob_meth: 'all' or 'part' subjects to use to compute probablistic map
    n_fold: split data into n_fold part, using first n_fold-1 part to get probabilistic map, then using rest part to evaluate overlap condition, by default is 2
    thr_range: pre-set threshold range to check the best maximum probabilistic threshold, the best threshold will search in this parameters, by default is [0,1,0.1], as the format of [start, stop, step]
    n_permuation: times of permutation, by default is 10

    Return:
    -------
    output_dice: all dice coefficient computed from function
                 output_dice consists as a 4 dimension array
                 permutation x threhold x subjects x regions
                 the first dimension permutation means the results of each permutation
                 the second dimension threhold means the results of pre-set threshold
                 the third dimension subjects means the results of each subject
                 the fourth dimension regions means the result of each region
    
    Example:
    --------
    >>> output_dice = get_maximum_threshold(imgdata, [2,4], labelnum = 4)
    """        
    assert (imgdata.ndim==2)|(imgdata.ndim==4), "imgdata should be 2/4 dimension"
    if imgdata.ndim == 4:
        imgdata = imgdata.reshape((imgdata.shape[0], imgdata.shape[3]))
    n_subj = imgdata.shape[1]
    if labelnum is None:
        labelnum = int(np.max(np.unique(imgdata)))
    assert (np.max(labels)<labelnum+1), "the maximum of labels should smaller than labelnum"
    output_dice = []
    for n in range(n_permutation):
        print("permutation {} starts".format(n+1))
        test_subj = np.sort(np.random.choice(range(n_subj), n_subj-n_subj/n_fold, replace = False)).tolist()
        verify_subj = [val for val in range(n_subj) if val not in test_subj]
        test_data = imgdata[:,test_subj]
        verify_data = imgdata[:,verify_subj]
        pm = make_pm(test_data, prob_meth, labelnum)
        pm_temp = []
        for i,e in enumerate(np.arange(thr_range[0], thr_range[1], thr_range[2])):
            print("threshold {} is verifing".format(e))
            mpm = make_mpm(pm, e)
            mpm_temp = []
            for j, vs in enumerate(verify_subj):
                mpm_temp.append([caldice(mpm, verify_data[:,j], lbl, lbl) for lbl in labels])
            pm_temp.append(mpm_temp)
        output_dice.append(pm_temp)
    output_dice = np.array(output_dice)
    return output_dice







