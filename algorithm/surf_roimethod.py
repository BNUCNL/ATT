# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import numpy as np
from surf_tools import caloverlap

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
        labels = range(1, int(np.max(mask))+1)
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

def make_mpm(pm, threshold, consider_baseline = False):
    """
    Make maximum probablistic map (mpm)
    
    Parameters:
    -----------
    pm: probabilistic map
        Note that pm.shape[3] should correspond to specific label of region
    threshold: threshold to filter vertex with low probability
    consider_baseline: whether consider baseline or not when compute mpm
                       if True, check vertices that contain several probabilities, if p1+p2+...+pn < 0.5, then discard it
                       Details see Liang Wang, et al., Probabilistic Maps of Visual Topology in Human Cortex, 2015
    
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
    if consider_baseline is True:
        vex_discard = [(np.count_nonzero(pm_temp[i,:])>1)&((np.sum(pm_temp[i,:]))<0.5) for i in range(pm_temp.shape[0])]
        vex_disind = [i for i,e in enumerate(vex_discard) if e]
        pm_temp[vex_disind,:] = 0 
    mpm = np.argmax(pm_temp, axis=1)
    mpm = mpm.reshape((mpm.shape[0], 1, 1))
    return mpm
    
def nfold_maximum_threshold(imgdata, labels, labelnum = None, prob_meth = 'part', n_fold=2, thr_range = [0,1,0.1], n_permutation=10):
    """
    Decide the maximum threshold from raw image data.
    Here using the cross validation method to decide threhold using for getting the maximum probabilistic map
    
    Parameters:
    -----------
    imgdata: A 2/4 dimensional data
    labels: list, label number used to extract dice coefficient
    labelnum: by default is None, label number size. We recommend to provide label number here.
    prob_meth: 'all' or 'part' subjects to use to compute probablistic map
    n_fold: split data into n_fold part, using first n_fold-1 part to get probabilistic map, then using rest part to evaluate overlap condition, by default is 2
    thr_range: pre-set threshold range to find the best maximum probabilistic threshold, the best threshold will search in this parameters, by default is [0,1,0.1], as the format of [start, stop, step]
    n_permuation: times of permutation, by default is 10

    Return:
    -------
    output_dice: all dice coefficient computed from function
                 output_dice consists of a 4 dimension array
                 permutation x threhold x subjects x regions
                 the first dimension permutation means the results of each permutation
                 the second dimension threhold means the results of pre-set threshold
                 the third dimension subjects means the results of each subject
                 the fourth dimension regions means the result of each region
    
    Example:
    --------
    >>> output_dice = nfold_maximum_threshold(imgdata, [2,4], labelnum = 4)
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
                mpm_temp.append([caloverlap(mpm, verify_data[:,j], lbl, lbl) for lbl in labels])
            pm_temp.append(mpm_temp)
        output_dice.append(pm_temp)
    output_dice = np.array(output_dice)
    return output_dice

def leave1out_maximum_threhold(imgdata, labels, labelnum = None, prob_meth = 'part', thr_range = [0,1,0.1]):
    """
    A leave one out cross validation metho for threshold to best overlapping in probabilistic map
    
    Parameters:
    -----------
    imgdata: A 2/4 dimensional data
    labels: list, label number used to extract dice coefficient
    labelnum: by default is None, label number size. We recommend to provide label number here.
    prob_meth: 'all' or 'part' subjects to use to compute probablistic map
    thr_range: pre-set threshold range to find the best maximum probabilistic threshold

    Return:
    -------
    output_dice: dice coefficient computed from function
                outputdice consists of a 3 dimension array
                subjects x threhold x regions
                the first dimension means the values of each leave one out (leave one subject out)
                the second dimension means the results of pre-set threshold
                the third dimension means the results of each region

    Example:
    --------
    >>> output_dice = leave1out_maximum_threhold(imgdata, [2,4], labelnum = 4)
    """
    if imgdata.ndim == 4:
        data = data.reshape(data.shape[0], data.shape[3])
    output_dice = []
    for i in range(data.shape[1]):
        data_temp = np.delete(data, i, axis=1)
        pm = make_pm(data_temp, prob_meth, labelnum)
        mpm_temp = []
        for j,e in enumerate(np.arange(thr_range[0], thr_range[1], thr_range[2])):
            mpm = make_mpm(pm, e)
            mpm_temp.append([caloverlap(mpm, imgdata[:,i], lbl, lbl) for lbl in labels])
        output_dice.append(mpm_temp)
    return np.array(output_dice)

def pm_overlap(pm, test_data, labels, thr_range = [0, 1, 0.1]):
    """
    Compute overlap(dice) between probabilistic map and test data
    
    Parameters:
    -----------
    pm: probabilistic map
    test_data: subject specific label data used as test data
    labels: list, label number used to extract overlap values 
    thr_range: pre-set threshold range to find the best maximum probabilistic threshold

    Return:
    -------
    output_dice: dice coefficient
                 outputdice consists of a 3 dimension array
                 subjects x thr_range x regions
                 the first dimension means the values of each leave one out (leave one subject out)
                 the second dimension means the results of pre-set threshold
                 the third dimension means the results of each region
                 
    Example:
    --------
    >>> output_dice = pm_overlap(pm, test_data, [2,4])
    """
    if test_data.ndim == 4:
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[-1])
    output_dice = []
    for i in range(test_data.shape[-1]):
        mpm_temp = []
        for j,e in enumerate(np.arange(thr_range[0], thr_range[1], thr_range[2])):
            mpm = make_mpm(pm, e)
            mpm_temp.append([caloverlap(mpm, test_data[:,i], lbl, lbl) for lbl in labels])
        output_dice.append(mpm_temp)
    return np.array(output_dice)












