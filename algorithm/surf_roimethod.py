# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import numpy as np
from tools import calc_overlap
import tools, surf_tools

def make_apm(act_merge, thr):
    """
    Compute activation probabilistic map

    Parameters:
    -----------
    act_merge: merged activation map
    thr_val: threshold of activation value

    Return:
    -------
    apm: activation probabilistic map

    Example:
    --------
    >>> apm = make_apm(act_merge, thr = 5.0)
    """
    import copy
    act_tmp = copy.deepcopy(act_merge)
    act_tmp[act_tmp<thr] = 0
    act_tmp[act_tmp!=0] = 1
    apm = np.mean(act_tmp,axis=-1)
    return apm

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
    pm = pm.reshape((pm.shape[0], 1, 1, pm.shape[-1]))
    return pm

def make_mpm(pm, threshold, keep_prob = False, consider_baseline = False):
    """
    Make maximum probablistic map (mpm)
    
    Parameters:
    -----------
    pm: probabilistic map
        Note that pm.shape[3] should correspond to specific label of region
    threshold: threshold to filter vertex with low probability
    keep_prob: whether to keep maximum probability but not transform it into labels
               If True, return map with maximum probability
               If False, return map with label from maxmum probability
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
    if not keep_prob: 
        mpm = np.argmax(pm_temp, axis=1)
    else:
        mpm = np.max(pm_temp, axis=1)
    mpm = mpm.reshape((-1, pm.ndim))
    return mpm
    
def nfold_location_overlap(imgdata, labels, labelnum = None, index = 'dice', thr_meth = 'prob', prob_meth = 'part', n_fold=2, thr_range = [0,1,0.1], n_permutation=1, controlsize = False, actdata = None):
    """
    Decide the maximum threshold from raw image data.
    Here using the cross validation method to decide threhold using for getting the maximum probabilistic map
    
    Parameters:
    -----------
    imgdata: A 2/4 dimensional data
    labels: list, label number used to extract dice coefficient
    labelnum: by default is None, label number size. We recommend to provide label number here.
    index: 'dice' or 'percent'
    thr_meth: 'prob', threshold probabilistic map by probabilistic values (MPM)
              'number', threshold probabilistic map by numbers of vertex
    prob_meth: 'all' or 'part' subjects to use to compute probablistic map
    n_fold: split data into n_fold part, using first n_fold-1 part to get probabilistic map, then using rest part to evaluate overlap condition, by default is 2
    thr_range: pre-set threshold range to find the best maximum probabilistic threshold, the best threshold will search in this parameters, by default is [0,1,0.1], as the format of [start, stop, step]
    n_permuation: times of permutation, by default is 10
    controlsize: whether control label data size with template mpm label size or not, by default is False.
    actdata: if controlsize is True, please input actdata as a parameter. By default is None.

    Return:
    -------
    output_overlap: dice coefficient/percentage computed from function
                    output_dice consists of a 4 dimension array
                    permutation x threhold x subjects x regions
                    the first dimension permutation means the results of each permutation
                    the second dimension threhold means the results of pre-set threshold
                    the third dimension subjects means the results of each subject
                    the fourth dimension regions means the result of each region
    
    Example:
    --------
    >>> output_overlap = nfold_location_overlap(imgdata, [2,4], labelnum = 4)
    """        
    assert (imgdata.ndim==2)|(imgdata.ndim==4), "imgdata should be 2/4 dimension"
    if imgdata.ndim == 4:
        imgdata = imgdata.reshape((imgdata.shape[0], imgdata.shape[3]))
    if actdata is not None:
        if actdata.ndim == 4:
            actdata = actdata.reshape((actdata.shape[0], actdata.shape[3]))
    n_subj = imgdata.shape[1]
    if labelnum is None:
        labelnum = int(np.max(np.unique(imgdata)))
    assert (np.max(labels)<labelnum+1), "the maximum of labels should smaller than labelnum"
    output_overlap = []
    for n in range(n_permutation):
        print("permutation {} starts".format(n+1))
        test_subj = np.sort(np.random.choice(range(n_subj), n_subj-n_subj/n_fold, replace = False)).tolist()
        verify_subj = [val for val in range(n_subj) if val not in test_subj]
        test_data = imgdata[:,test_subj]
        verify_data = imgdata[:,verify_subj]
        if actdata is not None:
            verify_actdata = actdata[...,verify_subj]
        else:
            verify_actdata = None
        pm = make_pm(test_data, prob_meth, labelnum)
        pm_temp = cv_pm_overlap(pm, verify_data, labels, labels, thr_meth = thr_meth, thr_range = thr_range, index = index, cmpalllbl = False, controlsize = controlsize, actdata = verify_actdata)
        output_overlap.append(pm_temp)
    output_overlap = np.array(output_overlap)
    return output_overlap

def leave1out_location_overlap(imgdata, labels, labelnum = None, index = 'dice', thr_meth = 'prob', prob_meth = 'part', thr_range = [0,1,0.1], controlsize = False, actdata = None):
    """
    A leave one out cross validation metho for threshold to best overlapping in probabilistic map
    
    Parameters:
    -----------
    imgdata: A 2/4 dimensional data
    labels: list, label number used to extract dice coefficient
    labelnum: by default is None, label number size. We recommend to provide label number here.
    index: 'dice' or 'percent'
    thr_meth: 'prob', threshold probabilistic map by probabilistic values (MPM)
              'number', threshold probabilistic map by numbers of vertex
    prob_meth: 'all' or 'part' subjects to use to compute probablistic map
    thr_range: pre-set threshold range to find the best maximum probabilistic threshold
    controlsize: whether control label data size with template mpm label size or not, by default is False.
    actdata: if controlsize is True, please input actdata as a parameter. By default is None.

    Return:
    -------
    output_overlap: dice coefficient/percentage computed from function
                    outputdice consists of a 3 dimension array
                    subjects x threhold x regions
                    the first dimension means the values of each leave one out (leave one subject out)
                    the second dimension means the results of pre-set threshold
                    the third dimension means the results of each region

    Example:
    --------
    >>> output_overlap = leave1out_location_overlap(imgdata, [2,4], labelnum = 4)
    """
    if imgdata.ndim == 4:
        imgdata = imgdata.reshape(imgdata.shape[0], imgdata.shape[-1])
    if actdata is not None:
        if actdata.ndim == 4:
            actdata = actdata.reshape(actdata.shape[0], actdata.shape[-1])
    output_overlap = []
    for i in range(imgdata.shape[-1]):
        data_temp = np.delete(imgdata, i, axis=1)
        testdata = np.expand_dims(imgdata[:,i],axis=1)
        if actdata is not None:
            test_actdata = np.expand_dims(actdata[:,i],axis=1)
        else:
            test_actdata = None
        pm = make_pm(data_temp, prob_meth, labelnum)
        pm_temp = cv_pm_overlap(pm, testdata, labels, labels, thr_meth = thr_meth, thr_range = thr_range, index = index, cmpalllbl = False, controlsize = controlsize, actdata = test_actdata)
        output_overlap.append(pm_temp)
    output_array = np.array(output_overlap)
    return output_array.reshape(output_array.shape[0], output_array.shape[2], output_array.shape[3])

def leave1out_magnitude(roidata, magdata, index = 'mean', thr_meth = 'prob', thr_range = [0,1,0.1], prob_meth = 'part'):
    """
    Function to use cross validation to extract magnitudes
    Compute probabilistic map to extract signal of the rest part of subject

    Parameters:
    ------------
    roidata: roidata used for probabilistic map
    magdata: magnitude data
    index: 'mean', 'std', 'ste', 'vertex', etc.
    thr_meth: 'prob', threshold probabilistic map by probabilistic values
              'number', threshold probabilistic map by numbers of vertex
    prob_meth: 'all' or 'part' used for probabilistic map generation
    thr_range: pre-set threshold range to compute thresholded labeled map

    Returns:
    --------
    mag_signals: magnitude signals
    
    Examples:
    ----------
    >>> mag_signals = leave1out_magnitude(roidata, magdata)
    """
    roidata = roidata.reshape(roidata.shape[0], roidata.shape[-1])
    magdata = magdata.reshape(magdata.shape[0], magdata.shape[-1])
    assert roidata.shape == magdata.shape, "roidata should have same shape as magdata"
    output_overlap = []
    n_subj = roidata.shape[-1]
    for i in range(roidata.shape[-1]):
        verify_subj = [i]
        test_subj = [val for val in range(n_subj) if val not in verify_subj]
        test_roidata = roidata[:, test_subj]
        verify_magdata = magdata[:, verify_subj]
        pm = make_pm(test_roidata, prob_meth)
        pm_temp = cv_pm_magnitude(pm, verify_magdata, index = index, thr_meth = thr_meth, thr_range = thr_range)
        output_overlap.append(pm_temp)
    output_array = np.array(output_overlap)
    output_array = output_array.reshape((-1,(thr_range[1]-thr_range[0])/thr_range[2]))
    return output_array

def nfold_magnitude(roidata, magdata, index = 'mean', thr_meth = 'prob', prob_meth = 'part', thr_range = [0,1,0.1], n_fold = 2, n_permutation = 1):
    """
    Using cross validation method to split data into nfold
    compute probabilistic map by first part of data, then extract signals of rest part of data using probabilistic map 

    Parameters:
    ------------
    roidata: roidata used for probabilistic map
    magdata: magnitude data
    index: 'mean', 'std', 'ste', 'vertex', etc.
    thr_meth: 'prob', threshold probabilistic map by probabilistic values (MPM)
              'number', threshold probabilistic map by numbers of vertex
    prob_meth: 'all' or 'part'. Subjects to use compute probabilistic map
    thr_range: pre-set threshold range to compute thresholded labeled map
    n_fold: split numbers for cross validation
    n_permutation: permutation times

    Returns:
    --------
    mag_signals: magnitude signals

    Examples:
    ---------
    >>> mag_signals = nfold_magnitude(roidata, magdata)
    """
    roidata = roidata.reshape(roidata.shape[0], roidata.shape[-1])
    magdata = magdata.reshape(magdata.shape[0], magdata.shape[-1])
    assert magdata.shape == roidata.shape, "roidata should have same shape as magdata"
    n_subj = roidata.shape[-1]
    output_overlap = []
    for n in range(n_permutation):
        print('permutation {} starts'.format(n+1))
        test_subj = np.sort(np.random.choice(n_subj, n_subj-n_subj/n_fold, replace = False)).tolist()
        verify_subj = [val for val in range(n_subj) if val not in test_subj]
        test_roidata = roidata[:, test_subj]
        verify_magdata = magdata[:, verify_subj]
        pm = make_pm(test_roidata, prob_meth)
        pm_temp = cv_pm_magnitude(pm, verify_magdata, index = index, thr_meth = thr_meth, thr_range = thr_range)
        output_overlap.append(pm_temp)
    output_array = np.array(output_overlap)
    return output_array

def pm_overlap(pm1, pm2, thr_range, option = 'number', index = 'dice'):
    """
    Analysis for probabilistic map overlap without using test data 
    The idea of this analysis is to control vertices number/threshold same among pm1 and pm2, binaried them then compute overlap
    
    Parameters:
    -----------
    pm1: probablistic map 1
    pm2: probabilistic map 2
    thr_range: threshold range, format as [min, max, step], which could be vertex numbers or probablistic threshold
    option: 'number', compute overlap between probablistic maps by multiple vertex numbers
            'threshold', compute overlap between probablistic maps by multiple thresholds
    index: 'dice', overlap indices as dice coefficient
           'percent', overlap indices as percent
    """
    assert (pm1.ndim == 1)|(pm1.ndim == 3), "pm1 should not contain multiple probablistic map"
    assert (pm2.ndim == 1)|(pm2.ndim == 3), "pm2 should not contain multiple probablistic map" 
    if pm1.ndim == 3:
        pm1 = pm1.reshape(pm1.shape[0], pm1.shape[-1])
    if pm1.ndim == 1:
        pm1 = np.expand_dims(pm1, axis=0)
    if pm2.ndim == 3:
        pm2 = pm2.reshape(pm2.shape[0], pm2.shape[-1])
    if pm2.ndim == 1:
        pm2 = np.expand_dims(pm2, axis=0)

    assert len(thr_range) == 3, "thr_range should be a 3 elements list, as [min, max, step]"
    if option == 'number':
        thre_func = tools.threshold_by_number
    elif option == 'threshold':
        thre_func = tools.threshold_by_values
    else:
        raise Exception('Missing option')

    output_overlap = []
    for i in np.arange(thr_range[0], thr_range[1], thr_range[2]):
        print('Computing overlap of vertices {}'.format(i))
        pm1_thr = thre_func(pm1, i)
        pm2_thr = thre_func(pm2, i)
        pm1_thr[pm1_thr!=0] = 1
        pm2_thr[pm2_thr!=0] = 1
        output_overlap.append(calc_overlap(pm1_thr, pm2_thr, 1, 1))
    output_overlap = np.array(output_overlap)
    output_overlap[np.isnan(output_overlap)] = 0
    return output_overlap
         
def cv_pm_overlap(pm, test_data, labels_template, labels_testdata, index = 'dice', thr_meth = 'prob', thr_range = [0, 1, 0.1], cmpalllbl = True, controlsize = False, actdata = None):
    """
    Compute overlap(dice) between probabilistic map and test data
    
    Parameters:
    -----------
    pm: probabilistic map
    test_data: subject specific label data used as test data
    labels_template: list, label number of template (pm) used to extract overlap values 
    label_testdata: list, label number of test data used to extract overlap values
    index: 'dice' or 'percent'
    thr_meth: 'prob', threshold probabilistic map by probabilistic values (MPM)
              'number', threshold probabilistic map by numbers of vertex
    thr_range: pre-set threshold range to find the best maximum probabilistic threshold
    cmpalllbl: compute all overlap label one to one or not.
               e.g. labels_template = [2,4], labels_testdata = [2,4]
                    if True, get dice coefficient of (2,2), (2,4), (4,2), (4,4)
                    else, get dice coefficient of (2,2), (4,4)
    controlsize: whether control label data size with template mpm label size or not, by default is False.
    actdata: if controlsize is True, please input actdata as a parameter. By default is None.

    Return:
    -------
    output_overlap: dice coefficient/percentage
                    outputdice consists of a 3 dimension array
                    subjects x thr_range x regions
                    the first dimension means the values of each leave one out (leave one subject out)
                    the second dimension means the results of pre-set threshold
                    the third dimension means the results of each region
                 
    Example:
    --------
    >>> output_overlap = cv_pm_overlap(pm, test_data, [2,4], [2,4])
    """
    if cmpalllbl is False:
        assert len(labels_template) == len(labels_testdata), "Notice that labels_template should have same length of labels_testdata if cmpalllbl is False"
    if test_data.ndim == 4:
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[-1])
    if actdata is not None:
        if actdata.ndim == 4:
            actdata = actdata.reshape(actdata.shape[0], actdata.shape[-1])
    output_overlap = []
    for i in range(test_data.shape[-1]):
        thrmp_temp = []
        if actdata is not None:
            verify_actdata = actdata[:,i]
        else:
            verify_actdata = None
        for j,e in enumerate(np.arange(thr_range[0], thr_range[1], thr_range[2])):
            print("threshold {} is verifing".format(e))
            if thr_meth == 'prob':    
                thrmp = make_mpm(pm, e)
            elif thr_meth == 'number':
                if pm.shape[-1] > 1:
                    raise Exception('only support 1 label for this situation')
                thrmp = tools.threshold_by_number(pm, e)
                thrmp[thrmp!=0] = 1
            if cmpalllbl is True:
                thrmp_temp.append([calc_overlap(thrmp.flatten(), test_data[:,i], lbltmp, lbltst, index, controlsize = controlsize, actdata = verify_actdata) for lbltmp in labels_template for lbltst in labels_testdata])
            else:
                thrmp_temp.append([calc_overlap(thrmp.flatten(), test_data[:,i], labels_template[idx], lbld, index, controlsize = controlsize, actdata = verify_actdata) for idx, lbld in enumerate(labels_testdata)])
        output_overlap.append(thrmp_temp)
    return np.array(output_overlap)

def cv_pm_magnitude(pm, test_magdata, index = 'mean', thr_meth = 'prob', thr_range = [0,1,0.1]):
    """
    Function to extract signals from probabilistic map with varied threshold

    Parameters:
    ------------
    pm: probablistic map
    test_magdata: magnitude data used as test dataset
    index: type of signals, by default is 'mean'
    thr_meth: 'prob', threshold probabilistic map using probabilistic threshold
              'number', threshold probabilistic map using numbers of vertex
    thr_range: threshold range

    Results:
    ---------
    signals: signals of each threshold 

    Example:
    ---------
    >>> signals = cv_pm_magnitude(pm, test_magdata)
    """
    test_magdata = test_magdata.reshape(test_magdata.shape[0], test_magdata.shape[-1])
    pm = pm.reshape(pm.shape[0], pm.shape[-1])
    signal = []
    for i in range(test_magdata.shape[-1]):
        signal_thr = []
        for j,e in enumerate(np.arange(thr_range[0], thr_range[1], thr_range[2])):
            if thr_meth == 'prob':
                thrmp = make_mpm(pm, e)
            elif thr_meth == 'number':
                if pm.shape[-1] > 1:
                    raise Exception('only support 1 label for this situation')
                thrmp = tools.threshold_by_number(pm, e)
                thrmp[thrmp!=0] = 1
            else:
                raise Exception('Threshold probability only contains by probability values or vertex numbers')
            signal_thr.append(surf_tools.get_signals(test_magdata[:,i], thrmp[:,0], method = index))
        signal.append(signal_thr)
    return np.array(signal)
                
def overlap_bysubject(imgdata, labels, subj_range, labelnum = None, prob_meth = 'part', index = 'dice'):
    """
    A function used for computing overlap between template (probilistic map created by all subjects) and probabilistic map of randomly chosen subjects.
    
    Parameters:
    -----------
    imgdata: label image data
    labels: list, label number indicated regions
    subj_range: range of subjects, the format as [minsubj, maxsubj, step]
    labelnum: label numbers, by default is None
    prob_meth: method for probabilistic map, 'all' to compute all subjects that contains non-regions, 'part' to compute part subjects that ignore subjects with non-regions.

    Returns:
    --------
    overlap_subj: overlap indices of each amount of subjects

    Example:
    --------
    >>> overlap_subj = overlap_bysubject(imgdata, [4], [0,100,10], labelnum = 4) 
    """
    nsubj = imgdata.shape[-1]
    pm = make_pm(imgdata, meth = prob_meth, labelnum = labelnum)
    overlap_subj = []
    for i in np.arange(subj_range[0], subj_range[1], subj_range[2]):
        subj_num = np.random.choice(nsubj, i, replace=False)        
        sub_imgdata = imgdata[...,subj_num]
        if sub_imgdata.ndim == 3:
            sub_imgdata = np.expand_dims(sub_imgdata, axis=-1)
        pm_sub = make_pm(sub_imgdata, meth = prob_meth, labelnum = labelnum)
        overlap_lbl = []
        for lbl in labels:
            pm_lbl = pm[...,lbl-1]
            pm_sub_lbl = pm_sub[...,lbl-1]
            pm_lbl[pm_lbl!=0] = 1
            pm_sub_lbl[pm_sub_lbl!=0] = 1
            overlap_lbl.append(calc_overlap(pm_lbl, pm_sub_lbl, 1, 1, index = index))
        overlap_subj.append(overlap_lbl)
    return np.array(overlap_subj)

class GetLblRegion(object):
    """
    A class to get template label regions
    
    Parameters:
    -----------
    template: template
    """
    def __init__(self, template):
        self._template = template

    def by_lblimg(self, lbldata):
        """
        Get specific template regions by rois given by user
        All regions overlapped with a specific label region will be covered

        Parameters:
        -----------
        lbldata: rois given by user

        Return:
        -------
        out_template: new template contains part of regions
                      if lbldata has multiple different rois, then new template will extract regions with each of roi given by user

        Example:
        --------
        >>> glr_cls = GetLblRegion(template)
        >>> out_template = glr_cls.by_lblimg(lbldata)
        """
        assert lbldata.shape == self._template.shape, "the shape of template should be equal to the shape of lbldata"
        labels = np.sort(np.unique(lbldata)[1:]).astype('int')
        out_template = np.zeros_like(lbldata)
        out_template = out_template[...,np.newaxis]
        out_template = np.tile(out_template, (1, len(labels)))
        for i,lbl in enumerate(labels):
            lbldata_tmp = tools.get_specificroi(lbldata, lbl)
            lbldata_tmp[lbldata_tmp!=0] = 1
            part_template = self._template*lbldata_tmp
            template_lbl = np.sort(np.unique(part_template)[1:])
            out_template[...,i] = tools.get_specificroi(self._template, template_lbl)
        return out_template

def get_border_vertex(data, faces, n = 2):
    """
    extract vertices that be in border of original data.

    Parameters:
    -----------
    data: original data (scalar data)
    faces: faces relationship  
    n: by default is 2. Neighboring ring number. 

    Returns:
    --------
    vx: vertices from border

    Examples:
    ---------
    >>> border_vertex = get_border_vertex(data, faces)
    """
    border_vertex = []
    data_vertex = np.where(data!=0)[0]
    one_ring_neighbor = surf_tools.get_n_ring_neighbor(data_vertex, faces, n)
    border_check = [not np.all(data[list(i)]) for i in one_ring_neighbor]   
    border_vertex = data_vertex[border_check]
    return border_vertex
    




