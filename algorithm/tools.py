# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import numpy as np
from scipy import stats, special
from scipy.spatial import distance
import copy
import pandas as pd


def _overlap(c1, c2, index='dice'):
    """
    Calculate overlap between two collections

    Parameters
    ----------
    c1, c2 : collection (list | tuple | set | 1-D array etc.)
    index : string ('dice' | 'percent')
        This parameter is used to specify index which is used to measure overlap.

    Return
    ------
    overlap : float
        The overlap between c1 and c2
    """
    set1 = set(c1)
    set2 = set(c2)
    intersection_num = float(len(set1 & set2))
    try:
        if index == 'dice':
            total_num = len(set1 | set2) + intersection_num
            overlap = 2.0 * intersection_num / total_num
        elif index == 'percent':
            overlap = 1.0 * intersection_num / len(set1)
        else:
            raise Exception("Only support 'dice' and 'percent' as overlap indices at present.")
    except ZeroDivisionError as e:
        overlap = np.nan
    return overlap

def calc_overlap(data1, data2, label1=None, label2=None, index='dice', controlsize = False, actdata = None):
    """
    Calculate overlap between two sets.
    The sets are acquired from data1 and data2 respectively.

    Parameters
    ----------
    data1, data2 : collection or numpy array
        label1 is corresponding with data1
        label2 is corresponding with data2
    
    label1, label2 : None or labels
        If label1 or label2 is None, the corresponding data is supposed to be
        a collection of members such as vertices and voxels.
        If label1 or label2 is a label, the corresponding data is always a numpy array with same shape and meaning.
        And we will acquire set1 elements whose labels are equal to label1 from data1
        and set2 elements whose labels are equal to label2 from data2.
    
    index : string ('dice' | 'percent')
        This parameter is used to specify index which is used to measure overlap.
    controlsize: True or False
        Whether control roi size or not when computing overlap index
        If controlsize is True, please give global image data, but not a collection
        Besides, actdata should be given.
    
    actdata: None or global image data
        If controlsize is True, please input actdata that correspond to data2

    Return
    ------
    overlap : float
        The overlap of data1 and data2

    Example:
    --------
    >>> overlap = calc_overlap(data1, data2, label1, label2, index = 'dice', controlsize = True, actdata = actdata)
    """
    if controlsize is True:
        if label1 is not None and label2 is not None:
            datasize1 = data1[data1==label1].shape[0]
            try:
                data2 = control_lbl_size(data2, actdata, datasize1, label = label2)
            except AttributeError:
                raise Exception('Please give actdata here!')
        else:
            raise Exception('Not support to control size of collection data')

    if label1 is not None:
        positions1 = np.where(data1 == label1)
        data1 = zip(*positions1)    

    if label2 is not None:
        positions2 = np.where(data2 == label2)
        data2 = zip(*positions2)

    # calculate overlap
    overlap = _overlap(data1, data2, index)

    return overlap


def calcdist(u, v, metric = 'euclidean', p = 1):
    """
    Compute distance between u and v
    ----------------------------------
    Parameters:
        u: vector u
        v: vector v
        method: distance metric
                For concrete metric, please check scipy.spatial.distance.pdist
        p: p for 'minkowski' distance only.
    Return:
        dist: distance between vector u and vector v
    """
    if isinstance(u, list):
        u = np.array(u)
    if isinstance(v, list):
        v = np.array(v)
    vec = np.vstack((u, v))
    if metric == 'minkowski':
        dist = distance.pdist(vec, metric, p)
    else:
        dist = distance.pdist(vec, metric)
    return dist

def eta2(a, b):
    """
    Compute eta2 between list a and list b
    
    eta2 = 1-sum((ai-mi)^2+(bi-mi)^2)/sum((ai-M)^2+(bi-M)^2)
    ai, value of each element in a
    bi, value of each element in b
    mi, 0.5*(ai+bi)
    M, average of sum (mean(mi))
    eta2 measures similarity between two lists/arrays (1 dim), note they need to comparable.
         higher eta2 means higher similarity
    
    Parameters:
    -----------
    a: list a 
    b: list b
       note that a, b should have the same length

    Output:
    -------
    eta: eta2

    Example:
    --------
    >>> eta = eta2(a, b)
    """
    a = np.array(a)
    b = np.array(b)
    mi = (a+b)/2
    M = np.mean(mi)
    sumwithin = np.sum((a-mi)**2+(b-mi)**2)
    sumtotal = np.sum((a-M)**2+(b-M)**2)
    return 1-1.0*(sumwithin)/sumtotal

def convert_listvalue_to_ordinal(listdata):
    """
    Convert list elements to ordinal values
    [5, 3, 3, 5, 6] --> [2, 1, 1, 2, 3]
    
    Parameters:
    -----------
    listdata: list data

    Return:
    -------
    ordinals: list with oridinal values

    Example:
    -------- 
    >>> ordinals = convert_listvalue_to_ordinal(listdata)
    """
    setdata = set(listdata)
    ordinal_map = {val: i for i, val in enumerate(setdata, 1)}
    ordinals = [ordinal_map[val] for val in listdata]
    return ordinals

def regressoutvariable(rawdata, covariate):
    """
    Regress out covariate variables from raw data
    -------------------------------------------------
    Parameters:
        rawdata: rawdata
        covariate: covariate to be regressed out
    Return:
        residue
    """
    if isinstance(rawdata, list):
        rawdata = np.array(rawdata)
    if isinstance(covariate, list):
        covariate = np.array(covariate)
    samp = ~np.isnan(rawdata * covariate)
    zfunc = lambda x: (x - np.nanmean(x))/np.nanstd(x)
    slope, intercept, r_value, p_value, std_err = stats.linregress(stats.zscore(rawdata[samp]), stats.zscore(covariate[samp]))
    residue = zfunc(rawdata) - slope*zfunc(covariate)
    return residue

def pearsonr(A, B):
    """
    A broadcasting method to compute pearson r and p
    Code reprint from stackflow
    -----------------------------------------------
    Parameters:
        A: matrix A, i*k
        B: matrix B, j*k
    Return:
        rcorr: matrix correlation, i*j
        pcorr: matrix correlation p, i*j
    Example:
        >>> rcorr, pcorr = pearsonr(A, B)
    """
    if isinstance(A,list):
        A = np.array(A)
    if isinstance(B,list):
        B = np.array(B)
    if np.ndim(A) == 1:
        A = np.expand_dims(A, axis=1).T
    if np.ndim(B) == 1:
        B = np.expand_dims(B, axis=1).T

    rcorr = 1.0 - distance.cdist(A, B, 'correlation')

    df = A.T.shape[1] - 2
    
    r_forp = rcorr*1.0
    r_forp[r_forp==1.0] = 0.0
    t_squared = rcorr.T**2*(df/((1.0-rcorr.T)*(1.0+rcorr.T)))
    pcorr = special.betainc(0.5*df, 0.5, df/(df+t_squared))
    return rcorr.T, pcorr

def r2z(r):
    """
    Perform the Fisher r-to-z transformation
    formula:
    z = (1/2)*(log(1+r) - log(1-r))
    se = 1/sqrt(n-3)
    --------------------------------------
    Parameters:
        r: r matrix or array
    Output:
        z: z matrix or array
    Example:
        >>> z = r2z(r)
    """
    from math import log
    func_rz = lambda r: 0.5*(log(1+r) - log(1-r))
    if isinstance(r,float):
        z = func_rz(r)
    else:
        r_flat = r.flatten()
        r_flat[r_flat>0.999] = 0.999
        z_flat = np.array([func_rz(rvalue) for rvalue in r_flat])
        z = z_flat.reshape(r.shape)
    return z

def z2r(z):
    """
    Perform the Fisher z-to-r transformation
    r = tanh(z)
    --------------------------------------------
    Parameters:
        z: z matrix or array
    Output:
        r: r matrix or array
    Example:
        >>> r = z2r(z)
    """
    from math import tanh
    if isinstance(z, float):
        r = tanh(z)
    else:
        z_flat = z.flatten()
        r_flat = np.array([tanh(zvalue) for zvalue in z_flat])
        r = r_flat.reshape(z.shape)
    return r

def hemi_merge(left_region, right_region, meth = 'single', weight = None):
    """
    Merge hemisphere data
    -------------------------------------
    Parameters:
        left_region: feature data extracted from left hemisphere
        right_region: feature data extracted from right hemisphere
        meth: 'single' or 'both'.
          'single' means if no paired feature data in subjects, keep exist data
          'both' means if no paired feature data in subjects, delete these                subjects
        weight: weights for feature data.
            Note that it's a (nsubj x 2) matrix
            weight[:,0] means left_region
            weight[:,1] means right_region
    Return:
        merge_region 
    """
    if left_region.shape[0] != right_region.shape[0]:
        raise Exception('Subject numbers of left and right feature data should be equal')
    nsubj = left_region.shape[0]
    leftregion_used = np.copy(left_region)
    rightregion_used = np.copy(right_region)
    if weight is None:
        weight = np.ones((nsubj,2))
        weight[np.isnan(leftregion_used),0] = 0.0
        weight[np.isnan(rightregion_used),1] = 0.0 
    if meth == 'single': 
        leftregion_used[np.isnan(leftregion_used)] = 0.0
        rightregion_used[np.isnan(rightregion_used)] = 0.0
        merge_region = (leftregion_used*weight[:,0] + rightregion_used*weight[:,1])/(weight[:,0] + weight[:,1])
    elif meth == 'both':
        total_weight = weight[:,0] + weight[:,1]
        total_weight[total_weight<2] = 0.0
        merge_region = (left_region*weight[:,0] + right_region*weight[:,1])/total_weight
    else:
        raise Exception('meth will be ''both'' or ''single''')
    merge_region[merge_region == 0] = np.nan
    return merge_region

def removeoutlier(data, meth = None, thr = [-2,2]):
    """
    Remove data as outliers by indices you set
    -----------------------------
    Parameters:
        data: data you want to remove outlier
        meth: 'iqr' or 'std' or 'abs'
        thr: outlier standard threshold.
             For example, when meth == 'iqr' and thr == [-2,2],
             so data should in [-2*iqr, 2*iqr] to be left
    Return:
        residue_data: outlier values will be set as nan
        n_removed: outlier numbers
    """
    residue_data = copy.copy(data)   
    if meth is None:
        residue_data = data
        outlier_bool = np.zeros_like(residue_data, dtype=bool) 
    elif meth == 'abs':
        outlier_bool = ((data<thr[0])|(data>thr[1]))
        residue_data[((residue_data<thr[0])|(residue_data>thr[1]))] = np.nan
    elif meth == 'iqr':
        perc_thr = np.percentile(data, [25,75])
        f_iqr = perc_thr[1] - perc_thr[0]
        outlier_bool = ((data < perc_thr[0] + thr[0]*f_iqr)|(data >= perc_thr[1] + thr[1]*f_iqr))
        residue_data[outlier_bool] = np.nan
    elif meth == 'std':
        f_std = np.nanstd(data)
        f_mean = np.nanmean(data)
        outlier_bool = ((data<(f_mean+thr[0]*f_std))|(data>(f_mean+thr[1]*f_std)))
        residue_data[(residue_data<(f_mean+thr[0]*f_std))|(residue_data>(f_mean+thr[1]*f_std))] = np.nan
    else:
        raise Exception('method should be ''iqr'' or ''abs'' or ''std''')
    n_removed = sum(i for i in outlier_bool if i) 
    return n_removed, residue_data

def listwise_clean(data):
    """
    Clean missing data by listwise method
    Parameters:
        data: raw data
    Return: 
        clean_data: no missing data
    """
    if isinstance(data, list):
        data = np.array(data)
    clean_data = pd.DataFrame(data).dropna().values
    return clean_data    

def ste(data, axis=None):
    """
    Calculate standard error
    --------------------------------
    Parameters:
        data: data array
    Output:
        standard error
    """
    if isinstance(data, float) | isinstance(data, int):
        return np.nanstd(data,axis)/np.sqrt(1)
    else:
        n = np.sum(~np.isnan(data),axis)
        ste = np.nanstd(data,axis)/np.sqrt(n)
        if isinstance(ste, np.ndarray):
            ste[np.isinf(ste)] = np.nan
        return ste

def get_specificroi(image, labellist):
    """
    Get specific roi from nifti image indiced by its label
    ----------------------------------------------------
    Parameters:
        image: nifti image data
        labellist: label you'd like to choose
    output:
        specific_data: data with extracted roi
    """
    logic_array = np.full(image.shape, False, dtype = bool)
    if isinstance(labellist, int):
        labellist = [labellist]
    for i,e in enumerate(labellist):
        logic_array += (image == e)
    specific_data = image*logic_array
    return specific_data

def make_lblmask_by_loc(mask, loclist, label = 1):
    """
    Generate a mask by loclist

    Parameters:
    -----------
    image: Provide a matrix as template, program will generate a same-shape mask as image
    loclist: location list

    Return:
    -------
    mask: output label mask

    Example:
    --------
    >>> mask = make_lblmask_by_loc(image, loclist)
    """
    try:
        mask[tuple(loclist),:] = label
    except IndexError:    
        mask = np.expand_dims(mask,axis=-1)
        mask[tuple(loclist),:] = label
        mask = mask[:,0]
    return mask

def list_reshape_bywindow(longlist, windowlen, step = 1):
    """
    A function to use window cut longlist into several pieces

    A list could be like below,
    [a, b, c, d, e]
    when windowlen as 2, step as 2,
    output as 
    [[a,b], [c,d]]
    when window len as 2, step as 1,
    output as 
    [[a,b], [b,c], [c,d], [d,e]]

    Parameters:
    -----------
    longlist: original long list
    windowlen: window length
    step: by default is 1
          window sliding step

    Returns:
    --------
    ic_list: intercept list

    Example:
    --------
    >>> ic_list = list_reshape_bywindow(longlist, windowlen = 3)
    """
    ic_list = []
    i = 0
    while len(longlist)>=(windowlen+step*i):
        ic_list.append(longlist[(step*i):(windowlen+step*i)])
        i+=1
    return ic_list

def lin_betafit(estimator, X, y, c, tail = 'both'):
    """
    Linear estimation using linear models
    -----------------------------------------
    Parameters:
        estimator: linear model estimator
        X: Independent matrix
        y: Dependent matrix
        c: contrast
        tail: significance tails
    Return:
        r2: determined values
        beta: slopes (scaled beta)
        t: tvals
        tpval: significance of beta
        f: f values of model test
        fpval: p values of f test 
    """
    try:
        from sklearn import preprocessing
    except ImportError:
        raise Exception('To call this function, please install sklearn')
    if isinstance(c, list):
        c = np.array(c)
    if c.ndim == 1:
        c = np.expand_dims(c, axis = 1)
    if X.ndim == 1:
        X = np.expand_dims(X, axis = 1)
    if y.ndim == 1:
        y = np.expand_dims(y, axis = 1)
    X = preprocessing.scale(X)
    y = preprocessing.scale(y)
    nsubj = X.shape[0]
    estimator.fit(X,y)
    beta = estimator.coef_.T
    y_est = estimator.predict(X)
    err = y - y_est
    errvar = (np.dot(err.T, err))/(nsubj - X.shape[1])
    t = np.dot(c.T, beta)/np.sqrt(np.dot(np.dot(c.T, np.linalg.inv(np.dot(X.T, X))),np.dot(c,errvar)))
    r2 = estimator.score(X,y)
    f = (r2/(X.shape[1]-1))/((1-r2)/(nsubj-X.shape[1]))
    if tail == 'both':
        tpval = stats.t.sf(np.abs(t), nsubj-X.shape[1])*2
        fpval = stats.f.sf(np.abs(f), X.shape[1]-1, nsubj-X.shape[1])*2
    elif tail == 'single':
        tpval = stats.t.sf(np.abs(t), nsubj-X.shape[1])
        fpval = stats.f.sf(np.abs(f), X.shape[1]-1, nsubj-X.shape[1])
    else:
        raise Exception('wrong pointed tail.')
    return r2, beta[:,0], t, tpval, f, fpval

def permutation_cross_validation(estimator, X, y, n_fold=3, isshuffle = True, cvmeth = 'shufflesplit', score_type = 'r2', n_perm = 1000):
    """
    An easy way to evaluate the significance of a cross-validated score by permutations
    -------------------------------------------------
    Parameters:
        estimator: linear model estimator
        X: IV
        y: DV
        n_fold: fold number cross validation
        cvmeth: kfold or shufflesplit. 
                shufflesplit is the random permutation cross-validation iterator
        score_type: scoring type, 'r2' as default
        n_perm: permutation numbers
    Return:
        score: model scores
        permutation_scores: model scores when permutation labels
        pvalues: p value of permutation scores
    """
    try:
        from sklearn import cross_validation
    except ImportError:
        raise Exception('To call this function, please install sklearn')
    if X.ndim == 1:
        X = np.expand_dims(X, axis = 1)
    if y.ndim == 1:
        y = np.expand_dims(y, axis = 1)
    X = preprocessing.scale(X)
    y = preprocessing.scale(y)
    if cvmeth == 'kfold':
        cvmethod = cross_validation.KFold(y.shape[0], n_fold, shuffle = isshuffle)
    elif cvmeth == 'shufflesplit':
        testsize = 1.0/n_fold
        cvmethod = cross_validation.ShuffleSplit(y.shape[0], n_iter = 100, test_size = testsize, random_state = 0)
    score, permutation_scores, pvalues = cross_validation.permutation_test_score(estimator, X, y, scoring = score_type, cv = cvmethod, n_permutations = n_perm)
    return score, permutation_scores, pvalues

class PCorrection(object):
    """
    Multiple comparison correction
    ------------------------------
    Parameters:
        parray: pvalue array
        mask: masks, by default is None
    Example:
        >>> pcorr = PCorrection(parray)
        >>> q = pcorr.bonferroni(alpha = 0.05) 
    """
    def __init__(self, parray, mask = None):
        if isinstance(parray, list):
            parray = np.array(parray)
        if mask is None:
            self._parray = np.sort(parray.flatten())
        else:
            self._parray = np.sort(parray.flatten()[mask.flatten()!=0])
        self._n = len(self._parray)
        
    def bonferroni(self, alpha = 0.05):
        """
        Bonferroni correction method
        p(k)<=alpha/m
        """
        return 1.0*alpha/self._n         

    def sidak(self, alpha = 0.05):
        """
        sidak correction method
        p(k)<=1-((1-alpha)**(1/m))
        """
        return 1.0-(1.0-alpha)**(1.0/self._n)
   
    def holm_bonferroni(self, alpha = 0.05):
        """
        Holm-Bonferroni correction method
        p(k)<=alpha/(m+1-k)
        """
        bool_array = [e>(alpha/(self._n-i)) for i,e in enumerate(self._parray)]
        if ~np.any(bool_array):
            return alpha
        else:
            return self._parray[np.argmax(bool_array)]
    
    def holm_sidak(self, alpha = 0.05):
        """
        Holm-Sidak correction method
        When the hypothesis tests are not negatively dependent
        p(k)<=1-(1-alpha)**(1/(m+1-k))
        """
        bool_array = [e>(1-(1-alpha)**(1.0/(self._n-i))) for i,e in enumerate(self._parray)]
        if ~np.any(bool_array):
            return alpha
        else:
            return self._parray[np.argmax(bool_array)]

    def fdr_bh(self, alpha = 0.05):
        """
        False discovery rate, Benjamini-Hochberg procedure
        Valid when all tests are independent, and also in various scenarios of dependence
        p(k) <= alpha*k/m
        FSL by-default option
        """
        bool_array = [e>(1.0*(i+1)*alpha/self._n) for i,e in enumerate(self._parray)]
        if ~np.any(bool_array):
            return alpha
        else:
            return self._parray[np.argmax(bool_array)]

    def fdr_bhy(self, alpha = 0.05, arb_depend = True):
        """
        False discovery rate, Benjamini-Hochberg-Yekutieli procedure
        p(k) <= alpha*k/m*c(m)
        if the tests are independent or positively correlated, c(m)=1, arb_depend = False
        in the case of negative correlation, c(m) = sum(1/i) ~= ln(m)+gamma+1/(2m), arb_depend = True, gamma = 0.577216
        """
        if arb_depend is False:   
            cm = 1
        else:
            gamma = 0.577216
            cm = np.log(self._n) + gamma + 1.0/(2*self._n)
        bool_array = [e>(1.0*(i+1)*alpha/(self._n*cm)) for i,e in enumerate(self._parray)] 
        if ~np.any(bool_array):
            return alpha
        else:
            return self._parray[np.argmax(bool_array)]

class NonUniformity(object):
    """
    Indices for non-uniformity
    -------------------------------
    Parameters:
        array: data arrays
    Example:
        >>> nu = NonUniformity(array)
    """
    def __init__(self, array):
        # normalize array to make it comparable
        self._array = array/sum(array)
        self._len = len(array)
    
    def entropy_meth(self):
        """
        Entropy method.
        Using Shannon Entropy to estimate non-uniformity, because uniformity has the highest entropy
        --------------------------------------------
        Parameters:
            None
        Output:
            Nonuniformity: non-uniformity index values
        Example:
            >>> values = nu.entropy_meth()
        """
        # create an array that have the highest entropy
        from math import log
        ref_array = np.array([1.0/self._len]*self._len)
        entropy = lambda array: -1*sum(array.flatten()*np.array([log(i,2) for i in array.flatten()]))
        ref_entropy = entropy(ref_array)
        act_entropy = entropy(self._array)
        nonuniformity = 1 - act_entropy/ref_entropy        
        return nonuniformity

    def l2norm(self):
        """
        Use l2 norm to describe non-uniformity.
        Assume elements in any vector sum to 1 (which transferred when initial instance), uniformity can be represented by L2 norm, which ranges from 1/sqrt(len) to 1.
        Here represented using: 
        (n*sqrt(d)-1)/(sqrt(d)-1)
        where n is the L2 norm, d is the vector length
        -----------------------------------------------
        Parameters:
            None
        Output:
            Nonuniformity: non-uniformity index values
        Example:
            >>> values = nu.l2norm()
        """
        return (np.linalg.norm(self._array)*np.sqrt(self._len)-1)/(np.sqrt(self._len)-1)

def threshold_by_number(imgdata, thr, threshold_type = 'number', option = 'descend'):
    """
    Threshold imgdata by a given number
    parameter option is 'descend', filter from the highest values
                        'ascend', filter from the lowest non-zero values
    Parameters:
        imgdata: image data
        thr: threshold, could be voxel number or voxel percentage
        threshold_type: threshold type.
                        'percent', threshold by percentage (fraction)
                        'number', threshold by node numbers
        option: default, 'descend', filter from the highest values
                'ascend', filter from the lowest values
    Return:
        imgdata_thr: thresholded image data
    Example:
        >>> imagedata_thr = threshold_by_number(imgdata, 100, 'number', 'descend')
    """
    if threshold_type == 'percent':
        voxnum = int(imgdata[imgdata!=0].shape[0]*thr)
    elif threshold_type == 'number':
        voxnum = thr
    else:
        raise Exception('Parameters should be percent or number')
    data_flat = imgdata.flatten()
    outdata_flat = np.zeros_like(data_flat)
    sortlist = np.sort(data_flat)[::-1]
    if option == 'ascend':
        data_flat[data_flat == 0] = sortlist[0]
    if option == 'descend':
        for i in range(voxnum):
            loc_flat = np.argmax(data_flat)
            outdata_flat[loc_flat] = sortlist[i]
            data_flat[loc_flat] = 0
    elif option == 'ascend':
        for i in range(voxnum):
            loc_flat = np.argmin(data_flat)
            outdata_flat[loc_flat] = sortlist[-1-i]
            data_flat[loc_flat] = sortlist[0]
    else:
        raise Exception('Wrong option inputed!')
    outdata = np.reshape(outdata_flat, imgdata.shape)
    return outdata

def threshold_by_value(imgdata, thr, threshold_type = 'value', option = 'descend'):
    """
    Threshold image data by values
    
    Parameters:
    -----------
    imgdata: activation image data
    thr: threshold, correponding to threshold_type
    threshold_type: 'value', threshold by absolute (not relative) values
                    'percent', threshold by percentage (fraction)
    option: 'descend', by default is 'descend', filter from the highest values
            'ascend', filter from the lowest values

    Return:
    -------
    imgdata_thr: thresholded image data
    
    Example:
    --------
    >>> imgdata_thr = threshold_by_values(imgdata, 2.3, 'value', 'descend')
    """
    if threshold_type == 'percent':
        if option == 'descend':
            thr_val = np.max(imgdata) - thr*(np.max(imgdata) - np.min(imgdata))
        elif option == 'ascend':
            thr_val = np.min(imgdata) + thr*(np.max(imgdata) - np.min(imgdata))
        else:
            raise Exception('No such parameter in option')
    elif threshold_type == 'value':
        thr_val = thr
    else:
        raise Exception('Parameters should be value or percent')
    if option == 'descend':
        imgdata_thr = imgdata*(imgdata>thr_val)
    elif option == 'ascend':
        imgdata_thr = imgdata*(imgdata<thr_val)
    else:
        raise Exception('No such parameter in option')
    return imgdata_thr

def control_lbl_size(labeldata, actdata, thr, label = None,  option = 'num'):
    """
    Threshold label data using activation mask (threshold activation data then binarized it to get mask to restrained raw label data range)
    
    Parameters:
    -----------
    labeldata: label data
    actdata: activation data, the activation data should correspond to label data
    thr: threshold, corresponding to parameter of option
    option: 'num', threshold value as vertex numbers, get mask with the largest values of thr-th vertices, by default is 'num' 
            'value', threshold with activation values, anywhere values smaller than thr will not be covered
            'percent_num', percentage of vertex numbers with the largest activation values
            'percent_value', percentage of activation values with the largest activation values

    Return:
    -------
    out_lbldata: new label data with region been thresholded        

    Example:
    --------
    >>> out_lbldata = control_lbl_size(labeldata, actdata, 125, label = 1, 'num')
    """
    # threshold activation data
    actdata = actdata*(labeldata == label)

    if option == 'num':
        outactdata = threshold_by_number(actdata, thr, 'number')
    elif option == 'value':
        outactdata = threshold_by_value(actdata, thr, 'value')
    elif option == 'percent_num':
        outactdata = threshold_by_number(actdata, thr, 'percent')
    elif option == 'percent_value':
        outactdata = threshold_by_value(actdata, thr, 'percent')
    else:
        raise Exception('No such option')

    out_lbldata = labeldata*(outactdata!=0)
    return out_lbldata

def permutation_corr_diff(r1_data, r2_data, n_permutation = 5000, method = 'pearson', tail = 'single'):
    """
    Do permutation test between r1_data and r2_data to check whether the difference between r1 (correlation coefficient) calculated from r1data and r2 calculated from r2data will be significant

    Parameters:
    -----------
    r1_data: raw data for correlation of r1
             the format of raw data as a array of Nx2, where N is the number of data point. The first column is data of x, the second column is data of y. r1 computed by x and y.
    r2_data: raw data for correlation of r2
    n_permutation: permutation times, by default is 5,000
    method: method for correlation and coefficient
            'pearson', pearson correlation coefficient
            'spearman', spearman correlation coefficient
    tail: 'single', one-tailed test
          'both', two-tailed test

    Return:
    -------
    r_dif: difference between r1 and r2
    permutation_scores: Scores obtained for each permutations
    pvalue: p values

    Example:
    ---------
    >>> r_dif, permutation_scores, pvalue = permutation_corr_diff(r1_data, r2_data)
    """
    import random
    if method == 'pearson':
        corr_method = stats.pearsonr
    elif method == 'spearman':
        corr_method = stats.spearman
    elif method == 'icc':
        corr_method = icc
    else:
        raise Exception('Only support pearson, spearman or intra-class correlation')
    r1 = corr_method(r1_data[:,0], r1_data[:,1])
    r2 = corr_method(r2_data[:,0], r2_data[:,1])
    if method == 'icc':
        r_dif = r1 - r2
    else:
        r_dif = r1[0] - r2[0]
    merged_data = np.concatenate((r1_data, r2_data))
    total_pt = merged_data.shape[0]
    permutation_scores = []
    for i in range(n_permutation):
        rd_lbl1 = tuple(np.sort(random.sample(range(total_pt), r1_data.shape[0])))
        rd_lbl2 = tuple([i for i in range(total_pt) if i not in rd_lbl1])
        r1_rddata = merged_data[rd_lbl1,:]
        r2_rddata = merged_data[rd_lbl2,:]
        r1_rd = corr_method(r1_rddata[:,0], r1_rddata[:,1])
        r2_rd = corr_method(r2_rddata[:,0], r2_rddata[:,1])
        if method == 'icc':
            permutation_scores.append(r1_rd - r2_rd)
        else:
            permutation_scores.append(r1_rd[0] - r2_rd[0])
    permutation_scores = np.array(permutation_scores)    
    if tail == 'single':
        pvalue = 1.0*(sum(permutation_scores>r_dif)+1)/(n_permutation+1)
    elif tail == 'both':
        pvalue = 1.0*(sum(np.abs(permutation_scores)>np.abs(r_dif))+1)/(n_pemutation+1)
    return r_dif, permutation_scores, pvalue

def permutation_diff(list1, list2, dist_method = 'mean', n_permutation = 1000, tail = 'single'):
    """
    Make permutation test for the difference of mean values between list1 and list2

    Parameters:
    -----------
    list1, list2: two lists contain data
    dist_method: 'mean', the difference between average of list1 and list2
                 'std', the difference between std of list1 and list2
                 -------------------Note-----------------------------------
                 The significance of correlation is from two-tailed test,
                 if you use 'pearson' or 'icc' as distance method, tail will
                 be set as 'both' compulsorily.
                 ----------------------------------------------------------
                 'pearson', the pearson correlation between list1 and list2
                 'icc', the intra-class correlation between list1 and list2
    n_permutation: permutation times
    tail: 'single', one-tailed test
          'both', two_tailed test
    
    Output:
    -------
    list_diff: difference between list1 and list2
    diff_scores: different values after the permutation
    pvalue: pvalues

    Examples:
    ----------
    >>> list_diff, diff_scores, pvalue = permutation_diff(list1, list2)
    """
    if not isinstance(list1, list):
        list1 = list(list1)
    if not isinstance(list2, list):
        list2 = list(list2)
    if (dist_method == 'pearson') or (dist_method == 'icc'):
        tail = 'both'
    list_diff = _dist_func(list1, list2, dist_method)
    list1_len = len(list1)
    list2_len = len(list2)
    list_total = np.array(list1+list2)
    list_total_len = len(list_total)
    
    diff_scores = []
    for i in range(n_permutation):
        list1_perm_idx = np.sort(np.random.choice(range(list_total_len), list1_len, replace=False))
        list2_perm_idx = np.sort(list(set(range(list_total_len)).difference(set(list1_perm_idx))))
        list1_perm = list_total[list1_perm_idx]
        list2_perm = list_total[list2_perm_idx]
        diff_scores.append(_dist_func(list1_perm, list2_perm, dist_method))
    if tail == 'single':
        pvalue = 1.0*(np.sum(diff_scores>list_diff)+1)/(n_permutation+1)
    elif tail == 'both':
        pvalue = 1.0*(np.sum(np.abs(diff_scores)>np.abs(list_diff))+1)/(n_permutation+1)
    return list_diff, diff_scores, pvalue

def _dist_func(list1, list2, dist_method = 'mean'):
    """
    An distance function for effect size of difference between list1 and list2
    """
    if dist_method == 'mean':
        diff_list = np.nanmean(list1) - np.nanmean(list2)
    elif dist_method == 'std':
        diff_list = np.nanstd(list1) - np.nanstd(list2)
    elif dist_method == 'pearson':
        assert len(list1) == len(list2), "The length of list1 and list2 must be same."
        diff_list = stats.pearsonr(list1, list2)[0]
    elif dist_method == 'icc':
        diff_list = icc(list1, list2)
    else:
        raise Exception('No such a option as dist_method!')
    return diff_list


def genroi_bytmp(raw_roi, template, thr, thr_idx = 'values', threshold_type = 'value', option = 'descend'):
    """
    Generate ROI map with different threshold by using a template with values

    Parameters:
    -----------
    raw_roi: original roi map
    template: a template with values, as our reference to threshold raw_roi
    thr: threshold
    thr_idx: 'values' or 'numbers'
             'values', threhold by values
             'numbers', threshold by vertex numbers (or voxel numbers)
    threshold_type: 'value' (by default), threshold by absolute (not relative) values
                    'percent', threhold by percentage (fraction)
    option: 'descend', by default is 'descend', filter from the highest values
            'ascend', filter from the lowest values

    Returns:
    --------
    new_roi: thresholded roi map

    Example:
    >>> new_roi = genroi_bytmp(raw_roi, template, thr = 4.0)
    """
    assert raw_roi.shape == template.shape, "the shape of raw_roi should be equal to template"

    if thr_idx == 'values':
        thrmethod = threshold_by_value
    elif thr_idx == 'numbers':
        thrmethod = threshold_by_number
    else:
        raise Exception('threshold by values or numbers, bad parameters were input')

    thr_template = thrmethod(template, thr, threshold_type = threshold_type, option = option)

    thr_template[thr_template!=0] = 1
    new_roi = raw_roi * thr_template
    return new_roi

def autocorr(x, t = 0, mode = 'point'):
    """
    Calculate the statistical correlation for a lag of t

    Paramters:
    ----------
    x: array, time series
    t: int, lag time, by default is t = 0
       Note that the largest t = len(x)-2
    mode: string, 'point': compute autocorrelation of one lag point
                  'fullcut': compute autocorrelation with whole points (without circular series) 
                  'circcut': compute autocorrelation with whole points (with circular series)

    Returns:
    --------
    r: pearson coefficient
    p: significance value
    """
    if mode == 'point':
        assert t<len(x)-1, "the largest t = len(x)-2, please be note of it"
        r, p = stats.pearsonr(x[0:len(x)-t], x[t:len(x)])
    elif mode == 'fullcut':
        r = []
        p = []
        for t0 in range(len(x)-1):
            r0, p0 = stats.pearsonr(x[0:len(x)-t0], x[t0:len(x)])
            r.append(r0)
            p.append(p0)
        r = np.array(r)
        p = np.array(p)
    elif mode == 'circcut':
        r = []
        p = []
        for t0 in range(len(x)):
            r0, p0 = stats.pearsonr(np.roll(x, t0), np.roll(x, len(x)-t0))
            r.append(r0)
            p.append(p0)
        r = np.array(r)
        p = np.array(p)
    else:
        raise Exception('No such a mode name')
    return r, p

def template_overlap(roimask, template, index='percent'):
    """
    Find the subregion of roimask that defined from template

    Parameters:
    -----------
    roimask: roi mask, several ROIs contained (max label: M). 
    template: template used for providing subregion (max label: N).
    index: 'percent', measured by percentage
           'dice', measured by dice coefficient

    Returns:
    --------
    label_dict, dict: record labels as subregion which belongs to template
    overlap_array, M*N array: record overlap values of each label.
    """
    roimask = roimask.astype('int')
    template = template.astype('int')

    overlap_array = np.zeros((np.max(roimask), np.max(template)))

    for i in np.sort(np.unique(roimask)[1:]):
        for j in np.sort(np.unique(template)[1:]):
            overlap_array[i-1, j-1] = calc_overlap(roimask, template, i, j, index = index)

    row, column = np.where(overlap_array)
    row += 1
    column += 1
    
    label_dict = {}
    for i in np.unique(row):
        label_dict[i] = column[row==i]

    return label_dict, overlap_array

def rearrange_matrix(matrix_data, index_list):
    """
    Rearrange square matrix by index list

    Parameters:
    ------------
    matrix_data(NxN matrix): square matrix data    
    index_list: index list

    Returns:
    --------
    rag_matrix: rearrange square matrix

    Examples:
    ----------
    >>> diag_data = [1,2,3,4]
    >>> matrix_data = np.diagflat(diag_data)
    >>> matrix_data
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]]) 
    >>> rag_matrix = rearrange_matrix(matrix_data, [2,3,1,0])
    >>> rag_matrix
    array([[3, 0, 0, 0],
           [0, 4, 0, 0], 
           [0, 0, 2, 0],
           [0, 0, 0, 1]])
    """
    assert matrix_data.ndim == 2, "Input a NxN matrix."
    assert matrix_data.shape[0] == matrix_data.shape[1], "row number should be equal to column number."
    index_list = np.array(index_list)
    tmp_data = matrix_data[index_list,:]
    rag_matrix = tmp_data[:,index_list]
    return rag_matrix

def anova_decomposition(Y):
    """
    Decompositing variance of dataset Y into Mean Square and its corresponded dof.

    The data Y are entered as a 'table' with subjects (targets) are in rows
    and repeated measure (judges) in columns

    Reference: P.E. Shrout & Joseph L. Fleiss (1979). "Intraclass Correlations: Uses in Assessing Rater Reliability". Psychological Bulletin 86 (2): 420-428.

    Source of variance: SST = SSW + SSB; SSW = SSBJ + SSE
    """
    [n_subjects, n_conditions] = Y.shape
    dfbt = n_subjects - 1
    dfbj = n_conditions - 1
    dfwt = n_subjects*dfbj
    dfe = dfbt * dfbj
    # SST
    mean_Y = np.mean(Y)
    SST = ((Y-mean_Y)**2).sum()
    # WMS (within-target mean square)
    Avg_WithinTarg = np.tile(np.mean(Y, axis=1), (n_conditions, 1)).T
    SSW = ((Y - Avg_WithinTarg)**2).sum()
    WMS = 1.0*SSW/dfwt
    # BMS (between-target mean square)
    SSB = ((Avg_WithinTarg - mean_Y)**2).sum()
    BMS = 1.0*SSB/dfbt
    # BJMS 
    Avg_BetweenTarg = np.tile(np.mean(Y,axis=0), (n_subjects, 1))
    SSBJ = ((Avg_BetweenTarg - mean_Y)**2).sum()
    BJMS = 1.0*SSBJ/dfbj
    # EMS
    SSE = SST - SSBJ - SSB
    EMS = 1.0*SSE/dfe
    
    # Package variables
    Output = {}
    Output['WMS'] = WMS
    Output['BMS'] = BMS
    Output['BJMS'] = BJMS
    Output['EMS'] = EMS
    Output['dof_bt'] = dfbt
    Output['dof_wt'] = dfwt
    Output['dof_bj'] = dfbj
    Output['dof_e'] = dfe
     
    return Output

def icc(Y, methods = '(2,1)'):    
    """
    Intra-correlation coefficient.
    The data Y are entered as a 'table' with subjects (targets) are in rows,
    and repeated measure (judges) in columns
    
    Reference: P.E. Shrout & Joseph L. Fleiss (1979). "Intraclass Correlations: Uses in Assessing Rater Reliability". Psychological Bulletin 86 (2): 420-428.

    Parameters:
    -----------
    Y: Original dataset, with its rows are targets and columns are judges.
    methods: Please see attached reference for details.
             (1,1), One-random effects
                    Each target is rated by a different set of k judges, 
                    randomly selected from a larger population of judges.
             (2,1), Two-way random effects
                    A random sample of k judges is selected from a larger 
                    population, and each judge rates each target, that is,
                    each judge rates n targets altogether.
             (3,1), Two-way mixed model
                    Each target is rated by each of the same k judges, who
                    are only judges of interest.

    Return: 
    -------
    r: intra-class correlation
    """
    decomp_var = anova_decomposition(Y)
    [n_targs, n_judges] = Y.shape
    if methods == '(1,1)':
        r = (decomp_var['BMS'] - decomp_var['WMS'])/(decomp_var['BMS']+(n_judges-1)*decomp_var['WMS'])
        F = decomp_var['BMS']/decomp_var['WMS']
        p = stats.f.sf(F, n_targs-1, n_targs*(n_judges-1))
    elif methods == '(2,1)':
        r = (decomp_var['BMS'] - decomp_var['EMS'])/(decomp_var['BMS']+(n_judges-1)*decomp_var['EMS']+n_judges*(decomp_var['BJMS']-decomp_var['EMS'])/n_targs)
        F = decomp_var['BMS']/decomp_var['EMS']
        p = stats.f.sf(F, n_targs-1, (n_judges-1)*(n_targs-1))
    elif methods == '(3,1)':
        r = (decomp_var['BMS'] - decomp_var['EMS'])/(decomp_var['BMS']+(n_judges-1)*decomp_var['EMS'])
        F = decomp_var['BMS']/decomp_var['EMS']
        p = stats.f.sf(F, n_targs-1, (n_targs-1)*(n_judges-1))
    else:
        raise Exception('Not support this method.')
    return r, p
