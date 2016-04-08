# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:

import numpy as np
import nibabel as nib
import os


def combine_sessid(id1, id2, method):
    """
    Input:
        id1, id2: id you want to combined with. Note that they can be 
                  a string or a list
        method: 'int' for intersection. 'union' for union. 'sub' for 
                substrate
    Output:
        sid: new sessid 
    """
    if isinstance(id1, str):
        id1 = open(id1, 'rb').read().splitlines()
    if isinstance(id2, str):
        id2 = open(id2, 'rb').read().splitlines()

    if method == 'int':
        sid = [val for val in id1 if val in id2]
    elif method == 'union':
        sid = list(set(id1 + id2))
    elif method == 'sub':
        sid = [val for val in id1 if val not in id2]
    else:
        raise 'error'
    return sid

def CalcEuclid(meas, sel, oripoint = [0,0,0]):
    """
    Calculate Euclid Distance.
    Formula: np.sqrt((x - x0)^2 + (y - y0)^2 + (z - z0)^2)
    --------------------------------
    Input:
         meas: subj x coordinate(x, y, z)
         sel: selection
         oripoint: original point, x0, y0, z0
    """
    outdist = np.empty((meas.shape[0], len(sel)/3))
    for i in range(len(sel)/3):
        outdist[:,i] = np.sqrt(np.square(meas[:, sel[3*i]] - oripoint[0]) + np.square(meas[:, sel[3*i+1]] - oripoint[1]) + np.square(meas[:, sel[3*i+2]] - oripoint[2]))
    return outdist

def calcorr(array1, array2, method = 'pearson'):
    """
    Calculate correlation between two array
    Note that array1.shape[0] should equal to array2.shape[0]
    ----------------------------------
    Input:
        array1: subj x features
        array2: subj x features
        method: do pearson correlation or spearman correlation
    """
    outcorr = np.empty((array1.shape[1], array2.shape[1]))
    outpval = np.copy(outcorr)
    n_sample = np.copy(outcorr)
    for i in range(array1.shape[1]):
        for j in range(array2.shape[1]):
            samp_sel = ~np.isnan(array1[:,i] * array2[:,j])
            n_sample[i, j] = np.count_nonzero(samp_sel)
            if method == 'pearson':
                [c, p] = stats.pearsonr(array1[samp_sel, i], array2[samp_sel, j])
            elif method == 'spearman':
                [c, p] = stats.spearmanr(array1[samp_sel, i], array2[samp_sel, j])
            else:
                raise Exception('Only support pearson or spearman correlation now!')
            outcorr[i, j] = c
            outpval[i ,j] = p
    return outcorr, outpval

def plot_corr(meas1, meas2, labels, method = 'pearson'):
    """
    Plot scatter and correlation picture between two arrays
    -------------------------------
    Parameters:
    meas1, meas2: two arrays
    labels: labels[0] is label of meas1, labels[1] is label of meas2
    method: Do pearson or spearman correlation
    """
    samp_sel = ~np.isnan(meas1 * meas2)
    x = meas1[samp_sel]
    y = meas2[samp_sel]
    if method == 'pearson':
        [corr, pval] = stats.pearsonr(x, y)
    elif method == 'spearman':
        [corr, pval] = stats.spearmanr(x, y)
    else:
        raise Exception('please point method as pearson or spearman')
    fig, ax = plt.subplots()
    plt.scatter(x, y)
    plt.plot(x, np.poly1d(np.polyfit(x,y,1))(x))
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))
    ax.text(x0+0.1*(x1-x0), y0+0.9*(y1-y0), 'r = %.3f, p = %.3f' % (corr, pval))
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title('Correlation')
    plt.show()

def sphere_roi(data, x, y, z, radius, value, atlas):
    """
    Generate a sphere roi which center in (x,y,z) 
    -------------------------------------------
    Parameters:
    data: targ_data
    x,y,z: peak coordinate of vox_x, vox_y, vox_z
    radius: sphere's radius
    value: area label - 1
    atlas_data: atlas data
    """
    for n_x in range(int(x-radius[0]), int(x+radius[0]+1)):
        for n_y in range(int(y-radius[1]), int(y+radius[1]+1)):
            for n_z in range(int(z-radius[2]), int(z+radius[2]+1)):
                n_coord = np.array((n_x, n_y, n_z))
                coord = np.array((x, y, z))
                minus = coord - n_coord
                if (np.square(minus) / np.square(np.array(radius)).astype(np.float)).sum() <= 1:
                    try:
                        if atlas[n_x, n_y, n_z] == value:
                            data[n_x, n_y, n_z] = value
                    except IndexError:
                        pass
    return data

def run_glocorr(temp1, temp2, mask, outpath, outname, method = 'pearson'):
    """
    Do global correlation between two 4D data
    Parameters
    ------------
    temp1, temp2: two 4D data
    mask: global mask (If you want to get roi's result, change your mask)
    outpath: output path
    outname: output name
    method: Do pearson or spearman correlation
    """

    if isinstance(mask, str):
        mask = nib.load(mask)
    if isinstance(temp1, str):
        temp1 = nib.load(temp1).get_data()
    if isinstance(temp2,str):
        temp2 = nib.load(temp2).get_data()
    header = mask.get_header()
    header.set_data_dtype('double')
    maskdata = mask.get_data()
    rvalue = np.empty(mask.shape)
    pvalue = np.empty(mask.shape)
    if method == 'pearson':
        corrfunc = stats.pearsonr
    elif method == 'spearman':
        corrfunc = stats.spearmanr
    else:
        raise Exception('Only support pearson or spearman now')
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                (r, p) = corrfunc(temp1[i,j,k,:], temp2[i,j,k,:])
                rvalue[i, j, k] = r
                pvalue[i, j, k] = p
    
    rvalue = rvalue*maskdata
    pvalue = pvalue*maskdata
    rvalue[np.isnan(rvalue)] = 0
    
    rvalue_img = nib.Nifti1Image(rvalue, None, header)
    pvalue_img = nib.Nifti1Image(pvalue, None, header)
    
    nib.save(rvalue_img, os.path.join(outpath, outname+'r.nii.gz'))
    nib.save(pvalue_img, os.path.join(outpath, outname+'p.nii.gz'))

def comp_similar(temp1, temp2, mask, method = 'pearson'):
    """
    Compare similarity between two image
    Parameters
    -----------------
    temp1, temp2: two target image
    mask: only compare voxel values containing in mask
    method: method
    Returns
    -----------------
    corr: similarity between two image
    """
    if isinstance(temp1, str):
        temp1 = nib.load(temp1).get_data()
    if isinstance(temp2, str):
        temp2 = nib.load(temp2).get_data()
    if isinstance(mask, str):
        mask = nib.load(mask).get_data()
    temp1_mask = temp1[mask == 1]
    temp2_mask = temp2[mask == 1]
    if method == 'pearson':
        corr = stats.pearsonr(temp1_mask, temp2_mask)
    else:
        raise Exception('Only support pearson correlation')
    return corr[0]


