# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import nibabel as nib
import os
import cPickle
from scipy.io import savemat

def save2csv(data, csv_file):
    """
    Save a 1/2D list data into a csv file.
    ------------------------------------
    Parameters:
        data: raw data
        csv_file: csv file name
    """
    if isinstance(data, list):
        try:
            f = open(csv_file, 'w')
        except IOError:
            print('Can not save file' + csv_file)
        else:
            for line in data:
                if isinstance(line, list):
                    line_str = [str(item) for item in line]
                    line_str = ','.join(line_str)
                else:
                    line_str = str(line)
                f.write(line_str + '\n')
            f.close()
    else:
        raise ValueError, 'Input must be a list.'        

def nparray2csv(data, labels = None, csv_file = None):
    """
    Save a np array into a csv file.
    ---------------------------------------------
    Parameters:
        data: raw data
        labels: Data names. Labels as a list.
        csv_file: csv filename.
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = np.expand_dims(data,axis=1)
        try:
            f = open(csv_file, 'w')
        except IOError:
            print('Can not save file' + csv_file)
        else:
            if isinstance(labels, list):
                labels = [str(item) for item in labels]
                labels = ','.join(labels)
                f.write(labels + '\n')
            for line in data:
                line_str = [str(item) for item in line]
                line_str = ','.join(line_str)
                f.write(line_str + '\n')
            f.close()
    else:
        raise ValueError, 'Input must be a numpy array.'

def save_pkl(data, outname, path = '.'):
    """
    Save data to .pkl
    ----------------------------
    Parameters:
        data: raw data
        outname: output filename
        path: output path
    """
    output = open(os.path.join(path, outname), 'wb')
    cPickle.dump(data, output)
    output.close()

def load_pkl(filename, path = '.'):
    """
    Load data from .pkl
    ------------------------------
    Parameters:
        filename: file name
        path: path of pointed pickle file
    Return:
        data
    """
    pkl_file = open(os.path.join(path, filename), 'rb')
    data = cPickle.load(pkl_file)
    pkl_file.close()
    return data

def save_mat(data_dict, outname, path = '.'):
    """
    Save data to .mat
    ---------------------------------------
    Parameters:
        data_dict: raw data dictionary
        outname: output filename (Note to add .mat as suffix)
        path: output path
    """
    savemat(os.path.join(path, outname), data_dict)     

def save_nifti(data, header, outname, outpath = './'):
    """
    Just for simplying operation, I write this function
    Parameters:
        data: saving data
        outname: file name
        outpath: saving folder
    """
    img = nib.Nifti1Image(data, None, header)
    nib.save(img, os.path.join(outpath, outname))




