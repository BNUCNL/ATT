# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import nibabel as nib
import os
import pickle
from scipy.io import savemat, loadmat
import csv

pjoin = os.path.join

def make_ioinstance(filename, filepath = '.'):
    """
    A function to pack IO factory, make it easier to use

    Parameters:
    -----------
    filename: filename
    filepath: filepath, by default is '.'

    Return:
    -------
    ioinstance: input/output instance contain save and load method

    Example:
    --------
    >>> ioinstance = make_ioinstance('data.csv')
    >>> ioinstance.save(data)
    """
    factory = _IOFactory()
    return factory.createfactory(filename, filepath)

class _IOFactory(object):
    """
    Make a factory for congruent read/write data
    Usage:
        >>>factory = iofiles.IOFactory()
        >>>factory.createfactory('data.csv')
    """
    def createfactory(self, filename, filepath = '.'):
        """
        Create your factory
        ----------------------------------------
        Input:
            filename: filenames
            filepath: filepath as reading/writing, by default is '.'
        Output: 
            A class
   
        Note:
            What support now is .csv, .pkl, .mat .nifti, .label
            # Note, we can read .gifti & .cifti data but can't save that
        """
        _comp_file = pjoin(filepath, filename)
        _lbl_cifti = False
        if _comp_file.endswith('csv'):
            return _CSV(_comp_file)
        elif _comp_file.endswith('txt'):
            return _TXT(_comp_file)
        elif _comp_file.endswith('pkl'):
            return _PKL(_comp_file)
        elif _comp_file.endswith('mat'):
            return _MAT(_comp_file)
        elif _comp_file.endswith('dscalar.nii') | _comp_file.endswith('dtseries.nii') | _comp_file.endswith('ptseries.nii') | _comp_file.endswith('dlabel.nii'):
            _lbl_cifti = True
            return _CIFTI(_comp_file)
        elif _comp_file.endswith('nii.gz') | (_comp_file.endswith('nii') & (_lbl_cifti is False)):
            return _NIFTI(_comp_file)
        elif _comp_file.endswith('gii'):
            return _GIFTI(_comp_file)
        elif _comp_file.endswith('label'):
            return _LABEL(_comp_file)
        else:
            return None

class _CSV(object):
    def __init__(self, _comp_file):
	    self._comp_file = _comp_file

    def save(self, data, labels = None):
        """
        Save a np array into a csv file.
        ---------------------------------------------
        Parameters:
            data: raw data
            labels: Data names. Labels as a list.
        """
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = np.expand_dims(data,axis=1)
            try:
                f = open(self._comp_file, 'w')
            except IOError:
                print('Can not save file' + self._comp_file)
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
            raise Exception('Input must be a numpy array.')

    def load(self, dtype = 'float'):
        """
        Read data from .csv
        ----------------------------------
        Parameters:
            dtype: read data as specific type, by default is 'float'
        """
        csv_reader = file(self._comp_file, 'rb')
        reader = csv.reader(csv_reader)
        outdata = []
        for row in reader:
            outdata.append(row)
        outdata = np.array(outdata)
        outdata = outdata.astype(dtype)
        return outdata

class _TXT(object):
    def __init__(self, _comp_file):
        self._comp_file = _comp_file
    
    def save(self, data):

        """
        Save data to .txt
        ------------------------
        Parameters:
            data: raw data
        """
        np.savetxt(self._comp_file, data)
    
    def load(self):
        """
        Load .txt data
        ------------------------
        """
        return np.loadtxt(self._comp_file)

class _PKL(object):
    def __init__(self, _comp_file):
        self._comp_file = _comp_file

    def save(self, data):
        """
        Save data to .pkl
        ----------------------------
        Parameters:
            data: raw data
        """
        output_class = open(self._comp_file, 'wb')
        pickle.dump(data, output_class)
        output_class.close()

    def load(self):
        """
        Load data from .pkl
        ------------------------------
        Parameters:
            filename: file name
            path: path of pointed pickle file
        Return:
            data
        """
        pkl_file = open(self._comp_file, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        return data


class _MAT(object):
    def __init__(self, _comp_file):
        self._comp_file = _comp_file

    def save(self, data):
        """
        Save data to .mat
        ---------------------------------------
        Parameters:
            data: raw data dictionary, note that data must be a dictionary data
        """
        savemat(self._comp_file, data)     

    def load(self):
        """
        Load data from .mat
        ---------------------------------------
        Parameters:
            outdata: output data
        """
        return loadmat(self._comp_file)

class _NIFTI(object):
    def __init__(self, _comp_file):
        self._comp_file = _comp_file

    def save(self, data, header):
        """
        Save nifti data
	Parameters:
            data: saving data
        """
        img = nib.Nifti1Image(data, None, header)
        nib.save(img, self._comp_file)

    def load(self, datatype = 'data'):
        """
        Load nifti data.
        Parameters:
        -----------------------------------
        datatype: data type to load.
                  By default, 'data', nifti image values
                  'affine', affine matrix
                  'header', header
                  'shape', matrix shapes
        """
        img = nib.load(self._comp_file)
        if datatype == 'data':
            outdata = img.get_data()
        elif datatype == 'affine':
            outdata = img.get_affine()
        elif datatype == 'header':
            outdata = img.get_header()
        elif datatype == 'shape':
            outdata = img.get_shape()
        else:
            raise Exception('Wrong datatype input')
        return outdata

class _CIFTI(object):

    def __init__(self, _comp_file):
        self._comp_file = _comp_file
    
    def get_brain_structure(self):
        """ 
        Return brain structure

        Parameters:
        -----------
        None

        Return:
        -------
        brain_structure[list]: brain_structure
        """
        img = nib.load(self._comp_file)

        header = img.header
        index_map = header.get_index_map(1)
        brain_models = [i for i in index_map.brain_models]
        brain_structure = [i.brain_structure for i in brain_models]

        return brain_structure                

    def load(self, structure = None):
        """
        Read cifti data. If your cifti data contains multiple contrast, you can input your contrast number and get value of this contrast.
 
        Parameters:
        --------------
        structure[string]: structure name

        Return:
        -------
        data[array]: cifti data
        vxidx[object]: vertex indices table
                       For matching vertex indices in surface, use vxidx.index(NUM) to get index in data. e.g. vxidx.index(32491) = 29695 in left cortex.

        """

        img = nib.load(self._comp_file)
        data = img.get_data()


        header = img.header
        index_map = header.get_index_map(1)
        brain_models = [i for i in index_map.brain_models]

        brain_structure = [i.brain_structure for i in brain_models]
        index_count = [i.index_count for i in brain_models]
        index_offset = [i.index_offset for i in brain_models]
        vertex_indices = [i.vertex_indices for i in brain_models]
        if structure is None:
            pass
        elif structure in brain_structure:
            idx = brain_structure.index(structure)
            data = data[:, (index_offset[idx]):(index_offset[idx]+index_count[idx])]
            vxidx = vertex_indices[idx]
        else:
            raise Exception('No such a structure in brain model')

        return data, vxidx
   
    def save(self, data, header):
        """
        Save cifti file

        Parameters:
        -----------
        data: output data
        header: cifti header
        """
        img = nib.Cifti2Image(data, header)
        img.to_filename(self._comp_file)
 
class _GIFTI(object):
    def __init__(self, _comp_file):
        self._comp_file = _comp_file
        
    def load(self):
        """
        read gifti data
        """
        img = nib.load(self._comp_file)
        if len(img.darrays) == 1:
            data = img.darrays[0].data
        else:
            data = []
            for i in range(0,len(img.darrays)):#files named *.midthickness may have two elements in darrays. one represents the mesh, and one represents the coordinates of vertex
                data.append(img.darrays[i].data)
        return data

    def save(self, data, hemisphere = None):
        """
        A method to save vector data as gifti image

        Save output name with [].func.gii or [].shape.gii

        Parameters:
        -----------
        data: vector to save as gifti file
        hemisphere: 'CortexLeft' or 'CortexRight', depending on hemisphere
        """
        assert hemisphere in ['CortexLeft', 'CortexRight']
        nvPair = nib.gifti.GiftiNVPairs(name='AnatomicalStructurePrimary', value=hemisphere)
        metaData = nib.gifti.GiftiMetaData(nvPair)

        data = np.squeeze(data.astype(np.float32))
        img = nib.gifti.GiftiImage(meta=metaData)
        gda = nib.gifti.GiftiDataArray(data=data)
        img.add_gifti_data_array(gda)

        nib.save(img, self._comp_file)
          

class _LABEL(object):
    def __init__(self, _comp_file):
        self._comp_file = _comp_file

    def load(self):
        """
        Load label data
        """
        label = nib.freesurfer.read_label(self._comp_file)
        return label

    def save(self, label, coords):
        """
        Save label data
        """
        with open(self._comp_file, 'w') as f:
            f.write('%d\n' % len(label))
            for lbl in label:
                x, y, z = coords[lbl]
                f.write('%d %f %f %f 0.000000\n' % (lbl, x, y, z))

