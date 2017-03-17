# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:

import numpy as np
import os
import nibabel as nib
import copy
from ATT.algorithm import roimethod
from ATT.algorithm import tools
from ATT.iofunc import iofiles

class ImageCalculator(object):
    def __init__(self):
        pass

    def merge4D(self, rawdatapath, outdatapath, outname, issave = True):    
        """
        Merge 3D images together
        --------------------------------------
        Parameters:
            rawdatapath: raw data path. Need to be a list contains path of each image
            outdatapath: output path.
            outname: output data name.
            issave: save data or not, by default is True
        Return:
            outdata: merged file
        """
        if isinstance(rawdatapath, np.ndarray):
            rawdatapath = rawdatapath.tolist()
        header = nib.load(rawdatapath[0]).get_header()
        datashape = nib.load(rawdatapath[0]).get_data().shape
        nsubj = len(rawdatapath)
        outdata = np.zeros((datashape[0], datashape[1], datashape[2], nsubj))
        for i in range(nsubj):
            if os.path.exists(rawdatapath[i]):
                outdata[...,i] = nib.load(rawdatapath[i]).get_data()
            else:
                raise Exception('File may not exist of %s' % rawdatapath[i])
        img = nib.Nifti1Image(outdata, None, header)
        if issave is True:
            if outdatapath.split('/')[-1].endswith('.nii.gz'):
                nib.save(img, outdatapath)
            else:
                outdatapath_new = os.path.join(outdatapath, outname)
                nib.save(img, outdatapath_new)
        return outdata

    def decompose_img(self, imagedata, header, outpath, outname=None):
        """
        Decompose 4D image into multiple 3D image to make it easy to show or analysis
        --------------------------------------------------------------------
        Parameters:
            imagedata: 4D image, the 4th dimension is the axis to decompose
            header: image header
            outpath: outpath
            outname: outname, should be a list. By default is None, system will distribute name into multiple 3D images automatically. If you want to generate images with meaningful name, please assign a list.
        Output:
            save outdata into multiple files
        Example:
            >>> imccls = ImageCalculator() 
            >>> imccls.decompose_img(imagedata, header, outpath)
        """
        assert np.ndim(imagedata) == 4, 'imagedata must be a 4D data'
        filenumber = imagedata.shape[3]
        if outname is None:
            digitname = range(1,filenumber+1,1)
            outname = [str(i) for i in digitname]
        else:
            assert len(outname) == filenumber, 'length of outname unequal to length of imagedata'
        for i,e in enumerate(outname):
            outdata = imagedata[...,i]
            img = nib.Nifti1Image(outdata, None, header)
            nib.save(img, os.path.join(outpath, e+'.nii.gz'))

    def combine_data(self, image1, image2, method = 'and'):
        """
        Combined data for 'and', 'or'
        ------------------------------------------
        Parameters:
            image1: dataset of the first image
            image2: dataset of the second image
            method: 'and' or 'or'
        """
        if (isinstance(image1, str) & isinstance(image2, str)):
            image1 = nib.load(image1).get_data()
            image2 = nib.load(image2).get_data()
        labels = np.unique(np.concatenate((np.unique(image1), np.unique(image2))))[1:]
        outdata = np.empty((image1.shape[0], image1.shape[1], image2.shape[2], labels.size))
        for i in range(labels.size):
            tempimage1 = copy.copy(image1)
            tempimage2 = copy.copy(image2)
            tempimage1[tempimage1 != labels[i]] = 0
            tempimage1[tempimage1 == labels[i]] = 1
            tempimage2[tempimage2 != labels[i]] = 0
            tempimage2[tempimage2 == labels[i]] = 1
            tempimage1.astype('bool')
            tempimage2.astype('bool')
            if method == 'and':
                tempimage = tempimage1 * tempimage2
            elif method == 'or':
                tempimage = tempimage1 + tempimage2
            else:
                raise Exception('Method support and, or now')
            outdata[...,i] = labels[i]*tempimage
        return outdata

class ExtractSignals(object):
    def __init__(self, atlas, regions = None):
        masksize = tools.get_masksize(atlas)
        
        self.atlas = atlas
        if regions is None:
            self.regions = masksize.shape[1]
        else:
            if isinstance(regions, int):
                self.regions = regions
            else:
                self.regions = len(regions)
        self.masksize = masksize

    def getsignals(self, targ, method = 'mean'):
        """
        Get measurement signals from target image by mask atlas.
        -------------------------------------------
        Parameters:
            targ: target image
            method: 'mean' or 'std', 'ste'(standard error), 'max' or 'voxel'
                    roi signal extraction method
        Return:
            signals: extracted signals
        """
        if targ.ndim == 3:
            targ = np.expand_dims(targ, axis = 3)
        signals = []
        
        for i in range(targ.shape[3]):
            if self.atlas.ndim == 3:
                signals.append(tools.get_signals(targ[...,i], self.atlas, method, self.regions))
            elif self.atlas.ndim == 4:
                signals.append(tools.get_signals(targ[...,i], self.atlas[...,i], method, self.regions))
        self.signals = np.array(signals)
        return np.array(signals)

    def getcoordinate(self, targ, size = [2,2,2], method = 'peak'):
        """
        Get peak coordinate signals from target image by mask atlas.
        -----------------------------------------------------------
        Parameters:
            targ: target image
            size: voxel size
            method: 'peak' or 'center'
                    coordinate extraction method
        """
        if targ.ndim == 3:
            targ = np.expand_dims(targ, axis = 3)
        coordinate = np.empty((targ.shape[3], self.regions, 3))

        for i in range(targ.shape[3]):
            if self.atlas.ndim == 3:
                coordinate[i, ...] = tools.get_coordinate(targ[...,i], self.atlas, size, method, self.regions)
            elif self.atlas.ndim == 4:
                coordinate[i, ...] = tools.get_coordinate(targ[...,i], self.atlas[...,i], size, method, self.regions)
        self.coordinate = coordinate
        return coordinate

    def getdistance_array2point(self, targ, pointloc, size = [2,2,2], coordmeth = 'peak', distmeth = 'euclidean'):
        """
        Get distance from each coordinate to a specific location
        -------------------------------------------------------
        Parameters:
            targ: target image
            pointloc: location of a specific voxel
            size: voxel size
            coordmeth: 'peak' or center
                       coordinate extraction method
            distmeth: distance method
        """
        if not hasattr(self, 'coordinate'):
            self.coordinate = tools.get_coordinate(targ, self.atlas, size, coordmeth)
        dist_point = np.empty((self.coordinate.shape[0], self.coordinate.shape[1]))
        pointloc = np.array(pointloc)
        if pointloc.shape[0] == 1:
            pointloc = np.tile(pointloc, [dist_point.shape[1],1])
        for i in range(dist_point.shape[0]):
            for j in range(dist_point.shape[1]):
                if not isinstance(pointloc[j], np.ndarray):
                    raise Exception('pointloc should be 2 dimension array or list')
                dist_point[i,j] = tools.calcdist(self.coordinate[i,j,:], pointloc[j], distmeth)
        self.dist_point = dist_point
        return dist_point

class MakeMasks(object):
    def __init__(self, header = None, issave = False, savepath = '.'):
        self._header = header
        self._issave = issave
        self._savepath = savepath

    def makepm(self, atlas, meth = 'all', maskname = 'pm.nii.gz'):
        """
        Make probabilistic maps
        ------------------------------
        Parameters:
            atlas: atlas mask
            meth: 'all' or 'part'
            maskname: output mask name, by default is 'pm.nii.gz'
        Return:
            pm
        """
        pm = roimethod.make_pm(atlas, meth)
        self._pm = pm
        if self._issave is True:
            iofactory = iofiles.IOFactory()
            factory = iofactory.createfactory(self._savepath, maskname)
            if maskname.endswith('gz') | maskname.endswith('nii'):
                factory.save_nifti(pm, self._header)
        return pm

    def makempm(self, threshold, maskname = 'mpm.nii.gz'):
        """
        Make maximum probabilistic maps
        --------------------------------
        Parameters:
            threshold: mpm threshold
            maskname: output mask name. By default is 'mpm.nii.gz'
        """
        if self._pm is None:
            raise Exception('please execute makepm first')
        mpm = roimethod.make_mpm(self._pm, threshold)
        if self._issave is True:
            iofactory = iofiles.IOFactory()
            factory = iofactory.createfactory(self._savepath, maskname)
            if maskname.endswith('gz') | maskname.endswith('nii'):
                factory.save_nifti(mpm, self._header)
        return mpm       
    
    def makemask_sphere(self, voxloc, radius, atlasshape = (91,109,91), maskname = 'spheremask.nii.gz'):
        """
        Make mask by means of roi sphere
        -------------------------------------------------
        Parameters:
            voxloc: peak voxel locations of each region
                    Note that it's a list
            radius: sphere radius, such as [3,3,3],etc.
            atlasshape: atlas shape
            maskname: Output mask name. By default is 'speremask.nii.gz'
        """ 
        spheremask = np.zeros(atlasshape)
        for i, e in enumerate(voxloc):
            spheremask, loc = roimethod.sphere_roi(e, radius, i+1, datashape = atlasshape, data = spheremask)
        if self._issave is True:
            iofactory = iofiles.IOFactory()
            factory = iofactory.createfactory(self._savepath, maskname)
            if maskname.endswith('gz') | maskname.endswith('nii'):
                factory.save_nifti(spheremask, self._header)
        return spheremask, loc

    def makemask_rgrowth(self, valuemap, coordinate, voxnum, maskname = 'rgmask.nii.gz'):
        """
        Make masks using region growth method
        -----------------------------------
        Parameters:
            valuemap: nifti data. Z map, cope map, etc.
            coordinate: region growth origin points
            voxnum: voxel numbers, integer or list
            maskname: output mask name.
        -----------------------------------
        Example:
            >>> outdata = INS.makemask_rgrowth(data, [[22,23,31], [22,22,31], [54,55,67]], 15)
        """
        import warnings
        warnings.simplefilter('always', UserWarning)       
 
        if isinstance(voxnum, int):
            voxnum = [voxnum]
        if len(voxnum) == 1:
            voxnum = voxnum*len(coordinate)
        if len(voxnum)!=len(coordinate):
            raise Exception('Voxnum length unequal to coodinate length.')
        rgmask = np.zeros_like(valuemap)
        for i,e in enumerate(coordinate):
            rg_image, loc = roimethod.region_growing(valuemap, e, voxnum[i])
            rg_image[rg_image!=0] = i+1
            if ~np.any(rgmask[rg_image!=0]):
                # all zeros
                rgmask += rg_image
            else:
                warnings.warn('coordinate {} has been overlapped! Label {} will missing'.format(e, i+1))
        if self._issave is True:
            iofactory = iofiles.IOFactory()
            factory = iofactory.createfactory(self._savepath, maskname)
            if maskname.endswith('gz') | maskname.endswith('nii'):
                factory.save_nifti(rgmask, self._header)
        return rgmask
                





