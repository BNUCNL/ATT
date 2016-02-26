# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:
from __future__ import division
import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fpath = os.path.join
  
class user_defined_exception(Exception):
    def __init__(self, str):
        Exception.__init__(self)
        self._str = str

class AtlasDB(object):
    def __init__(self):
        data = []

    def read_from_pkl(self, path, filename):
        """
        
        Parameters
        ----------
        read data from .pkl 
        path    parent path
        filename    filename.pkl

        Returns
        Whole data from .pkl
        -----------

        """
        file = open(fpath(path, filename),'r')
        data = cPickle.load(file)
        file.close()
        self.data = data

    def output_data(self, modal = 'geo', param = 'volume', stem = 'zstat'):
        """
        
        Parameters
        ----------
        Getting data by keys of modal and param
        modal
        param

        Returns
        data
        ----------
        """
        if (modal == 'geo') & (param == 'volume'):
            return self.data[modal][param]['act_volume']
        if self.data.has_key(modal):
            if self.data[modal].has_key(param):
                outdata = self.data[modal][param][stem+'_'+param]
                if modal == 'rest':
                    outdata[outdata == 0] = np.nan

        return outdata

class AtlasDescribe(object):
    def __init__(self, data, areaname):
        self.data = data
        self.areaname = areaname
        self.datamean = []
        self.datastd = []
    def subjexist(self, area):
        """

        Parameters
        ------------
        find num of exist data of each areas
        please use the result that the inputting data is volume


        Returns
        existnum: existed subject numbers
        existperc: existed subject percentage
        -------

        """
        totalsubj = self.data.shape[0]
        existnum = sum(self.data[:,self.areaname.index(area)] != 0 )
        existperc = existnum/float(totalsubj)

        self.existnum = existnum
        self.existperc = existperc

    def paradescrib(self, area):
        """

        Parameters
        -----------
        calculate mean and std from raw data


        Returns
        datamean
        datastd
        ------------

        """
        dataarea = self.data[:,self.areaname.index(area)]

        # datacal is the data removed nan and zeros
        datacal = deloutlier(dataarea)
        self.datamean = datacal.mean(axis=0)
        self.datastd = datacal.std(axis=0)

    def calhemLi(self, area):
        """

        Parameters
        ----------
        Attention: the area here contains left and right hemisphere
        For example,'MT' or 'V3' or something else

        Returns
        -------
        livalue
        """
        l_dataarea = self.data[:, self.areaname.index('l'+area)]
        r_dataarea = self.data[:, self.areaname.index('r'+area)]

        livalue = callivalue(l_dataarea, r_dataarea)
        self.livalue = livalue


class DataFigure(object):
    def __init__(self, areaname):
        self.areaname = areaname
    def plotcorr(self, data1, data2, area):
        pass

    def plotbar(self):
        pass



def deloutlier(data):
    """

    Parameters
    ----------
    Remove data that contains nan
    data

    Returns
    -------
    datoutlier
    """
    dataoutlier = [i for i in data if (str(i) != 'nan')]
    dataoutlier = np.array(dataoutlier)
    return dataoutlier

def callivalue(a,b):
    """

    Parameters
    ----------
    calculate LI between a and b
    a
    b

    Returns
    -------
    livalue
    """
    livalue = (a-b)/(a+b)
    return livalue












