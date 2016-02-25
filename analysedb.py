# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:

""

import cPickle
import os
import numpy as np

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
                return self.data[modal][param][stem+'_'+param]

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
        datacal = [i for i in dataarea if (str(i)!='nan')&(i!=0)]
        datacal = np.array(datacal)
        self.datamean = datacal.mean(axis=0)
        self.datastd = datacal.std(axis=0)



	














