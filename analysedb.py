# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:

""

import cPickle
import os

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




	
