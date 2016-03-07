# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



class Analyzer(object):
    def __init__(self, meas, subj_id, roi_name, feature_name):
        self.meas = meas
        self.subj_id = subj_id
        self.roi_name = roi_name
        self.feature_name  = feature_name

    def feature_desp(self, meth, figure):
        """
        feature description and plot
        Parameters
        ----------
        figure

        Returns
        -------

        """

        fmean = np.nanmean(self.meas)
        fstd  = np.nanstd(self.meas)

        nSubj = self.meas.shape[0] # number of subjects
        nRoi = len(self.roi_id) # number of ROI
        nFeat =  len(self.feature_name) # number of feature
        for f in np.arange(nFeat):
            ## plot for feature
            roi_name = self.feature_name[np.floor(np.divide(f, nROI))]
            feat_name = self.roi_name[np.mod(f, nROI)]



        return fmean, fstd



    def feature_relation(self, meth, figure):
        """

        Parameters
        ----------
        meth
        figure

        Returns
        -------

        """

       feat_corr = np.corrcoef(self.meas)

       # plot feat_corr


    def behavior_predict1(self, meth, behavior):
        """
        Univariate predict
        Parameters
        ----------
        meth
        behavior

        Returns
        -------

        """



    def behavior_predict2(self, meth, behavior):
        """
        Multivariate predict
        Parameters
        ----------
        meth
        behavior

        Returns
        -------

        """




    def topymvpa(self):
        """
        Generate pymvpa dataset
        Returns
        -------

        """



    def outlier_remove(self, meth, figure):
        """
        remove outlier
        Parameters
        ----------
        meth
        figure

        Returns
        -------

        """
