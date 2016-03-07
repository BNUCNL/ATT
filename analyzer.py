# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.spatial.distance.cdist as cdist


class Analyzer(object):
    def __init__(self, meas, subj_id, roi_name, feature_name):
        self.meas = meas
        self.subj_id = subj_id
        self.roi_name = roi_name
        self.feature_name  = feature_name

    def feature_description(self, feat_sel=None, figure=True):
        """
        feature description and plot
        Parameters
        ----------
        figure

        Returns
        -------

        """

        nFeat = len(self.feature_name) # number of feature
        if feat_sel is None:
            feat_sel = np.arange(nFeat)

        fmean = np.nanmean(self.meas[:, feat_sel])
        fstd  = np.nanstd(self.meas[:, feat_sel])

        if figure:
            nRoi = len(self.roi_id) # number of ROI
            for f in feat_sel:
                roi_name = self.feature_name[np.floor(np.divide(f, nRoi))]
                feat_name = self.roi_name[np.mod(f, nRoi)]

                plt.hist(np.isnan(self.meas[:, f]))
                plt.xlabel(roi_name+'-'+feat_name)
                plt.ylabel('Probability')
                plt.title('Histogram for ROI feature')
                plt.show()

        return fmean, fstd



    def feature_relation(self, feat_sel=None, figure=True):
        """

        Parameters
        ----------
        meth
        figure

        Returns
        -------

        """
        nFeat = len(self.feature_name) # number of feature
        if feat_sel is None:
            feat_sel = np.arange(nFeat)

        feat_corr = np.corrcoef(self.meas[:, feat_sel])

        # plot feat_corr
        if figure:
            fig, ax = plt.subplots()
            heatmap = ax.pcolor(feat_corr)
            plt.show()

        return feat_corr


    def behavior_predict1(self, feat_sel, behavior):
        """
        Univariate predict
        Parameters
        ----------
        meth
        behavior

        Returns
        -------

        """

        nFeat = len(self.feature_name) # number of feature
        if feat_sel is None:
            feat_sel = np.arange(nFeat)

        feat_beh_corr = 1 - cdist(self.meas[:, feat_sel], behavior)


        if figure:
            fig, ax = plt.subplots()
            heatmap = ax.pcolor(feat_beh_corr])
            plt.show()

        return feat_beh_corr



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
        pass




    def topymvpa(self):
        """
        Generate pymvpa dataset
        Returns
        -------

        """
        pass



    def outlier_remove(self, outlier_sel):
        """
        remove outlier
        Parameters
        ----------
        outlier_sel: outlier index, 1-d np.array

        Returns
        -------

        """
        nSubj = self.meas.shape[0] # number of subjects
        good_subj = np.ones(nSubj, dtype=bool)
        good_subj[outlier_sel] = False
        self.meas = self.meas[good_subj, :]


        return self.meas
