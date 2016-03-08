# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


class Analyzer(object):
    def __init__(self, meas, subj_id, roi_name, feat_name):
        self.meas = meas
        self.subj_id = subj_id
        self.roi_name = roi_name
        self.feat_name = feat_name

    def feature_description(self, feat_sel=None, figure=True):
        """
        feature description and plot
        Parameters
        ----------
        feat_sel: feature selection, index for feature of interest, a np.array
        figure :  to indicate whether to plot figures, True or False

        Returns
        -------
        feat_mean: feature mean, a np.array
        feat_std:  feature std, a np.array

        """

        if feat_sel is None:
            feat_sel = np.arange(self.meas.shape[1])

        feat_mean = np.nanmean(self.meas[:, feat_sel])
        feat_std = np.nanstd(self.meas[:, feat_sel])

        if figure:
            plt.close('all')
            nRoi = len(self.roi_name)  # number of ROI
            for f in feat_sel:
                feat_name = self.feat_name[np.floor(np.divide(f, nRoi)).astype(int)]
                roi_name = self.roi_name[np.mod(f, nRoi).astype(int)]

                plt.hist(self.meas[~np.isnan(self.meas[:, f]), f], normed=True)
                plt.xlabel(roi_name+'-'+feat_name)
                plt.ylabel('Probability')
                plt.title('Histogram')
                plt.show()

        return feat_mean, feat_std

    def feature_relation(self, feat_sel=None, figure=True):
        """
        relations among features
        Parameters
        ----------
        feat_sel: feature selection, index for feature of interest, a np.array
        figure :  to indicate whether to plot figures, True or False

        Returns
        -------
        feat_corr: correlation matrix of features, nFeat x nFeat np.array

        """
        if feat_sel is None:
            feat_sel = np.arange(self.meas.shape[1])

        samp_sel = ~np.isnan(np.prod(self.meas, axis=1))
        feat_corr = np.corrcoef(self.meas[np.ix_(samp_sel, feat_sel)].T)

        # plot feat_corr
        if figure:
            plt.close('all')
            fig, ax = plt.subplots()
            ax.pcolor(feat_corr)
            plt.title('Feature relation')
            plt.show()

        return feat_corr


    def behavior_predict1(self, beh_meas, feat_sel=None, figure=True):
        """
        Univariate predict
        Parameters
        ----------
        beh_meas: behavior measures, nSubj x nBeh np.array
        feat_sel: feature selection, index for feature of interest, a np.array

        Returns
        -------
        feat_beh_corr: correlation matrix between brain measurements and behavior measurements, nFeat x nBeh

        """

        if feat_sel is None:
            feat_sel = np.arange(self.meas.shape[1])

        if beh_meas.ndim == 1:
            beh_meas = np.tile(beh_meas, (1, 1))

        samp_sel = ~np.isnan(np.prod(self.meas, axis=1))
        feat_beh_corr = 1 - distance.cdist(self.meas[np.ix_(samp_sel, feat_sel)].T,  beh_meas[:, samp_sel])

        if figure:
            plt.close('all')
            fig, ax = plt.subplots()
            heatmap = ax.pcolor(feat_beh_corr)
            plt.title('Brain-behavior relation')
            plt.show()

            nRoi = len(self.roi_name)  # number of ROI
            for f in feat_sel:
                feat_name = self.feat_name[np.floor(np.divide(f, nRoi)).astype(int)]
                roi_name = self.roi_name[np.mod(f, nRoi).astype(int)]

                x = self.meas[samp_sel, f]
                y = np.squeeze(beh_meas[:, samp_sel])
                plt.scatter(x, y)
                plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x))

                plt.xlabel(roi_name+'-'+feat_name)
                plt.ylabel('Behavior Score')
                plt.title('Behavior predict')
                plt.show()

        return feat_beh_corr

    def behavior_predict2(self, beh_meas):
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
        self.meas: de-outlierd measurements

        """
        nSamp = self.meas.shape[0]  # number of sample
        good_samp = np.ones(nSamp, dtype=bool)
        good_samp[outlier_sel] = False
        self.meas = self.meas[good_samp, :]


        return self.meas
