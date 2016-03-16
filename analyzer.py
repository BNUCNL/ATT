# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from atlas import UserDefinedException
from sklearn.linear_model import LinearRegression


def plot_mat(mat, title, xlabels, ylabels):
    """

    Parameters
    ----------
    mat : matrix to be plotted, a 2d np.array
    title : title for the fig
    xlabels: labels for x axis
    ylabels: labels for y axis

    Returns
    -------

    """
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(mat)
    ax.set_xticks(np.arange(mat.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(mat.shape[0]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(xlabels, minor=False)
    ax.set_yticklabels(ylabels, minor=False)

    plt.xticks(rotation=45)
    ax.grid(False)

    cbar = plt.colorbar(heatmap)
    #cbar.set_label('Pearson correlation')

    # turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    plt.title(title)
    plt.show()


def plot_bar(data, title, xlabels, ylabels, err=None):
    """

    Parameters
    ----------
    data : data to be plotted, a 1d np.array
    err : error for data, same shape as data
    title
    xlabels
    ylabels

    Returns
    -------

    """
    ind = np.arange(data.shape[0])
    width = 0.35
    fig, ax = plt.subplots()
    if err is None:
        rects1 = ax.bar(ind, data, width, color='r')
    else:
        rects1 = ax.bar(ind, data, width, color='r', yerr=err)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))
    ax.set_ylabel(ylabels)
    ax.set_xticks(ind + width)
    plt.xticks(rotation=45)
    ax.set_xticklabels(xlabels)
    ax.set_title(title)
    plt.show()


def cohen_d(x, y):
    nx, ny = x.shape[0], y.shape[0]
    dof = nx + ny - 2
    d = (np.mean(x) - np.mean(y)) / \
        np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

    return d


class Analyzer(object):
    def __init__(self, meas, meas_name, roi_name, subj_id, subj_gender):
        self.meas = meas
        self.subj_id = subj_id
        self.roi_name = roi_name
        self.meas_name = meas_name
        self.subj_gender = subj_gender

        self.feat_name = []
        n_roi = len(self.roi_name)  # number of ROI
        for f in np.arange(meas.shape[1]):
            meas_name = self.meas_name[np.floor(np.divide(f, n_roi)).astype(int)]
            roi_name = self.roi_name[np.mod(f, n_roi).astype(int)]
            self.feat_name.append(roi_name + '-' + meas_name)

    def hemi_merge(self, meth='both'):
        """

        Parameters
        ----------
        meth : 'single' or 'both'.single, keep roi which appear in a single hemisphere;
        both, only keep roi which appear in both hemisphere

        Returns
        -------

        """
        self.roi_name = [self.roi_name[i] for i in np.arange(0, len(self.roi_name), 2)]
        odd_f = np.arange(0, len(self.feat_name), 2)
        self.feat_name = [self.feat_name[i] for i in odd_f]

        if meth is 'single':
            for f in odd_f:
                meas = self.meas[:, (f, f+1)]
                bool_nan = np.isnan(meas)
                index = np.logical_xor(bool_nan[:, 0], bool_nan[:, 1])
                value = np.where(np.isnan(meas[index, 0]), meas[index, 1], meas[index, 0])
                meas[index, :] = np.repeat(value[..., np.newaxis], 2, axis=1)
        elif meth is 'both':
            pass

        self.meas = (self.meas[:, odd_f] + self.meas[:, (odd_f + 1)])/2

    def feature_description(self, feat_sel=None, figure=False):
        """
        feature description and plot
        Parameters
        ----------
        feat_sel: feature selection, index for feature of interest, a np.array
        figure :  to indicate whether to plot figures, True or False

        Returns
        -------
        feat_stats:  statistics for each feature, a 5xnFeat np.array
        rows are [mean, std, n_sample, t, p], respectively.

        """

        if feat_sel is None:
            feat_sel = np.arange(self.meas.shape[1])

        feat_stats = np.zeros((5, feat_sel.shape[0]))
        for f in feat_sel:
            meas = self.meas[:, f]
            meas = meas[~np.isnan(meas)]
            [t, p] = stats.ttest_1samp(meas, 0)
            feat_stats[:, f] = [np.mean(meas), np.std(meas), meas.shape[0], t, p]

        if figure:
            for f in feat_sel:
                feat_name = self.feat_name[f]
                meas = self.meas[:, f]
                meas = meas[~np.isnan(meas)]
                if meas.shape[0] < 100:
                    n_bin = 10
                else:
                    n_bin = np.fix(meas.shape[0]/10)
                fig, ax = plt.subplots()
                plt.hist(meas, bins=n_bin)
                plt.xlabel(feat_name)
                plt.ylabel('Frequency counts')
                plt.title('Histogram')
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                ax.set_aspect((x1-x0)/(y1-y0))
                plt.show()

        return feat_stats

    def feature_relation(self, feat_sel=None, figure=False):
        """
        relations among features
        Parameters
        ----------
        feat_sel: feature selection, index for feature of interest, a np.array
        figure :  to indicate whether to plot figures, True or False

        Returns
        -------
        feat_corr: correlation matrix of features, nFeat x nFeat np.array
        n_sample: number of samples which have all features

        """
        if feat_sel is None:
            feat_sel = np.arange(self.meas.shape[1])

        corr = np.zeros((feat_sel.shape[0], feat_sel.shape[0]))
        pval = np.copy(corr)
        n_sample = np.copy(corr)
        for i in np.arange(feat_sel.shape[0]):
            for j in np.arange(i+1, feat_sel.shape[0], 1):
                meas1 = self.meas[:, feat_sel[i]]
                meas2 = self.meas[:, feat_sel[j]]
                samp_sel = ~np.isnan(meas1 * meas2)
                n_sample[i, j] = np.count_nonzero(samp_sel)
                x = meas1[samp_sel]
                y = meas2[samp_sel]
                [c, p] = stats.pearsonr(x, y)
                corr[i, j] = c
                pval[i, j] = p

        if figure:
            labels = [self.feat_name[i] for i in feat_sel]
            plot_mat(corr, 'Feature correlation', labels, labels)
            # plot for each feature
            for i in np.arange(feat_sel.shape[0]):
                for j in np.arange(i+1, feat_sel.shape[0], 1):
                    meas1 = self.meas[:, feat_sel[i]]
                    meas2 = self.meas[:, feat_sel[j]]
                    samp_sel = ~np.isnan(meas1 * meas2)
                    x = meas1[samp_sel]
                    y = meas2[samp_sel]
                    fig, ax = plt.subplots()
                    plt.scatter(x, y)
                    plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x))
                    x0, x1 = ax.get_xlim()
                    y0, y1 = ax.get_ylim()
                    ax.set_aspect((x1-x0)/(y1-y0))
                    plt.xlabel(labels[i])
                    plt.ylabel(labels[j])
                    plt.title('Feature correlation')
                    plt.show()

        return corr, pval, n_sample

    def behavior_predict1(self, beh_meas, beh_name, feat_sel=None, figure=False):
        """
        Univariate feature-wise predict for behavior
        Parameters
        ----------
        beh_meas: behavior measures, nSubj x nBeh np.array
        beh_name: behavior name, a list
        feat_sel: feature selection, index for feature of interest, a np.array
        figure: true or false

        Returns
        -------
        corr: correlation matrix between brain measurements and behavior measurements,
        nFeat x nBeh np.array
        p: p value matrix
        n_sample: number of samples

        """

        if feat_sel is None:
            feat_sel = np.arange(self.meas.shape[1])

        if beh_meas.ndim == 1:
            beh_meas = np.tile(beh_meas, (1, 1)).T

        corr = np.zeros((feat_sel.shape[0], beh_meas.shape[1]))
        pval = np.copy(corr)
        n_sample = np.copy(corr)
        for f in np.arange(feat_sel.shape[0]):
            for b in np.arange(beh_meas.shape[1]):
                meas = self.meas[:, feat_sel[f]]
                beh = beh_meas[:, b]
                samp_sel = ~np.isnan(meas * beh)
                n_sample[f, b] = np.count_nonzero(samp_sel)
                [c, p] = stats.pearsonr(meas[samp_sel], beh[samp_sel])
                corr[f, b] = c
                pval[f, b] = p

        if figure:
            beh_labels = beh_name
            feat_labels = [self.feat_name[i] for i in feat_sel]
            plot_mat(corr, 'Feature correlation', beh_labels, feat_labels)

            # plot for each feature
            for f in feat_sel:
                for b in np.arange(beh_meas.shape[1]):
                    meas = self.meas[:, feat_sel[f]]
                    beh = beh_meas[:, b]
                    samp_sel = ~np.isnan(meas * beh)
                    x = meas[samp_sel]
                    y = beh[samp_sel]
                    fig, ax = plt.subplots()
                    plt.scatter(x, y)
                    plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x))
                    x0, x1 = ax.get_xlim()
                    y0, y1 = ax.get_ylim()
                    ax.set_aspect((x1-x0)/(y1-y0))
                    plt.xlabel(self.feat_name[f])
                    plt.ylabel(beh_name[b])
                    plt.title('Behavior predict')
                    plt.show()

        return corr, pval, n_sample

    def behavior_predict2(self, beh_meas, beh_name, feat_sel=None, figure=False):
        """

        Parameters
        ----------
        beh_meas
        beh_name
        feat_sel
        figure

        Returns
        -------
        slope: slope for regression,nBeh x nFeat np.array,
        rows are behaviors, columns are features

        """
        if feat_sel is None:
            feat_sel = np.arange(self.meas.shape[1])

        if beh_meas.ndim == 1:
            beh_meas = np.tile(beh_meas, (1, 1)).T

        samp_sel = ~np.isnan(np.prod(self.meas, axis=1))
        slope = np.zeros((beh_meas.shape[1], feat_sel.shape[0]))
        for b in np.arange(beh_meas.shape[1]):
            beh_sel = ~np.isnan(beh_meas[:, b])
            sel = np.logical_and(samp_sel, beh_sel)
            x = self.meas[sel, :]
            y = np.tile(beh_meas[sel, b], (1, 1)).T
            glm = LinearRegression(copy_X=True, fit_intercept=True, normalize=True)
            glm.fit(x, y)
            slope[b, :] = glm.coef_

        if figure:
            labels = [self.feat_name[i] for i in feat_sel]
            for b in np.arange(slope.shape[0]):
                plot_bar(slope[b, :], 'Behavior predict for %s' % beh_name[b], labels, 'Slope')

        return slope

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

    def hemi_asymmetry(self, feat_sel=None, figure=False):
        """

        Parameters
        ----------
        feat_sel
        figure

        Returns
        -------
        li_stats: stats for laterality index, 5xnFeat np.array,
        rows are [mean, std, n_sample, t, p], columns are features
        """

        if feat_sel is None:
            feat_sel = np.arange(self.meas.shape[1])

        if (feat_sel.shape[0] % 2) != 0:
            raise UserDefinedException('The number of feature should be even and paired next each other')

        li_stats = np.zeros((5, feat_sel.shape[0]/2))
        for f in np.arange(0, feat_sel.shape[0], 2):
            meas = self.meas[:, feat_sel[f:f+2]]
            meas = meas[~np.isnan(np.prod(meas, axis=1)), :]
            li = (meas[:, 0] - meas[:, 1])/(meas[:, 0] + meas[:, 1])
            [t, p] = stats.ttest_1samp(li, 0)
            li_stats[:, f/2] = [np.nanmean(li), np.nanstd(li), li.shape[0], t, p]

        if figure:
            feat_labels = [self.feat_name[i] for i in feat_sel[::2]]
            plot_bar(li_stats[0, :], 'Laterality index', feat_labels, 'LI score', li_stats[1, :])

        return li_stats

    def gender_diff(self, feat_sel=None, figure=False):
        """

        Parameters
        ----------
        feat_sel
        figure

        Returns
        -------
        gd_stats: statistics for gender difference, 5xnFeat
        rows are [cohen_d, n_male, n_female, t, p]; columns are features
        """

        if feat_sel is None:
            feat_sel = np.arange(self.meas.shape[1])

        subj_gender = np.ones(len(self.subj_gender), dtype=bool)
        f_idx = [i for i, g in enumerate(self.subj_gender) if g == 'f']
        subj_gender[f_idx] = False

        gd_stats = np.zeros((5, feat_sel.shape[0]))
        for f in feat_sel:
            meas = self.meas[:, f]
            idx = ~np.isnan(meas)
            meas = meas[idx]
            gender = subj_gender[idx]

            n_male = np.count_nonzero(gender)
            n_female = meas.shape[0] - n_male
            d = cohen_d(meas[gender], meas[~gender])
            [t, p] = stats.ttest_ind(meas[gender], meas[~gender], equal_var=0)

            gd_stats[:, f] = [d, n_male, n_female, t, p]

        if figure:
            xlabels = [self.feat_name[i] for i in feat_sel]
            plot_bar(gd_stats[0, :], 'Gender differences', xlabels, 'Cohen d')

        return gd_stats
