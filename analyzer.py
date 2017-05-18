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
    def __init__(self, meas, meas_type, meas_name, roi_name, subj_id, subj_gender):
        """

        Parameters
        ----------
        meas :  n_subj x n_feature 2d array
        meas_type: scalar or geometry
        meas_name: list which keep measures name
        roi_name: list which keep roi name
        subj_id: list which keep subject id
        subj_gender: list which keep subject gender

        Returns
        -------

        """
        self.meas = meas
        self.type = meas_type
        self.subj_id = subj_id
        self.roi_name = roi_name
        self.meas_name = meas_name
        self.subj_gender = subj_gender

        self.feat_name = []
        n_roi = len(self.roi_name)  # number of ROI
        if self.type is 'scalar':
            for f in np.arange(meas.shape[1]):
                meas_name = self.meas_name[np.floor(np.divide(f, n_roi)).astype(int)]
                roi_name = self.roi_name[np.mod(f, n_roi).astype(int)]
                self.feat_name.append(roi_name + '-' + meas_name)
        elif self.type is 'geometry':
            geo = ['x', 'y', 'z']
            for f in np.arange(meas.shape[1]):
                meas_name = self.meas_name[np.floor(np.divide(f, n_roi*3)).astype(int)]
                roi_name = self.roi_name[np.mod(np.floor(f/3), n_roi).astype(int)]
                geo_name = geo[np.mod(f, 3)]
                self.feat_name.append(roi_name + '-' + meas_name + '-' + geo_name)
        else:
            raise UserDefinedException('Measure type is error!')

    def outlier_remove(self, meth='std', interval=[-2, 2], feat_sel=None, figure=False):
        """

        Parameters
        ----------
        meth
        interval
        feat_sel
        figure

        Returns
        -------

        """
        if feat_sel is None:
            feat_sel = np.arange(self.meas.shape[1])
        elif isinstance(feat_sel, list):
            feat_sel = np.array(feat_sel)

        n_outlier = np.zeros(feat_sel.shape[0])
        if meth is 'std':
            for f in np.arange(feat_sel.shape[0]):
                meas = self.meas[:, feat_sel[f]]
                f_std = np.nanstd(meas)
                f_mean = np.nanmean(meas)
                outlier = np.logical_or(meas < f_mean + interval[0] * f_std,
                                        meas >= f_mean + interval[1] * f_std)
                n_outlier[f] = np.count_nonzero(outlier)
                meas[outlier] = np.nan
        elif meth is 'iqr':
            for f in np.arange(feat_sel.shape[0]):
                meas = self.meas[:, feat_sel[f]]
                percentile = np.nanpercentile(meas, [25, 75])
                f_iqr = percentile[1] - percentile[0]
                outlier = np.logical_or(meas < percentile[0] + interval[0] * f_iqr,
                                        meas >= percentile[1] + interval[1] * f_iqr)
                n_outlier[f] = np.count_nonzero(outlier)
                meas[outlier] = np.nan
        else:
            raise UserDefinedException('Wrong method!')

        if figure:
            labels = [self.feat_name[i] for i in feat_sel]
            plot_bar(n_outlier, 'Number of outlier', labels, 'Count')

        return n_outlier

    def hemi_merge(self, meth='single', weight=None):
        """

        Parameters
        ----------
        meth : 'single' or 'both'.single, keep roi which appear in a single hemisphere;
        both, only keep roi which appear in both hemisphere
        weight: weight for each roi, n_subj x n_roi np.array

        Returns
        -------

        """
        if self.type is 'scalar':
            self.roi_name = [self.roi_name[i] for i in np.arange(0, len(self.roi_name), 2)]
            odd_f = np.arange(0, len(self.feat_name), 2)
            self.feat_name = [self.feat_name[i] for i in odd_f]

            if weight is None:
                weight = np.ones(self.meas.shape)
                weight[np.isnan(self.meas)] = 0
            else:
                weight = np.repeat(weight, self.meas.shape[1]/weight.shape[1], axis=1)

            if meth is 'single':
                for f in odd_f:
                    meas = self.meas[:, f:f+2]
                    bool_nan = np.isnan(self.meas)
                    index = np.logical_xor(bool_nan[:, 0], bool_nan[:, 1])
                    value = np.where(np.isnan(meas[index, 0]), meas[index, 1], meas[index, 0])
                    meas[index, :] = np.repeat(value[..., np.newaxis], 2, axis=1)
            elif meth is 'both':
                    pass

            odd_meas = self.meas[:, odd_f] * weight[:, odd_f]
            even_meas = self.meas[:, odd_f+1] * weight[:, odd_f+1]
            self.meas = (odd_meas + even_meas)/(weight[:, odd_f] + weight[:, odd_f+1])
        else:
            self.roi_name = [self.roi_name[i] for i in np.arange(0, len(self.roi_name), 2)]
            n_subj, n_feat = self.meas.shape
            meas = np.reshape(self.meas, (n_subj, -1, 3))
            odd_f = np.arange(0, meas.shape[1], 2)
            f_index = []
            for i in np.arange(0, meas.shape[1], 2):
                for j in [0, 1, 2]:
                    f_index.append(i*3+j)
            self.feat_name = [self.feat_name[i] for i in f_index]

            if meth is 'single':
                for f in odd_f:
                    f_meas = meas[:, f:f+2, :]
                    bool_nan = np.isnan(np.prod(f_meas, axis=2))
                    index = np.logical_xor(bool_nan[:, 0], bool_nan[:, 1])
                    value = np.where(np.isnan(f_meas[index, 0, :]), f_meas[index, 1, :], f_meas[index, 0, :])
                    meas[index, f:f+2, :] = np.repeat(value[:, np.newaxis, :], 2, axis=1)
                meas[:, odd_f+1, 0] = -meas[:, odd_f+1, 0]
            elif meth is 'both':
                meas[:, odd_f+1, 0] = -meas[:, odd_f+1, 0]

            self.meas = np.reshape((meas[:, odd_f, :] + meas[:, odd_f+1, :])/2, (n_subj, -1))

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
        elif isinstance(feat_sel, list):
            feat_sel = np.array(feat_sel)

        feat_stats = np.zeros((5, feat_sel.shape[0]))
        for f in np.arange(feat_sel.shape[0]):
            meas = self.meas[:, feat_sel[f]]
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
        elif isinstance(feat_sel, list):
            feat_sel = np.array(feat_sel)

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
            plot_mat(corr.T, 'Feature correlation', labels, labels)
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
                    ax.text(x0+0.1*(x1-x0), y0+0.9*(y1-y0), 'r = %.3f, p = %.3f' % (corr[i, j], pval[i, j]))
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
        elif isinstance(feat_sel, list):
            feat_sel = np.array(feat_sel)

        if beh_meas.ndim == 1:
            beh_meas = np.expand_dims(beh_meas, axis=1)

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
                    meas = self.meas[:, f]
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
                    ax.text(x0+0.1*(x1-x0), y0+0.9*(y1-y0),'r = %.3f, p = %.3f') % (corr[f, b], pval[f, b])
                    plt.xlabel(self.feat_name[f])
                    plt.ylabel(beh_name[b])
                    plt.title('Behavior predict')
                    plt.show()

        return corr, pval, n_sample

    def behavior_predict2(self, beh_meas, beh_name, contrast=None, feat_sel=None, figure=False):
        """

        Parameters
        ----------
        beh_meas: matrix for behavior measurements, n_subj x n_feat
        beh_name: name for behavior measurements
        contrast: contrast matrix, each row is a contrast, n_contrast x n_feat.
        if contrast is set as None, we will contrast each feature to zero
        feat_sel
        figure

        Returns
        -------
        stats: stats for the regression,(n_beh*3) x n_contrast np.array,
        for each behavior, 1st row is slope, 2nd is t, 3rd is p
        rows are behaviors, columns are features
        dof: degree of freedom, n_beh x 1
        r2 : r square of the fit, n_beh x 1

        """
        if feat_sel is None:
            feat_sel = np.arange(self.meas.shape[1])
        elif isinstance(feat_sel, list):
            feat_sel = np.array(feat_sel)

        if contrast is None:
            contrast = np.identity(feat_sel.shape[0])

        if beh_meas.ndim == 1:
            beh_meas = np.expand_dims(beh_meas, axis=1)

        samp_sel = ~np.isnan(np.prod(self.meas, axis=1))
        slope_stats = np.zeros((beh_meas.shape[1]*3, feat_sel.shape[0]))
        for b in np.arange(beh_meas.shape[1]):
            beh_sel = ~np.isnan(beh_meas[:, b])
            sel = np.logical_and(samp_sel, beh_sel)
            dof = np.count_nonzero(sel) - contrast.shape[0]
            x = stats.zscore(self.meas[np.ix_(sel, feat_sel)], axis=0)
            y = np.expand_dims(beh_meas[sel, b], axis=1)
            glm = LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
            glm.fit(x, y)
            y_pred = glm.predict(x)
            # total sum of squares
            sst = ((y - y.mean())**2).sum()
            # sum of squares of error
            sse = ((y - y_pred)**2).sum()
            r2 = 1 - sse/sst

            beta = glm.coef_
            slope_stats[b*3, :] = beta
            t = np.zeros(beta.shape[1])
            for i in np.arange(contrast.shape[0]):
                c = np.expand_dims(contrast[i, :], axis=0)
                t[i] = np.dot(c, beta.T)/np.sqrt(((sse/dof) * np.dot(np.dot(c, np.linalg.inv(np.dot(x.T, x))), c.T)))

            slope_stats[b*3+1, :] = t
            slope_stats[b*3+2, :] = stats.t.sf(np.abs(t), dof)*2

        if figure:
            labels = [self.feat_name[i] for i in feat_sel]
            for b in np.arange(0, slope_stats.shape[0], 3):
                plot_bar(slope_stats[b, :], 'Behavior predict for %s' % beh_name[b], labels, 'Slope')

        return slope_stats, r2, dof


    def hemi_asymmetry(self, feat_sel=None, figure=False):
        """

        Parameters
        ----------
        feat_sel: feature selection which hold the feature index of interest.
        when meas type is scalar, selection index should be paired; when meas
        type is geometry, selection index should be triple paired, and ordered
        as x, y, z in each triple group
        figure

        Returns
        -------
        li_stats: stats for laterality index, 5xnFeat np.array,
        rows are [mean, std, n_sample, t, p], columns are features
        """

        if feat_sel is None:
            feat_sel = np.arange(self.meas.shape[1])
        elif isinstance(feat_sel, list):
            feat_sel = np.array(feat_sel)

        if self.type is 'scalar':
            if (feat_sel.shape[0] % 2) != 0:
                raise UserDefinedException('Feature index should be paired')

            li_stats = np.zeros((5, feat_sel.shape[0]/2))
            for f in np.arange(0, feat_sel.shape[0], 2):
                meas = self.meas[:, feat_sel[f:f+2]]
                meas = meas[~np.isnan(np.prod(meas, axis=1)), :]
                li = (meas[:, 0] - meas[:, 1])/(meas[:, 0] + meas[:, 1])
                [t, p] = stats.ttest_1samp(li, 0)
                li_stats[:, f/2] = [np.mean(li), np.std(li), li.shape[0], t, p]
        else:
            if (feat_sel.shape[0] % 2) != 0 and (feat_sel.shape[0] % 3) != 0:
                raise UserDefinedException('Feature index should triple paired')
            li_stats = np.zeros((5, feat_sel.shape[0]/2))
            n_subj, n_feat = self.meas.shape
            meas = self.meas[:, feat_sel]
            meas = np.reshape(meas, (n_subj, -1, 3))
            for f in np.arange(0, meas.shape[1], 2):
                f_meas = meas[:, feat_sel[f:f+2], :]
                f_meas = f_meas[~np.isnan(np.prod(f_meas, axis=(1, 2))), :, :]
                f_meas[:, :, 0] = np.absolute(f_meas[:, :, 0])
                li = np.squeeze(f_meas[:, 0, :] - f_meas[:, 1, :])
                [t, p] = stats.ttest_1samp(li, 0)
                f_stats = np.vstack((np.mean(li, axis=0), np.std(li, axis=0), np.repeat(li.shape[0], 3), t, p))
                li_stats[:, np.arange((f/2)*3, ((f/2)+1)*3)] = f_stats

        if figure:
            if self.type is 'scalar':
                feat_labels = [self.feat_name[i] for i in feat_sel[::2]]
            else:
                feat_labels = []
                for i in np.arange(0, feat_sel.shape[0]/3, 2):
                    for j in [0, 1, 2]:
                        feat_labels.append(self.feat_name[i*3+j])

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
        elif isinstance(feat_sel, list):
            feat_sel = np.array(feat_sel)

        subj_gender = np.ones(len(self.subj_gender), dtype=bool)
        f_idx = [i for i, g in enumerate(self.subj_gender) if g == 'f']
        subj_gender[f_idx] = False

        gd_stats = np.zeros((5, feat_sel.shape[0]))
        for f in np.arange(feat_sel.shape[0]):
            meas = self.meas[:, feat_sel[f]]
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
