# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

class FigureFactory(object):
    def __init__(self):
	pass

    def __str__(self):
        return 'A factory for ploting figures'

    def createfactory(self, figuretype):
        """
        Create factory by this function
        ------------------------------------
        Parameters:
            figuretype: 'corr', correlation plots
                        'mat', matrix plot
                        'bar', plot bar
                        'hist', histogram
                        'hierarchy', hierarchy maps
                        'line', line maps
        """
        figure = self._Figures()
        if figuretype == 'corr':
            figuror = figure._corr_ploting
        elif figuretype == 'mat':
            figuror = figure._mat_ploting
        elif figuretype == 'bar':
            figuror = figure._bar_ploting
        elif figuretype == 'hist':
            figuror = figure._hist_ploting
        elif figuretype == 'hierarchy':
            figuror = figure._hierarchy_ploting
        elif figuretype == 'line':
            figuror = figure._simpleline_ploting
        else:
              raise Exception('wrong parameter input!')
        return figuror

    class _Figures(object):
        def __init__(self):
            pass    
    
        def _corr_ploting(self, meas1, meas2, labels, method = 'pearson'):
            """
            Make scatter plot and give a fit on it.
            ------------------------------------------
            Paramters:
                meas1: feature measurement
                meas2: feature measurement
                labels: A list contains two labels.
                        labels[0] means label of meas1, labels[1] means label of meas2.
                method: 'pearson' or 'spearman' correlation
            """
            if (meas1.dtype != 'O') | (meas2.dtype != 'O'):
                samp_sel = ~np.isnan(meas1*meas2)
                x = meas1[samp_sel]
                y = meas2[samp_sel]
            else:
                x = meas1
                y = meas2
            if method == 'pearson':
                corr, pval = stats.pearsonr(x, y)
            elif method == 'spearman':
                corr, pval = stats.spearmanr(x, y)
            else:
                raise Exception('Wrong method you used')
            fig, ax = plt.subplots()
            plt.scatter(x, y)
            plt.plot(x, np.poly1d(np.polyfit(x,y,1))(x))
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.set_aspect((x1-x0)/(y1-y0))
            ax.text(x0+0.1*(x1-x0), y0+0.9*(y1-y0), 'r = %.3f, p = %.3f' % (corr, pval))
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            plt.title(method.capitalize()+' Correlation')
            plt.show()

        def _mat_ploting(self, data, xlabel, ylabel):
            """
            Plot matrix using heatmap
            ------------------------------------
            Paramters:
                data: raw data
                xlabel: xlabels
                ylabel: ylabels
            """
            sns.heatmap(data, xticklabels = xlabel, yticklabels = ylabel)
            plt.show()

        def _bar_ploting(self, data, title, xlabels, ylabels, legendname, legendpos = 'upper left', err=None):
            """
            Do barplot
            --------------------------
            Parameters:
                data: raw data
                title:
                xlabels, ylabels: 
                err: error of data estimation. Used for errorbar
            """
            color = ['#BDBDBD', '#575757', '#404040', '#080808', '#919191']
            if isinstance(data, list):
                data = np.array(data)
            if data.ndim == 1:
                data = np.expand_dims(data, axis = 1)
            ind = np.arange(data.shape[0])
            width = 0.70/data.shape[1]
            fig, ax = plt.subplots() 
            rects = []
            if err is None:
                for i in range(data.shape[1]):
                    rects = ax.bar(ind + i*width, data[:,i], width, color = color[i%5], label = legendname[i])
                    ax.legend(loc=legendpos) 
            else:
                for i in range(data.shape[1]):
                    rects = ax.bar(ind + i*width, data[:,i], width, color = color[i%5], yerr = err[:,i], error_kw=dict(ecolor = '#757575', capthick=1), label = legendname[i])
                    ax.legend(loc=legendpos)
            
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.set_aspect((x1-x0)/(y1-y0))
            ax.set_ylabel(ylabels)
            ax.set_xticks(ind + width)
            if np.min(data)<0:
                ax.set_ylim([1.33*np.min(data), 1.33*np.max(data)])
            else:
                ax.set_ylim([0, 1.33*np.max(data)])
            plt.xticks(rotation = 45)
            ax.set_xticklabels(xlabels)
            ax.set_title(title, fontsize=12)
            plt.show()

        def _hist_ploting(self, n_scores, legend_label, *oppar):
            """
            Plot histogram of given data
            Parameters:
            ----------------------------------
                n_scores: scores
                legend_label: data legend label
                score: Optional choice. used for permutation cross validation results.In permutation cross validation, n_scores means value of permutation scores, score means actual score.
                pval: Optional choice. Need to use with score. p values of permutation test.
            """
            if len(oppar) == 0:
                plt.hist(n_scores, 50, label = legend_label)
                ylim = plt.ylim()
            elif len(oppar) == 2:
                plt.hist(n_scores, 50, label = 'permutation scores')
                ylim = plt.ylim()
                plt.plot(2*[oppar[0]], ylim, '--k', linewidth = 3,
                         label = 'Scores'
                         ' (pvalue %s)' % str(oppar[1]))
                plt.ylim(ylim)
            else:
                raise Exception('parameter numbers should be 2 or 4!')
            plt.legend()
            plt.xlabel('Score')
            plt.show()
          
        def _hierarchy_ploting(self, distance, regions):
            """
            Plot hierarchy structure of specific indices between regions
            -------------------------------
            Parameters:
                distance: distance array, distance array by using scipy.pdist
                regions: region name        
            """
            Z = linkage(distance, 'average')
            dendrogram(Z, labels = regions)
            plt.show()

        def _simpleline_ploting(self, dataarray, xlabel, ylabel, ylim = None):
            """
            Plot an array using simple lines
            For better showing, rescaling each array into range of 0 to 1
            --------------------------------------
            Parameters:
                dataarray: data array, a x*y array, y means number of lines
                xlabel: xlabel
                ylabel: ylabel
                ylim: By default is None, if ylim exists, limit y values of figure
            """
            fig, ax = plt.subplots()
            ax.set_color_cycle(['red', 'blue', 'yellow', 'black', 'green'])
            dataarray_scaling = np.zeros_like(dataarray)
            for i in range(dataarray.shape[1]):
                dataarray_scaling[:,i] = (dataarray[:,i]-np.min(dataarray[:,i]))/(np.max(dataarray[:,i])-np.min(dataarray[:,i]))
            plt.plot(dataarray_scaling)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if ylim is not None:
                plt.ylim(ylim)
            plt.show()


