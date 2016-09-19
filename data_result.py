# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from ATT.core import analysebase
from ATT.util import plotfig
from ATT.io import iofiles

pjoin = os.path.join

data_parpath = '/nfs/j3/userhome/huangtaicheng/hworkingshop/parcellation_MT/Summary_result/program/mttask/data'

##-----------------------------------------------------------------------------
## Plot dice coefficient
# data_stem = 'data_spec/dice'
# dice_lzg_htc = iofiles.load_pkl(pjoin(data_parpath, data_stem, 'dice_lzg_htc.pkl'))
# dice_lzg_zgf = iofiles.load_pkl(pjoin(data_parpath, data_stem, 'dice_lzg_zgf.pkl'))
# dice_zgf_htc = iofiles.load_pkl(pjoin(data_parpath, data_stem, 'dice_zgf_htc.pkl'))
# dice_mean = []
# dice_err = []
# nonnan_lzg_htc = np.sum(~np.isnan(dice_lzg_htc), axis = 0)
# nonnan_lzg_zgf = np.sum(~np.isnan(dice_lzg_zgf), axis = 0)
# nonnan_zgf_htc = np.sum(~np.isnan(dice_zgf_htc), axis = 0)

# dice_mean.append(np.nanmean(dice_lzg_htc, axis = 0))
# dice_mean.append(np.nanmean(dice_lzg_zgf, axis = 0))
# dice_mean.append(np.nanmean(dice_zgf_htc, axis = 0))
# dice_mean = np.array(dice_mean)

# dice_err.append(np.nanstd(dice_lzg_htc, axis = 0)/np.sqrt(nonnan_lzg_htc))
# dice_err.append(np.nanstd(dice_lzg_zgf, axis = 0)/np.sqrt(nonnan_lzg_zgf))
# dice_err.append(np.nanstd(dice_zgf_htc, axis = 0)/np.sqrt(nonnan_zgf_htc))
# dice_err = np.array(dice_err)

# plotfig.plot_bar(dice_mean.T, 'Reliability', ['rV3', 'lV3', 'rMT', 'lMT'], 'Dice Coefficient', ['expert 1-2','expert 1-3','expert 2-3'], dice_err.T)

# ----------------------------------------------------------------------------
# Describe activation data
# data_stem = 'data_spec/measuredata'
# masksize = iofiles.load_pkl(pjoin(data_parpath, data_stem, 'masksize_spec.pkl'))
# masksize = masksize*(2*2*2)/1000
# zvalue = iofiles.load_pkl(pjoin(data_parpath, data_stem, 'zval_spec.pkl'))
# pscvalue = iofiles.load_pkl(pjoin(data_parpath, data_stem, 'psc_spec.pkl'))
# pscvalue = pscvalue/100
# coordinate_value = iofiles.load_pkl(pjoin(data_parpath, data_stem, 'coordinate_spec.pkl'))

# size_mean = np.nanmean(masksize, axis = 0)
# nonnan_size = np.sum(~np.isnan(masksize), axis = 0)
# size_err = np.nanstd(masksize,axis = 0)/np.sqrt(nonnan_size)

# zval_mean = np.nanmean(zvalue, axis = 0)
# nonnan_zval = np.sum(~np.isnan(zvalue), axis = 0)
# zval_err = np.nanstd(zvalue,axis = 0)/np.sqrt(nonnan_zval)

# psc_mean = np.nanmean(pscvalue, axis = 0)
# nonnan_psc = np.sum(~np.isnan(pscvalue), axis = 0)
# psc_err = np.nanstd(pscvalue,axis = 0)/np.sqrt(nonnan_psc)

# coordinate lateralization
# coord_lat = np.empty((coordinate_value.shape[0], coordinate_value.shape[1]/2, coordinate_value.shape[2]))
# for i in range(coord_lat.shape[1]):
#     coord_lat[:,i,:] = np.abs(coordinate_value[:,2*i+1,:]) - np.abs(coordinate_value[:,2*i,:])
# coordlat_mean = np.nanmean(coord_lat,axis = 0)
# nonnan_coordlat = np.sum(~np.isnan(coord_lat), axis = 0)
# coordlat_err = np.nanstd(coord_lat, axis = 0)/np.sqrt(nonnan_coordlat)

# coordinate 
# coord_stdsum = []
# for i in range(coordinate_value.shape[1]):
#     coord_stdsum.append(np.sum(np.nanstd(coordinate_value[:,i,:], axis = 0)))
# coord_stdsum = np.array(coord_stdsum)

# plotfig.plot_bar(size_mean.reshape(2,2), 'activation size', ['V3', 'MT'], 'Volume (cm3)', ['Left', 'Right'], err = size_err.reshape(2,2))
# plotfig.plot_bar(zval_mean.reshape(2,2), 'Z values', ['V3', 'MT'], 'Z values', ['Left', 'Right'], err = zval_err.reshape(2,2))
# plotfig.plot_bar(psc_mean.reshape(2,2), 'Percentage signal change', ['V3', 'MT'], 'Percentage signal change (%)', ['Left', 'Right'], err = psc_err.reshape(2,2))
# plotfig.plot_bar(coordlat_mean, 'Peak Coordinate Lateralization', ['V3', 'MT'], 'Coordinate difference (mm)', ['X','Y','Z'], err = coordlat_err)
# plotfig.plot_bar(coord_stdsum.reshape(2,2), 'Sum of SD of Coordinate', ['V3', 'MT'], 'SD of Coordinates (mm)', ['Left', 'Right'])

# ----------------------------------------------------------------------------
# feature prediction 1, simple linear correlation analysis
data_stem = 'data_spec/measuredata/MT'
# data_dist_stem = 'data_spec/measuredata'
data_pd = pd.read_csv(pjoin(data_parpath, data_stem, 'sum_data.csv'))
data_pd['mtsys_bnc_same'] = data_pd['mtsys_bnc'].values
# data_pd['dist_all0_lMT'] = data_dist[:,3]
# data_pd['dist_all0_rMT'] = data_dist[:,2]
data_allmea = data_pd[['act-mean-lMT', 'act-mean-rMT', 'size_lMT', 'size_rMT', 'cop-mean-lMT', 'cop-mean-rMT', 'alff-mean-lMT', 'alff-mean-rMT', 'mtsys_bnc', 'mtsys_bnc_same', 'vbm-amount-mean-lMT', 'vbm-amount-mean-rMT', 'thick_lMT', 'thick_rMT', 'area_lMT', 'area_rMT']]
measname_all_comb = ['zvalue', 'size', 'psc', 'alff', 'bnc', 'vbm-amount', 'corical thick', 'cortical area']
measfeature_all = analysebase.FeatureRelation(data_allmea.values, measname_all_comb, mergehemi = True, figure = True)
meascorr, measpval = measfeature_all.feature_prediction1()

glm = linear_model.LinearRegression()
# ridge = linear_model.Ridge(alpha = 1)

data_z = data_pd[['act-mean-lMT', 'act-mean-rMT', 'alff-mean-lMT', 'alff-mean-rMT', 'mtsys_bnc', 'mtsys_bnc_same', 'vbm-amount-mean-lMT', 'vbm-amount-mean-rMT', 'thick_lMT', 'thick_rMT', 'area_lMT', 'area_rMT']]
measname_z_comb = ['zvalue', 'alff', 'bnc', 'vbm-amount', 'cortical thick', 'cortical area']
measfeature_z = analysebase.FeatureRelation(data_z.values, measname_z_comb, mergehemi = True, figure = True)
r2_z, beta_z, tval_z, tpval_z, f_z, fpval_z = measfeature_z.feature_prediction2(glm)
score_z, n_score_z, pval_z = measfeature_z.feature_prediction3(glm)

data_psc = data_pd[['cop-mean-lMT', 'cop-mean-rMT', 'alff-mean-lMT', 'alff-mean-rMT', 'mtsys_bnc', 'mtsys_bnc_same', 'vbm-amount-mean-lMT', 'vbm-amount-mean-rMT', 'thick_lMT', 'thick_rMT', 'area_lMT', 'area_rMT']]
measname_psc_comb = ['psc', 'alff', 'bnc', 'vbm-amount', 'cortical thick', 'cortical area']
measfeature_psc = analysebase.FeatureRelation(data_psc.values, measname_psc_comb, mergehemi = True, figure = True)
r2_psc, beta_psc, tval_psc, tpval_psc, f_psc, fpval_psc = measfeature_psc.feature_prediction2(glm)
score_psc, n_score_psc, pval_psc = measfeature_psc.feature_prediction3(glm)

# data_size = data_pd[['size_lMT', 'size_rMT', 'alff-mean-lMT', 'alff-mean-rMT', 'mtsys_bnc', 'mtsys_bnc_same', 'vbm-amount-mean-lMT', 'vbm-amount-mean-rMT', 'thick_lMT', 'thick_rMT', 'area_lMT', 'area_rMT']]
# measname_size_comb = ['masksize', 'alff', 'bnc', 'vbm-amount', 'cortical thick', 'cortical area']
# measfeature_size = analysebase.FeatureRelation(data_size.values, measname_size_comb, mergehemi = True, figure = True)
# r2_size, beta_size, tval_size, tpval_size, f_size, fpval_size = measfeature_size.feature_prediction2(glm)
# score_size, n_score_size, pval_size = measfeature_size.feature_prediction3(glm)

# data_dist = data_pd[['dist_all0_lMT', 'dist_all0_rMT', 'alff-mean-lMT', 'alff-mean-rMT', 'mtsys_bnc', 'mtsys_bnc_same', 'vbm-amount-mean-lMT', 'vbm-amount-mean-rMT', 'thick_lMT', 'thick_rMT', 'area_lMT', 'area_rMT']]
measname_dist_comb = ['dist', 'alff', 'bnc', 'vbm-amount', 'cortical thick', 'cortical area']
data_dist = data_pd[['act-mean-dist-lMT', 'act-mean-dist-rMT', 'alff-mean-lMT', 'alff-mean-rMT', 'mtsys_bnc', 'mtsys_bnc_same', 'vbm-amount-mean-lMT', 'vbm-amount-mean-rMT', 'thick_lMT', 'thick_rMT', 'area_lMT', 'area_rMT']]
# measname_dist_comb = ['dist', 'alff', 'bnc']
measfeature_dist = analysebase.FeatureRelation(data_dist.values, measname_dist_comb, mergehemi = True, figure = True)
r2_dist, beta_dist, tval_dist, tpval_dist, f_dist, fpval_dist = measfeature_dist.feature_prediction2(glm)
score_dist, n_score_dist, pval_dist = measfeature_dist.feature_prediction3(glm)
# ---------------------------------------------------------------------------
## Functional connectivity hierachy structure and activation pattern similarity hierachy strucutre
# data_stem = 'data_heatmap'
# actpat_data = iofiles.load_pkl(pjoin(data_parpath, data_stem, 'actsignal_rg.pkl'))
# restpat_data = iofiles.load_pkl(pjoin(data_parpath, data_stem, 'restsignal_rg.pkl'))
# region_comb = ['OFA', 'pFus', 'pSTS', 'MT']
# actpatmap_comb = analysebase.ComPatternMap(actpat_data, region_comb, mergehemi = True, figure = True)
# restpatmap_comb = analysebase.ComPatternMap(restpat_data, region_comb, mergehemi = True, figure = True)
# corr_actmap, distance_actmap = actpatmap_comb.patternmap()
# corr_restmap, distance_restmap = restpatmap_comb.patternmap()



