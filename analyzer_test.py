# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:

from analyzer import *
import os
import cPickle

# load data from disk to RAM
data_path = './mydata'
file_name = 'mt-zstat-falffsph0.pkl'
mt_file = open(os.path.join(data_path, file_name), 'r')
data = cPickle.load(mt_file)
mt_file.close()


# prep inputs
meas = data['meas_mean'][:100, :]
subj_id = data['subj_id']
roi_name = data['roi_name']
meas_name = data['meas_name']
subj_gender = data['subj_gender']
# meas_type = 'geometry'
meas_type = 'scalar'

# generate an analyzer
mt_analyzer = Analyzer(meas, meas_type, meas_name, roi_name, subj_id, subj_gender)

# remove outlier
mt_analyzer.outlier_remove(meth='std', figure=True)

"""
# merge data from two hemisphere
mt_analyzer.hemi_merge()

sel = np.arange(mt_analyzer.meas.shape[1])


# description for each features, and save stats
feat_stats = mt_analyzer.feature_description(feat_sel=sel, figure=True)
np.savetxt(os.path.join(data_path, 'feat_stats.txt'), feat_stats, fmt='%.3f')


# description for the relation among each pair of feature
feat_corr, feat_pval, n_sample = mt_analyzer.feature_relation(feat_sel=sel, figure=False)

# uni-variate  behavior predict
beh_meas = np.mean(meas, axis=1) # fake behavior use mean brain meas
beh_corr, beh_pval, beh_nsamp = mt_analyzer.behavior_predict1(beh_meas, ['fakeBeh'], feat_sel=sel, figure=False)

# multivariate behavior predict
beh_meas = np.random.randn(meas.shape[0], 1)
slop_stats, r2, dof = mt_analyzer.behavior_predict2(beh_meas, ['RandBeh'], feat_sel=sel, figure=False)

# calculate hemisphere asymmetry
li_stats = mt_analyzer.hemi_asymmetry(feat_sel=sel, figure=False)

# calculate gender difference
gd_stats = mt_analyzer.gender_diff(feat_sel=sel, figure=False)
"""