# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:


from analyzer import *
import os
import cPickle


data_path = './data'
file_name = 'mt-zstat-falff.pkl'
mt_file = open(os.path.join(data_path, file_name), 'r')
data = cPickle.load(mt_file)
mt_file.close()

meas = data['meas_mean']
subj_id = data['subj_id']
roi_name = data['roi_name']
meas_name = data['meas_name']
subj_gender = data['subj_gender']

mt_analyzer = Analyzer(meas, meas_name, roi_name, subj_id, subj_gender)

mt_analyzer.hemi_merge()

plt.close('all')
# feat_stats = mt_analyzer.feature_description()

# feat_corr, feat_pval, n_sample = mt_analyzer.feature_relation(figure=True)

# fake behavior use mean brain meas
#beh_meas = np.mean(meas, axis=1)
#beh_corr, beh_pval, beh_nsamp = mt_analyzer.behavior_predict1(beh_meas, ['fakeBeh'])

#beh_meas = np.random.randn(meas.shape[0], 1)
#reg_stats = mt_analyzer.behavior_predict2(beh_meas, ['RandBeh'], figure=True)

#li_stats = mt_analyzer.hemi_asymmetry(figure=True)

# gd_stats = mt_analyzer.gender_diff(figure=True)