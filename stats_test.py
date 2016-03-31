# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:

from analyzer import *
from statistic import *
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


mt_stats = stats(mt_analyzer)
beh_meas = np.mean(meas, axis=1) # fake behavior use mean brain meas
boot_stats = mt_stats.bootstrap('behavior_predict1', beh_meas)

# merge data from two hemisphere
# mt_analyzer.hemi_merge()

# sel = np.arange(mt_analyzer.meas.shape[1])
