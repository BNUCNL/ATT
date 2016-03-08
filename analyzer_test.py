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
feat_name = data['feat_name']

mt_analyzer = Analyzer(meas, subj_id, roi_name, feat_name)

mt_analyzer.feature_description()

mt_analyzer.feature_relation()

# fake behavior use mean brain meas
beh_meas = np.mean(meas, axis=1)
mt_analyzer.behavior_predict1(beh_meas)





