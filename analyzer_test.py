# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:


from analyzer import *
import os
import cPickle


data_path = './data'
mt_data_path = os.path.join(data_path, 'mt.pkl')

mt_file = open(mt_data_path, 'r')
data = cPickle.load(mt_file)
file.close()

roi_name = data.roi_name
subj_id = data.subj_id
roi_name = data.roi_name
meas = data.meas
feature_name = data.feature_name

mt_analyzer = Analyzer(meas, subj_id, roi_name, feature_name)
mt_analyzer.feature_description()
mt_analyzer.feature_relation()
mt_analyzer.behavior_predict1(beh_meas)





