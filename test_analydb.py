# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:

"test analysedb.py"

from analysedb import *

pklpath = '/nfs/j3/userhome/huangtaicheng/workingdir/program/ATT/data'
pklfile = 'index_data.pkl'
areaname = ['rV3', 'lV3', 'rMT', 'lMT']
areanum = [1,2,3,4]


atlas = AtlasDB()
atlas.read_from_pkl(pklpath, pklfile)

volume = atlas.output_data('geo', 'volume')
mean_alff = atlas.output_data('rest', 'alff', 'mean')
zstat_peakcoor = atlas.output_data('geo', 'peakcoor', 'zstat')

atlasdes = AtlasDescribe(volume, areaname)
atlasdes.subjexist('lMT')
atlasdes.paradescrib('lMT')







