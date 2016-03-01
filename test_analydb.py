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

# volume
volume = atlas.output_data('geo', 'volume')
# zvalue
mean_zstat = atlas.output_data('act', 'zstat', 'mean')
peak_zstat = atlas.output_data('act', 'zstat', 'max')
std_zstat = atlas.output_data('act', 'zstat', 'std')
# psc value
mean_psc = atlas.output_data('act', 'psc', 'mean')
peak_psc = atlas.output_data('act', 'psc', 'max')
std_psc = atlas.output_data('act', 'psc', 'std')
# alff
mean_alff = atlas.output_data('rest', 'alff', 'mean')
peak_alff = atlas.output_data('rest', 'alff', 'max')
std_alff = atlas.output_data('rest', 'alff', 'std')
# falff
mean_falff = atlas.output_data('rest', 'falff', 'mean')
peak_falff = atlas.output_data('rest', 'falff', 'max')
std_falff = atlas.output_data('rest', 'falff', 'std')
# reho
mean_reho = atlas.output_data('rest', 'reho', 'mean')
peak_reho = atlas.output_data('rest', 'reho', 'max')
std_reho = atlas.output_data('rest', 'reho', 'std')
# peakcoor
zstat_peakcoor = atlas.output_data('geo', 'peakcoor', 'zstat')
psc_peakcoor = atlas.output_data('geo', 'peakcoor', 'psc')
alff_peakcoor = atlas.output_data('geo', 'peakcoor', 'alff')
falff_peakcoor = atlas.output_data('geo', 'peakcoor', 'falff')
reho_peakcoor = atlas.output_data('geo', 'peakcoor', 'reho')

#----------------volume------------------------------------
atlasdes_volume = AtlasDescribe(volume, areaname)
#------------------------------
"""
Calculate "percent subject" and "number of subject"
"""
existnum = np.empty(len(areaname))
existperc = np.empty(len(areaname))
for i in range(len(areaname)):
    atlasdes_volume.subjexist(areaname[i])
    existnum[i] = atlasdes_volume.existnum
    existperc[i] = atlasdes_volume.existperc
#--------------------------------
volumemean = []
volumestd = []
volumeli = []
for areai in areaname:
    atlasdes_volume.paradescrib(areai)
    volumemean.append(atlasdes_volume.datamean)
    volumestd.append(atlasdes_volume.datastd)
volumemean = np.array(volumemean)
volumestd = np.array(volumestd)

for i in range(0, len(areaname)-1, 2):
    atlasdes_volume.calhemLi(areaname[i][1:])
    volumeli.append(atlasdes_volume.livalue)
volumeli = np.array(volumeli)
#------------------------------------------------------------

atlasdes = AtlasDescribe(mean_alff, areaname)



















atlasdes.subjexist('lMT')
atlasdes.paradescrib('lMT')
atlasdes.calhemLi('V3')






