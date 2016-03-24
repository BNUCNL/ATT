
from ImageCalculator import  *
import os

data_path = './data'


falff = nib.load(os.path.join(data_path, 'zstat.nii.gz'))
zstat = nib.load(os.path.join(data_path, 'zstat.nii.gz'))

img_calc = ImageCalculator(4)

"""
ic = img_calc.add(falff, zstat)
nib.save(ic, os.path.join(data_path, 'add.nii.gz'))
"""

ip = img_calc.pearsonr(falff, zstat)
nib.save(ip, os.path.join(data_path, 'pearsonr.nii.gz'))