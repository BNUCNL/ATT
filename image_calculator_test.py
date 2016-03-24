
from image_calculator import *
import os

data_path = './data'


falff = nib.load(os.path.join(data_path, 'zstat.nii.gz'))
zstat = nib.load(os.path.join(data_path, 'zstat.nii.gz'))

# construct 4d calculator
img_calc = ImageCalculator(4)

# image add and save
ic = img_calc.add(falff, zstat)
nib.save(ic, os.path.join(data_path, 'add.nii.gz'))

# image correlation and save
ip = img_calc.pearsonr(falff, zstat)
nib.save(ip, os.path.join(data_path, 'pearsonr.nii.gz'))