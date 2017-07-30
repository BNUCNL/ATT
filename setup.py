# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et;

from distutils.core import setup

setup(name='ATT',
      version='1.0',
      description='Toolbox for nifti data analysis',
      author='Taicheng Huang',
      author_email='taicheng_huang@sina.cn',
      url='https://github.com/helloTC/ATT',
      packages=['algorithm', 'graph', 'iofunc', 'surface', 'util', 'volume'],
      install_requires=['numpy', 'scipy', 'nibabel', 'six', 'sklearn', 'pandas', 'matplotlib', 'seaborn']
      )
