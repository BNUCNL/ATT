# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et;

from distutils.core import setup

setup(name='ATT',
      version='0.5',
      description='Toolbox for neuroimaging data analysis',
      author='Taicheng Huang',
      author_email='taicheng_huang@mail.bnu.edu.cn',
      url='https://github.com/helloTC/ATT',
      packages=['ATT.algorithm', 'ATT.iofunc', 'ATT.util'],
      install_requires=['numpy', 'scipy', 'nibabel', 'six', 'sklearn', 'pandas', 'matplotlib', 'seaborn']
      )
