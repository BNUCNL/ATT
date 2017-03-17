# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
ATT package is a toolbox used for analyzing nifti data

Modules
------------
corefunc: core operation package of ATT
algorithm: algorithm storehouse of ATT. Main completion in corefunc comes from here
iofunc: input/output package, used for load/save files
utilfunc: utility package, contains figure module (plotfig), file operation module (fileoperate) and decorator

Usage:
    >>> from ATT.corefunc import atlasbase
"""


__all__ = []
from . import algorithm
from . import corefunc
from . import utilfunc
from . import iofunc





