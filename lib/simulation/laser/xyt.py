# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 10:58:17 2018

@author: XuBo
"""

from __future__ import division, print_function

import numpy as np

def normgau(t, x, y, wt, wx, wy=None):
    if wy is None:
        wy = wx
        
    return np.exp(-(x**2)/2/wx)*np.exp(-(y**2)/2/wy)*np.exp(-t**2/2/wt)+0j

