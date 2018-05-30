# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 10:49:31 2018

@author: XuBo
"""

from __future__ import division, print_function

import numpy as np

def sfg(z, y, arg):
    return np.array([1j*arg[0]*np.conj(y[1])*y[2]*np.exp(1j*arg[3]*z), 
                     1j*arg[1]*np.conj(y[0])*y[2]*np.exp(1j*arg[3]*z), 
                     1j*arg[2]*y[0]*y[1]*np.exp(-1j*arg[3]*z)])

