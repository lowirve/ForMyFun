# -*- coding: utf-8 -*-
"""
Created on Tue May 29 08:16:20 2018

@author: XuBo
"""
from __future__ import division, print_function

import sys
import numpy as np

sys.path.append(r'C:\Users\xub\Desktop\Python project\Packages')
#sys.path.append(r'E:\xbl_Berry\Desktop\Python project\Packages')

from timeit import default_timer as timer
from lib.ode import ode
from lib.nonlinear import sfg

def Gau(w0, wt, x, y, t):
    return np.exp(-(x**2+y**2)/2/w0)*np.exp(-t**2/2/wt) if wt != 0 else np.exp(-(x**2+y**2)/2/w0)

def para(wl, d, n):
    return d/wl/n

xsize = 256
ysize = 256

w0 = 500 # 500um

ns = np.array([1.617, 1.617, 1.617])
deff = 4.47e-7
wls =np.array([.607, .607, .303])

args = np.append(para(wls, deff, ns),0) # 4 element array

x = np.linspace(-128*1, 127*1, xsize)
y = np.linspace(-128*1, 127*1, ysize)

xx, yy = np.meshgrid(x, y, indexing='ij')

E = 100*Gau(w0, 10, xx, yy, 0)

E1 = np.empty((xsize, ysize, 3), dtype=np.complex128)

x0, x1 = 0, 5000 

A = E.copy()
B = E.copy()
C = np.zeros_like(E,dtype=np.complex128)

start = timer()
y0 = A, B, C
E1[:,:,0], E1[:,:,1], E1[:,:,2] = ode(x0, x1, y0, sfg, args)[0]

end = timer()    

print (end-start)   


