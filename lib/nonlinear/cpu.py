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


if __name__ == "__main__": 
    
    import sys

    sys.path.append(r'C:\Users\xub\Desktop\Python project\Packages\lib')
    #sys.path.append(r'E:\xbl_Berry\Desktop\Python project\Packages\lib')
    
    from timeit import default_timer as timer
    from ode.cpu import ode

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


