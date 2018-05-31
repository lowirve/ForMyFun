
from __future__ import division, print_function

import numpy as np
from cmath import exp

from numba import cuda

from gpu import ode



@cuda.jit(device=True)
def gf(x, y, arg, h):#x is scalar, y is a tuple, arg is a tuple
    dydx0 = 1j*arg[0]*y[1].conjugate()*y[2]*exp(1j*arg[3]*x)*h
    dydx1 = 1j*arg[1]*y[0].conjugate()*y[2]*exp(1j*arg[3]*x)*h
    dydx2 = 1j*arg[2]*y[0]*y[1]*exp(-1j*arg[3]*x)*h
    return dydx0, dydx1, dydx2 

def Gau(w0, wt, x, y, t):
    return np.exp(-(x**2+y**2)/2/w0)*np.exp(-t**2/2/wt) if wt != 0 else np.exp(-(x**2+y**2)/2/w0)

def para(wl, d, n):
    return d/wl/n

if __name__ == "__main__": 
    from timeit import default_timer as timer
    xsize = 128
    ysize = 128
    
    w0 = 500 # 500um
   
    ns = np.array([1.617, 1.617, 1.617])
    deff = 4.47e-7
    wls =np.array([.607, .607, .303])
    
    args = np.append(para(wls, deff, ns),0) # 4 element array
    args = tuple(args)
    
    x = np.linspace(-128*1, 127*1, xsize)
    y = np.linspace(-128*1, 127*1, ysize)
    
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    E = 100*Gau(w0, 10, xx, yy, 0)
    
    E = E.astype(np.complex128)  
    
    A = E.copy()
    B = E.copy()
    C = np.zeros_like(E,dtype=np.complex128)
    
    E2 = np.empty((xsize, ysize, 3), dtype=np.complex128)
    
    start = timer()
    
    test = ode(0, 5000, gf, 3, args)#overhead
    
    end = timer()   
    print (end-start)    
    
    start = timer()
    
    test.move([A, B, C], [E2[:,:,i] for i in range(3)])
    
    end = timer()    
    print (end-start)