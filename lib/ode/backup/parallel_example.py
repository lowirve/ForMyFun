# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 10:56:39 2018

@author: XuBo
"""

from __future__ import division, print_function

from scipy.integrate import ode
import numpy as np
import multiprocessing as mp
from functools import partial

#   It is unclear why, but it only works when all the functions are declared 
#   on top level of if __name__ == "__main__" section
def Gau(w0, wt, x, y, t):
    return np.exp(-(x**2+y**2)/2/w0)*np.exp(-t**2/2/wt) if wt != 0 else np.exp(-(x**2+y**2)/2/w0)

def para(wl, d, n):
    return d/wl/n

def f(z, y, arg):
    return np.array([1j*arg[0]*np.conj(y[1])*y[2]*np.exp(1j*arg[3]*z), 1j*arg[1]*np.conj(y[0])*y[2]*np.exp(1j*arg[3]*z), 
                         1j*arg[2]*y[0]*y[1]*np.exp(-1j*arg[3]*z)])


def codeint(y0, x0, x1, f, args):
        
    r = ode(f).set_integrator('zvode')
    r.set_initial_value(y0, x0).set_f_params(args)
    sol = r.integrate(x1)

    if not r.successful():
        raise RuntimeError("ode failed")
    else:
        return sol    
    
    
def mp_ode(y0, x0, x1, f, args):
        
    cores = int(mp.cpu_count()-2)
    
    para = {'x0':x0, 'x1':x1, 'f':f, 'args':args}
        
    _f = partial(codeint, **para)
    
    p = mp.Pool(cores)
    sol = p.map(_f, y0)
    
    return np.array(sol)
   
    
if __name__ == "__main__": 
    
    from timeit import default_timer as timer

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

#   CPU serial computing   
    start = timer()
    for j, _  in enumerate(y):
        for i, _ in enumerate(x):
            y0 = np.array([E[i, j], E[i, j], 0])
            E1[i,j,:] = codeint(y0, x0, x1, f, args)
    
    end = timer()
    
    print (end-start)

#  CPU parallel computing
    start = timer()
    EEE = np.dstack([E,E,np.zeros_like(E)]).reshape(-1,3)
#    EEE = E1[:,:,:,0].T.reshape(-1,3) #not sure how to work yet. More revisions are required.
    
    sol = mp_ode(EEE, x0, x1, f, args)

    E2 = np.empty_like(E1)
    E2[:] = sol.reshape(xsize, ysize, 3)
    
    end = timer()
    
    print (end-start)
    
    