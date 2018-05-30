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
from lib.ode.gpu import ode       
from lib.tools.plot.xy import image
from lib.nonlinear.gpu import sfg
  
xsize = 128
ysize = 128
tsize = 64

z = 5000

w0 = 500 # 500um
wt = 30 # 30ps

def Gau(w0, wt, x, y, t):
    return np.exp(-(x**2+y**2)/2/w0)*np.exp(-t**2/2/wt) if wt != 0 else np.exp(-(x**2+y**2)/2/w0)

def para(wl, d, n):
    return d/wl/n

ns = np.array([1.605, 1.605, 1.605])
deff = 8.32e-7
wls =np.array([1.064, 1.064, 1.064])

args = np.append(para(wls, deff, ns), 0) # 4 element array
args = tuple(args)

x = np.linspace(-128*1, 127*1, xsize)
y = np.linspace(-128*1, 127*1, ysize)
t = np.linspace(-64*1, 63*1, tsize)

xx, yy, tt= np.meshgrid(x, y, t, indexing='ij')

E = 100*Gau(w0, wt, xx, yy, tt)

E = E.astype(np.complex128)  

#    image(E[:,:,tsize//2])

A = E.copy()
B = E.copy()
C = np.zeros_like(E,dtype=np.complex128)

start = timer()

test = ode(0, 5000, sfg, 3, args)#overhead

end = timer()   
print (end-start)    

start = timer()

test.move([A, B, C])

sol = test.get()[2]

image(A[xsize//2,:,:])
image(B[xsize//2,:,:])
image(C[xsize//2,:,:])
image(sol[xsize//2,:,:])

end = timer()    
print (end-start)
