# -*- coding: utf-8 -*-
"""
Created on Tue May 29 08:28:12 2018

@author: XuBo
"""
from __future__ import division, print_function

import sys
sys.path.append(r'C:\Users\xub\Desktop\Python project\Packages')
#sys.path.append(r'E:\xbl_Berry\Desktop\Python project\Packages')
    
from timeit import default_timer as timer
from lib.crystals.data import lbo3
from lib.crystals.crystals import crystal
from lib.coordinate import xy, xyt
from lib.propagator.gpu import xypropagator, xytpropagator, propagator

import numpy as np

def Gau(w0, x, y, wt=None, t=None):
    return np.exp(-(x**2+y**2)/2/w0) if ((wt is None) or (t is None)) else np.exp(-(x**2+y**2)/2/w0)*np.exp(-t**2/2/wt)

wl = 1.064 #1.064 um
w0 = 500 # 500um
wt = 30 # 30ps
dz = 50000 # 500um

xsize = 128
ysize = 128
tsize = 64
size = 3
    
crys = crystal(lbo3, wl*1e3, 25, 90, 45)
crys.show()
    
key = 'hi'

x = np.linspace(-256*size, 256*size, xsize, dtype=np.float32)
y = np.linspace(-256*size, 256*size, ysize, dtype=np.float32)

xx, yy = np.meshgrid(x, y, indexing='ij')

E = Gau(w0, xx, yy) 
E = E.astype(np.complex128)

start = timer()

sol1 = xypropagator(E, x, y, dz, crys, key)

end = timer()
print(end - start)

space1 = xy(x, y)

E = Gau(w0, space1.xx, space1.yy)
E = E.astype(np.complex128)

start = timer()

sol = propagator(crys, space1, key)

sol.load(dz)
sol.propagate(E)

sol2 = sol.get()

end = timer()
print(end - start)

print(np.allclose(sol1, sol2)) 
   
t = np.linspace(-64*1, 63*1, tsize, dtype=np.float32)

xx, yy, tt = np.meshgrid(x, y, t, indexing='ij')

E = Gau(w0, xx, yy, wt, tt)
E = E.astype(np.complex128)

ref = True

start = timer()

sol3 = xytpropagator(E, x, y, t, dz, crys, key, ref)

end = timer()
print(end - start) 

space2 = xyt(x, y, t)

E = Gau(w0, space2.xxx, space2.yyy, wt, space2.ttt)
E = E.astype(np.complex128)

start = timer()

sol = propagator(crys, space2, key)
sol.load_ref(sol.para['dw'])

sol.load(dz)
sol.propagate(E)
 
sol4 = sol.get()

end = timer()
print(end - start)

print(np.allclose(sol3, sol4))
