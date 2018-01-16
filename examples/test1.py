# -*- coding: utf-8 -*-
"""

"""

from __future__ import division, print_function

import sys

sys.path.append(r'C:\Users\xub\Desktop\Python project\Packages\lib')
#sys.path.append(r'E:\xbl_Berry\Desktop\Python project\Packages\lib')

from numba import cuda
import numpy as np

from timeit import default_timer as timer
from simulation.nonlinear.gpu import sfg  
from simulation.laser.xy import normgau
from simulation.coordinate import xy

from simulation.crystals import crystal
from simulation.crystals.data import lbo3

from simulation.propagator.gpu import propagator
from simulation.ode.gpu import ode

#from simulation.crystals.phasematch


from plot.xy import image
      
xsize = 256
ysize = 256

z = 5000

w0 = 500 # 500um

def para(wl, d, n):
    return d/wl/n

ns = np.array([1.605, 1.605, 1.605])
deff = 8.32e-7
wls =np.array([1.064, 1.064, 1.064])

args = np.append(para(wls, deff, ns), 0) # 4 element array
args = tuple(args)

x = np.linspace(-128*1, 127*1, xsize)
y = np.linspace(-128*1, 127*1, ysize)

space = xy(x, y)

E = 100*normgau(space.xx, space.yy, w0)

E = E.astype(np.complex128)  

A = E.copy()
B = E.copy()
C = np.zeros_like(E,dtype=np.complex128)

start = timer()

red1 = crystal(lbo3, 1064, 25, 90, 11.4)
red2 = crystal(lbo3, 1064, 25, 90, 11.4)
blue = crystal(lbo3, 532, 25, 90, 11.4)

red1 = propagator(red1, space, 'hi')
red2 = propagator(red2, space, 'hi')
blue = propagator(blue, space, 'lo')

h = z/100

stream = cuda.stream()

test = ode(0, h, sfg, 3, args)

red1.load(h/2, stream)
red2.load(h/2, stream)
blue.load(h/2, stream)

red1.propagate(A)
red2.propagate(B)
blue.propagate(C)

test.move([red1._get(), red2._get(), blue._get()], stream)

red1.load(h, stream)
red2.load(h, stream)
blue.load(h, stream)
    
for i in range(int(z//h)-1):
    temp = test._get()
    
    red1.propagate(temp[0])
    red2.propagate(temp[1])
    blue.propagate(temp[2])
    
    test.move([red1._get(), red2._get(), blue._get()], stream)

red1.load(h/2, stream)
red2.load(h/2, stream)
blue.load(h/2, stream)

temp = test._get()

red1.propagate(temp[0])
red2.propagate(temp[1])
blue.propagate(temp[2])

end = timer()   
print (end-start)    

start = timer()

image(blue.get())

end = timer()    
print (end-start)


    