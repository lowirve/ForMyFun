# -*- coding: utf-8 -*-
"""

"""

from __future__ import division, print_function

import sys

sys.path.append(r'C:\Users\xub\Desktop\Python project\Packages\lib')
#sys.path.append(r'E:\xbl_Berry\Desktop\Python project\Packages\lib')

from numba import cuda, complex128, float64
from cmath import exp
import numpy as np

from timeit import default_timer as timer
from simulation.nonlinear.gpu import sfg  
from simulation.laser.xy import normgau
from simulation.coordinate import xy

from simulation.crystals import crystal
from simulation.crystals.data import lbo3

from simulation.propagator.gpu import propagator
from simulation.ode.gpu64 import ode

#from simulation.crystals.phasematch


from plot.xy import image
      
xsize = 128
ysize = 128

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

h = 5000/100

test = ode(0, h, sfg, 3, args)

red1.load(A, h/2)
red2.load(B, h/2, stream=red1.stream)
blue.load(C, h/2, stream=red1.stream)

#test._load([red1._get(), red2._get(), blue._get()], stream=red1.stream)
    
for i in range(100):
    red1.move()
    red2.move()
    blue.move()
    
    test._load([red1._get(), red2._get(), blue._get()], stream=red1.stream)
    test.move()
    
    temp = test._get()
    
    red1._load(temp[0], h, stream=test.stream)
    red2._load(temp[1], h, stream=test.stream)
    blue._load(temp[2], h, stream=test.stream)

#red1._load(temp[0], h/2, stream=test.stream)
#red2._load(temp[1], h/2, stream=test.stream)
#blue._load(temp[2], h/2, stream=test.stream)

red1.move()
red2.move()
blue.move()

end = timer()   
print (end-start)    

start = timer()

image(blue.get())

end = timer()    
print (end-start)


    