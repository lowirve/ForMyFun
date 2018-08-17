# -*- coding: utf-8 -*-
"""

"""

from __future__ import division, print_function

import sys

sys.path.append(r'C:\Users\xub\Desktop\Python project\Packages\temp')
#sys.path.append(r'E:\xbl_Berry\Desktop\Python project\Packages\lib')

from numba import cuda
import numpy as np

from timeit import default_timer as timer
from nonlinear.gpu import sfg  
from tools.functions import normgau
from coordinate import xyt#, xy

from crystals.crystals import crystal
from crystals.data import lbo3

from propagator.gpu import propagator
from ode.gpu import ode

#from simulation.crystals.phasematch

#from plot.xy import image
    

def evolution(space, crystals, keys, lasers, args, gf, z, step):

    start = timer()
    
    A, B, C = lasers
    
    red1, red2, blue = crystals
    
    red1 = propagator(red1, space, keys[0])
    ref = red1.para['dw']*space.www
    red1.load_ref(ref)
    red2 = propagator(red2, space, keys[1], ref)
    blue = propagator(blue, space, keys[2], ref)
    
    h = z/step
    
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
        
    for i in range(step-1):
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
    time = end-start
    print (time)
    
#    print(red1.get().dtype)
#    image(E[:,:,tsize//2])  
#    
#    image(red1.get()[:,:,tsize//2])
#    
#    image(red1.get()[:,ysize//2,:])
#    image(red2.get()[:,ysize//2,:])
#    image(blue.get()[:,ysize//2,:])

    return time
 
    
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    sizes = [(64,64,64)]#, (256, 256, 128)] 
    
    times = []
    
    steps = [50, 100, 200]#, 1000]
    
    dtype = np.float64
    
    fig, ax = plt.subplots()
    
    xaxis = [np.prod(size) for size in sizes]
    
    xaxis.insert(0, 0)
    
    xaxis = np.array(xaxis)
    
    for i, step in enumerate(steps):
        
        print('Trial {0}, step number is {1}:'.format(i, step))
        
        temp = []
        
        for size in sizes:
            
            xsize, ysize, tsize = size
            
            dxy = 1
            dt = 1
            
            z = 5000
            
            w0 = 500 # 500um
            wt = 30 # 30ps
            
            def para(wl, d, n):
                return d/wl/n
            
            ns = np.array([1.605, 1.605, 1.605])
            deff = 8.32e-7
            wls =np.array([1.064, 1.064, 1.064])
            
            args = np.append(para(wls, deff, ns), 0) # 4 element array
            args = tuple(args)
            
            x = np.linspace(-128*dxy, 127*dxy, xsize, dtype=dtype)
            y = np.linspace(-128*dxy, 127*dxy, ysize, dtype=dtype)
            t = np.linspace(-64*dt, 63*dt, tsize, dtype=dtype)
            
            space = xyt(x, y, t)
            
            E = 100*normgau(t=space.ttt, x=space.xxx, y=space.yyy, wt=wt, wx=w0)
            
            print(E.dtype)
            
#            E = E.astype(np.complex64)  
            
            A = E.copy()
            B = E.copy()
            C = np.zeros_like(E)
            
            print(C.dtype)
            
            keys = ['hi', 'hi', 'lo']
            
            red1 = crystal(lbo3, 1064, 25, 90, 11.4)
            red2 = crystal(lbo3, 1064, 25, 90, 11.4)
            blue = crystal(lbo3, 532, 25, 90, 11.4)
                
            crystals = red1, red2, blue
            
            lasers = A, B, C
            
            temp.append(evolution(space, crystals, keys, lasers, args, sfg, z, step))
            
#        p = np.polyfit(xaxis[1:], temp, 1)
#    
#        ax.plot(xaxis[1:], temp, linestyle='-', marker='o')
#    
#        ax.plot(xaxis, p[0]*xaxis+p[1], linestyle='-')
#        
#        times.append(temp)
#        print()
#        
#    times = np.array(times)    
#    
#    np.savetxt('data.csv', times, delimiter='\t')        
#
#    ax.set_xscale('log')
#    ax.set_yscale('log')
#    
#    plt.tight_layout()
#    
#    plt.show()
#    
    
    