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
from simulation.laser.xyt import normgau
from simulation.coordinate import xyt, xy

from simulation.crystals import crystal
from simulation.crystals.data import lbo3

from simulation.propagator.gpu import propagator
from simulation.ode.gpu import ode

#from simulation.crystals.phasematch

from plot.xy import image
    

def evolution(space, crystals, keys, lasers, args, gf, z, step):

    start = timer()
    
    A, B, C = lasers
    
    red1, red2, blue = crystals
    
    red1 = propagator(red1, space, keys[0])
    red1.load_ref(red1.para['dw'])
    red2 = propagator(red2, space, keys[1], red1.para['dw'])
    blue = propagator(blue, space, keys[2], red1.para['dw'])
    
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
    time = end-start
    print (time)
    
#    image(E[:,:,tsize//2])  
#    
#    image(red1.get()[:,:,tsize//2])
#    
#    image(red1.get()[:,ysize//2,:])
#    image(red2.get()[:,ysize//2,:])
#    image(blue.get()[:,ysize//2,:])

    return time
 
    
if __name__ == '__main__':

    sizes = [(32,32,32), (64,32,32), (64,64,32), (64,64,64), (128, 128, 64), (128, 128, 128), (256, 256, 128)] 
    
    times = []
    
    steps = [10, 50, 100, 200, 1000]
    
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
            
            x = np.linspace(-128*dxy, 127*dxy, xsize)
            y = np.linspace(-128*dxy, 127*dxy, ysize)
            t = np.linspace(-64*dt, 63*dt, tsize)
            
            space = xyt(x, y, t)
            
            E = 100*normgau(space.ttt, space.xxx, space.yyy, wt, w0)
            
            E = E.astype(np.complex128)  
            
            A = E.copy()
            B = E.copy()
            C = np.zeros_like(E,dtype=np.complex128)
            
            keys = ['hi', 'hi', 'lo']
            
            red1 = crystal(lbo3, 1064, 25, 90, 11.4)
            red2 = crystal(lbo3, 1064, 25, 90, 11.4)
            blue = crystal(lbo3, 532, 25, 90, 11.4)
                
            crystals = red1, red2, blue
            
            lasers = A, B, C
            
            temp.append(evolution(space, crystals, keys, lasers, args, sfg, z, step))
        
        times.append(temp)
        print()
        
    times = np.array(times)    
    
    np.savetxt('data.csv', times, delimiter='\t')
        
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    
    xaxis = np.array([np.prod(np.array(size)) for size in sizes])
    
    line1, = ax.plot(xaxis, times[2], linestyle='-', marker='o')
    
    p = np.polyfit(xaxis[:3], times[2][:3], 1)
    
    line2, = ax.plot(xaxis, p[0]*xaxis+p[1], linestyle='-')

    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    plt.show()
    
    
    