# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 10:58:17 2018

@author: XuBo
"""

from __future__ import division, print_function

import numpy as np

def normgau(dimension, width, shift=[0, 0, 0]):
    
    sol = 1
    
    for i in range(len(dimension)):
        sol *= np.exp(-2*((dimension[i]-shift[i])/width[i])**2)
        
    return sol + 0j    

def integrate(values, steps):
    return np.sum(values*np.prod(steps))


if __name__ == '__main__':
    
    import sys

    sys.path.append(r'C:\Users\xub\Desktop\Python project\Packages\lib')
    #sys.path.append(r'E:\xbl_Berry\Desktop\Python project\Packages\lib')
    
    from coordinate import xyt, xy
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
    xsize, ysize, tsize = 128, 128, 128
    
    dxy = 1
    dt = 1
    
    w0 = 50 # 500um
    wt = 30 # 30ps
    
    dtype = np.float64
    
    x = np.linspace(-128*dxy, 127*dxy, xsize, dtype=dtype)
    y = np.linspace(-128*dxy, 127*dxy, ysize, dtype=dtype)
    t = np.linspace(-64*dt, 63*dt, tsize, dtype=dtype)
    
    a = xy(x, y)
    b = xyt(x, y, t)
    
    ia = normgau([a.xx, a.yy], [w0, w0])
    ib = normgau([b.xxx, b.yyy, b.ttt], [w0, w0, wt])   
    
    print('numerical result: {}'.format(integrate(abs(ia),(a.dx,a.dy))))
    print('analytical result: {}'.format(np.pi/2*w0**2))    
    
    print()
    
    print('numerical result: {}'.format(integrate(abs(ib),(b.dx,b.dy,b.dt))))
    print('analytical result: {}'.format((np.pi/2)**1.5*w0**2*wt))     
    
    fig, (ax1, ax2) = plt.subplots(1,2,subplot_kw={'projection': '3d'})
    
#    ax1.pcolor(np.abs(ia))
    
    ax1.plot_wireframe(a.xx, a.yy, np.abs(ia[:,:]), rstride=8, cstride=8)
    ax2.plot_wireframe(b.xx, b.yy, np.abs(ib[:,ysize//2,:]), rstride=8, cstride=8)
    
#    ax1.axis('equal')
#    ax2.axis('equal')
    
    plt.tight_layout()
    
    plt.show()
    
    
    
    
    