# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 10:58:17 2018

@author: XuBo
"""

from __future__ import division, print_function

import numpy as np

def normgau(x, y, wx, t=None, wt=None, wy=None):
    if wy is None:
        wy = wx
    
    if wt is None or t is None:
        t = 0
        wt = 1
        
    return np.exp(-2*(x/wx)**2)*np.exp(-2*(y/wy)**2)*np.exp(-2*(t/wt)**2)+0j    


if __name__ == '__main__':
    
    import sys

    sys.path.append(r'C:\Users\xub\Desktop\Python project\Packages\lib')
    #sys.path.append(r'E:\xbl_Berry\Desktop\Python project\Packages\lib')
    
    from simulation.coordinate import xyt, xy
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
    
    ia = normgau(a.xx, a.yy, w0)
    ib = normgau(b.xxx, b.yyy, w0, b.ttt, wt)
    
    
    
    print('numerical result: {}'.format(np.sum(abs(ia)*a.dx*a.dy)))
    print('analytical result: {}'.format(np.pi/2*w0**2))    
    
    print()
    
    print('numerical result: {}'.format(np.sum(abs(ib)*b.dx*b.dy*b.dt)))
    print('analytical result: {}'.format((np.pi/2)**1.5*w0**2*wt)) 
    
    
    fig, (ax1, ax2) = plt.subplots(1,2,subplot_kw={'projection': '3d'})
    
#    ax1.pcolor(np.abs(ia))
    
    ax1.plot_wireframe(a.xx, a.yy, np.abs(ia[:,:]), rstride=8, cstride=8)
    ax2.plot_wireframe(b.xx, b.yy, np.abs(ib[:,ysize//2,:]), rstride=8, cstride=8)
    
#    ax1.axis('equal')
#    ax2.axis('equal')
    
    plt.tight_layout()
    
    plt.show()
    
    
    
    
    