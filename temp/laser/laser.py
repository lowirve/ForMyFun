# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:30:28 2018

@author: XuBo
"""
from __future__ import division, print_function

import sys

sys.path.append(r'C:\Users\xub\Desktop\Python project\Packages\temp')
#sys.path.append(r'E:\xbl_Berry\Desktop\Python project\Packages\lib')

import numpy as np
from tools.functions import integrate

c = 2.99792458e5 #um/ns
e0 = 8.854187817e-18 #F/um (V/C/um)

class laser(object):
    
    @property
    def Eprofile(self):
        return self._Eprofile
    
    @Eprofile.setter
    def Eprofile(self, Eprofile):
        self._Eprofile = Eprofile
        
    
    def __init__(self, Eprofile, coord, pe, n=1, phase=0):#pe is short for pulse energy or power
        
        self.coordinate = coord
        
        self.Iprofile = pe/integrate(Iprofile, coord.step())*Iprofile
        
        self.Eprofile = Eprofile
        
        
    def EtoI(self, Eprofile, n=1):
        return 0.5*np.abs(Eprofile)**2*c*e0*n
        
    def ItoE(self, Iprofile, phase=0, n=1):
        return np.sqrt(2*Iprofile/c/e0/n)*np.exp(1j*phase)
        
    def _pe(self):
        return integrate(self.Iprofile, self.coordinate.step())
        

class cw(laser):
    
    def __init__(self):
        pass
    

class pulse(laser):
    
    def __init__(self):
        pass
    
    
    
if __name__ == '__main__':
    
    from coordinate import xyt, xy
    from tools.functions import normgau
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
    
    ia = laser(ia, a, 1e-5)
    ib = laser(ib, b, 1e-5)
    
    fig, (ax1, ax2) = plt.subplots(1,2,subplot_kw={'projection': '3d'})
    
#    ax1.pcolor(np.abs(ia))
    
    ax1.plot_wireframe(a.xx, a.yy, np.abs(ia.Eprofile), rstride=8, cstride=8)
    ax2.plot_wireframe(b.xx, b.yy, np.abs(ib.Eprofile[:,ysize//2,:]), rstride=8, cstride=8)
    
#    ax1.axis('equal')
#    ax2.axis('equal')
    
    plt.tight_layout()
    
    plt.show()