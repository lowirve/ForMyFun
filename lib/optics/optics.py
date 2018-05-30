# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:31:23 2018

@author: XuBo
"""

from __future__ import division, print_function
import numpy as np

from ..coordinate import xy

class lens(object):
    
    def __init__(self, f, wl=1.064):
        self.f = f
        self.wl = wl#in um
    
    def phase(self, coord):
        return np.exp(-1j*np.pi/self.wl/self.f*(coord.xx**2+coord.yy**2))
        
def propagate(Ein, dz, wl, space): # wl and z are in um
  
    kE = np.fft.fftn(Ein)
    
    k = 2*np.pi/wl
      
    kE2 = kE*np.exp(1j*dz*np.sqrt(k**2-space.kxx**2-space.kyy**2))
      
    Eout = np.fft.ifftn(kE2)
    
    return Eout


if __name__ == '__main__':
    
    from coordinate import xy
#    from tools.plot.xy import image
    import matplotlib.pyplot as plt  
    from scipy.optimize import curve_fit
    
    def Gau(w0, x, y, wt=None, t=None):
        return np.exp(-(x**2+y**2)/2/w0**2) if ((wt is None) or (t is None)) else np.exp(-(x**2+y**2)/2/w0**2)*np.exp(-t**2/2/wt**2)
    
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))

    wl = .532 #1.064 um
    w0 = 3000/2/1.414 # 500um
    wt = 30 # 30ps
    dz = 100e3 # 100um
    
    xsize = 2048
    ysize = 2048
        
    x = np.linspace(-256*16, 256*16, xsize)
    y = np.linspace(-256*16, 256*16, ysize)
    
    space1 = xy(x, y)
   
    E = Gau(w0, space1.xx, space1.yy)
    
#    image(E)
    
    p0 = 1, 0, w0
    
    coeff, _ = curve_fit(gauss, x, E[:,ysize//2], p0=p0)
    
    print(coeff)
#    
#    plt.figure()
#    
#    plt.plot(np.abs(E[:,ysize//2]))  
    
    l1 = lens(dz, wl)
    
    E = E*l1.phase(space1)
   
    sol = propagate(E, dz+5000, wl, space1)
    
#    image(sol)
    
#    plt.figure()
#    
#    plt.plot(np.abs(sol[:,ysize//2]))    
    
    p0 = 152, 0, 7.5
    
    coeff, _ = curve_fit(gauss, x, np.abs(sol[:,ysize//2]), p0=p0)
    
    print(coeff[2]*2.828)

