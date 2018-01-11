# -*- coding: utf-8 -*-
"""

"""
from __future__ import division, print_function

import numpy as np
from numpy import fft

class xy(object):   
    def __init__(self, x, y):
        self.xsize = x.size
        self.ysize = y.size
        
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        
        self.x = x
        self.y = y
        
        self.kx = 2*np.pi*fft.fftfreq(self.xsize, d = self.dx)#Beware that the fft space is swapped.
        self.ky = 2*np.pi*fft.fftfreq(self.ysize, d = self.dy)#Beware that the fft space is swapped.
        
        self.xy = np.meshgrid(self.x, self.y, indexing = 'ij')
        self.kxy = np.meshgrid(self.kx, self.ky, indexing = 'ij')#Beware that the fft space is swapped.
        
class xyt(xy):
    def __init__(self, x, y, t):
        xy.__init__(self, x, y)
        
        self.tsize = t.size
        
        self.t = t
        self.dt = t[1] - t[0]
        
        self.w = 2*np.pi*fft.fftfreq(self.tsize, d = self.dt)#Beware that the fft space is swapped.
        
        self.xyt = np.meshgrid(self.x, self.y, self.t, indexing = 'ij')
        self.kxyw = np.meshgrid(self.kx, self.ky, self.w, indexing = 'ij')#Beware that the fft space is swapped.
        
if __name__ == '__main__':
    xsize = 256
    ysize = 256
    tsize = 128      
        
    x = np.linspace(-256*1, 256*1, xsize)
    y = np.linspace(-256*1, 256*1, ysize)
    t = np.linspace(-64*1, 63*1, tsize)
    
    test1 = xy(x, y)
    test2 = xyt(x, y, t)
    
    
    
    