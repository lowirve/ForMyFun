# -*- coding: utf-8 -*-
"""
import crystal is not necessary. Maybe rewrite kpara and make it compatible with other data format.
The same work for phase function or propagator class.
"""

from __future__ import division, print_function  

import numpy as np

from lib.coordinate import xy, xyt#one solution to solve the mutual import within one package is using
                                  #relative import, such as "from .coordinate import xy, xyt". The problem
                                  #is that relative import only works for package. So module file written in
                                  #this way will run in Spyder individually. The other way is to use absolute
                                  #import. But in this way, the directory must be added to the intepretor first.

c = 2.99792458e2 # unit is (um/ps) 

def k(wl, n): 
    return 2*np.pi*n/wl

def kpara(crystal, key):
    # In an (x, y, z) axis system where z is parallel to k,x is parallel to hi eigenpolarizatoin walkoff,
    # and y is parallel to lo eigenpolarization walkoff.
    
    if not (key in ('lo', 'hi')):
        raise ValueError("key must be either 'hi' or 'lo'.")
        
    nx = crystal.nx
    ny = crystal.ny
    nz = crystal.nz
        
    nlo = crystal.nhl[1]
    nhi = crystal.nhl[0]
    
    rlo = crystal.rhl[1]
    rhi = crystal.rhl[0]
    
    gihi = crystal.gihl[0]
    gilo = crystal.gihl[1]
    
    gvdhi = crystal.gvdhl[0]
    gvdlo = crystal.gvdhl[1]
    
    if key == 'hi':
        return {'kx':np.tan(rhi), 'kxky':np.tan(rhi)*np.tan(rlo)*(nhi**2/(nhi**2-nlo**2)),
              'ky2':-(1-(nlo**2/(nhi**2-nlo**2))*np.tan(rhi)**2), 
              'kx2':-(-(nhi**2/(nhi**2-nlo**2))*np.tan(rlo)**2+np.tan(rhi)**2+nhi**4*nlo**2/(nx*ny*nz)**2),
              'dw':gihi/c, 'dw2': gvdhi/2e9, 'kc': k(crystal.wl/1e3, crystal.nhl[0])}    
    else:
        return {'ky':np.tan(rlo), 'kxky':-np.tan(rhi)*np.tan(rlo)*(nlo**2/(nhi**2-nlo**2)),
              'kx2':-(1-(-nhi**2/(nhi**2-nlo**2))*np.tan(rlo)**2), 
              'ky2':-(-(-nlo**2/(nhi**2-nlo**2))*np.tan(rhi)**2+np.tan(rlo)**2+nhi**2*nlo**4/(nx*ny*nz)**2),
              'dw':gilo/c, 'dw2': gvdlo/2e9, 'kc': k(crystal.wl/1e3, crystal.nhl[1])} 
        
        
def phase(kx, ky, para, key, dw=None):
    if not (key in ('lo', 'hi')):
        raise ValueError("key must be either 'hi' or 'lo'.") 
        
    kc = para['kc']
    
    if dw is not None:
        if key == 'hi':    
            return kx*para['kx']+kx*ky*para['kxky']+ky**2/2/kc*para['ky2']+kx**2/2/kc*para['kx2']+dw*para['dw']+dw**2*para['dw2']
        else:
            return ky*para['ky']+kx*ky*para['kxky']+kx**2/2/kc*para['kx2']+ky**2/2/kc*para['ky2']+dw*para['dw']+dw**2*para['dw2']
    else:
        if key == 'hi':    
            return kx*para['kx']+kx*ky*para['kxky']+ky**2/2/kc*para['ky2']+kx**2/2/kc*para['kx2']#+kc#full phase
        else:
            return ky*para['ky']+kx*ky*para['kxky']+kx**2/2/kc*para['kx2']+ky**2/2/kc*para['ky2']#+kc#full phase
 
    
class propagator(object):   
    
    def __init__(self, crys, coord, key):
        self.para = kpara(crys, key)
        self.key = key
        
        if type(coord) == xy:
            self.phase = phase(coord.kxx, coord.kyy, self.para, key)
        
        elif type(coord) == xyt: #xy is the superclass of xyt. So isinstance(xyt(), xy) returns true. Type(), on the other hand, returns false.
            self.phase = phase(coord.kxxx, coord.kyyy, self.para, key, coord.www)
            
    def load(self, dz, stream=None): #The extra useless parameter, 'stream', is to make the class compatible with gpu version in the applications.            
        self._move = np.exp(1j*self.phase*dz)
        
    def propagate(self, init):                     
        if self.phase.shape != init.shape:
            raise ValueError("Input doesn't match.")
            
        self._data = np.fft.fftn(init)
        self._data = self._data*self._move
        
    def get(self):
        return np.fft.ifftn(self._data)
    
    def _get(self): #The extra useless function is to make the class compatible with gpu version in the applications.
        return self.get()
        

def xypropagator(E, x, y, dz, crys, key):
        
    kx = 2*np.pi*np.fft.fftfreq(x.size, d = (x[1]-x[0])) # k in x
    ky = 2*np.pi*np.fft.fftfreq(y.size, d = (y[1]-y[0])) # k in y   
    
    kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij') #xx, yy in k space
    
    para = kpara(crys, key)
    kphase = phase(kxx, kyy, para, key)
    
    kE = np.fft.fftn(E)
    
    kE2 = kE*np.exp(1j*kphase*dz)

    sol = np.fft.ifftn(kE2)
    
    return sol

def xytpropagator(E, x, y, t, dz, crys, key, ref=False):
    
    kx = 2*np.pi*np.fft.fftfreq(x.size, d = (x[1]-x[0])) # k in x
    ky = 2*np.pi*np.fft.fftfreq(y.size, d = (y[1]-y[0])) # k in y
    dw = 2*np.pi*np.fft.fftfreq(t.size, d = (t[1]-t[0])) # dw in t

    kxx, kyy, dww = np.meshgrid(kx, ky, dw, indexing = 'ij')

    para = kpara(crys, key)
    
    if ref:
        kphase = phase(para=para, kx=kxx, ky=kyy, dw=dww, key=key) - dww*para['dw']#most time-consuming step
    else:
        kphase = phase(para=para, kx=kxx, ky=kyy, dw=dww, key=key)
  
    kE = np.fft.fftn(E)
      
    kE2 = kE*np.exp(1j*kphase*dz)
      
    sol = np.fft.ifftn(kE2)
    
    return sol
