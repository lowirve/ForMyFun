# -*- coding: utf-8 -*-
"""
tst
"""
from __future__ import division, print_function

from numba import cuda
from pyculib import fft

import numpy as np
from cmath import exp

from ..coordinate import xy, xyt

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
        return {'kx':np.tan(rhi), 'ky': 0, 'kxky':np.tan(rhi)*np.tan(rlo)*(nhi**2/(nhi**2-nlo**2)),
              'ky2':-(1-(nlo**2/(nhi**2-nlo**2))*np.tan(rhi)**2), 
              'kx2':-(-(nhi**2/(nhi**2-nlo**2))*np.tan(rlo)**2+np.tan(rhi)**2+nhi**4*nlo**2/(nx*ny*nz)**2),
              'dw':gihi/c, 'dw2': gvdhi/2e9, 'kc': k(crystal.wl/1e3, crystal.nhl[0])}    
    else:
        return {'kx': 0, 'ky':np.tan(rlo), 'kxky':-np.tan(rhi)*np.tan(rlo)*(nlo**2/(nhi**2-nlo**2)),
              'kx2':-(1-(-nhi**2/(nhi**2-nlo**2))*np.tan(rlo)**2), 
              'ky2':-(-(-nlo**2/(nhi**2-nlo**2))*np.tan(rhi)**2+np.tan(rlo)**2+nhi**2*nlo**4/(nx*ny*nz)**2),
              'dw':gilo/c, 'dw2': gvdlo/2e9, 'kc': k(crystal.wl/1e3, crystal.nhl[1])} 
        
        
def phase(para, **kwargs):
        
    kc = para['kc']
    kx = kwargs.pop('kx', 0)
    ky = kwargs.pop('ky', 0)
    dw = kwargs.pop('dw', 0)
    
    return kx*para['kx']+ky*para['ky']+kx*ky*para['kxky']+ky**2/2/kc*para['ky2']+kx**2/2/kc*para['kx2']+dw*para['dw']+dw**2*para['dw2']

        
@cuda.jit#('void(complex128[:,:], complex128[:,:])')
def multiple2(A, B):
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < A.shape[1]:
        A[i,j] *= exp(B[i,j])  
        
@cuda.jit#('void(complex128[:,:,:], complex128[:,:,:])')
def multiple3(A, B):
    i, j, k = cuda.grid(3)
    if i < A.shape[0] and j < A.shape[1] and k < A.shape[2]:
        A[i,j,k] *= exp(B[i,j,k]) 
 
    
class propagator(object):    

    _streamsource = 'internal'
    _solcalculated = False
    
    def __init__(self, crys, coord, key, ref=None):
        self.para = kpara(crys, key)
        self.key = key
        self.coord = coord
        
        if type(coord) == xy:
            self.phase = phase(self.para, kx=coord.kxx, ky=coord.kyy)
            
            nnn = np.prod(self.phase.shape, dtype=np.int32) 
            
            TPB = np.array([16, 16])
            BPG = np.array(self.phase.shape)/TPB
            BPG = BPG.astype(np.int)
            
            self.gridim = tuple(BPG)
            self.threadim = tuple(TPB)
            
            @cuda.jit#('void(complex128[:,:])')
            def division(A):
                i, j = cuda.grid(2)
                if i < A.shape[0] and j < A.shape[1]:
                    A[i,j] /= nnn 
                    
            self._multiple =multiple2 
            
        
        elif type(coord) == xyt: #xy is the superclass of xyt. So isinstance(xyt(), xy) returns true. Type(), on the other hand, returns false.
            self.phase = phase(self.para, kx=coord.kxxx, ky=coord.kyyy, dw=coord.www)
                     
            if ref:
                self.load_ref(ref)
            
            nnn = np.prod(self.phase.shape, dtype=np.int32) 
                    
            TPB = np.array([8, 8, 4])
            BPG = np.array(self.phase.shape)/TPB
            BPG = BPG.astype(np.int)
            
            self.gridim = tuple(BPG)
            self.threadim = tuple(TPB)
            
            @cuda.jit#('void(complex64[:,:,:])')
            def division(A):
                i, j, k = cuda.grid(3)
                if i < A.shape[0] and j < A.shape[1] and k < A.shape[2]:
                    A[i,j,k] /= nnn
                    
            self._multiple =multiple3
            
        self._division = division
        
    def load_ref(self, ref):
        self.phase -= self.coord.www*ref
            
    def load(self, dz, stream=None):   
        """load dz and stream. Hence once either dz or stream needs to be updated, run this function.
           if stream is not assigned here, input in propagate must be np.array and cannot be a devicearrya."""
           
        self._phase = 1j*dz*self.phase
        
        if stream is None:
            self.stream = cuda.stream() 
        else:
            self.stream = stream
            self._streamsource = 'external'
        
        self._move = cuda.to_device(self._phase, stream=self.stream)
        
    def propagate(self, init):
        #If no stream is assigned in self.load() function, it should not be allowed to assign devicearray in here!
        #If a stream is assigned in self.load() function, it better guarantee that the devicearray given here shares the same stream. 
        
        if self.phase.shape != init.shape:
            raise ValueError("Input doesn't match.")      
            
        if cuda.devicearray.is_cuda_ndarray(init): 
            if self._streamsource == 'internal':
                raise ValueError("Input cannot be a devicearray.") 
            self._data = init             
        else:            
            self._data = cuda.to_device(init, stream=self.stream)                      
            
#        fft.FFTPlan(shape=self.phase.shape, itype=init.dtype, otype=init.dtype, stream=self.stream)            

        fft.fft_inplace(self._data, stream=self.stream)         

        self.stream.synchronize()                   

        self._multiple[self.gridim, self.threadim, self.stream](self._data, self._move)

        self.stream.synchronize()
        
        self._solcalculated = False
        
    def get(self):
        """Once this function is performed, the device (gpu) memory handle is recycled."""
        
        if not self._solcalculated:
    
            self.sol = self._get().copy_to_host(stream=self.stream)

            self.stream.synchronize()
            
            self._solcalculated = True
                   
        return self.sol
    
    def _get(self):
        #Performing this function gives other external functions access to change the devicearray, and hence
        #subsequently running self.get() function may not have the correct answer anymore. Avoid running self.get() 
        #after self._get().
        
        fft.ifft_inplace(self._data, stream=self.stream)
#        self.stream.synchronize()
        
        self._division[self.gridim, self.threadim, self.stream](self._data)
        
        self.stream.synchronize()
        
        return self._data
    

def xypropagator(E, x, y, dz, crys, key):
        
    kx = 2*np.pi*np.fft.fftfreq(x.size, d = (x[1]-x[0])).astype(x.dtype) # k in x
    ky = 2*np.pi*np.fft.fftfreq(y.size, d = (y[1]-y[0])).astype(y.dtype) # k in y
    
    kxx, kyy = np.meshgrid(kx, ky, indexing='ij') #xx, yy in k space
    
    para = kpara(crys, key)
    kphase = phase(para, kx=kxx, ky=kyy)
    
    P = 1j*dz*kphase
    
    TPB = np.array([16, 16])
    BPG = np.array([x.size, y.size])/TPB
    BPG = BPG.astype(np.int)
    
    gridim = tuple(BPG)
    threadim = tuple(TPB)
    
    stream = cuda.stream()
    
#    fft.FFTPlan(shape=E.shape, itype=E.dtype, otype=E.dtype, stream=stream)  

    dE = cuda.to_device(E, stream=stream)
    dP = cuda.to_device(P, stream=stream)
    
    fft.fft_inplace(dE, stream=stream)
#    stream.synchronize()
    
    multiple2[gridim, threadim, stream](dE, dP)
#    stream.synchronize()
    
    fft.ifft_inplace(dE, stream=stream)
#    stream.synchronize()
    
    sol = dE.copy_to_host(stream=stream)/np.prod(E.shape, dtype=kphase.dtype)
    stream.synchronize()
    
    return sol

def xytpropagator(E, x, y, t, dz, crys, key, ref=False):
    
    kx = 2*np.pi*np.fft.fftfreq(x.size, d = (x[1]-x[0])).astype(x.dtype) # k in x
    ky = 2*np.pi*np.fft.fftfreq(y.size, d = (y[1]-y[0])).astype(y.dtype) # k in y
    dw = 2*np.pi*np.fft.fftfreq(t.size, d = (t[1]-t[0])).astype(t.dtype) # dw in t
    
    kxx, kyy, dww = np.meshgrid(kx, ky, dw, indexing='ij')
    
    para = kpara(crys, key)
    
    if ref:
        kphase = phase(para, kx=kxx, ky=kyy, dw=dww) - dww*para['dw']#most time-consuming step
    else:
        kphase = phase(para, kx=kxx, ky=kyy, dw=dww)
    
    P = 1j*dz*kphase
    
    TPB = np.array([8, 8, 4])
    BPG = np.array([x.size, y.size, t.size])/TPB
    BPG = BPG.astype(np.int)
    
    gridim = tuple(BPG)
    threadim = tuple(TPB)
    
    stream = cuda.stream()
    
#    fft.FFTPlan(shape=E.shape, itype=E.dtype, otype=E.dtype, stream=stream)   

    dE = cuda.to_device(E, stream=stream)
    dP = cuda.to_device(P, stream=stream)
    
    fft.fft_inplace(dE, stream=stream)
#    stream.synchronize()
    
    multiple3[gridim, threadim, stream](dE, dP)
#    stream.synchronize()
    
    fft.ifft_inplace(dE, stream=stream)
#    stream.synchronize()
    
    sol = dE.copy_to_host(stream=stream)/np.prod(E.shape, dtype=kphase.dtype)
    stream.synchronize()
    
    return sol
        
