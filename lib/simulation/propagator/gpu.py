# -*- coding: utf-8 -*-
"""
tst
"""
from __future__ import division, print_function

from numba import cuda
from pyculib import fft

import numpy as np
from cmath import exp

import sys
sys.path.append(r'C:\Users\xub\Desktop\Python project\Packages\lib')
#sys.path.append(r'E:\xbl_Berry\Desktop\Python project\Packages\lib')

from simulation.crystals import crystal
from simulation.coordinate import xy, xyt


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
    
    def __init__(self, crys, coord, key):
        self.para = kpara(crys, key)
        self.key = key
        
        if type(coord) == xy:
            self.phase = phase(coord.kxx, coord.kyy, self.para, key)
            
            nnn = np.prod(self.phase.shape) 
            
            TPB = np.array([16, 16])
            BPG = np.array(self.phase.shape)/TPB
            BPG = BPG.astype(np.int)
            
            self.gridim = tuple(BPG)
            self.threadim = tuple(TPB)
            
            @cuda.jit#('void(complex128[:,:], complex128[:,:])')
            def division(A):
                i, j = cuda.grid(2)
                if i < A.shape[0] and j < A.shape[1]:
                    A[i,j] /= nnn 
            
        
        elif type(coord) == xyt: #xy is the superclass of xyt. So isinstance(xyt(), xy) returns true. Type(), on the other hand, returns false.
            self.phase = phase(coord.kxxx, coord.kyyy, self.para, key, coord.www)
            
            nnn = np.prod(self.phase.shape)
                    
            TPB = np.array([16, 16, 4])
            BPG = np.array(self.phase.shape)/TPB
            BPG = BPG.astype(np.int)
            
            self.gridim = tuple(BPG)
            self.threadim = tuple(TPB)
            
            @cuda.jit#('void(complex128[:,:,:], complex128[:,:,:])')
            def division(A):
                i, j, k = cuda.grid(3)
                if i < A.shape[0] and j < A.shape[1] and k < A.shape[2]:
                    A[i,j,k] /= nnn
            
        self._division = division
            
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
        
        if len(self.phase.shape) == 2:
            multiple2[self.gridim, self.threadim, self.stream](self._data, self._move)
            self.stream.synchronize()
            
        elif len(self.phase.shape) == 3:
            multiple3[self.gridim, self.threadim, self.stream](self._data, self._move)
            self.stream.synchronize()        
        
    def get(self):
        """Once this function is performed, the device (gpu) memory handle is recycled."""
        
        fft.ifft_inplace(self._data, stream=self.stream)
#        self.stream.synchronize()
        
        sol = self._data.copy_to_host(stream=self.stream)/np.prod(self.phase.shape) 
        
        self.stream.synchronize()
        
        return sol
    
    def _get(self):
        
        fft.ifft_inplace(self._data, stream=self.stream)
#        self.stream.synchronize()
        
        self._division[self.gridim, self.threadim, self.stream](self._data)
        
        self.stream.synchronize()
        
        return self._data
    

def xypropagator(E, x, y, dz, crys, key):
        
    kx = 2*np.pi*np.fft.fftfreq(x.size, d = (x[1]-x[0])) # k in x
    ky = 2*np.pi*np.fft.fftfreq(y.size, d = (y[1]-y[0])) # k in y   
    
    kxx, kyy = np.meshgrid(kx, ky, indexing='ij') #xx, yy in k space
    
    para = kpara(crys, key)
    kphase = phase(kxx, kyy, para, key)
    
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
    
    sol = dE.copy_to_host(stream=stream)/np.prod(E.shape)
    stream.synchronize()
    
    return sol

def xytpropagator(E, x, y, t, dz, crys, key):
    
    kx = 2*np.pi*np.fft.fftfreq(x.size, d = (x[1]-x[0])) # k in x
    ky = 2*np.pi*np.fft.fftfreq(y.size, d = (y[1]-y[0])) # k in y
    dw = 2*np.pi*np.fft.fftfreq(t.size, d = (t[1]-t[0])) # dw in t
    
    kxx, kyy, dww = np.meshgrid(kx, ky, dw, indexing='ij')
    
    para = kpara(crys, key)

    kphase = phase(kxx, kyy, para, key, dww) #most time-consuming step
    
    P = 1j*dz*kphase
    
    TPB = np.array([16, 16, 4])
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
    
    sol = dE.copy_to_host(stream=stream)/np.prod(E.shape)
    stream.synchronize()
    
    return sol
        


if __name__ == '__main__':    
    
    from timeit import default_timer as timer
    from simulation.crystals.data import lbo3
    
    def Gau(w0, x, y, wt=None, t=None):
        return np.exp(-(x**2+y**2)/2/w0) if ((wt is None) or (t is None)) else np.exp(-(x**2+y**2)/2/w0)*np.exp(-t**2/2/wt)

    wl = 1.064 #1.064 um
    w0 = 500 # 500um
    wt = 30 # 30ps
    dz = 500 # 500um
    
    xsize = 128
    ysize = 128
    tsize = 64
        
    crys = crystal(lbo3, wl*1e3, 25, 90, 45)
    crys.show()
        
    key = 'hi'
    
    x = np.linspace(-256*1, 256*1, xsize)
    y = np.linspace(-256*1, 256*1, ysize)
    
    xx, yy = np.meshgrid(x, y)
    
    E = Gau(w0, xx, yy) 
    E = E.astype(np.complex64)
    
    start = timer()
    
    sol1 = xypropagator(E, x, y, dz, crys, key)

    end = timer()
    print(end - start)
    
    space1 = xy(x, y)
    
    E = Gau(w0, space1.xx, space1.yy)
    E = E.astype(np.complex64)
    
    start = timer()

    sol = propagator(crys, space1, key)
    
    sol.load(dz)
    sol.propagate(E)
    
    sol2 = sol.get()
    
    end = timer()
    print(end - start)
    
    print(np.allclose(sol1, sol2)) 
   
    x = np.linspace(-256*1, 256*1, xsize)
    y = np.linspace(-256*1, 256*1, ysize)
    t = np.linspace(-64*1, 63*1, tsize)
    
    xx, yy, tt = np.meshgrid(x, y, t)

    E = Gau(w0, xx, yy, wt, tt)
    E = E.astype(np.complex64)

    start = timer()
    
    sol3 = xytpropagator(E, x, y, t, dz, crys, key)

    end = timer()
    print(end - start) 
    
    space2 = xyt(x, y, t)
    
    E = Gau(w0, space2.xxx, space2.yyy, wt, space2.ttt)
    E = E.astype(np.complex64)
    
    start = timer()
    
    sol = propagator(crys, space2, key)
    
    sol.load(dz)
    sol.propagate(E)
 
    sol4 = sol.get()
    
    end = timer()
    print(end - start)
    
    print(np.allclose(sol3, sol4))
    