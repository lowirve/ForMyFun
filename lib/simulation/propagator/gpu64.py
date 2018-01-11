# -*- coding: utf-8 -*-
"""

"""
from __future__ import division, print_function

from numba import cuda
from pyculib import fft

import numpy as np

import sys
sys.path.append(r'C:\Users\xub\OneDrive - Coherent, Inc\Python project\Packages\lib')
#sys.path.append(r'E:\xbl_Berry\Desktop\Python project\Packages\lib')

from simulation.crystals import crystal
from plot.xy import image

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
              'dw':gihi/c, 'dw2': gvdhi/2e9}    
    else:
        return {'ky':np.tan(rlo), 'kxky':-np.tan(rhi)*np.tan(rlo)*(nlo**2/(nhi**2-nlo**2)),
              'kx2':-(1-(-nhi**2/(nhi**2-nlo**2))*np.tan(rlo)**2), 
              'ky2':-(-(-nlo**2/(nhi**2-nlo**2))*np.tan(rhi)**2+np.tan(rlo)**2+nhi**2*nlo**4/(nx*ny*nz)**2),
              'dw':gilo/c, 'dw2': gvdlo/2e9} 
        
        
def phase(kx, ky, kc, para, key, dw=None):
    if not (key in ('lo', 'hi')):
        raise ValueError("key must be either 'hi' or 'lo'.") 
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
   

def xypropagator(E, x, y, dz, crys, key):
    
    @cuda.jit('void(complex128[:,:], complex128[:,:])')
    def multiple(A, B):
        i, j = cuda.grid(2)
        if i < A.shape[0] and j < A.shape[1]:
            A[i,j] *= B[i,j]
        
    kx = 2*np.pi*np.fft.fftfreq(x.size, d = (x[1]-x[0])) # k in x
    ky = 2*np.pi*np.fft.fftfreq(y.size, d = (y[1]-y[0])) # k in y   
    
    kxx, kyy = np.meshgrid(kx, ky) #xx, yy in k space
    
    if key == 'hi':
        n = crys.nhl[0]
    if key == 'lo':
        n = crys.nhl[1]
        
    kc = k(crys.wl/1e3, n) 
    
    para = kpara(crys, key)
    kphase = phase(kxx, kyy, kc, para, key)
    
    TPB = np.array([16, 16])
    BPG = np.array([x.size, y.size])/TPB
    BPG = BPG.astype(np.int)
    
    gridim = tuple(BPG)
    threadim = tuple(TPB)
    
    stream = cuda.stream()
    
    fft.FFTPlan(shape=E.shape, itype=np.complex128, otype=np.complex128, stream=stream)
    
    P = np.exp(1j*kphase*dz)   

    dE = cuda.to_device(E, stream=stream)
    dP = cuda.to_device(P, stream=stream)
    
    fft.fft_inplace(dE, stream=stream)
    stream.synchronize()
    
    multiple[gridim, threadim, stream](dE, dP)
    stream.synchronize()
    
    fft.ifft_inplace(dE, stream=stream)
    stream.synchronize()
    
    sol = dE.copy_to_host(stream=stream)/np.prod(E.shape)
    
    return sol

def xytpropagator(E, x, y, t, dz, crys, key):
    
    @cuda.jit('void(complex128[:,:,:], complex128[:,:,:])')
    def multiple(A, B):
        i, j, k = cuda.grid(3)
        if i < A.shape[0] and j < A.shape[1] and k < A.shape[2]:
            A[i,j,k] *= B[i,j,k]    
    
    kx = 2*np.pi*np.fft.fftfreq(x.size, d = (x[1]-x[0])) # k in x
    ky = 2*np.pi*np.fft.fftfreq(y.size, d = (y[1]-y[0])) # k in y
    dw = 2*np.pi*np.fft.fftfreq(t.size, d = (t[1]-t[0])) # dw in t
    
    kxx, kyy, dww = np.meshgrid(kx, ky, dw)
    
    if key == 'hi':
        n = crys.nhl[0]
    if key == 'lo':
        n = crys.nhl[1]
    
    kc = k(crys.wl/1e3, n) 
    
    para = kpara(crys, key)
    kphase = phase(kxx, kyy, kc, para, key, dww)
    
    P = np.exp(1j*kphase*dz)
    
    TPB = np.array([16, 16, 4])
    BPG = np.array([x.size, y.size, t.size])/TPB
    BPG = BPG.astype(np.int)
    
    gridim = tuple(BPG)
    threadim = tuple(TPB)
    
    stream = cuda.stream()
    
    fft.FFTPlan(shape=E.shape, itype=np.complex128, otype=np.complex128, stream=stream)   

    dE = cuda.to_device(E, stream=stream)
    dP = cuda.to_device(P, stream=stream)
    
    fft.fft_inplace(dE, stream=stream)
    stream.synchronize()
    
    multiple[gridim, threadim, stream](dE, dP)
    stream.synchronize()
    
    fft.ifft_inplace(dE, stream=stream)
    stream.synchronize()
    
    sol = dE.copy_to_host(stream=stream)/np.prod(E.shape)
    
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
    
    xsize = 256
    ysize = 256
    tsize = 128
        
    crys = crystal(lbo3, wl*1e3, 25, 90, 45)
    crys.show()
        
    key = 'hi'
    
    x = np.linspace(-256*1, 256*1, xsize, dtype=np.complex128)
    y = np.linspace(-256*1, 256*1, ysize, dtype=np.complex128)
    
    xx, yy = np.meshgrid(x, y)
    
    E = Gau(w0, xx, yy)   
    
    image(E)
    
    start = timer()
    
    sol = xypropagator(E, x, y, dz, crys, key)

    end = timer()
    print(end - start)
    
    image(sol)
    
    x = np.linspace(-256*1, 256*1, xsize, dtype=np.complex128)
    y = np.linspace(-256*1, 256*1, ysize, dtype=np.complex128)
    t = np.linspace(-64*1, 63*1, tsize, dtype=np.complex128)
    
    xx, yy, tt = np.meshgrid(x, y, t)

    E = Gau(w0, xx, yy, wt, tt)
    
    image(E[:,:,tsize//2])

    start = timer()
    
    sol = xytpropagator(E, x, y, t, dz, crys, key)

    end = timer()
    print(end - start) 
    
    image(sol[:,:,tsize//2])
    