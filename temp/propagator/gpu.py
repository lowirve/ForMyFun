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

from crystals import crystal
from coordinate import xy, xyt


c = 2.99792458e2 # unit is (um/ps) 

def crystal_wrap(func):
    def wrapper(crys, key):        
        return func(key, **crys.parameters())   
    return wrapper

@crystal_wrap
def kpara(key, **kwargs):
    # In an (x, y, z) axis system where z is parallel to k,x is parallel to hi eigenpolarizatoin walkoff,
    # and y is parallel to lo eigenpolarization walkoff.
    
    """
    The function is to calculate the coefficients of taylor expansion for the phase term.
    So far, it only works for principle plan cut.
    """
    
    if not (key in ('lo', 'hi')):
        raise ValueError("key must be either 'hi' or 'lo'.")
        
    def k(wl, n): 
        return 2*np.pi*n/wl    
    
    #below parameters are mandatory.     
    nx = kwargs.pop('nx')
    ny = kwargs.pop('ny')
    nz = kwargs.pop('nz')
        
    nlo = kwargs.pop('nlo')
    nhi = kwargs.pop('nhi')
    
    wl = kwargs.pop('wl')
    
    #below parameters are optional.
    rlo = kwargs.pop('rlo', 0)
    rhi = kwargs.pop('rhi', 0)
    
    gihi = kwargs.pop('gihi', 0)
    gilo = kwargs.pop('gilo', 0)
    
    gvdhi = kwargs.pop('gvdhi', 0)
    gvdlo = kwargs.pop('gvdlo', 0)          
    
    #there seems to exist high-order symmetry between the functions of hi and lo. Need to dig in more.
    if key == 'hi':
        return {'kx':np.tan(rhi), 'ky': 0, 'kxky':np.tan(rhi)*np.tan(rlo)*(nhi**2/(nhi**2-nlo**2)),
              'ky2':-(1-(nlo**2/(nhi**2-nlo**2))*np.tan(rhi)**2), 
              'kx2':-(-(nhi**2/(nhi**2-nlo**2))*np.tan(rlo)**2+np.tan(rhi)**2+nhi**4*nlo**2/(nx*ny*nz)**2),
              'dw':gihi/c, 'dw2': gvdhi/2e9, 'kc': k(wl/1e3, nhi)}    
    else:
        return {'kx': 0, 'ky':np.tan(rlo), 'kxky':-np.tan(rhi)*np.tan(rlo)*(nlo**2/(nhi**2-nlo**2)),
              'kx2':-(1-(-nhi**2/(nhi**2-nlo**2))*np.tan(rlo)**2), 
              'ky2':-(-(-nlo**2/(nhi**2-nlo**2))*np.tan(rhi)**2+np.tan(rlo)**2+nhi**2*nlo**4/(nx*ny*nz)**2),
              'dw':gilo/c, 'dw2': gvdlo/2e9, 'kc': k(wl/1e3, nlo)} 
        
        
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

def phase_wrap(cls):
    """This wrapper changes the propagator class from a general class to a specified class for birefringent crystals"""
    class wrapper(cls) :             
        def __init__(self, crys, coord, key, ref=None):
            para = kpara(crys, key)
                
            if type(coord) == xy:
                ph = phase(para, kx=coord.kxx, ky=coord.kyy)
                
            elif type(coord) == xyt: #xy is the superclass of xyt. So isinstance(xyt(), xy) returns true. Type(), on the other hand, returns false.
                ph = phase(para, kx=coord.kxxx, ky=coord.kyyy, dw=coord.www)  
            
            super(wrapper, self).__init__(ph, ref)
            self.para = para
    wrapper.__name__ = cls.__name__    
    return wrapper
        
@phase_wrap
class propagator(object):    

    _streamsource = 'internal'
    _solcalculated = False
    
    def __init__(self, ph, ref=None):
        self.phase = ph
           
        if ref:
            self.load_ref(ref)
            
        nnn = np.prod(self.phase.shape, dtype=np.int32) 
        
        if len(self.phase.shape) == 2:           
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

        elif len(self.phase.shape) == 3:                     
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
        self.phase -= ref #self.coord default is one. So it will not alter the class performance
            
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


if __name__ == '__main__':    
    
    from timeit import default_timer as timer
    from crystals.data import lbo3
    
    def Gau(w0, x, y, wt=None, t=None):
        return np.exp(-(x**2+y**2)/2/w0) if ((wt is None) or (t is None)) else np.exp(-(x**2+y**2)/2/w0)*np.exp(-t**2/2/wt)

    wl = 1.064 #1.064 um
    w0 = 500 # 500um
    wt = 30 # 30ps
    dz = 50000 # 500um
    
    xsize = 128
    ysize = 128
    tsize = 64
    size = 3
        
    crys = crystal(lbo3, wl*1e3, 25, 90, 45)
    crys.show()
        
    key = 'hi'
    
    x = np.linspace(-256*size, 256*size, xsize, dtype=np.float32)
    y = np.linspace(-256*size, 256*size, ysize, dtype=np.float32)
    
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    E = Gau(w0, xx, yy) 
    E = E.astype(np.complex128)
    
    start = timer()
    
    sol1 = xypropagator(E, x, y, dz, crys, key)

    end = timer()
    print(end - start)
    
    space1 = xy(x, y)
    
    E = Gau(w0, space1.xx, space1.yy)
    E = E.astype(np.complex128)
    
    start = timer()

    sol = propagator(crys, space1, key)
    
    sol.load(dz)
    sol.propagate(E)
    
    sol2 = sol.get()
    
    end = timer()
    print(end - start)
    
    print(np.allclose(sol1, sol2)) 
   
    t = np.linspace(-64*1, 63*1, tsize, dtype=np.float32)
    
    xx, yy, tt = np.meshgrid(x, y, t, indexing='ij')

    E = Gau(w0, xx, yy, wt, tt)
    E = E.astype(np.complex128)
    
    ref = True

    start = timer()
    
    sol3 = xytpropagator(E, x, y, t, dz, crys, key, ref)

    end = timer()
    print(end - start) 
    
    space2 = xyt(x, y, t)
    
    E = Gau(w0, space2.xxx, space2.yyy, wt, space2.ttt)
    E = E.astype(np.complex128)
    
    start = timer()
    
    sol = propagator(crys, space2, key)
    sol.load_ref(sol.para['dw']*space2.www)
    
    sol.load(dz)
    sol.propagate(E)
 
    sol4 = sol.get()
    
    end = timer()
    print(end - start)
    
    print(np.allclose(sol3, sol4))
