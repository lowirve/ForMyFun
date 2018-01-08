# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 10:47:15 2018

@author: XuBo
"""

from __future__ import division, print_function

from numba import cuda, complex64, float32
from cmath import exp
import numpy as np


class ode(object):
    """The current structure is time-invariant. 
    If the ode is affected by the initial state (x0), the current structure won't work."""

    def __init__(self, x0, x1, gf, fsize, arg, hstart=10, nmax=10000, eps=1e-8, hmin=None, TPB=[16, 16]):
        #VERY HIGH OVERHEAD
        self.x0 = x0
        self.xq = x1
        self.gf = gf
        self.arg = arg
        self.hstart = hstart              
        self.nmax = nmax
        self.eps = eps
        self.TPB = TPB
        self.fsize = fsize
    
        if hmin is None:
            hmin = (x1-x0)/1e7
            self.hmin = hmin

        @cuda.jit('float32(float32[:], int32)' ,device=True)
        def dmax(y,fsize):
            temp = y[0]
            for i in range(1, fsize):
                if y[i] > temp:
                    temp = y[i]
            return temp

        @cuda.jit(device=True)
        def grkck(x, y, arg, h, sol, err):
            a = (1/5, 3/10, 3/5, 1, 7/8)
            b1 = (1/5, )
            b2 = (3/40, 9/40)
            b3 = (3/10, -9/10, 6/5)
            b4 = (-11/54, 5/2, -70/27, 35/27)
            b5 = (1631/55296, 175/521, 575/13824, 44275/110592, 253/4096)
            
            c = (37/378, 0, 250/621, 125/594, 0, 512/1771)
            d = (2825/27648-37/378,0,18575/48384-250/621,13525/55296-125/594,277/14336,1/4-512/1771)
            
            k1 = gf(x,y,arg,h)

#            fsize = len(k1) #doesn't work
            
            ytemp = cuda.local.array(fsize, dtype=complex64)
            
            for i in range(fsize):
                ytemp[i] = y[i]+b1[0]*k1[i]
           
            k2 = gf(x+h*a[0],ytemp,arg,h)

            for i in range(fsize):
                ytemp[i] = y[i]+b2[0]*k1[i]+b2[1]*k2[i]        
        
            k3 = gf(x+h*a[1],ytemp,arg,h)

            for i in range(fsize):
                ytemp[i] = y[i]+b3[0]*k1[i]+b3[1]*k2[i]+b3[2]*k3[i]          
        
            k4 = gf(x+h*a[2], ytemp, arg, h)

            for i in range(fsize):
                ytemp[i] = y[i]+b4[0]*k1[i]+b4[1]*k2[i]+b4[2]*k3[i]+b4[3]*k4[i]      
        
            k5 = gf(x+h*a[3], ytemp, arg, h)

            for i in range(fsize):
                ytemp[i] = y[i]+b5[0]*k1[i]+b5[1]*k2[i]+b5[2]*k3[i]+b5[3]*k4[i]+b5[4]*k5[i]
        
            k6 = gf(x+h*a[4], ytemp, arg, h)
            
            for i in range(fsize):
                sol[i] = y[i]+c[0]*k1[i]+c[2]*k3[i]+c[3]*k4[i]+c[5]*k6[i]

            for i in range(fsize):
                err[i] = d[0]*k1[i]+d[1]*k2[i]+d[2]*k3[i]+d[3]*k4[i]+d[4]*k5[i]+d[5]*k6[i]
        
            
        @cuda.jit(device=True)
        def grkqs(x, y, arg, h, eps, yscal):
            SAFETY = 0.9
            PSHRNK = -0.25
            PGROW = -0.2
            ERRON = 1.89e-4

            sol = cuda.local.array(fsize, dtype=complex64) 
            yerr = cuda.local.array(fsize, dtype=complex64) 
            temp = cuda.local.array(fsize, dtype=float32)
        
            while(1):
                grkck(x,y,arg,h,sol,yerr)
                
                for i in range(fsize):
                    temp[i] = abs(yerr[i]/yscal[i])
                
                errmax = dmax(temp,fsize)
#                errmax = max(abs(yerr[0]/yscal[0]),abs(yerr[1]/yscal[1]),abs(yerr[2]/yscal[2])) # evaluate accuracy
                errmax /= eps
                if errmax <= 1:
                    hdid = h
                    break
                htemp = SAFETY*h*errmax**PSHRNK        #truncation error too large, reduce stepsize
                h = max(h/10, htemp) if h >= 0 else min(h/10, htemp)#no more than a factor of 10
                
            if errmax > ERRON:
                hnext = SAFETY*h*errmax**PGROW # feedback for next stepsize
            else:
                hnext = 5.*h #No more than a factor of 5 increase
                
            y[0] = sol[0]
            y[1] = sol[1]
            y[2] = sol[2]
            
            return hdid, hnext #return the integration over a valid stepsize, the used stepsize, and the next stepsize
        
        @cuda.jit('void(complex64[:,:], complex64[:,:], complex64[:,:])')
        def godeint(A, B, C):
            TINY = 1e-30
            
            i, j = cuda.grid(2)
            if i < A.shape[0] and j < A.shape[1]:
                
                xtemp = x0
            
                ytemp = cuda.local.array(fsize, dtype=complex64)

                ytemp[0] = A[i,j]
                ytemp[1] = B[i,j]
                ytemp[2] = C[i,j]
                    
                h = abs(hstart)*(x1-x0)/abs(x1-x0) # h must be a real number, this step ensures that the sign of step is right.        
                
                jj = 0
                
                for ii in range(nmax):
                    jj += 1
                    if abs(h) <= abs(hmin):
                        raise ValueError("Step size too small in odeint")
                    temp = gf(xtemp, ytemp, arg, h)
                    
                    yscal = cuda.local.array(fsize, dtype=float32)
                    for k in range(fsize):
                        yscal[k] = abs(ytemp[k])+abs(temp[k])+TINY
                        
                    if (xtemp+h-x1)*(xtemp+h-x0) > 0:
                        h = x1-xtemp
                    hdid, hnext = grkqs(xtemp, ytemp, arg, h, eps, yscal)
                    xtemp += hdid
                    if (xtemp-x1)*(x1-x0) >= 0:
                        A[i,j] = ytemp[0]
                        B[i,j] = ytemp[1]
                        C[i,j] = ytemp[2]
                        break
                    h = hnext     
                    
                if jj >= nmax:
                    raise ValueError("Too many steps in routine odeint") 
            
        self.f = godeint

                        
    def move(self, lin, lout):
        TPB = np.array(self.TPB)
        BPG = np.array(lin[0].shape)/TPB
        BPG = BPG.astype(np.int)
        
        gridim = tuple(BPG)
        threadim = tuple(TPB)
        
        stream = cuda.stream()
        
        dA = cuda.to_device(lin[0], stream=stream)
        dB = cuda.to_device(lin[1], stream=stream)
        dC = cuda.to_device(lin[2], stream=stream)
        
        self.f[gridim, threadim, stream](dA, dB, dC)
        stream.synchronize()
    
        lout[0][:] = dA.copy_to_host(stream=stream)
        lout[1][:] = dB.copy_to_host(stream=stream)
        lout[2][:] = dC.copy_to_host(stream=stream)  

      
    
if __name__ == "__main__":
    
    from timeit import default_timer as timer

    @cuda.jit(device=True)
    def gf(x, y, arg, h):#x is scalar, y is a tuple, arg is a tuple
        dydx0 = 1j*arg[0]*y[1].conjugate()*y[2]*exp(1j*arg[3]*x)*h
        dydx1 = 1j*arg[1]*y[0].conjugate()*y[2]*exp(1j*arg[3]*x)*h
        dydx2 = 1j*arg[2]*y[0]*y[1]*exp(-1j*arg[3]*x)*h
        return dydx0, dydx1, dydx2 
    
    xsize = 128
    ysize = 128

    w0 = 500 # 500um
    
    def Gau(w0, wt, x, y, t):
        return np.exp(-(x**2+y**2)/2/w0)*np.exp(-t**2/2/wt) if wt != 0 else np.exp(-(x**2+y**2)/2/w0)
    
    def para(wl, d, n):
        return d/wl/n
    
    ns = np.array([1.617, 1.617, 1.617])
    deff = 4.47e-7
    wls =np.array([.607, .607, .303])
    
    args = np.append(para(wls, deff, ns),0) # 4 element array
    args = tuple(args)
    
    x = np.linspace(-128*1, 127*1, xsize)
    y = np.linspace(-128*1, 127*1, ysize)
    
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    E = 100*Gau(w0, 10, xx, yy, 0)
    
    E = E.astype(np.complex64)  
    
    A = E.copy()
    B = E.copy()
    C = np.zeros_like(E,dtype=np.complex64)
    
    E2 = np.empty((xsize, ysize, 3), dtype=np.complex64)

    start = timer()
    
    test = ode(0, 5000, gf, 3, args)#overhead
    
    end = timer()   
    print (end-start)    

    start = timer()
    
    test.move([A, B, C], [E2[:,:,i] for i in range(3)])
    
    end = timer()    
    print (end-start)


    