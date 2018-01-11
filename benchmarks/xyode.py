"""

"""

from __future__ import division, print_function

import numpy as np
import sys

sys.path.append(r'C:\Users\xub\Desktop\Python project\Packages\lib')
#sys.path.append(r'E:\xbl_Berry\Desktop\Python project\Packages\lib')

from simulation.ode import cpu
from simulation.ode import gpu32
from simulation.ode import parallel as pl

from cmath import exp
from numba import cuda

def Gau(w0, wt, x, y, t):
    return np.exp(-(x**2+y**2)/2/w0)*np.exp(-t**2/2/wt) if wt != 0 else np.exp(-(x**2+y**2)/2/w0)

def para(wl, d, n):
    return d/wl/n

def f(z, y, arg):
    return np.array([1j*arg[0]*np.conj(y[1])*y[2]*np.exp(1j*arg[3]*z), 1j*arg[1]*np.conj(y[0])*y[2]*np.exp(1j*arg[3]*z), 
                         1j*arg[2]*y[0]*y[1]*np.exp(-1j*arg[3]*z)])
    
@cuda.jit(device=True)
def gf(x, y, arg, h):
    dydx0 = 1j*arg[0]*y[1].conjugate()*y[2]*exp(1j*arg[3]*x)*h
    dydx1 = 1j*arg[1]*y[0].conjugate()*y[2]*exp(1j*arg[3]*x)*h
    dydx2 = 1j*arg[2]*y[0]*y[1]*exp(-1j*arg[3]*x)*h
    return dydx0, dydx1, dydx2 
    
def comparison(xsize, ysize):
    
    from timeit import default_timer as timer
    
    w0 = 500 # 500um
    
    ns = np.array([1.617, 1.617, 1.617])
    deff = 4.47e-7
    wls =np.array([.607, .607, .303])
    
    args = np.append(para(wls, deff, ns),0) # 4 element array
    args = tuple(args) #cuda.jit only takes tuple. CRITICAL!!!
    
    x = np.linspace(-128*1, 127*1, xsize)
    y = np.linspace(-128*1, 127*1, ysize)
    
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    E = 100*Gau(w0, 10, xx, yy, 0)
    
    E = E.astype(np.complex64) #declaration of the data type is CRITITCAL for cuda.jit
    
    E1 = np.empty((xsize, ysize, 3), dtype=np.complex64)
    
    x0, x1 = 0, 5000 
    
# cpu parallel computing -> E1
    start = timer()
    EEE = np.dstack([E,E,np.zeros_like(E)]).reshape(-1,3)
    
    sol = pl.mp_ode(EEE, x0, x1, f, args)

    E1[:] = sol.reshape(xsize, ysize, 3)
    
    end = timer()
    cpu_t = end-start
    print("cpu parallel time-cost: {} s".format(cpu_t))
#    
    A = E.copy()
    B = E.copy()
    C = np.zeros_like(E,dtype=np.complex64)
#    
    E2 = np.empty((xsize, ysize, 3), dtype=np.complex64)
    
# gpu parallel computing -> E2
    start = timer()
    
    test = gpu32.ode(x0, x1, gf, 3, args)#overhead
    
    test.move([A, B, C], [E2[:,:,i] for i in range(3)])
    
    end = timer()    
    gpu_t = end-start
    print ("gpu parallel time-cost: {} s".format(gpu_t))
    
# cpu numpy broadcast computing -> E3
    E3 = np.empty((xsize, ysize, 3), dtype=np.complex64)
    
    start = timer()
    y0 = A, B, C
    E3[:,:,0], E3[:,:,1], E3[:,:,2] = cpu.ode(x0, x1, y0, f, args)[0]
    
    end = timer()    
    numpy_t = end-start
    print("numpy time-cost: {} s".format(numpy_t))
    
    print("cpu vs gpu results are same: {}".format(np.allclose(E1, E2)))
    print("cpu vs numpy results are same: {}".format(np.allclose(E1, E3)))
    print()
    
#    xy.image(E1[:,:,2])
#    xy.image(E2[:,:,2])
#    xy.image(E3[:,:,2])
    
    return [cpu_t, gpu_t, numpy_t]
    
if __name__ == "__main__": 

    tests = [(16,16), (32,32), (64,64), (128,128), (256,256), (512,512), (1024,1024)]  
    
    c = []
    g = []
    n = []
    
    for i, test in enumerate(tests):
        print('trail {0}\ntest: {1}'.format(i+1,test))
        temp = comparison(*test)
        c.append(temp[0])
        g.append(temp[1])
        n.append(temp[2])
        
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    xaxis = [test[0]*test[1] for test in tests]
    line1, = ax.plot(xaxis, c, marker='s', label='cpu multipleprocessing')    
    line2, = ax.plot(xaxis, g, marker='o', label='gpu cuda')  
    line3, = ax.plot(xaxis, n, marker='v', label='cpu numpy')
    ax.legend(loc='upper left')
    ax.set_xscale('log')
    plt.show()      
        
        
        