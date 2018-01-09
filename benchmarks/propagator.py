"""

"""

from __future__ import division, print_function

import numpy as np
import sys

sys.path.append(r'C:\Users\xub\OneDrive - Coherent, Inc\Python project\Packages\lib')
#sys.path.append(r'E:\xbl_Berry\Desktop\Python project\Packages\lib')

from simulation.propagate import cpu
from simulation.propagate import gpu64

from simulation.crystals import crystal
from simulation.crystals.data import lbo3

from timeit import default_timer as timer


def Gau(w0, x, y, wt=None, t=None):
    return np.exp(-(x**2+y**2)/2/w0) if ((wt is None) or (t is None)) else np.exp(-(x**2+y**2)/2/w0)*np.exp(-t**2/2/wt)

def comparison(xsize, ysize, tsize):
    
    wl = 1.064 #1.064 um
    w0 = 500 # 500um
    wt = 30 # 30ps
    dz = 500 # 500um
        
    crys = crystal(lbo3, wl*1e3, 25, 90, 45)
        
    key = 'lo'
    
    x = np.linspace(-256*1, 256*1, xsize, dtype=np.complex128)
    y = np.linspace(-256*1, 256*1, ysize, dtype=np.complex128)
    t = np.linspace(-64*1, 63*1, tsize, dtype=np.complex128)
    
    xx, yy, tt = np.meshgrid(x, y, t)

    E = Gau(w0, xx, yy, wt, tt)


    start = timer()
    
    sol1 = cpu.xytpropagator(E, x, y, t, dz, crys, key)

    end = timer()
    cpu_t = end - start
    print('cpu-based numpy: {}'.format(cpu_t))
    
    start = timer()
    
    sol2 = gpu64.xytpropagator(E, x, y, t, dz, crys, key)

    end = timer()
    gpu_t = end - start
    print('numpy gpu combination: {}'.format(gpu_t))     
    
    print(np.allclose(sol1, sol2))
    print()
    
    return [cpu_t, gpu_t]


if __name__ == '__main__':
    
    tests = [(16,16,16), (32,32,16), (64,64,32), (128,128,64), (256,256,128), (512,512,256)]  
    
    c = []
    g = []
    
    for i, test in enumerate(tests):
        print('trail {0}\ntest: {1}'.format(i+1,test))
        temp = comparison(*test)
        c.append(temp[0])
        g.append(temp[1])
        
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    xaxis = [test[0]*test[1]*test[2] for test in tests]
    line1, = ax.plot(xaxis, c, marker='s', label='cpu numpy')    
    line2, = ax.plot(xaxis, g, marker='o', label='gpu numpy')  
    ax.legend(loc='upper left')
    ax.set_xscale('log')
    plt.show()
    
    

