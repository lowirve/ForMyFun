"""

"""

from __future__ import division, print_function

import numpy as np
import sys

sys.path.append(r'C:\Users\xub\Desktop\Python project\Packages\lib')
#sys.path.append(r'E:\xbl_Berry\Desktop\Python project\Packages\lib')

from simulation.propagator import cpu
from simulation.propagator import gpu

from simulation.crystals import crystal
from simulation.crystals.data import lbo3
from simulation.coordinate import xyt

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
    
    space2 = xyt(x, y, t)
    
    E = Gau(w0, space2.xxx, space2.yyy, wt, space2.ttt)
    E = E.astype(np.complex128)

    start = timer()
    
    sol1 = cpu.xytpropagator(E, x, y, t, dz, crys, key)

    end = timer()
    cpu_t1 = end - start
    print('cpu-based numpy: {}'.format(cpu_t1))
    
    start = timer()
    
    sol2 = gpu.xytpropagator(E, x, y, t, dz, crys, key)

    end = timer()
    gpu_t1 = end - start
    print('numpy gpu combination: {}'.format(gpu_t1))     
    
    print(np.allclose(sol1, sol2))
    print()
    
    start = timer()
    
    sol = cpu.propagator(crys, space2, key)
    
    sol.load(dz)
    sol.move(E)
 
    sol3 = sol.get()

    end = timer()
    cpu_t2 = end - start
    print('cpu-based numpy class: {}'.format(cpu_t2))
    
    start = timer()
    
    sol = gpu.propagator(crys, space2, key)
    
    sol.load(dz)
    sol.move(E)
 
    sol4 = sol.get()

    end = timer()
    gpu_t2 = end - start
    print('numpy gpu combination class: {}'.format(gpu_t2))     
    
    print(np.allclose(sol3, sol4))
    print()
    
    return [cpu_t1, gpu_t1, cpu_t2, gpu_t2]


if __name__ == '__main__':
    
    tests = [(16,16,16), (32,32,16), (64,64,32), (128,128,64), (256,256,128), (512,256,256)]  
    
    c1 = []
    g1 = []
    c2 = []
    g2 = []
    
    for i, test in enumerate(tests):
        print('trail {0}\ntest: {1}'.format(i+1,test))
        temp = comparison(*test)
        c1.append(temp[0])
        g1.append(temp[1])
        c2.append(temp[2])
        g2.append(temp[3])
        
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    xaxis = [test[0]*test[1]*test[2] for test in tests]
    line1, = ax.plot(xaxis, c1, marker='s', label='cpu numpy')    
    line2, = ax.plot(xaxis, g1, marker='o', label='gpu numpy')  
    line3, = ax.plot(xaxis, c2, marker='+', label='cpu numpy class')    
    line4, = ax.plot(xaxis, g2, marker='x', label='gpu numpy class') 
    ax.legend(loc='upper left')
    ax.set_xscale('log')
    plt.show()
    
    

