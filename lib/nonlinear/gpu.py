# -*- coding: utf-8 -*-
"""

"""

from __future__ import division, print_function

from numba import cuda
from cmath import exp


@cuda.jit(device=True)
def sfg(x, y, arg, h):#x is scalar, y is a tuple, arg is a tuple
    dydx0 = 1j*arg[0]*y[1].conjugate()*y[2]*exp(1j*arg[3]*x)*h
    dydx1 = 1j*arg[1]*y[0].conjugate()*y[2]*exp(1j*arg[3]*x)*h
    dydx2 = 1j*arg[2]*y[0]*y[1]*exp(-1j*arg[3]*x)*h
    return dydx0, dydx1, dydx2 


    