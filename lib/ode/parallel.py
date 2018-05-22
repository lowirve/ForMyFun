# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 10:56:39 2018

@author: XuBo
"""

from __future__ import division, print_function

from scipy.integrate import ode
import numpy as np
import multiprocessing as mp
from functools import partial


def codeint(y0, x0, x1, f, args):
        
    r = ode(f).set_integrator('zvode')
    r.set_initial_value(y0, x0).set_f_params(args)
    sol = r.integrate(x1)

    if not r.successful():
        raise RuntimeError("ode failed")
    else:
        return sol    
    
    
def mp_ode(y0, x0, x1, f, args):
        
    cores = int(mp.cpu_count()-2)
    
    para = {'x0':x0, 'x1':x1, 'f':f, 'args':args}
        
    _f = partial(codeint, **para)
    
    p = mp.Pool(cores)
    sol = p.map(_f, y0)
    
    return np.array(sol)
