# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 10:49:31 2018

@author: XuBo
"""

from __future__ import division, print_function

import numpy as np

def rkdumb(x,y,f,arg,h,n):
    """Driver routine
        rk4 with no error estimate and hence no adaptive step method
        code is baesd on Numerical Receipe, P713"""
        
    def rk4(x,y,f,arg,h):
        """Algorithm routine"""        
        k1 = h*f(x,y,arg)
        hh = h/2
        k2 = h*f(x+hh,y+k1/2,arg)
        k3 = h*f(x+hh,y+k2/2,arg)
        k4 = h*f(x+h,y+k3,arg)
        return y+k1/6+k2/3+k3/3+k4/6
        
    for i in range(n):
        y = rk4(x,y,f,arg,h)        
        x += h
    return y


def ode(x0, x1, y, f, arg, hstart=10, nmax=10000, eps=1e-8, hmin=None):
    """Driver routine with adaptive method
    
       RK-CK method for ODE and error estimate, and hence doable for adaptive method
       Code is from Numerical Receipe, P719"""
    
    def rkck(x,y,f,arg,h):
        """Algorithm routine"""    
        
        a = [1/5,3/10,3/5,1,7/8]
        b1 = np.array([1/5])
        b2 = np.array([3/40,9/40])
        b3 = np.array([3/10,-9/10,6/5])
        b4 = np.array([-11/54,5/2,-70/27,35/27])
        b5 = np.array([1631/55296,175/521,575/13824,44275/110592,253/4096])
        
        c = np.array([37/378,0,250/621,125/594,0,512/1771])
        ci = np.array([2825/27648,0,18575/48384,13525/55296,277/14336,1/4])
        
        d = c-ci
        
        k1 = h*f(x,y,arg)
        k2 = h*f(x+h*a[0],y+b1[0]*k1,arg)
        k3 = h*f(x+h*a[1],y+b2[0]*k1+b2[1]*k2,arg)
        k4 = h*f(x+h*a[2],y+b3[0]*k1+b3[1]*k2+b3[2]*k3,arg)
        k5 = h*f(x+h*a[3],y+b4[0]*k1+b4[1]*k2+b4[2]*k3+b4[3]*k4,arg)
        k6 = h*f(x+h*a[4],y+b5[0]*k1+b5[1]*k2+b5[2]*k3+b5[3]*k4+b5[4]*k5,arg)
        
        sol = y+c[0]*k1+c[1]*k2+c[2]*k3+c[3]*k4+c[4]*k5+c[5]*k6
        err = d[0]*k1+d[1]*k2+d[2]*k3+d[3]*k4+d[4]*k5+d[5]*k6
        
        return sol, err    
       
    def rkqs(x, y, f, arg, h, eps, yscal):
        """Stepper routine"""
        
        SAFETY = 0.9
        PSHRNK = -0.25
        PGROW = -0.2
        ERRON = 1.89e-4
        
        n=0
        while(1):
            n += 1
            sol, yerr = rkck(x,y,f,arg,h)
            errmax = np.nanmax(np.abs(yerr/yscal)) # evaluate accuracy
            errmax /= eps
            if errmax <= 1:
                hdid = h
                break
            htemp = SAFETY*h*np.power(errmax, PSHRNK)        #truncation error too large, reduce stepsize
            h = max(h/10, htemp) if h >= 0 else min(h/10, htemp)#no more than a factor of 10
            xnew = x+h
            if xnew == x:
                raise ValueError('Stepsize underflow in rkqs. runction error')
            
        if errmax > ERRON:
            hnext = SAFETY*h*np.power(errmax, PGROW) # feedback for next stepsize
        else:
            hnext = 5.*h #No more than a factor of 5 increase
        
        return sol, hdid, hnext, n #return the integration over a valid stepsize, the used stepsize, and the next stepsize    
    
    TINY = 1e-30
    
    if hmin is None:
        hmin = (x1-x0)/1e8
        
    h = np.fabs(hstart)*np.sign(x1-x0) # h must be a real number, this step ensures that the sign of step is right.
    xtemp = x0
    ytemp = y
    
    ncount = 0
    
    for i in range(nmax):
#        print(h)
        if np.fabs(h) <= np.fabs(hmin):
            raise ValueError("Step size too small in odeint")
        yscal = np.abs(ytemp)+np.abs(h*f(xtemp, ytemp, arg))+TINY
        if (xtemp+h-x1)*(xtemp+h-x0) > 0:
            h = x1-xtemp
        ytemp, hdid, hnext, nint = rkqs(xtemp, ytemp, f, arg, h, eps, yscal)
        xtemp += hdid
        ncount += nint
        if (xtemp-x1)*(x1-x0) >= 0:
            return ytemp, (xtemp, ncount)
        h = hnext
    
    raise ValueError("Too many steps in routine odeint")


