# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:57:35 2017

Including data for LBO

@author: XuBo
"""

import numpy as np

from collections import namedtuple

ntCrys = namedtuple('NLCrystal', 'name Sellmeier dtensor wlrange cclass comment') #namedtuple used to store information for each crystal
ntCrys.__new__.__defaults__ = (None,)*len(ntCrys._fields) #set the default value to None in case of missed info

#############################################################################################################################################

lbo1 = ntCrys(
name = 'LBO1',
#Sellmeier equations in the order of nx, ny and nz. wl must be in um and tt must be in Centidegree.
Sellmeier =(lambda wl, tt: np.sqrt(2.454140+0.011249/(wl**2-0.011350)-0.014591*wl**2-6.60e-5*wl**4)-9.3e-6*(tt-25),
            lambda wl, tt: np.sqrt(2.539070+0.012711/(wl**2-0.012523)-0.018540*wl**2+2.0e-4*wl**4)-13.6e-6*(tt-25),
            lambda wl, tt: np.sqrt(2.586179+0.013099/(wl**2-0.011893)-0.017968*wl**2-2.26e-4*wl**4)-(6.3-2.1*wl)*1e-6*(tt-25)),
dtensor = np.array([[0, 0, 0, 0, 0, -0.67],[-0.67, 0.04, 0.85, 0, 0, 0],[0, 0, 0, 0.85, 0, 0]]),
wlrange = (160, 2600),
comment = 'Data is from Castech.')

#############################################################################################################################################

lbo2 = ntCrys(
name = 'LBO2',
#Sellmeier equations in the order of nx, ny and nz. wl must be in um and tt must be in Centidegree.
Sellmeier =(lambda wl, tt: np.sqrt(2.4542+0.01125/(wl**2-0.01135)-0.01388*wl**2)-(3.76*wl-2.3)*1e-6*(tt-20),
            lambda wl, tt: np.sqrt(2.5390+0.01277/(wl**2-0.01189)-0.01849*wl**2+4.3025e-5*wl**4-2.9131e-5*wl**6)-(19.40-6.01*wl)*1e-6*(tt-20),
            lambda wl, tt: np.sqrt(2.5865+0.01310/(wl**2-0.01223)-0.01862*wl**2+4.5778e-5*wl**4-3.2526e-5*wl**6)-(9.70-1.50*wl)*1e-6*(tt-20)),
dtensor = np.array([[0, 0, 0, 0, 0, -0.67],[-0.67, 0.04, 0.85, 0, 0, 0],[0, 0, 0, 0.85, 0, 0]]),
wlrange = (160, 2600),
comment = 'Springer, “Nonlinear Optical Crystals: A Complete Survey”, 2005.')

#############################################################################################################################################

lbo3 = ntCrys(
name = 'LBO3',
#Sellmeier equations in the order of nx, ny and nz. wl must be in um and tt must be in Centidegree.
Sellmeier =(lambda wl, tt: np.sqrt(2.4542+0.01125/(wl**2-0.01135)-0.01388*wl**2)-(3.76*wl-2.3)*1e-6*((tt-20)+29.13e-3*(tt-20)**2),
            lambda wl, tt: np.sqrt(2.5390+0.01277/(wl**2-0.01189)-0.01849*wl**2+4.3025e-5*wl**4-2.9131e-5*wl**6)-(19.40-6.01*wl)*1e-6*((tt-20)-32.89e-4*(tt-20)**2),
            lambda wl, tt: np.sqrt(2.5865+0.01310/(wl**2-0.01223)-0.01862*wl**2+4.5778e-5*wl**4-3.2526e-5*wl**6)-(9.70-1.50*wl)*1e-6*((tt-20)-74.49e-4*(tt-20)**2)),
dtensor = np.array([[0, 0, 0, 0, 0, -0.67],[-0.67, 0.04, 0.85, 0, 0, 0],[0, 0, 0, 0.85, 0, 0]]),
wlrange = (160, 2600),
comment = 'K. Kato, IEEE JQE v30 p2950 (1994). The temperature calibration is improved')







