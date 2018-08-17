# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:57:35 2017

Including data for LBO, bbo

It is planned to merge this with crystal module eventually.

@author: XuBo

Considering the expandability in the future, the current structure needs to be changed:
    1) Change the data class to adopt the fact that in most case only refractive index (Sellmeier) and its variance due to temperature is 
        variable from case to case. Maybe unify them in one object, but include all the sources within it.
    2) Change Sellmeier function to a univariable function, and put other parameters, such as temperature, pressure, and etc as optional.
    
"""

import numpy as np

from collections import namedtuple

ntCrys = namedtuple('NLCrystal', 'name Sellmeier dtensor wlrange ttrange birefringence comment') #namedtuple used to store information for each crystal
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
birefringence = 'Biaxial',
comment = 'n: Data is from Castech.')

#############################################################################################################################################

lbo2 = ntCrys(
name = 'LBO2',
#Sellmeier equations in the order of nx, ny and nz. wl must be in um and tt must be in Centidegree.
Sellmeier =(lambda wl, tt: np.sqrt(2.4542+0.01125/(wl**2-0.01135)-0.01388*wl**2)-(3.76*wl-2.3)*1e-6*(tt-20),
            lambda wl, tt: np.sqrt(2.5390+0.01277/(wl**2-0.01189)-0.01849*wl**2+4.3025e-5*wl**4-2.9131e-5*wl**6)-(19.40-6.01*wl)*1e-6*(tt-20),
            lambda wl, tt: np.sqrt(2.5865+0.01310/(wl**2-0.01223)-0.01862*wl**2+4.5778e-5*wl**4-3.2526e-5*wl**6)-(9.70-1.50*wl)*1e-6*(tt-20)),
dtensor = np.array([[0, 0, 0, 0, 0, -0.67],[-0.67, 0.04, 0.85, 0, 0, 0],[0, 0, 0, 0.85, 0, 0]]),
wlrange = (160, 2600),
birefringence = 'Biaxial',
comment = 'n: Springer, “Nonlinear Optical Crystals: A Complete Survey”, 2005.')

#############################################################################################################################################

lbo3 = ntCrys(
name = 'LBO3',
#Sellmeier equations in the order of nx, ny and nz. wl must be in um and tt must be in Centidegree.
Sellmeier =(lambda wl, tt: np.sqrt(2.4542+0.01125/(wl**2-0.01135)-0.01388*wl**2)-(3.76*wl-2.3)*1e-6*((tt-20)+29.13e-3*(tt-20)**2),
            lambda wl, tt: np.sqrt(2.5390+0.01277/(wl**2-0.01189)-0.01849*wl**2+4.3025e-5*wl**4-2.9131e-5*wl**6)-(19.40-6.01*wl)*1e-6*((tt-20)-32.89e-4*(tt-20)**2),
            lambda wl, tt: np.sqrt(2.5865+0.01310/(wl**2-0.01223)-0.01862*wl**2+4.5778e-5*wl**4-3.2526e-5*wl**6)-(9.70-1.50*wl)*1e-6*((tt-20)-74.49e-4*(tt-20)**2)),
dtensor = np.array([[0, 0, 0, 0, 0, -0.67],[-0.67, 0.04, 0.85, 0, 0, 0],[0, 0, 0, 0.85, 0, 0]]),
wlrange = (160, 2600),
birefringence = 'Biaxial',
comment = 'n: K. Kato, IEEE JQE v30 p2950 (1994). The temperature calibration is improved')

#############################################################################################################################################

bbo1 = ntCrys(
name = 'beta-BBO1',
#Sellmeier equations in the order of nx, ny and nz. wl must be in um and tt must be in Centidegree.
Sellmeier =(lambda wl, tt: np.sqrt(3.63357+0.01878/(wl**2-0.01822)+60.9129/(wl**2-67.8505))+(-0.0137/wl**3+0.0607/wl**2-0.1334/wl-1.5287)*1e-5*(tt-20),
            lambda wl, tt: np.sqrt(3.63357+0.01878/(wl**2-0.01822)+60.9129/(wl**2-67.8505))+(-0.0137/wl**3+0.0607/wl**2-0.1334/wl-1.5287)*1e-5*(tt-20),
            lambda wl, tt: np.sqrt(3.33469+0.01237/(wl**2-0.01647)+79.0672/(wl**2-82.2919))+(0.0413/wl**3-0.2119/wl**2+0.4408/wl-1.2749)*1e-5*(tt-20)),
dtensor = np.array([[0, 0, 0, 0, 0.08, 2.2],[2.2, -2.2, 0, 0.08, 0, 0],[0.08, 0.08, 0, 0, 0, 0]]),
wlrange = (205, 2600),
birefringence = 'Negative Uniaxial',
comment = 'n: K. Kato, et al, Sellmeier and thermo-optic dispersion formulas for β-BaB2O4 (revisited)')

#############################################################################################################################################

bbo2 = ntCrys(
name = 'beta-BBO2',
#Sellmeier equations in the order of nx, ny and nz. wl must be in um and tt must be in Centidegree.
Sellmeier =(lambda wl, tt: np.sqrt(2.7359+0.01878/(wl**2-0.01822)-0.01471*wl**2+0.0006081*wl**4-6.74e-5*wl**6)-(16.6)*1e-6*(tt-20),
            lambda wl, tt: np.sqrt(2.7359+0.01878/(wl**2-0.01822)-0.01471*wl**2+0.0006081*wl**4-6.74e-5*wl**6)-(16.6)*1e-6*(tt-20),
            lambda wl, tt: np.sqrt(2.3753+0.01224/(wl**2-0.01667)-0.01627*wl**2+0.0005716*wl**4-6.305e-5*wl**6)-(9.3)*1e-6*(tt-20)),
dtensor = np.array([[0, 0, 0, 0, 0.08, 2.2],[2.2, -2.2, 0, 0.08, 0, 0],[0.08, 0.08, 0, 0, 0, 0]]),
wlrange = (185, 2600),
ttrange= (20, 80), #Celsius
birefringence = 'Negative Uniaxial',
comment = 'n: Springer, “Nonlinear Optical Crystals: A Complete Survey”, 2005.')

#############################################################################################################################################

abbo = ntCrys(
name = 'alpha-BBO',
#Sellmeier equations in the order of nx, ny and nz. wl must be in um and tt must be in Centidegree.
Sellmeier =(lambda wl: np.sqrt(2.7471+0.01878/(wl**2-0.01822)-0.01354*wl**2),
            lambda wl: np.sqrt(2.7471+0.01878/(wl**2-0.01822)-0.01354*wl**2),
            lambda wl: np.sqrt(2.37153+0.01224/(wl**2-0.01667)-0.01516*wl**2)),
dtensor = None,
wlrange = (185, 2600),
ttrange= None, #Celsius
birefringence = 'Negative Uniaxial',
comment = 'n: Data is from Castech.')

#############################################################################################################################################

clbo = ntCrys(
name = 'CLBO',
#Sellmeier equations in the order of nx, ny and nz. wl must be in um and tt must be in Centidegree.
Sellmeier =(lambda wl, tt: np.sqrt(2.2104+0.01018/(wl**2-0.01424)-0.01258*wl**2)+(-0.328/wl-12.48)*1e-6*(tt-20),
            lambda wl, tt: np.sqrt(2.2104+0.01018/(wl**2-0.01424)-0.01258*wl**2)+(-0.328/wl-12.48)*1e-6*(tt-20),
            lambda wl, tt: np.sqrt(2.0588+0.00838/(wl**2-0.01363)-0.00607*wl**2)+(0.014/wl**3-0.039/wl**2+0.047/wl-8.36)*1e-6*(tt-20)),
dtensor = np.array([[0, 0, 0, 0.61, 0, 0],[0, 0, 0, 0, 0.61, 0],[0, 0, 0, 0, 0, 0.74]]),#from SNLO
wlrange = (191.4, 2090),
ttrange= (20, 80), #Celsius
birefringence = 'Negative Uniaxial',
comment = 'n: Springer, “Nonlinear Optical Crystals: A Complete Survey”, 2005.')

#############################################################################################################################################

lb4 = ntCrys(
name = 'LB4',
#Sellmeier equations in the order of nx, ny and nz. wl must be in um and tt must be in Centidegree.
Sellmeier =(lambda wl: np.sqrt(2.56431+0.0112337/(wl**2-0.013103)-0.019075*wl**2),
            lambda wl: np.sqrt(2.56431+0.0112337/(wl**2-0.013103)-0.019075*wl**2),
            lambda wl: np.sqrt(2.38651+0.010664/(wl**2-0.012878)-0.012813*wl**2)),
dtensor = np.array([[0, 0, 0, 0, 0.12, 0],[0, 0, 0, 0.12, 0, 0],[0.12, 0.12, 0.47, 0, 0, 0]]),
wlrange = (160, 2090),
birefringence = 'Negative Uniaxial',
comment = 'n: Springer, “Nonlinear Optical Crystals: A Complete Survey”, 2005.')

#############################################################################################################################################


