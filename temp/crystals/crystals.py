# -*- coding: utf-8 -*-
"""
@author: XuBo

Class Crystal is created to perform all calculations related to crystals' optical properties. 
It needs Sellmeiers equation and tensor d, which is stored in a namedtuple for each crystal.

#reorganize the structure of data ??

"""
from __future__ import division, print_function

import numpy as np

from data import ntCrys, lbo3

from scipy.optimize import bisect


class crystal(object):
    _c = 2.99792458 # unit is 10^8 m/s
    _oa = None # optical axis
    _nhl = None # eigenpolarization refractive indices
    _rhl = None # walkoff
    _gihl = None # group indices
    _gvdhl = None # group velocity dispersion, unit is fs^2/mm
    _wl = 1.064 # wavelength, unite is um.
    _tt = 25. # temperature, based on centidegree
    _theta = 0. # rad
    _phi = 0. # rad
    _delta = 0. # rad
    _material = None # the crystal we are seeking PM
    _a = 0
    _b = 0
    _d = 0
    
    @property
    def material(self): # material data
        return self._material
    
    @material.setter
    def material(self, value):
        if isinstance(value, ntCrys):
            self._material = value
            self._fnx, self._fny, self._fnz = value.Sellmeier
            self._dtensor = value.dtensor
            self._wlrange = value.wlrange
        else:
            raise ValueError('Input must be a defined crystal')           
        
    @property # return the order of no/ne corresponding to nhi nlo. 0 is hi, 1 is low.
              # for are defined in the principle planes, otherwise shows None
    def oe(self): 
        return self.foe(self.nhl, self.nx, self.ny, self.nz)
    
    @property
    def theta(self): 
        return np.rad2deg(self._theta)
    
    @theta.setter
    def theta(self, value):
        if value >= 0. and value <= 180.:
            self._theta = np.deg2rad(value)
        else:
            raise ValueError('Theta must be >= 0 and <= 180')
        
    @property
    def phi(self):
        return np.rad2deg(self._phi)
    
    @phi.setter
    def phi(self, value):
        if value >= -180. and value <= 180.:
            self._phi = np.deg2rad(value)               
        else:
            raise ValueError('Phi must be >= -180 and <= 180')  
            
    @property #optical axis
    def oa(self):
        self._oa = self.foa(self.nx, self.ny, self.nz)
        return np.rad2deg(self._oa)  
    
    @property # eigenpolarization value, return (nhi, nlo)
    def nhl(self): 
        self._nhl = self.fnhl(self._theta, self._phi, self.nx, self.ny, self.nz)
        return self._nhl  

    @property # walkoff for each eigenpolarization, return (rho_hi, rho_lo)
    def rhl(self): 
        self._nhl = self.nhl
        self._rhl = np.array(map(lambda x: self.fwalkoff(self._theta, self._phi, self.nx, self.ny, self.nz, x), self._nhl))
        return self._rhl #rad   
    
    @property # group indices (ghi, glo)
    def gihl(self): 
        self._gihl = self.fgindex(self._wl, self._tt, self._theta, self._phi)
        return self._gihl   
    
    @property # group velocity dispersion (GVD_hi, GVD_lo), unit is fs^2/mm
    def gvdhl(self): 
        self._gvdhl = self.fgvd(self._wl, self._tt, self._theta, self._phi)
        return self._gvdhl
    
    @property # wavelength
    def wl(self):
        return self._wl*1e3
    
    @wl.setter
    def wl(self, value):#unit must be nm
        if value > self._wlrange[0] and value < self._wlrange[1]:
            self._wl = value/1e3
            self.nx, self.ny, self.nz = self.fupdateN(self._wl, self._tt)
        else:
            raise ValueError('Wavelength must be > {0} and < {1}'.format(*self._wlrange))
            
    @property
    def brewster(self):
        return np.rad2deg(self.fbrewster())

    @property # temperature
    def tt(self):
        return self._tt
    
    @tt.setter
    def tt(self, value):
        if value > 0. and value < 200.:
            self._tt = value
            self.nx, self.ny, self.nz = self.fupdateN(self._wl, self._tt)
        else:
            raise ValueError('Temperature must be > 0 C and < 200 C')   
    
    def __init__(self, Crys, wl=None, tt=None, theta=None, phi=None): # Crys must be a ntCrys type
        """To initialize the crystal class."""           
        self.material = Crys # Crystal info from ntCrys type is necessary. Other parameters can be given separately.
        if wl:
            self.wl = wl
            
        if tt:
            self.tt = tt
            
        if theta:
            self.theta = theta
            
        if phi:
            self.phi = phi
            
    def fupdateN(self, wl, tt): 
        """This function is to calculate nx, ny, nz based on wavelength and temperature"""
        return self._fnx(wl, tt), self._fny(wl, tt), self._fnz(wl, tt)

    def foa(self, nx, ny, nz):  
        """This function is to calculate the optical axis"""
        return np.arcsin(np.sqrt((nx**(-2)-ny**(-2))/(nx**(-2)-nz**(-2))))
    
    def fnhl(self, theta, phi, nx, ny, nz):
        """This function is to calculate nhi and nlo based on given direction angles and nx, ny, nz"""
        cosT = np.cos(theta)
        cosP = np.cos(phi)
        sinP = np.sin(phi)
        sinT = np.sin(theta)
        
        self._a = cosT**2*cosP**2/nx**2+cosT**2*sinP**2/ny**2+sinT**2/nz**2
        self._b = 2*cosT*sinP*cosP*(ny**(-2)-nx**(-2))
        self._d = sinP**2*nx**(-2)+cosP**2*ny**(-2)
        
#        calculate delta angle. Not useful now.
#        self._delta = 0.5*(np.arctan(cosT*np.sin(2*phi)/(
#                np.tan(self._oa)**(-2)*sinT**2+sinP**2-cosT**2*cosP**2)))

#       Calculate based on the analytical solutions
        nhi = np.sqrt(2./(self._a+self._d-np.sqrt((self._a-self._d)**2+self._b**2)))
        nlo = np.sqrt(2./(self._a+self._d+np.sqrt((self._a-self._d)**2+self._b**2)))
        
        return np.array((nhi, nlo))

#       An alternative way to calculate by solving the Fresnel equation.        
#        def fFresnel(n):
#            return sinT**2*cosP**2*(1/n**2-1/ny**2)*(1/n**2-1/nz**2)+sinT**2*sinP**2*(
#                    1/n**2-1/nx**2)*(1/n**2-1/nz**2)+cosT**2*(1/n**2-1/nx**2)*(1/n**2-1/ny**2)
#        
#        from scipy import optimize
#        x0 = np.array([nz, nx])
#        return optimize.fsolve(fFresnel, x0)
    
    
    def fwalkoff(self, theta, phi, nx, ny, nz, n):  
        """This function is to calculate the walkoff at a given refractive index"""
        for ni in (nx, ny, nz):
            if abs(ni-n) < 1e-6:
                return 0

        cosT = np.cos(theta)
        cosP = np.cos(phi)
        sinP = np.sin(phi)
        sinT = np.sin(theta)        
        
        def fDprime(n):
            return np.sqrt((sinT*cosP/(1/n**2-1/nx**2))**2+(sinT*sinP/(1/n**2-1/ny**2))**2+(
                    cosT/(1/n**2-1/nz**2))**2)
            
        return np.arctan(n**2/fDprime(n))
    
    def fgindex(self, wl, tt, theta, phi):
        """This functoin is to numerically calculate the group velocity index at given angles and (wavelength, temperature)"""
        dwl = 1e-6
        wl2 = wl + dwl
        nhl = self.fnhl(theta, phi, *self.fupdateN(wl, tt))
        return (nhl - (self.fnhl(theta, phi, *self.fupdateN(wl2, tt))-nhl)/dwl*wl) #ng(wl) = n(wl) - wl*dn/dwl
    
    def fgvd(self, wl, tt, theta, phi):
        """This functoin is to numerically calculate the group velocity dispersion at given angles and (wavelength, temperature)"""
        dwl = 1e-5
        wl2 = wl + dwl
        wl3 = wl - dwl
        #return -(self.fgindex(wl2, tt, theta, phi) - self.fgindex(wl, tt, theta, phi))/dwl*wl**2*1e5/np.pi/self._c**2/2 #-2wl^2/(2pic^2)*dng/dwl
        return (self.fnhl(theta, phi, *self.fupdateN(wl2, tt)) + self.fnhl(theta, phi, *self.fupdateN(wl3, tt)
                           ) - 2* self.fnhl(theta, phi, *self.fupdateN(wl, tt)))/dwl**2*wl**3*1e5/2./np.pi/self._c**2
                          #wl^3/(2pic^2)*d^2n/dwl^2 produces more accurate in principle                          
                          
    def foe(self, nhl, nx, ny, nz): 
        """"This function is to identify ne/no for uniaxial crystals or for the principle plan cut of biaxial crystals"""
        def ftest(n, nx, ny, nz): # test if n is one of nx, ny, nz
            for ni in (nx, ny, nz):
                if abs(ni-n) < 1e-3: # the tolerance cannot be too large, otherwise it causes trouble for other function.
                    return True    
                
        if ftest(nhl[0], nx, ny, nz):
                return {'o':0, 'e':1}
            
        if ftest(nhl[1], nx, ny, nz):
                return {'o':1, 'e':0}       
        
        #Be aware that principle axis cut is an exception in this function, since both nhi and nlo will be no in the definition.
        return {'o':None, 'e':None}
    
    
    def show(self):
        print ("nx, ny, nz: {:.8f} {:.8f} {:.8f}".format(self.nx, self.ny, self.nz))
        print ("nhi, nlo: {:.4f} {:.4f}".format(*self.nhl))
        print ("oa: {:.3f} deg".format(self.oa))
        print ("walkoff hi, lo: {:.2f} {:.2f} mrad".format(*(self.rhl*1e3)))
        print ("Refractive index hi, lo: {:.6f} {:.6f}".format(*self.nhl))
        print ("group index hi, lo: {:.5f} {:.5f}".format(*self.gihl))
        print ("gvd hi, lo: {:.3f} {:.3f}".format(*self.gvdhl))
        print 
        
    def fbrewster(self):
        sol = []
        
        _f = lambda x: self.nhl[0]*np.sin(x)-np.sin(np.pi/2-x)
        
        sol.append(bisect(_f, 0, np.pi/2))
        
        _f = lambda x: self.nhl[1]*np.sin(x)-np.sin(np.pi/2-x)
        
        sol.append(bisect(_f, 0, np.pi/2))
        
        return np.array(sol)
        
    def parameters(self):
        kwargs = dict(nx = self.nx, ny = self.ny, nz = self.nz, nlo = self.nhl[1], 
                      nhi = self.nhl[0], wl = self.wl, rlo = self.rhl[1], 
                      rhi = self.rhl[0], gihi = self.gihl[0], gilo = self.gihl[1], 
                      gvdhi = self.gvdhl[0], gvdlo = self.gvdhl[1])
        return kwargs

            
if __name__ == '__main__':     
       
#    lbo = crystal(lbo2)
#    for wl, tt in [(1064, 25), (532, 25)]:
#        for theta, phi in [(90,0), (45, 45)]:
#            lbo.wl = wl
#            lbo.tt = tt
#            lbo.theta = theta
#            lbo.phi = phi
#            print ('wl: {}, tt: {}, theta: {}, phi: {}'.format(wl, tt, theta, phi))
#            test1(lbo)            
    
    lbo = crystal(lbo3, 532, 25, 90, 11.4)
#    lbo.tt = 25
#    lbo.wl = 532
#    lbo.delta = 90
#    lbo.phi = 11.6   
    lbo.show()
    
    print(lbo.brewster)
    
    lbo.tt = 149
    lbo.wl = 1064
    print(lbo.parameters())

        

