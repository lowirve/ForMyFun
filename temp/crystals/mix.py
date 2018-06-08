# -*- coding: utf-8 -*-
"""
Phase matching class

@author: XuBo

Introduce a new ubiquitous function to calculate deff
Introduce a new universal function to calculate phase matching at principle cuts

"""
from __future__ import division, print_function

import numpy as np
from crystals import crystal
from data import ntCrys
from scipy.optimize import minimize_scalar#, bisect    


class phasematch(object):
    
    _wls = [1064, 1064, 532] # wavelength pair, unite is nm.
    _angles = None # a dictionary to store all phase match angle, unit is centigrade
    _material = None # the crystal we are seeking PM
    _dtensor = None # d tensor to calculate deff
    _data = None # crystals properties at the given PM angles
    _tt = 25. # temperature, based on centidegree
    _crystals = None # a list of three crystals correponding to three wavelengths
    
    @property
    def data(self):
        self._angles = self.angles
        self._data = self._fdata(self._crystals, self._wls, self._tt, self._angles)
        return self._data
    
    @property
    def angles(self):
        self._angles = self.fPMangle(self._crystals, self._slant)
        return self._angles
    
    @property
    def wls(self):
        return self._wls
    
    @wls.setter
    def wls(self, lvalue): # input must be a list
        self._wls = self.fwls(lvalue)
        for cry, wl in zip(self._crystals, self._wls):
            cry.wl = wl
    
    @property
    def tt(self):
        return self._tt
    
    @tt.setter
    def tt(self, value):
        self._tt = value
        for cry in self._crystals:
            cry.tt = value
            
    @property
    def material(self):
        return self._material
    
    @material.setter
    def material(self, value):
        if isinstance(value, ntCrys):
            self._material = value
            self._crystals = self._finit(self._material, self._wls, self._tt)
            self._dtensor = value.dtensor
        else:
            raise ValueError('Input must be a defined crystal')  
    
    def __init__(self, Crys, wls=None, tt=None, slant=0):
        self.material = Crys  
        self._slant = slant
        if wls:
            self.wls = wls
        
        if tt:
            self.tt = tt
            
    def _finit(self, material, wls, tt):
        wl1, wl2, wl3 = wls
        
        a = crystal(material, wl1, tt)

        b = crystal(material, wl2, tt)

        c = crystal(material, wl3, tt)

        return [a, b, c]

    def fwls(self, lvalue):
        """calculate and organize the wavelengths group"""
        try:
            if lvalue.count(0) != 1:
                raise ValueError
        except ValueError:
            print("Input must have one zero. Please try again")
            return None
    
        index = lvalue.index(0)
        
        if index == 0:
            lvalue[0] = 1./(1./lvalue[2]-1./lvalue[1])
        elif index == 1:
            lvalue[1] = 1./(1./lvalue[2]-1./lvalue[0])
        else:
            lvalue[2] = 1./(1./lvalue[1]+1./lvalue[0])
            
        return sorted(lvalue, reverse=True)

    def fPMangle(self, crystals, slant=0): # 
        '''Phase matching calculation'''
        
        def deltak(crystals, axes, **angles):
            
            def _deltak(crystals, axes):
                a, b, c = [int(axis=='l') for axis in axes]
                return np.pi*2*crystals[0].nhl[a]/crystals[0].wl+np.pi*2*crystals[1].nhl[b]/crystals[1].wl\
                       -np.pi*2*crystals[2].nhl[c]/crystals[2].wl
            
            for cry in crystals:
                cry.theta = angles['Theta']
                cry.phi = angles['Phi']
            
            return np.abs(_deltak(crystals, axes))*1e3
        
        def collinear(crystals, **angles):
            
            if self.wls[0] == self.wls[1]:
                axess = ['hhl', 'hll', 'hlh', 'llh']
            else:
                axess = ['hhl', 'hll', 'hlh', 'lhh', 'lhl', 'llh']
            
            result = {}
                
            for axes in axess:
                if 'Phi' in angles.keys():
                    f = lambda x: deltak(crystals, axes, Theta=x, Phi=angles['Phi'])
                else:
                    f = lambda x: deltak(crystals, axes, Theta=angles['Theta'], Phi=x)
                    
#                root = bisect(f,0,90) #not working as it requires the opposite signs on the two endpoints, which is not straightforward.
                
                sol = minimize_scalar(f, bounds=(0,90), method='bounded')
                fsol = f(sol.x)
                
                if fsol <= 1e-3:
                    if 'Phi' in angles.keys():
                        result[axes] = {'Theta':sol.x, 'Phi':angles['Phi']}
                    else:
                        result[axes] = {'Theta':angles['Theta'], 'Phi':sol.x}
            
            return result
        
        def _updateOE(sol):            
                temp = crystals[0]
                result = {}
                for axes in sol.keys():
                    temp.theta = sol[axes]['Theta']
                    temp.phi = sol[axes]['Phi']
                    oe = temp.oe
                    keys = {}
                    for key in oe.keys():
                        if oe[key] == 0:
                            keys['h'] = key
                        else:
                            keys['l'] = key
                    pols = ''.join([keys[st] for st in axes])

                    result[pols] = temp.theta, temp.phi
                    
                return result        
    
        if slant == 0:#collinear
            def _XY():
                sol = collinear(crystals, Theta=90)               
                sol = _updateOE(sol)
                
                for key in sol.keys():
                    if np.abs(self.fdeff(crystals, key, 'XY', *sol[key])) < 1e-2:
                        sol.pop(key)
                
                return sol

            
            def _YZ():
                sol = collinear(crystals, Phi=90)               
                sol = _updateOE(sol)
                
                for key in sol.keys():
                    if np.abs(self.fdeff(crystals, key, 'YZ', *sol[key])) < 1e-2:
                        sol.pop(key)
                
                return sol
                
            def _XZ():
                sol = collinear(crystals, Phi=0)               
                sol = _updateOE(sol)
                
                for key in sol.keys():
                    if np.abs(self.fdeff(crystals, key, 'XZ', *sol[key])) < 1e-2:
                        sol.pop(key)
                
                return sol
        
        return {'XY':_XY(),
                'YZ':_YZ(),
                'XZ':_XZ()}
        
    def _fdata(self, crystals, wls, tt, angles):
        result = {}
        for key1, value1 in angles.iteritems():
            result[key1] = {}
            for key2, value2 in value1.iteritems():
                self._update(crystals, wls, tt, *value2)
                _temp = {}
                _temp['angle'] = value2
                _temp['polarization'] = key2
                _temp['n'] = tuple(cry.nhl[cry.oe[pol]] for cry, pol in zip(crystals, key2))
                _temp['gi'] = tuple(cry.gihl[cry.oe[pol]] for cry, pol in zip(crystals, key2))
                _temp['gvd'] = tuple(cry.gvdhl[cry.oe[pol]] for cry, pol in zip(crystals, key2))
                _temp['walkoff'] = tuple(cry.rhl[cry.oe[pol]] for cry, pol in zip(crystals, key2))
                _temp['angtol'] = self.ftol(crystals, key2, wls, tt, value2[0], value2[1], 'theta'), self.ftol(
                        crystals, key2, wls, tt, value2[0], value2[1], 'phi')
                _temp['tttol'] = self.ftol(crystals, key2, wls, tt, value2[0], value2[1], 'tt')
                _temp['bw'] = self.ftol(crystals, key2, wls, tt, value2[0], value2[1], 'wls')
                    
                _temp['deff'] = -self.fdeff(crystals, key2, key1, *value2) #walkoff angle is not ignored.
                
                result[key1][key2] = _temp
            
        return result

    def show(self, key='all'):
        if key == 'all':
            data = self.data
        else:
            data = {key:self.data[key]}
        for key1, value1 in data.iteritems():
            print(key1)
            for key2, value2 in value1.iteritems():
#                if abs(value2['deff']) > 1e-2:
                print ('                       {0:5.1f} ({1}) + {2:5.1f} ({3}) = {4:5.1f} ({5})'.format(
                        *[item for sublist in zip(self._wls, value2['polarization']) for item in sublist]))
                print ('Walkoff:               {0:>5.2f}         {1:>5.2f}       {2:>5.2f} mrad'.format(*(value*1e3 for value in value2['walkoff'])))
                print ('Phase indices:         {0:5.3f}         {1:5.3f}       {2:5.3f}'.format(*value2['n']))
                print ('Group indices:         {0:5.3f}         {1:5.3f}       {2:5.3f}'.format(*value2['gi']))
                print ('GVD:                   {0:5.1f}         {1:5.1f}       {2:5.1f} fs^2/mm'.format(*value2['gvd']))
                print ('theta, phi:            {0:5.1f}         {1:5.1f} deg'.format(*value2['angle']))
                print ('deff:                  {0:5.3f} pm/V'.format(value2['deff']))
                print ('ang. tol. theta, phi:  {0:5.2f}        {1:5.2f} mradxcm'.format(*value2['angtol']))
                print ('temperature tol.:      {0:5.2f} Kxcm'.format(value2['tttol']))
                print ('accpt bw 1&3 and 2&3:  {0:5.2f}        {1:5.2f} cm-1xcm'.format(*value2['bw']))
                print()

    # Be careful that during the calculation, the crystals internal status (tt, wls, theta, phi) can be altered.
    # Hence it is unsafe to use these result without reassignment.
    def ftol(self, crystals, polarizations, wls, tt, theta, phi, arg):
        if arg == 'wls':
            self._update(crystals, wls, tt, theta, phi)
            def fabw(a, b):
                return np.abs(1/(a-b))*2.784/2./np.pi
                
            gis = [cry.gihl[cry.oe[pol]] for cry, pol in zip(crystals, polarizations)]
            
            return fabw(gis[2],gis[0]), fabw(gis[2],gis[1])
                    
        x = 1e-4 # if the value is too small, numerical error arises. delta_h is about 1e-3 of one unit.
        if arg == 'tt':
#            f1 = lambda x: np.abs(self.fdeltak(crystals, polarizations, wls, tt+x, theta, phi)) - 2.784e-7
#            return np.deg2rad(optimize.fsolve(f1, 0))*1e3
            dkdtt = np.abs((self.fdeltak(crystals, polarizations, wls, tt+x, theta, phi)-self.fdeltak(
                    crystals, polarizations, wls, tt, theta, phi))/x)
            return (2.784e-7/dkdtt)
        
        if arg == 'theta':
#            f2 = lambda x: np.abs(self.fdeltak(crystals, polarizations, wls, tt, theta+x, phi)) - 2.784e-7
#            return np.deg2rad(optimize.fsolve(f2, 0))*1e3            
            dkdtheta = np.abs((self.fdeltak(crystals, polarizations, wls, tt, theta+x, phi)-self.fdeltak(
                    crystals, polarizations, wls, tt, theta, phi))/x)
            dk2dtheta = np.abs((self.fdeltak(crystals, polarizations, wls, tt, theta+x, phi)+self.fdeltak(
                    crystals, polarizations, wls, tt, theta-x, phi)-2*self.fdeltak(crystals, polarizations, wls, tt, theta, phi))/x**2)
            
            #In this method, the first two items in the taylor expansion are used. Although for most cases wheren the angle is 
            #far away from NCPM this is redundent, it solves the issue when the angle is close to NCPM (+/- 0.5 deg).
            _temp = np.roots([0.5*dk2dtheta, dkdtheta, -2.784e-7])
            return 2*np.deg2rad(_temp[_temp>0][0])*1e3
        
#           #an alternative way but not accurate enough. It assumes only at NCPM the second derivative is used. 
#           #However, it was found when the angle is close to NCPM (+/- 0.5 deg), 
#           #the error of the first derivative is becoming large enough to distore the result.         
#            if theta in (0, 90):
#                return 2*np.deg2rad(np.sqrt(2*2.784e-7/dk2dtheta))*1e3
#            else:
#                return 2*np.deg2rad(2.784e-7/dkdtheta)*1e3
        
        if arg == 'phi':
#            f3 = lambda x: np.abs(self.fdeltak(crystals, polarizations, wls, tt, theta, phi+x)) - 2.784e-7
#            return np.deg2rad(optimize.fsolve(f3, 0))*1e3
            dkdphi = np.abs((self.fdeltak(crystals, polarizations, wls, tt, theta, phi+x)-self.fdeltak(
                    crystals, polarizations, wls, tt, theta, phi))/x)
            dk2dphi = np.abs((self.fdeltak(crystals, polarizations, wls, tt, theta, phi+x)+self.fdeltak(
                    crystals, polarizations, wls, tt, theta, phi-x)-2*self.fdeltak(crystals, polarizations, wls, tt, theta, phi))/x**2)
            
            #In this method, the first two items in the taylor expansion are used. Although for most cases wheren the angle is 
            #far away from NCPM this is redundent, it solves the issue when the angle is close to NCPM (+/- 0.5 deg).
            _temp = np.roots([0.5*dk2dphi, dkdphi, -2.784e-7])
            return 2*np.deg2rad(_temp[_temp>0][0])*1e3
        
#           #an alternative way but not accurate enough. It assumes only at NCPM the second derivative is used. 
#           #However, it was found when the angle is close to NCPM (+/- 0.5 deg), 
#           #the error of the first derivative is becoming large enough to distore the result.          
#            if phi in (0, 90):
#                return 2*np.deg2rad(np.sqrt(2*2.784e-7/dk2dphi))*1e3
#            else:
#                return 2*np.deg2rad(2.784e-7/dkdphi)*1e3
        
    
    def fdeltak(self, crystals, polarizations, wls, tt, theta, phi): 
        """This function is designed to calculation the variation of delta_k as a function of tt, theta, or phi"""
    
        #tt, theta, and phi are considered univariale for the crystals, unlike wls. This is based on collinear geometry.     
        self._update(crystals, wls, tt, theta, phi)
            
        def _fdeltak(crystals, polarizations, wls):
            def fk(n, wl):
                return np.pi*2*n/wl

            ns = [cry.nhl[cry.oe[pol]] for cry, pol in zip(crystals, polarizations)]      
            _temp = [fk(n, wl) for n, wl in zip(ns, wls)] # wls are in unit of nm.
            return _temp[0]+_temp[1]-_temp[2] # unit of k is nm-1
        
        return _fdeltak(crystals, polarizations, wls)
    
    def _update(self, crystals, wls, tt, theta, phi):
        for cry, wl in zip(crystals, wls):
            cry.tt = tt
            cry.theta = theta
            cry.phi = phi
            cry.wl = wl

    def fdeff(self, crystals, pols, plane, theta, phi):
        """a function to calculate deff based on dtensor, theta, phi, and delta"""
        
        for cry in crystals:
            cry.theta = theta
            cry.phi = phi
        
        def projection(theta, phi, delta):
            
            theta = np.deg2rad(theta)
            phi = np.deg2rad(phi)
            delta = np.deg2rad(delta)
            
            return [np.sin(phi)*np.sin(delta)-np.cos(theta)*np.cos(phi)*np.cos(delta),
                    -np.cos(phi)*np.sin(delta)-np.cos(theta)*np.sin(phi)*np.cos(delta),
                    np.sin(theta)*np.cos(delta)]        
        
        if plane == 'XY':
            deltas = [90 if pol=='e' else 0 for pol in pols]
            sol = [projection(theta, phi+np.rad2deg(cry.rhl[cry.oe[pol]]), delta) for cry, pol, delta in zip(crystals, pols, deltas)]
        
        if plane == 'XZ':
            deltas = [90 if pol=='o' else 0 for pol in pols]
            sol = [projection(theta-np.rad2deg(cry.rhl[cry.oe[pol]]), phi, delta) for cry, pol, delta in zip(crystals, pols, deltas)]
            
        if plane == 'YZ':
            deltas = [90 if pol=='o' else 0 for pol in pols]
            sol = [projection(theta-np.rad2deg(cry.rhl[cry.oe[pol]]), phi, delta) for cry, pol, delta in zip(crystals, pols, deltas)]           
        
        e3 = np.array(sol[2])
        e12 = np.array([sol[0][0]*sol[1][0], sol[0][1]*sol[1][1], sol[0][2]*sol[1][2],
                        sol[0][1]*sol[1][2]+sol[0][2]*sol[1][1],
                        sol[0][0]*sol[1][2]+sol[0][2]*sol[1][0],
                        sol[0][0]*sol[1][1]+sol[0][1]*sol[1][0]])
        temp = np.einsum('ij,j->i', self._dtensor, e12)
        
        return np.einsum('i,i->', e3, temp)

   
if __name__ == '__main__':    
    from data import lbo3
    
    p = phasematch(lbo3, [1035, 1035, 0], 40)
    p.show('all')
    print(p.angles)
    
    

    