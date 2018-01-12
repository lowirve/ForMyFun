# -*- coding: utf-8 -*-
"""
Phase matching class

@author: XuBo
"""
import numpy as np
from scipy import optimize
from crystals import *
from data import *

def xyztensor(tensor):
    """a function to transfer d tensor from numpy.array to a dictionary based on xyz."""   

    return {'xxx':tensor[0,0],'yxx':tensor[1,0], 'zxx':tensor[2,0], 
            'xyy':tensor[0,1],'yyy':tensor[1,1], 'zyy':tensor[2,1], 
            'xzz':tensor[0,2],'yzz':tensor[1,2], 'zzz':tensor[2,2], 
            'xyz':tensor[0,3],'yyz':tensor[1,3], 'zyz':tensor[2,3], 
            'xxz':tensor[0,4],'yxz':tensor[1,4], 'zxz':tensor[2,4], 
            'xxy':tensor[0,5],'yxy':tensor[1,5], 'zxy':tensor[2,5], }


def fdeff(d):
    """a function to return the deff function in a dictionary of which the keys are the polarization combinations"""
    
    def ooe(theta, phi):
        cT = np.cos(np.deg2rad(theta))
        sT = np.sin(np.deg2rad(theta))
        cP = np.cos(np.deg2rad(phi))
        sP = np.sin(np.deg2rad(phi))
        return -d['xxx']*cT*cP*sT**2-d['xyy']*cT*cP**3+d['xxy']*2*cT*cP**2*sP-d['yxx']*cT*sP**3-d['yyy']*cT*cP**2*sP+\
                d['yyz']*2*cT*cP*sP**2+d['zxx']*sT*sP**2+d['zyy']*sT*cP**2-d['zxy']*2*sT*cP*sP 
                #?d['yyz'] is uncertain. In the book of SNLO, it IS d['yyx'], which actually doesn't exist.
    
    def oee(theta, phi):
        cT = np.cos(np.deg2rad(theta))
        sT = np.sin(np.deg2rad(theta))
        cP = np.cos(np.deg2rad(phi))
        sP = np.sin(np.deg2rad(phi))
        return +d['xxx']*cT**2*cP**2*sP-d['xyy']*cT**2*cP**2*sP+d['xyz']*cT*sT*cP**2-d['xxz']*cT*sT*cP*sP+\
                d['xxy']*(cT**2*cP*sP**2-cT**2*cP**3)+d['yxx']*cT**2*cP*sP**2-d['yyy']*cT**2*cP*sP**2+\
                d['yyz']*cT*sT*cP*sP-d['yxz']*cT*sT*sP**2+d['yxy']*(cT**2*sP**3-cT**2*cP**2*sP)-\
                d['zxx']*cT*sT*cP*sP+d['zyy']*cT*sT*cP*sP-d['zyz']*sT**2*cP+d['zxz']*sT**2*sP+d['zxy']*(cT*sT*cP**2-cT*sT*sP**2)
    
    def oeo(theta, phi):
        cT = np.cos(np.deg2rad(theta))
        sT = np.sin(np.deg2rad(theta))
        cP = np.cos(np.deg2rad(phi))
        sP = np.sin(np.deg2rad(phi))
        return -d['xxx']*cT*cP*sP**2+d['xyy']*cT*cP*sP**2-d['xyz']*sT*cP*sP+d['xxz']*sT*sP**2+\
                d['xxy']*(cT*cP**2*sP-cT*sP**3)+d['yxx']*cT*cP**2*sP-d['yyy']*cT*cP**2*sP+\
                d['yyz']*sT*cP**2-d['yxz']*sT*cP*sP**2+d['yxy']*(cT*cP*sP**2-cT*cP**3)
                
    def eeo(theta, phi):
        cT = np.cos(np.deg2rad(theta))
        sT = np.sin(np.deg2rad(theta))
        cP = np.cos(np.deg2rad(phi))
        sP = np.sin(np.deg2rad(phi))
        return +d['xxx']*cT**2*cP**2*sP+d['xyy']*cT**2*sP**3+d['xzz']*sT**2*sP-d['xyz']*2*cT*sT*sP**2-\
                d['xxz']*2*cT*sT*cP*sP+d['xxy']*2*cT**2*cP*sP**2-d['yxx']*cT**2*cP**3-d['yyy']*cT**2*cP*sP**2-\
                d['yzz']*sT**2*cP+d['yyz']*2*cT*sT*cP*sP+d['yxz']*2*cT*sT*cP**2-d['yxy']*2*cT**2*cP**2*sP
                
    return {'ooe':ooe, 'oee':oee, 'oeo':oeo, 'eoe':oee, 'eeo':eeo, 'eoo':oeo}
    

class phasematch(object):
    
    _wls = [1064, 1064, 532] # wavelength pair, unite is nm.
    _angles = None # a dictionary to store all phase match angle, unit is centigrade
    _material = None # the crystal we are seeking PM
    _tensor = None # d tensor to calculate deff
    _fdeff = None # deff functions
    _data = None # crystals properties at the given PM angles
    _tt = 25. # temperature, based on centidegree
    _crystals = None # a list of three crystals correponding to three wavelengths
    
    @property
    def data(self):
        self._angles = self.angles
        self._data = self._fdata(self._crystals, self._wls, self._tt, self._angles, self._fdeff)
        return self._data
    
    @property
    def angles(self):
        self._angles = self.fPMangle(self._crystals)
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
            self._tensor = xyztensor(value.dtensor)
            self._fdeff = fdeff(self._tensor)
        else:
            raise ValueError('Input must be a defined crystal')  
    
    def __init__(self, Crys, wls=None, tt=None):
        self.material = Crys  
        if wls:
            self.wls = wls
        
        if tt:
            self.tt = tt
            
    def _finit(self, material, wls, tt):
        wl1, wl2, wl3 = wls
        a = crystal(material)
        a.wl = wl1
        a.tt = tt
        b = crystal(material)
        b.wl = wl2
        b.tt = tt
        c = crystal(material)
        c.wl = wl3
        c.tt = tt
        return [a, b, c]

    def fwls(self, lvalue):
        """calculate and organize the wavelengths group"""
        try:
            if lvalue.count(0) != 1:
                raise ValueError
        except ValueError:
            print "Input must have one zero. Please try again"
            return None
    
        index = lvalue.index(0)
        
        if index == 0:
            lvalue[0] = 1./(1./lvalue[2]-1./lvalue[1])
        elif index == 1:
            lvalue[1] = 1./(1./lvalue[2]-1./lvalue[0])
        else:
            lvalue[2] = 1./(1./lvalue[1]+1./lvalue[0])
            
        return sorted(lvalue, reverse=True)


    def fPMangle(self, crystals): # 
        '''Phase matching calculation for principle planes'''
        
        a, b, c = crystals # from a to c, wl is decreasing
        
        la, lb, lc = a.wl, b.wl, c.wl
        
        def _fU(dXY):
            return ((dXY['A']+dXY['B'])/dXY['C'])**2
        
        def _fW(dXY):
            return ((dXY['A']+dXY['B'])/dXY['F'])**2
        
        def _fR(dXY):
            return ((dXY['A']+dXY['B'])/(dXY['B']+dXY['D']))**2
        
        def _fQ(dXY):
            return ((dXY['A']+dXY['B'])/(dXY['A']+dXY['E']))**2
           
        def _fS(dYZ):
            return ((dYZ['A']+dYZ['B'])/(dYZ['D']+dYZ['E']))**2
        
        def _fV(dYZ):
            return ((dYZ['B'])/(dYZ['C']-dYZ['A']))**2
        
        def _fY(dYZ):
            return ((dYZ['B'])/(dYZ['E']))**2 
        
        def _fT(dYZ):
            return ((dYZ['A'])/(dYZ['C']-dYZ['B']))**2
        
        def _fZ(dYZ):
            return ((dYZ['A'])/(dYZ['D']))**2    
            
        def _feeo(dYZ):
            U = _fU(dYZ['eeo'])
            S = _fS(dYZ['eeo'])
            
            _temp = (1-U)/(U-S)
            
            if _temp < 0:
                return None
            else:
                return np.rad2deg(np.arctan(np.sqrt(_temp))) 
            
        def _foeo(dYZ):
            V = _fV(dYZ['oeo'])
            Y = _fY(dYZ['oeo'])
            
            _temp = (1-V)/(V-Y)
            
            if _temp < 0:
                return None
            else:
                return np.rad2deg(np.arctan(np.sqrt(_temp)))           
    
        def _feoo(dYZ):
            T = _fT(dYZ['eoo'])
            Z = _fZ(dYZ['eoo'])
            
            _temp = (1-T)/(T-Z)
            
            if _temp < 0:
                return None
            else:
                return np.rad2deg(np.arctan(np.sqrt(_temp))) 
            
        def _fooe(dXY):
            U = _fU(dXY['ooe'])
            W = _fW(dXY['ooe'])
            
            _temp = (1-U)/(W-1)
            
            if _temp < 0:
                return None
            else:
                return np.rad2deg(np.arctan(np.sqrt(_temp))) 
            
        def _feoe(dXY):
            U = _fU(dXY['eoe'])
            W = _fW(dXY['eoe'])
            R = _fR(dXY['eoe'])
            
            _temp = (1-U)/(W-R)
            
            if _temp < 0:
                return None
            else:
                return np.rad2deg(np.arctan(np.sqrt(_temp)))           
    
        def _foee(dXY):
            U = _fU(dXY['oee'])
            W = _fW(dXY['oee'])
            Q = _fQ(dXY['oee'])
            
            _temp = (1-U)/(W-Q)
            
            if _temp < 0:
                return None
            else:
                return np.rad2deg(np.arctan(np.sqrt(_temp)))
        
        def _fXY():        
            XY = {'ooe':{'A':a.nz/la, 'B':b.nz/lb, 'C': c.ny/lc, 'F': c.nx/lc},
                  'eoe':{'A':a.ny/la, 'B':b.nz/lb, 'C': c.ny/lc, 'D': a.nx/la, 'F': c.nx/lc},
                  'oee':{'A':a.nz/la, 'B':b.ny/lb, 'C': c.ny/lc, 'E': b.nx/lb, 'F': c.nx/lc}}        
     
            result = {'ooe':_fooe(XY),
                    'eoe':_feoe(XY),
                    'oee':_foee(XY)} #theta = 90, phi is variable.
            
            return {key:(90, value) for key, value in result.iteritems() if value!= None}
    
        def _fYZ():        
            YZ = {'eeo':{'A':a.ny/la, 'B':b.ny/lb, 'C': c.nx/lc, 'D': a.nz/la, 'E': b.nz/lb},
                  'oeo':{'A':a.nx/la, 'B':b.ny/lb, 'C': c.nx/lc, 'E': b.nz/lb},
                  'eoo':{'A':a.ny/la, 'B':b.nx/lb, 'C': c.nx/lc, 'D': a.nz/la}}
      
            result = {'eeo':_feeo(YZ),
                    'oeo':_foeo(YZ),
                    'eoo':_feoo(YZ)} #phi = 90, theta is variable.    
                      
            return {key:(value, 90) for key, value in result.iteritems() if value!= None}
    
        def _fXZ():        
            XZ = {'ooe':{'A':a.ny/la, 'B':b.ny/lb, 'C': c.nx/lc, 'F': c.nz/lc},
                  'eoe':{'A':a.nx/la, 'B':b.ny/lb, 'C': c.nx/lc, 'D': a.nz/la, 'F': c.nz/lc},
                  'oee':{'A':a.ny/la, 'B':b.nx/lb, 'C': c.nx/lc, 'E': b.nz/lb, 'F': c.nz/lc},
                  'eeo':{'A':a.nx/la, 'B':b.nx/lb, 'C': c.ny/lc, 'D': a.nz/la, 'E': b.nz/lb},
                  'oeo':{'A':a.ny/la, 'B':b.nx/lb, 'C': c.ny/lc, 'E': b.nz/lb},
                  'eoo':{'A':a.nx/la, 'B':b.ny/lb, 'C': c.ny/lc, 'D': a.nz/la}}
            
            result = {'ooe':_fooe(XZ),
                    'eoe':_feoe(XZ),
                    'oee':_foee(XZ),
                    'eeo':_feeo(XZ),
                    'oeo':_foeo(XZ),
                    'eoo':_feoo(XZ)} #phi = 0, theta is variable. 
                      
            return {key:(value, 0) for key, value in result.iteritems() if value!= None}
            
        _XY = _fXY()
        _YZ = _fYZ()
        _XZ = _fXZ()
        
        return {'XY':_XY,
                'YZ':_YZ,
                'XZ':_XZ}
        
    def _fdata(self, crystals, wls, tt, angles, deff):
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
                if key1 == 'XY':
                    pol = ''.join(['o' if st == 'e' else 'e' for st in key2])
                else:
                    pol = key2
                    
                _temp['deff'] = -deff[pol](*value2) #walkoff angle is ignored, different from SNLO.
                
                result[key1][key2] = _temp
            
        return result

    def fprint(self, key='all'):
        if key == 'all':
            data = self.data
        else:
            data = {key:self.data[key]}
        for key1, value1 in data.iteritems():
            print key1
            for key2, value2 in value1.iteritems():
                print '{0:.1f} ({1}) + {2:.1f} ({3}) = {4:.1f} ({5})'.format(
                        *[item for sublist in zip(self._wls, value2['polarization']) for item in sublist])
                print 'Walkoff:       {0:.2f} {1:.2f} {2:.2f} mrad'.format(*value2['walkoff'])
                print 'Phase indices: {0:.3f} {1:.3f} {2:.3f}'.format(*value2['n'])
                print 'Phase indices: {0:.3f} {1:.3f} {2:.3f}'.format(*value2['gi'])
                print 'GVD:           {0:.1f} {1:.1f} {2:.1f} fs^2/mm'.format(*value2['gvd'])
                print 'theta, phi:    {0:.1f} {1:.1f} deg'.format(*value2['angle'])
                print 'deff:          {0:.3f} pm/V'.format(value2['deff'])
                print 'ang. tol. theta, phi:  {0:.2f} {1:.2f} mradxcm'.format(*value2['angtol'])
                print 'temperature tol.:      {0:.2f} Kxcm'.format(value2['tttol'])
                print 'accpt bw 1&3 and 2&3:  {0:.2f} {1:.2f} cm-1xcm'.format(*value2['bw'])
                print

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

   
if __name__ == '__main__':    
    p = phasematch(lbo3, [1064, 1064, 0], 25)
    p.fprint('XY')
    
    

    