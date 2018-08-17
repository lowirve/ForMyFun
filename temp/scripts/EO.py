# -*- coding: utf-8 -*-
"""
Created on Mon Aug 06 17:56:31 2018

@author: XuBo
"""
from __future__ import division, print_function

import numpy as np
from scipy.optimize import newton
from scipy.linalg import eig

def Jacobi(M, n=10, tolerance=1e-10, flag=False):
    
#    def amax(matrix):
#        i = (0, 1)
#        j = (0, 2)
#        k = (1, 2)
#        
#        temp = matrix[i]
#        ind = i
#        
#        if matrix[j] > temp:
#            temp = matrix[j]
#            ind = j
#        
#        if matrix[k] > temp:
#            temp = matrix[k]
#            ind = k
#            
#        return ind

    #Rotation matrix. See the reference for the formula.    
    rotation = {(0,1):lambda x: np.array([  [np.cos(x), np.sin(x), 0],
                                            [-np.sin(x), np.cos(x), 0],
                                            [0, 0, 1]]),
                (0,2):lambda x: np.array([  [np.cos(x), 0, np.sin(x)],
                                            [0, 1, 0],
                                            [-np.sin(x), 0, np.cos(x)]]),
                (1,2):lambda x: np.array([  [1, 0, 0],
                                            [0, np.cos(x), np.sin(x)],
                                            [0, -np.sin(x), np.cos(x)]])}
    
    sol = np.identity(3)
    
    #iternation no more than 10 times
    while(n):
        n -= 1
        
        temp = np.abs(np.triu(M,1))
        
        if np.amax(temp) < tolerance:
            break
        #find the largest element on upper triangle only as the matrix is symmetric.
        ind = np.unravel_index(np.argmax(temp, axis=None), temp.shape) #this line may contain float number precision issue
        
#        #less fancy, but suppose to be more reliable method, given the array has only three elements that matter.       
#        ind = amax(temp)
        
        #The angle should be restricted in the range [-pi/4, pi/4], hence arctan2 by nature is not suitable.
        #However, arctan doesn't have protection over 1/0. Extra effort is needed, which is a compromise, 
        #although the error is not consistent due to the float number precision.                  
        if M[ind[0], ind[0]] == M[ind[1], ind[1]]:
            theta = np.pi/4
        else:
            theta = np.arctan(2*M[ind]/(M[ind[0], ind[0]]-M[ind[1], ind[1]]))/2
        
        if flag:
#            print(2*M[ind], (M[ind[0], ind[0]]-M[ind[1], ind[1]]))
            print('rotation angle {0:.4f} deg'.format(np.rad2deg(theta)))
            print('rotation plane {}\n'.format(ind))
        
        T = rotation[ind](theta)
        
        M = np.einsum('ij,jk->ik', T, np.einsum('ij,jk->ik', M, T.T))
        
        sol = np.einsum('ij,jk->ik', sol, T.T)
        
    return sol, M


def halfwave(R, n, l, wl):
    # l in mm
    # wl in um
    
    def E(x):
#######################################################################
        return np.array([x, 0, 0]) #V/m, subject to change when the electric field is from different axis
#######################################################################          
    
    def f(x):
      
        n1, n2, n3 = n
        
        P = np.einsum('ij,j->i', R, E(x))
        
        N = np.array([[1/n1**2+P[0], P[5], P[4]],
                      [P[5], 1/n2**2+P[1], P[3]],
                      [P[4], P[3], 1/n3**2+P[2]]])
    
        sol = np.sqrt(np.abs(1/np.diag(Jacobi(N)[1])))
        
        deltaRI = np.abs(sol[0]-sol[1])
        
        return np.pi/2 - np.pi*2*deltaRI*l*1e3/wl
    
    return E(newton(f, 1e6))
       

if __name__ == '__main__':
    
    #LN  @m/V
    r11 = 0
    r12 = -3.4
    r13 = 8.6
    r21 = 0
    r22 = 3.4
    r23 = 8.6
    r31 = 0
    r32 = 0
    r33 = 30.8    
    r41 = 0
    r42 = 28
    r43 = 0    
    r51 = 28
    r52 = 0
    r53 = 0
    r61 = -3.4
    r62 = 0
    r63 = 0
    
#    #KDP  @m/V
#    r11 = 0
#    r12 = 0
#    r13 = 0
#    r21 = 0
#    r22 = 0
#    r23 = 0
#    r31 = 0
#    r32 = 0
#    r33 = 0    
#    r41 = 8.77
#    r42 = 0
#    r43 = 0    
#    r51 = 0
#    r52 = 8.77
#    r53 = 0
#    r61 = 0
#    r62 = 0
#    r63 = -10.5
    
    R = 1e-12*np.array([[r11, r12, r13],
                  [r21, r22, r23],
                  [r31, r32, r33],
                  [r41, r42, r43],
                  [r51, r52, r53],
                  [r61, r62, r63]])

    
    n1 = 2.232
    n2 = 2.232
    n3 = 2.156
    
    n = n1, n2, n3
    
    E = halfwave(R, n, 25, 1.064)

    print('Half-wave Electric Field: {} V/m\n'.format(E))     
    
    P = np.einsum('ij,j->i', R, E)
    
    N = np.array([[1/n1**2+P[0], P[5], P[4]],
                  [P[5], 1/n2**2+P[1], P[3]],
                  [P[4], P[3], 1/n3**2+P[2]]])
    
    print(N)
    print()
    
    result = Jacobi(N, flag=True)
    
    print(result[0])
    print()
    print(result[1])
    print()
    print(np.sqrt(np.abs(1/np.diag(result[1]))))     
        
   
    
##Eigenvalue/vector direct calculation method shows highly arbitrary eigenvalue/vector order, 
##making the principle axes reorientation unstable and unidentifiable, hence, discarded.    
#    result = eig(N, left=False, right=True)
#      
#    print(result[0])
#    print(np.sqrt(np.abs(1/result[0])))
#    print(result[1])
#    
#    
#    #a = np.abs(result[1])
#    #
#    #ind = np.unravel_index(np.argmin(a, axis=None), a.shape)
#    #
#    #print(ind)
#    
#    b = np.array([[1, 0, 0],
#                  [0, 1, 0],
#                  [0, 0, -1]])
#    
#    a = result[1][:,(1,2,0)]
#    
#    a = np.einsum('ij,jk->ik', a, b)
#
#    #below approach assumes that the rotation always follows XYZ in order. And then decompose the matrix accordingly.    
#    angleX = np.rad2deg(np.arctan2(a[2,1], a[2,2]))
#    angleY = np.rad2deg(np.arctan2(-a[2,0], np.sqrt(a[2,1]**2+a[2,2]**2)))
#    angleZ = np.rad2deg(np.arctan2(a[1,0], a[0,0]))
#    
#    print('\nFirst rotate about X at {0:.4f} deg'.format(angleX))
#    print('Second rotate about Y at {0:.4f} deg'.format(angleY))
#    print('Last rotate about Z at {0:.4f} deg'.format(angleZ))
    
    
