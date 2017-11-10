# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:13:09 2017

@author: dtp17
"""

import numpy as np
import scipy.sparse as spa
import matplotlib.pyplot as plt
import scipy.optimize as opt

def RK4(RHS,y,dt,args=()):
    '''Runge-Kutta integrator'''
    k1 = RHS(y            ,*args)
    k2 = RHS(y + dt/2 * k1,*args)
    k3 = RHS(y + dt/2 * k2,*args)
    k4 = RHS(y + dt   * k3,*args)
    
    ynew = y + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ynew

def getDiffMatrix(M,dx):
#    Dmat = spa.diags([np.ones(M-1),-np.ones(M-1)],[1,-1])
    Dmat = np.diag(np.ones(M-1),1) + np.diag(-np.ones(M-1),-1)
    Dmat[0,0] = -1
    Dmat[-1,-1] = 1
    Dmat[0,:] *= 2
    Dmat[-1,:] *= 2
    return 1./(2*dx) * Dmat

def RHS(hvec,avec):
    return D * Mmat.dot((hvec+bvec)**3 * Mmat.dot(hvec)) + E * avec

def RHS2(hvec,avec):
    qvec = - D * (hvec+bvec)**3 * Mmat.dot(hvec)
    qvec = enforceBCq(qvec)
    rhsvec = - Mmat.dot(qvec) + E * avec
    return rhsvec

def solFuncBT(hnew,hold,avec,dt):
    return hnew - hold - dt * RHS(hnew,avec)

def solFuncCN(hm,hold,avec,dt):
    return 2*hm - hold - dt * RHS(hm,avec)

def stepFT(hvec,avec,dt):
    '''Forward Euler step'''
    
    hvecnew = hvec + dt * RHS(hvec,avec)
    return hvecnew

def stepRK4(hvec,avec,dt):
    '''RK4 step'''
    hvecnew = RK4(RHS,hvec,dt,[avec])

def stepBT(hvec,avec,dt):
    '''Backward Euler step'''
    hvecnew = opt.fsolve(solFuncBT,hvec,args=(hvec,avec,dt))
    return hvecnew

def stepCN(hvec,avec,dt):
    '''Crank-Nicholson step'''
    hmnew = opt.fsolve(solFuncCN,hvec,args=(hvec,avec,dt))
    hvecnew = 2 * hmnew - hvec
    return hvecnew

def enforceBCq(qvec):
    
def enforceBCh(hvec):


def step(hvec,avec,dt):
    ''' '''
    qvec = -D * (hvec+bvec)**3 * difffunc(hvec)
    

# Define various constants
    
N = 100
dt = 1
M = 200
dx = 0.1
cfl = 
D = 1
E = 1
Mmat = getDiffMatrix(M,dx)

# Bed elevation
bvec = np.zeros(M)
# Mass balance
avec = np.zeros(M)

# ice elevation in time and space

# Specify boundary conditions
# Choose two boundary conditions to fix
FixQLeft  = True
FixQRight = False
FixHLeft  = False
FixHRight = True

# specify BC values either as constants or functions taking time as the argument
QLeft  = 0
QRight = 0
HLeft  = H0
HRight = H0




