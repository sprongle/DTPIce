# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:13:09 2017

@author: dtp17
"""

import numpy as np
import scipy.sparse as spa
from scipy.integrate import odeint
from numba import jit
import matplotlib.pyplot as plt
import scipy.optimize as opt

def hsteady(x,H0,q0,D):
#    return (H0**4 - 4 * q0 * (1 - x))**0.25
    return (H0**4 - 4 * q0/D * (1 - x))**0.25

def hgradsteady(x,H0,q0,D):
    #return q0 * (H0**4 - 4 * q0 * (1 - x))**(-0.75)
    return q0/D * (H0**4 - 4 * q0/D * (1 - x))**(-0.75)


#def RK4(RHS,y,dt,args=()):
#    '''Runge-Kutta integrator'''
#    k1 = RHS(y            ,*args)
#    k2 = RHS(y + dt/2 * k1,*args)
#    k3 = RHS(y + dt/2 * k2,*args)
#    k4 = RHS(y + dt   * k3,*args)
#    
#    ynew = y + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
#    return ynew

def getDiffMatrix(M,dx):
#    Dmat = spa.diags([np.ones(M-1),-np.ones(M-1)],[1,-1]).tocsc()
    Dmat = np.diag(np.ones(M-1),1) + np.diag(-np.ones(M-1),-1)
    Dmat[0,0] = -1
    Dmat[-1,-1] = 1
    Dmat[0,:] *= 2
    Dmat[-1,:] *= 2
    return 1./(2*dx) * Dmat


#def RHSRK42(hvec,avec,qvec,qindices,hindices):
#    '''RHS for RK4 integrator'''
#    qvecarr = qvec.copy()
#    qvecarr[qindices] = (- DD * (hvec+bvec)**3 * Mmat.dot(hvec))[qindices]
##    return -Mmat.dot(qvecarr) + EE * avec
#    hvecnew = np.zeros(hvec.shape)
#    hvecnew[hindices] = (-Mmat.dot(qvecarr) + EE * avec)[hindices]
#    return hvecnew

@jit(nopython=True)
def RHSodeint(hvec,t,avec,qvec,qindices,hindices):
    '''RHS for odeint integrator'''
    qvecarr = qvec.copy()
    qvecarr[qindices] = (- DD * (hvec+bvec)**3 * np.dot(Mmat,hvec))[qindices]
#    hvecnew = np.copy(hvec)
    hvecnew = np.zeros(hvec.shape)
    hvecnew[hindices] = (-np.dot(Mmat,qvecarr) + EE * avec)[hindices]
    return hvecnew

#def integrateRK4(hvec0,time,avec,qvec,qindices,hindices):
#    '''RK4 integration'''
#    dt = time[1] - time[0]
#    harr = np.zeros([len(time),len(hvec0)])
#    harr[:,:] = hvec0
#    for ti in range(len(time)-1):
#        hnew = RK4(RHSRK42,harr[ti,:],dt,args=[avec,qvec,qindices,hindices])
#        harr[ti+1,hindices] = hnew[hindices]
#    return harr    

@jit(nopython=True)
def solFuncBT2(hnew,hold,avec,qvec,qindices,hindices,dt):
    '''hnew here only includes those indices in h to be solved for, i.e. excluding fixed BC'''
    hnew2 = hold.copy()
    hnew2[hindices] = hnew
#    qnew2 = qvec.copy()
#    qnew2[qindices] = 
    qvec[qindices] = (-DD * (hnew2+bvec)**3 * np.dot(Mmat,hnew2))[qindices]
    rhs = - np.dot(Mmat,qvec) + EE * avec
    return (hnew2 - hold - dt * rhs)[hindices]

def integrateBT(hvec0,time,avec,qvec,qindices,hindices):
    '''Backward Euler integration'''
    dt = time[1] - time[0]
    harr = np.zeros([len(time),len(hvec0)])
    qvecarr = qvec.copy()
    harr[:,:] = hvec0
    for ti in range(len(time)-1):
        harr[ti+1,hindices] = opt.fsolve(solFuncBT2,harr[ti,hindices],args=(harr[ti,:],avec,qvecarr,qindices,hindices,dt))
    return harr

def integrateODE(hvec0,time,avec,qvec0,qindices,hindices,full=False):
    '''Using scipy.integrate.odeint integrator'''
    return odeint(RHSodeint,hvec0,time,args=(avec,qvec0,qindices,hindices),full_output=full)

#class IceModel(object):
#    
#    def __init__(DD,EE):
#        
#    def setup
#        
#    def integrate(time,)


# Get dimensionless numbers
g = 9.81 # m s^-2
T = 60**2 * 24 * 365.25 * 100 # s
H = 1000 # m
L = 10e3 # m
nu = 2.6e13 # Pa s
A = 1 / T# m s^-1

DD = g * T * H**3 / (3 * nu * L**2)
DD = 0.1
EE = T * A / H

# Define Geometry
N = 200
M = 500 #200
#cfl = 

# space and time arrays
xarr = np.linspace(0,1,M+1)
#tarr = np.linspace(0,1,N+1)
tarr = np.linspace(0,30,N*30+1)
dx = xarr[1] - xarr[0]
dt = tarr[1] - tarr[0]

# Differentiation matrix
Mmat = getDiffMatrix(M+1,dx)

# Bed elevation
bvec = np.zeros(M+1)
# Mass balance
avec = np.zeros(M+1)
# Initial ice surface elevation and ice flux
hvec0 = np.zeros(M+1)
qvec0 = np.zeros(M+1)

# Specify boundary conditions
# Choose two boundary conditions to fix
FixQLeft  = True
FixQRight = False
FixHLeft  = False
FixHRight = True

# specify BC values either as constants or functions taking time as the argument
QLeft  = DD #0.1 * 10
QRight = 0
HLeft  = 0.1
HRight = 0.5

qindices = np.arange(0+FixQLeft,M+1-FixQRight)
hindices = np.arange(0+FixHLeft,M+1-FixHRight)

# Assign BCs
if FixQLeft == True:
    qvec0[0] = QLeft
if FixQRight == True:
    qvec0[-1] = QRight
if FixHLeft == True:
#    hvec0[0] = HLeft
    hvec0[:] = HLeft
if FixHRight == True:
#    hvec0[-1] = HRight
    hvec0[:] = HRight

# Integratiion
#harrrk4 = integrateRK4(hvec0,tarr,avec,qvec0,qindices,hindices)
#harrbt = integrateBT(hvec0,tarr,avec,qvec0,qindices,hindices)
harrode = integrateODE(hvec0,tarr,avec,qvec0,qindices,hindices)

# Plot
fractions = [0,0.25,0.5,0.75,1]
fractions = [0,0.01,0.03,0.05,0.1,0.15,0.2,0.25,0.3,0.5]
tindices = [np.where(abs(tarr-frac*tarr[-1])==abs(tarr-frac*tarr[-1]).min())[0][0] for frac in fractions]
fig, ax = plt.subplots(1)
[ax.plot(xarr,harrode[ti,:],label='t=%.2f' % tarr[ti]) for ti in tindices]
#ax.plot(xarr,hsteady(xarr,HRight,-QLeft/DD),'k--')
ax.plot(xarr,hsteady(xarr,HRight,-QLeft,DD),'k--')
ax.legend()







