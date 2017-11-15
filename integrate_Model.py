#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:08:07 2017

@author: matthias
"""
import numpy as np
import scipy.sparse as spa
from scipy.integrate import odeint
from numba import jit
import matplotlib.pyplot as plt
import scipy.optimize as opt
import icefunctions as icef




# Get dimensionless numbers
g = 9.81 # m s^-2
T = 60**2 * 24 * 365.25 * 100 # s
H = 1000 # m
L = 10e3 # m
nu = 2.6e13 # Pa s
A = 1 / T# m s^-1

DD = g * T * H**3 / (3 * nu * L**2)
DD = 1.#0.1
EE = T * A / H

# Define Geometry
N = 70 # 200
M = 1000 #500 #200

icem = icef.IceModel(DD,EE)
icem.setupGeometry(M,N*30,tmax=30)

# Bottom topography
b0 = 0.1 #0.2
b1 = -0.15/2 #-0.3
b2 = 0.16/3 #0.1
k = np.pi
#bvec = b0 + b1 * icem.xarr
bvec = b0 + b1 * icem.xarr + b2 * np.sin(k*icem.xarr)

# Surface Mass Balance
avec = np.zeros(M+1)

icem.setupPhysics(bvec,avec)

# Boundary Conditions
BCchoices = {
        'FixQLeft':DD*0.03,
        'FixQRight':False,
        'FixHLeft':False,
        'FixHRight':0.125} #0.2

icem.setupFixedBC(BCchoices)

# Integrate
hout = icem.integrateODE(icem.hvec0)
#hout = icem.integrateODE(icem.bvec+0.1)

# Plot
fractions = [0,0.25,0.5,0.75,1]
fractions = [0,0.01,0.03,0.05,0.1,0.15,0.2,0.25,0.3,0.5]
tindices = [np.where(abs(icem.tarr-frac*icem.tarr[-1])==abs(icem.tarr-frac*icem.tarr[-1]).min())[0][0] for frac in fractions]
fig, ax = plt.subplots(1)
[ax.plot(icem.xarr,icem.res[ti,:],label='t=%.2f' % icem.tarr[ti]) for ti in tindices]
#ax.plot(xarr,hsteady(xarr,HRight,-QLeft/DD),'k--')
ax.plot(icem.xarr,icem.bvec,'k-')
#ax.plot(icem.xarr,icef.hsteady(icem.xarr,BCchoices['FixHRight'],-BCchoices['FixQLeft'],icem.DD),'k--')
ax.legend()

