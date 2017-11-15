# Ice functions

import numpy as np
from scipy.integrate import odeint
from numba import jit

def hsteady(x,H0,q0,D):
#    return (H0**4 - 4 * q0 * (1 - x))**0.25
    return (H0**4 - 4 * q0/D * (1 - x))**0.25

def hgradsteady(x,H0,q0,D):
    #return q0 * (H0**4 - 4 * q0 * (1 - x))**(-0.75)
    return q0/D * (H0**4 - 4 * q0/D * (1 - x))**(-0.75)


class IceModel(object):
    
    def __init__(self,DD,EE):
        self.DD = DD
        self.EE = EE
        
    def setupGeometry(self,M,N,xmax=1,tmax=1):
        self.M = M
        self.N = N
        self.xarr = np.linspace(0,xmax,M+1)
        self.tarr = np.linspace(0,tmax,N+1)
        self.dx = self.xarr[1] - self.xarr[0]
        self.dt = self.tarr[1] - self.tarr[0]
        self.Mmat = self.getDiffMatrix(M+1,self.dx)
        
    def getDiffMatrix(self,M,dx):
        Dmat = np.diag(np.ones(M-1),1) + np.diag(-np.ones(M-1),-1)
        Dmat[0,0] = -1
        Dmat[-1,-1] = 1
        Dmat[0,:] *= 2
        Dmat[-1,:] *= 2
        return 1./(2*dx) * Dmat
    
    def setupPhysics(self,bvec,avec):
        self.bvec = bvec
        self.avec = avec
        
    def setupFixedBC(self,BCs):
        '''Set up time-independent boundary conditions for h and q. Require as input
        BCs= {
                'FixQLeft':DD,
                'FixQRight':False,
                'FixHLeft':False,
                'FixHRight':0.5}
        '''
        hvec0 = np.zeros(self.M+1)
        qvec0 = np.zeros(self.M+1)

#        qindices = np.arange(0+FixQLeft,M+1-FixQRight)
#        hindices = np.arange(0+FixHLeft,M+1-FixHRight)
        qindices = np.arange(0+(BCs['FixQLeft']!=False),self.M+1-(BCs['FixQRight']!=False))
        hindices = np.arange(0+(BCs['FixHLeft']!=False),self.M+1-(BCs['FixHRight']!=False))
        
        # Assign BCs
        if not BCs['FixQLeft'] == False:
            qvec0[0] = BCs['FixQLeft']
        if not BCs['FixQRight'] == False:
            qvec0[-1] = BCs['FixQRight']
        if not BCs['FixHLeft'] == False:
            hvec0[:] = BCs['FixHLeft']
        if not BCs['FixHRight'] == False:
            hvec0[:] = BCs['FixHRight']
            
        self.hvec0 = hvec0
        self.qvec0 = qvec0
        self.qindices = qindices
        self.qmask = [i for i in np.arange(self.M+1) if not i in qindices]
        self.hindices = hindices
        self.hmask = [i for i in np.arange(self.M+1) if not i in hindices]
        self.FixedBC = True

    #@jit(nopython=True)
    def RHSodeintFixBC(self,hvec,t): #,avec,qvec,qindices,hindices
        '''RHS for odeint integrator.
        Fixed BC and A'''
        qvecarr = self.qvec0.copy()
        qvecarr[self.qindices] = (- self.DD * (hvec+self.bvec)**3 * np.dot(self.Mmat,hvec))[self.qindices]
        hvecnew = np.zeros(hvec.shape)
        hvecnew[self.hindices] = (-np.dot(self.Mmat,qvecarr) + self.EE * self.avec)[self.hindices]
        
        return hvecnew

    @jit(nopython=True)
    def RHSodeintFixBCvaryA(self,hvec,t): #,avec,qvec,qindices,hindices
        '''RHS for odeint integrator
        Fixed BC, A as time-dependent function'''
        qvecarr = self.qvec0.copy()
        qvecarr[self.qindices] = (- self.DD * (hvec+self.bvec)**3 * np.dot(self.Mmat,hvec))[self.qindices]
        hvecnew = np.zeros(hvec.shape)
        hvecnew[self.hindices] = (-np.dot(self.Mmat,qvecarr) + self.EE * self.avec(t))[self.hindices]
        return hvecnew

    @jit(nopython=True)
    def RHSodeintFixHvaryAvaryQ(self,hvec,t): #,avec,qvec,qindices,hindices
        '''RHS for odeint integrator
        Fixed BC, A as time-dependent function. Requires manually setting the function self.qfunc(t)'''
        qvecarr = np.zeros(hvec.shape) #self.qvec0.copy()
        qvecarr[self.qmask] = self.qfunc(t)
        qvecarr[self.qindices] = (- self.DD * (hvec+self.bvec)**3 * np.dot(self.Mmat,hvec))[self.qindices]
        hvecnew = np.zeros(hvec.shape)
        hvecnew[self.hindices] = (-np.dot(self.Mmat,qvecarr) + self.EE * self.avec(t))[self.hindices]
        return hvecnew

    def integrateODE(self,hvec0,full=False):
        '''Using scipy.integrate.odeint integrator'''
        # select RHS function depending on choices
        if isinstance(self.avec,np.ndarray):
            RHSfunc = self.RHSodeintFixBC
            print('Fixed BC and forcing')
        elif isinstance(self.avec,type(print)):
            RHSfunc = self.RHSodeintFixBCvaryA
            print('Fixed BC, vary forcing')
        elif isinstance(self.avec,type(print)) and isinstance(self.qfunc,type(print)):
            print('Fixed H BC, varying Q BC, varying forcing')
            RHSfunc = self.RHSodeintFixHvaryAvaryQ
            
        self.res = odeint(RHSfunc,hvec0,self.tarr,full_output=full)
        outdict = {
                'qvec0':self.qvec0,
                'qindices':self.qindices,
                'DD':self.DD,
                'EE':self.EE,
                'avec':self.avec,
                'bvec':self.bvec,
                'xarr':self.xarr,
                'tarr':self.tarr,
                'res':self.res
                }
        modelID = np.random.randint(1000,9999)
        print('Model and run saved as res_%i.npy' % modelID)
        np.save('res_%i.npy' % modelID,outdict)
        return self.res

