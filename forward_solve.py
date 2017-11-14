import numpy as np
import chebyshev as cheb
from DaConstant import *
import matplotlib.pyplot as plt
def iceFlow(h, diff_mat, di, tl = 0, a = [], bedrock = []):
    ''' Calculates rhs for a non-dimensional forward_euler timestep scheme and returns value
    of ice flow, q, plus any deposition or removal of ice thickness as given by
    a '''
    hdiff = diff_mat.dot(h)
    if not a:
        a = np.zeros(np.shape(h)[0])
    if not bedrock:
        bedrock = np.zeros(np.shape(h)[0])
    hb = (h + bedrock)**3
    hb = diff_mat.dot(hb*hdiff)
    return di*hb + tl*a

def forwardStep(h, diff_mat, di, dt, func_in = iceFlow, a_in = [], bedrock_in = [], tl_in = 0):
    ''' Simple forward-euler timestepping scheme '''
    return h + dt*func_in(h,diff_mat,di,a=a_in,bedrock=bedrock_in, tl = tl_in)

def integrateForward(dt,di,h_grid_in,N):
    ''' Integrating forward-euler timestepping scheme for N iterations'''
    step_no = 0
    t = [0]
    h_save = h_grid_in.data
    while step_no <= N:
        print(step_no)
        h_grid_in.data = forwardStep(h_grid_in.data,h_grid_in.diff_mat,di,dt)
        t.append(t[-1]+dt)
        h_save = np.vstack((h_save,h_grid_in.data))
        step_no += 1
    return t, h_save


n = 1000
di = DaConstant(3.15*(10**8),100000,1000,2.6*(10**13))
h = np.linspace(1,0.001,n)
h_grid_init = cheb.ChebGrid(n,data_in = h)
h_grid_init.diff_mat[[0,-1],:] = 0
t, saved_h = integrateForward(0.0001,di,h_grid_init,500)
print(saved_h[3,:])
plt.plot(h_grid_init.x_pts[::-1],saved_h[-1,:])
plt.show()
