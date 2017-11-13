import numpy as np
import chebyshev as cheb

def forwardStep(h, diff_mat, di, dt, func_in = iceFlow, a_in = None, bedrock_in = None, tl_in = 0):
    return dt*func_in(h,diff_mat,di,a=a_in,bedrock=bedrock_in, tl = tl_in)


def iceFlow(h, diff_mat, di, tl = 0, a = None, bedrock = None):
    hdiff = diff_mat.dot(h)
    if not a:
        a = np.zeros(np.shape(h)[0])
    if not bedrock:
        bedrock = np.zeros(np.shape(h)[0])
    hb = (h + b)**3
    hb = diff_mat.dot(hb*hdiff)
    return di*hb + tl*a

def intForward(dt,h_grid_in,n_steps)
