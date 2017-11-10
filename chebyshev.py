import numpy as np

def chebPts(n):
    ''' Returns a vector of Chebyshev points between 1 -> -1'''
    x = np.zeros(n)
    for i in range(len(x)):
        x[i] = np.cos((i*np.pi)/(n-1))
    return x

def chebdiffmat(n):
    ''' Returns an nxn differentiation matrix for chebyshev points, using
    forward difference at the end points and centered difference for the interior
    points '''
    x_pts = chebPts(n)
    diff_mat = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            if i == j:
                if i != 0 and i != n-1:
                    diff_mat[i,i] = -0.5*x_pts[i]/(1-(x_pts[i]**2))
                elif i == 0:
                    diff_mat[i,i] = ((2*(n-1)**2)+1)/6
                elif i == n-1:
                    diff_mat[i,i] = -((2*(n-1)**2)+1)/6
            else:
                if (i != 0 and i!= n-1 and j != 0 and j!= n-1) or (i == 0 and j==n-1) or (i == n-1 and j == 0):
                    diff_mat[i,j] = ((-1)**(i+j))/(x_pts[i] - x_pts[j])
                elif i == 0 or i == n-1:
                    diff_mat[i,j] = 2*((-1)**(i+j))/(x_pts[i] - x_pts[j])
                elif j == 0 or j == n-1:
                    diff_mat[i,j] = 0.5*((-1)**(i+j))/(x_pts[i] - x_pts[j])

    return diff_mat

class ChebGrid():
    '''Defines a grid of n Chebyshev points, self.xPts between x=0 and x=1,
    which comes with a differentiation matrix self.xDiff which can be applied to
    h to get a gradient dh/dx'''

    def __init__(self, n):
        self.x_pts = chebPts(n)
        self.data = np.zeros(n)
        self.n_pts = n
        self.diff_mat = chebDiffMat(n)
        self.x_deriv = self.diffSelf()

    def diffSelf(self):
        ''' Applies the Chebyshev differentiation matrix to the data held in self.data
        and saves the result to self.xDeriv '''
        self.x_deriv = self.diff_mat.dot(self.data)
