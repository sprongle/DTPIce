import numpy as np

def chebpts(n):
    ''' Returns a vector of Chebyshev points between 1 -> -1'''
    x = np.zeros(n)
    for i in range(len(x)):
        x[i] = np.cos((i*np.pi)/(n-1))
    return x

def chebdiffmat(n):
    ''' Returns an nxn differentiation matrix for chebyshev points, using
    forward difference at the end points and centered difference for the interior
    points '''
    xPts = chebpts(n)
    diffMat = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            if i == j:
                if i != 0 and i != n-1:
                    diffMat[i,i] = -0.5*xPts[i]/(1-(xPts[i]**2))
                elif i == 0:
                    diffMat[i,i] = ((2*(n-1)**2)+1)/6
                elif i == n-1:
                    diffMat[i,i] = -((2*(n-1)**2)+1)/6
            else:
                if (i != 0 and i!= n-1 and j != 0 and j!= n-1) or (i == 0 and j==n-1) or (i == n-1 and j == 0):
                    diffMat[i,j] = ((-1)**(i+j))/(xPts[i] - xPts[j])
                elif i == 0 or i == n-1:
                    diffMat[i,j] = 2*((-1)**(i+j))/(xPts[i] - xPts[j])
                elif j == 0 or j == n-1:
                    diffMat[i,j] = 0.5*((-1)**(i+j))/(xPts[i] - xPts[j])

    return diffMat

class chebgrid():
    '''Defines a grid of n Chebyshev points, self.xPts between x=0 and x=1,
    which comes with a differentiation matrix self.xDiff which can be applied to
    h to get a gradient dh/dx'''

    def __init__(self, n):
        self.xPts = chebpts(n)
        self.data = np.zeros(n)
        self.nPts = n
        self.diffMat = chebdiffmat(n)
        self.xDeriv = self.diffself()

    def diffself(self):
        ''' Applies the Chebyshev differentiation matrix to the data held in self.data
        and saves the result to self.xDeriv '''
        self.xDeriv = self.diffMat.dot(self.data)
