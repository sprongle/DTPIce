
def DaConstant(T, L, H, mu):
    g = 9.81  # gravitational constant
    rho = 900 # Approximate density of ice
    ''' T = time in seconds
    L = length in meters
    H = height in meters
    mu = dynamic viscosity in Pa * s '''
    nu = mu/rho
    D = ( g*T*H**3)/(3*nu*L**2)
    return D
