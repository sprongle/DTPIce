
def DaConstant(T, L, H, nu):
    g = 9.81  # gravitational constant
    # T = time in seconds
    # L = length in meters
    # H = height in meters
    # nu = kinematic viscosity in Pa * s
    D = ( g*T*H**3)/(3*nu*L**2)
    return D