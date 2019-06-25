
__all__ = ['boost', 'boost_jac', 'vdp', 'vdp_jac', 'vdp_extrema', 'vdp_auto', 'hr']

import numpy as np

def jacobian_finite_differences(fun,t,y):
    n = len(y)
    J = np.zeros((n,n))
    ref = fun(t,y)
    eps = 1e-8
    for i in range(n):
        dy = np.zeros(n)
        dy[i] = eps
        pert = fun(t,y+dy)
        J[:,i] = (pert-ref)/eps
    return J


def boost_matrixes(R, L, C, Rs, Vin):
    A1 = np.array([ [-1/(R*C), 0],   [0, -Rs/L]    ])
    A2 = np.array([ [-1/(R*C), 1/C], [-1/L, -Rs/L] ])
    B  = np.array([0, Vin/L])
    return A1, A2, B

def boost(t, y, T=20e-6, DC=0.5, R=5, L=10e-6, C=47e-6, Rs=0, Vin=5):
    A1, A2, B = boost_matrixes(R, L, C, Rs, Vin)
    if (t % T) < (DC * T):
        A = A1
    else:
        A = A2
    return np.matmul(A,y) + B

def boost_jac(t, y, T=20e-6, DC=0.5, R=5, L=10e-6, C=47e-6, Rs=0, Vin=5):
    A1, A2, _ = boost_matrixes(R, L, C, Rs, Vin)
    if (t % T) < (DC * T):
        return A1
    return A2

def vdp(t,y,epsilon,A,T):
    F = [1./tt for tt in T]
    ydot = np.array([
        y[1],
        epsilon*(1-y[0]**2)*y[1] - y[0]
    ])
    n = len(A)
    for i in range(n):
        ydot[1] += A[i]*np.cos(2*np.pi*F[i]*t)
    return ydot

def vdp_jac(t,y,epsilon):
    return np.array([
        [0,1],
        [-2*epsilon*y[0]*y[1]-1,epsilon*(1-y[0]**2)]
    ])

def vdp_extrema(t,y,epsilon,A,T,coord):
    ydot = vdp(t,y,epsilon,A,T)
    return ydot[coord]

#def polar(t,y,k,rho,L,T):
#    root = np.square(y[0]**2 + y[1]**2)
#    return np.array([
#        k * (rho*L*y[0]/root - L*y[0]) - 2*np.pi*y[1]/T,
#        k * (rho*L*y[1]/root - L*y[1]) - 2*np.pi*y[0]/T
#        ])

def vdp_auto(t,y,epsilon,A,T):
    rho = 1
    N_forcing = len(A)
    N_eq = 2 + N_forcing*2
    ydot = np.zeros(N_eq)
    ydot[0] = y[1]
    ydot[1] = epsilon*(1-y[0]**2)*y[1] - y[0]
    for i in range(N_forcing):
        j = 2*(i+1)
        ydot[1] += A[i] * y[j]
        sum_of_squares = y[j]**2 + y[j+1]**2
        ydot[j+1] = 2*np.pi/T[i]*y[j] + y[j+1]*((rho-sum_of_squares)/(2*sum_of_squares))
        ydot[j] = (rho - sum_of_squares - 2*y[j+1]*ydot[j+1]) / (2*y[j])
    return ydot

def polar(t,y,rho,T):
    sum_of_squares = y[0]**2 + y[1]**2
    ydot = np.zeros(2)
    ydot[1] = 2*np.pi/T*y[0] + y[1]*((rho-sum_of_squares)/(2*sum_of_squares))
    ydot[0] = (rho - sum_of_squares - 2*y[1]*ydot[1]) / (2*y[0])
    return ydot

def burster(t,y,alpha,R,TF,I0,T,A,F,frac):
    square = lambda t,T,frac: 1 if t%T < frac*T else 0
    V1 = A*np.sin(2*np.pi*F*t)
    V2 = square(t,T,frac)
    A = I0 * np.arctanh(alpha*y[0])
    return np.array([-(y[0]/R + A + V1*V2) / (alpha * TF * I0 * (1-A*A))])

def colpitts(t,y,Q,g,k,Q0,alpha):
    n = lambda x: np.exp(-x)-1.
    #return np.array([
    #    g/(Q*(1-k)) * (-np.exp(-y[1]) + 1 + y[2]),
    #    g/(Q*k) * y[2],
    #    -Q*k*(1-k)/g*(y[0]+y[1]) - y[2]/Q
    #    ])
    x1,x2,x3 = y[0],y[1],y[2]
    return np.array([
        g/(Q*(1-k)) * (-alpha*n(x2) + x3),
        g/(Q*k) * ((1-alpha)*n(x2) + x3) - Q0*(1-k)*x2,
        -Q*k*(1-k)/g * (x1+x2) - x3/Q
        ])

def colpitts_jac(t,y,Q,g,k):
    return np.array([
        [0,g*np.exp(y[1])/(Q*(k-1)),g/(Q*(1-k))],
        [0,0,g/(k*Q)],
        [-Q*k*(1-k)/g,-Q*k*(1-k)/g,-1./Q]
        ])

def hr(t,y,I,b,mu=0.01,s=4,x_rest=-1.6):
    return np.array([
        y[1] - y[0]**3 + b*y[0]**2 + I - y[2],
        1 - 5*y[0]**2 - y[1],
        mu * (s * (y[0] - x_rest) - y[2])
        ])

def vanderpol_test():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    epsilon = 1e-3
    A = [10,0.1]
    T = [10,1000]
    y0 = [2e-3,0,1,0,1,0]
    tend = 3000
    fun = lambda t,y: vdp_auto(t,y,epsilon,A,T)
    sol = solve_ivp(fun, [0,tend], y0, method='RK45', atol=1e-6, rtol=1e-8)
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.show()

def burster_test():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    I0 = 0.4e-3
    alpha = 1
    TF = 2e-9
    R = 1e3
    amp = 4.5e-3
    T = 100e-9
    F = 1e9
    tend = 200e-9
    frac = 0.5
    fun = lambda t,y: burster(t,y,alpha,R,TF,I0,T,amp,F,frac)
    y0 = [1e-3]
    sol = solve_ivp(fun, [0,tend], y0, method='RK45', atol=1e-8, rtol=1e-10)
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.show()

def hr_test():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    b = 3
    I = 5
    tend = 500
    fun = lambda t,y: hr(t,y,I,b)
    y0 = [0,1,0.1]
    sol = solve_ivp(fun, [0,tend], y0, method='RK45', atol=1e-8, rtol=1e-10)
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.show()

if __name__ == '__main__':
    hr_test()
