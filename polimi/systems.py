
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

def vdp(t,y,epsilon,A,T):
#    F = 1./T
#    return np.array([
#        y[1],
#        epsilon*(1-y[0]**2)*y[1] - y[0] + A*np.cos(2*np.pi*F*t)
#    ])
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

def y0min(t,y):
    return y[0]
y0min.direction = 1

def y1min(t,y):
    return y[1]
y1min.direction = 1

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
