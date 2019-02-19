
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.optimize import newton
from scipy.special import binom


#### Adams-Bashforth and Adams-Moulton coefficients

#gamma = lambda j: (-1)**j*quad(lambda s,j: binom(-s,j), 0, 1, args=[j])[0]
#def beta(k,i):
#    tmp = 0
#    for j in range(i-1,k):
#        tmp += gamma(j)*binom(j,i-1)
#    return (-1)**(i-1) * tmp

A = np.zeros((6,6))
B = np.zeros((6,6))
Bstar = np.zeros((6,6))
A[:,0] = 1.
B[0,0] = 1.
B[1,:2] = np.array([3.,-1.])/2.
B[2,:3] = np.array([23.,-16.,5.])/12.
B[3,:4] = np.array([55.,-59.,37.,-9.])/24.
B[4,:5] = np.array([1901.,-2774.,2616.,-1274.,251.])/720.
B[5,:6] = np.array([4277.,-7923.,9982.,-7298.,2877.,-475.])/1440.
Bstar[0,0] = 1.
Bstar[1,:2] = np.array([1.,1.])/2.
Bstar[2,:3] = np.array([5.,8.,-1.])/12.
Bstar[3,:4] = np.array([9.,19.,-5.,1.])/24.
Bstar[4,:5] = np.array([251.,646.,-264.,106.,-19.])/720.
Bstar[5,:6] = np.array([475.,1427.,-798.,482.,-173.,27.])/1440.


def _envelope_system(t,y,fun,T,rtol=1e-8,atol=None):
    N = len(y)
    if atol is None:
        atol = 1e-10 + np.zeros(N)
    one_period = solve_ivp(fun,[0,T],y,rtol=rtol,atol=atol)
    G = 1./T * (one_period['y'][:,-1] - one_period['y'][:,0])
    return G

    
def envelope(fun,t_span,y0,T,H,order=3,rtol=1e-8,atol=None):
    env_fun = lambda t,y: _envelope_system(t,y,fun,T)
    return bdf(env_fun, t_span, y0, H, order=4)


def envelope_ext(fun,t_span,y0,T,H,order=3,rtol=1e-8,atol=None):
    env_fun = lambda t,y: _envelope_system(t,y,fun,T,rtol,atol)
    return solve_ivp(env_fun, t_span, y0, method='BDF', rtol=rtol, atol=atol)


def envelope_full(fun,t_span,y0,T,H,order=3,rtol=1e-8,atol=None):
    import ipdb
    # number of dimensions of the system
    N = len(y0)
    if atol is None:
        atol = 1e-10 + np.zeros(N)
    # number of time steps to take
    n_steps = int(np.ceil(np.diff(t_span)/H)) + 1
    # the solution dictionary that will be returned
    sol = {'t': np.array([]),
           'y': np.array([[] for i in range(N)]),
           'T': np.arange(n_steps)*H + t_span[0],
           'Z': np.zeros((N,n_steps)),
           'G': np.zeros((N,n_steps))}
    # initialization
    sol['Z'][:,0] = y0
    y = solve_ivp(fun,[0,T],y0,rtol=rtol,atol=atol)
    sol['G'][:,0] = 1./T * (y['y'][:,-1] - y['y'][:,0])
    sol['t'] = np.concatenate((sol['t'],t_span[0]+y['t']))
    sol['y'] = np.concatenate((sol['y'],y['y']),axis=1)
    ### R-K steps
    import sys
    for i in range(1,order):
        sys.stdout.write('R-K iteration #%d ' % i)
        y = solve_ivp(fun,[0,T],sol['Z'][:,i-1],rtol=rtol,atol=atol)
        sys.stdout.write('+')
        k1 = 1./T * (y['y'][:,-1] - y['y'][:,0])
        print('k1 = (%f,%f)' % tuple(k1))
        y = solve_ivp(fun,[0,T],sol['Z'][:,i-1]+H/2*k1,rtol=rtol,atol=atol)
        sys.stdout.write('+')
        k2 = 1./T * (y['y'][:,-1] - y['y'][:,0])
        print('k2 = (%f,%f)' % tuple(k2))
        y = solve_ivp(fun,[0,T],sol['Z'][:,i-1]+H/2*k2,rtol=rtol,atol=atol)
        sys.stdout.write('+')
        k3 = 1./T * (y['y'][:,-1] - y['y'][:,0])
        print('k3 = (%f,%f)' % tuple(k3))
        #if i == 2:
        #    ipdb.set_trace()
        y = solve_ivp(fun,[0,T],sol['Z'][:,i-1]+H*k3,rtol=rtol,atol=atol)
        sys.stdout.write('+')
        k4 = 1./T * (y['y'][:,-1] - y['y'][:,0])
        print('k4 = (%f,%f)' % tuple(k4))
        sol['Z'][:,i] = sol['Z'][:,i-1] + H*(k1+2*k2+2*k3+k4)/6
        y = solve_ivp(fun,[0,T],sol['Z'][:,i],rtol=rtol,atol=atol)
        sys.stdout.write('*')
        #ipdb.set_trace()
        sol['G'][:,i] = 1./T * (y['y'][:,-1] - y['y'][:,0])
        sol['t'] = np.concatenate((sol['t'],sol['T'][i]+y['t']))
        sol['y'] = np.concatenate((sol['y'],y['y']),axis=1)
        sys.stdout.write(' done.\n')
    print('After R-K steps.')
    for i in range(order,n_steps):
        ### predictor ###
        z_p = np.zeros(N)
        for j in range(order):
            z_p += A[order-1,j]*sol['Z'][:,i-1-j] + H*B[order-1,j]*sol['G'][:,i-1-j]
        ### corrector ###
        y = solve_ivp(fun,[0,T],z_p,rtol=rtol,atol=atol)
        g_tmp = 1./T * (y['y'][:,-1] - y['y'][:,0])
        z_c = A[order-1,0]*sol['Z'][:,i-1] + H*Bstar[order-1,0]*g_tmp
        for j in range(1,order):
            z_c += A[order-1,j]*sol['Z'][:,i-j] + H*Bstar[order-1,j]*sol['G'][:,i-j]
        sol['Z'][:,i] = z_c
        sys.stdout.write('.')
        if (i-order+1)%50 == 0:
            sys.stdout.write('\n')
        y = solve_ivp(fun,[0,T],sol['Z'][:,i],rtol=rtol,atol=atol)
        sol['G'][:,i] = 1./T * (y['y'][:,-1] - y['y'][:,0])
        sol['t'] = np.concatenate((sol['t'],sol['T'][i]+y['t']))
        sol['y'] = np.concatenate((sol['y'],y['y']),axis=1)
    if (i-order+1)%50:
        sys.stdout.write('\n')
    return sol


