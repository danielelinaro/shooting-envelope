
import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from polimi.systems import jacobian_finite_differences


def _extended_system(t,y,fun,jac,T,autonomous):
    if autonomous:
        N = int(np.max(np.roots([1,2,-len(y)])))
    else:        
        N = int(np.max(np.roots([1,1,-len(y)])))
    J = jac(t,y[:N])
    phi = np.reshape(y[N:N+N**2],(N,N))
    ydot = np.concatenate((T*fun(t*T,y[:N]),T*np.matmul(J,phi).flatten()))
    if autonomous:
        dxdt = y[-N:]
        ydot = np.concatenate((ydot,T*np.matmul(J,dxdt)+fun(t,y[:N])))
    return ydot


def shooting(fun,y0_guess,T_guess,autonomous,jac=None,max_iter=100,tol=1e-6,rtol=1e-6,atol=None,do_plot=False):
    # original number of dimensions of the system
    N = len(y0_guess)
    # number of dimensions of the extended system
    N_ext = N**2 + N
    X = y0_guess
    if autonomous:
        N_ext += N
        X = np.append(X,T_guess)
    else:
        # the period is fixed if the system is non-autonomous
        T = T_guess
    if jac is None:
        jac = lambda t,y: jacobian_finite_differences(fun, t, y)
    if atol is None:
        atol = 1e-8 + np.zeros(N_ext)
    for i in range(max_iter):
        #print('y0 = (%f,%f)' % (X[0],X[1]))
        y0_ext = np.concatenate((X[:N],np.eye(N).flatten()))
        if autonomous:
            y0_ext = np.concatenate((y0_ext,np.zeros(N)))
            T = X[-1]
        sol = solve_ivp(lambda t,y: _extended_system(t,y,fun,jac,T,autonomous),
                        [0,1], y0_ext, method='BDF', atol=atol, rtol=rtol)
        r = np.array([x[-1]-x[0] for x in sol['y'][:N]])
        phi = np.reshape(sol['y'][N:N**2+N,-1],(N,N))
        if autonomous:
            b = sol['y'][-N:,-1]
            M = np.zeros((N+1,N+1))
            M[:N,:N] = phi - np.eye(N)
            M[:N,-1] = b
            M[-1,:N] = b
            r = np.append(r,0.)
        else:
            M = phi - np.eye(N)
        X_new = X - np.matmul(inv(M),r)
        if do_plot:
            if i == 0:
                plt.subplot(1,2,2)
                plt.plot(sol['y'][1,:],sol['y'][0,:],'r')
                plt.subplot(1,2,1)
                plt.plot(sol['t'],sol['y'][0,:],'r')
            else:
                if np.max(np.abs(X_new-X)) < 1e-8:
                    plt.plot(sol['t'],sol['y'][0,:],'k')
                else:
                    plt.plot(sol['t'],sol['y'][0,:])
        if np.max(np.abs(X_new-X)) < tol:
            break
        X = X_new
    if do_plot:
        plt.xlabel('Time')
        plt.ylabel('x')
        plt.subplot(1,2,2)
        plt.plot(sol['y'][1,:],sol['y'][0,:],'k')
        plt.xlabel('y')
        plt.show()

    if autonomous:
        return X[:N],X[-1],phi,i+1
    return X,phi,i+1


