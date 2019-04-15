
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from polimi.envel import TrapEnvelope, BEEnvelope
from polimi.systems import vdp, vdp_jac

import ipdb


def variational_system(fun, jac, t, y, T):
    N = int((-1 + np.sqrt(1 + 4*len(y))) / 2)
    J = jac(t,y[:N])
    phi = np.reshape(y[N:N+N**2],(N,N))
    return np.concatenate((T * fun(t*T, y[:N]), \
                           T * np.matmul(J,phi).flatten()))


def linear(t,y,A,B,T):
    u = lambda t: np.array([np.cos(2*np.pi*t/TT) for TT in T])
    return np.matmul(A,y) + np.matmul(B,u(t))


def variational_linear():
    A = np.array([[0,1],[-1,-0.001]])
    B = np.array([[0],[10]])
    T_small = 2*np.pi
    T_large = 100*T_small
    T = [T_large]
    fun = lambda t,y: linear(t,y,A,B,T)
    jac = lambda t,y: A
    var_fun = lambda t,y: variational_system(fun, jac, t, y, T_large)
    
    t_span = [0,T_large]
    sol = solve_ivp(fun, t_span, [1,0], method='RK45', rtol=1e-8, atol=1e-10)
    y0 = sol['y'][:,-1]
    t_span_var = [0,1]
    y0_var = np.concatenate((y0,np.eye(len(y0)).flatten()))
    
    sol = solve_ivp(fun, t_span, y0, method='RK45', rtol=1e-8, atol=1e-10)
    envelope_solver = TrapEnvelope(fun, t_span, y0, T_guess=None,
                                   T=T_small, rtol=1e-2, atol=1e-4)
    env = envelope_solver.solve()

    plt.plot(sol['t'],sol['y'][0],'k',label='y_1')
    plt.plot(env['t'],env['y'][0],'ro',label='y_1 env')
    plt.plot(sol['t'],sol['y'][1],'m',label='y_2')
    plt.plot(env['t'],env['y'][1],'go',label='y_2 env')
    plt.legend(loc='best')
    plt.show()

    atol = [1e-2,1e-2,1e-6,1e-6,1e-6,1e-6]
    var_sol = solve_ivp(var_fun, t_span_var, y0_var, rtol=1e-7, atol=1e-8,
                        dense_output=True)
    var_envelope_solver = TrapEnvelope(var_fun, t_span_var, y0_var, T_guess=None,
                                       T=T_small/T_large, rtol=1e-1, atol=atol)
    var_env = var_envelope_solver.solve()

    plt.plot(t_span_var,[0,0],'b')
    plt.plot(var_sol['t'],var_sol['y'][3],'k')
    plt.plot(var_env['t'],var_env['y'][3],'mo')
    plt.show()


def variational_vdp():
    epsilon = 1e-3
    A = [10,1]
    T = [4,400]
    T_large = max(T)
    T_small = min(T)
    fun = lambda t,y: vdp(t,y,epsilon,A,T)
    jac = lambda t,y: vdp_jac(t,y,epsilon)
    var_fun = lambda t,y: variational_system(fun, jac, t, y, T_large)

    t_span_var = [0,1]
    y0 = np.array([-5.8133754 ,  0.13476983])
    y0_var = np.concatenate((y0,np.eye(len(y0)).flatten()))

    atol = [1e-2,1e-2,1e-6,1e-6,1e-6,1e-6]
    var_sol = solve_ivp(var_fun, t_span_var, y0_var, rtol=1e-7, atol=1e-8,
                        dense_output=True)
    var_envelope_solver = TrapEnvelope(var_fun, t_span_var, y0_var, T_guess=None,
                                       T=T_small/T_large, rtol=1e-1, atol=atol)
    var_env = var_envelope_solver.solve()

    plt.plot(t_span_var,[0,0],'b')
    plt.plot(var_sol['t'],var_sol['y'][2],'k')
    plt.plot(var_env['t'],var_env['y'][2],'mo')
    plt.show()

    
def variational_vdp_one_freq():
    epsilon = 1e-3
    A = [1]
    T_small = 2*np.pi
    T_large = 100*T_small
    T = [T_large]
    fun = lambda t,y: vdp(t,y,epsilon,A,T)
    jac = lambda t,y: vdp_jac(t,y,epsilon)
    var_fun = lambda t,y: variational_system(fun, jac, t, y, T_large)

    t_span_var = [0,1]
    y0 = np.array([-0.41338977, -0.00832443])
    y0_var = np.concatenate((y0,np.eye(len(y0)).flatten()))

    atol = [1e-2,1e-2,1e-6,1e-6,1e-6,1e-6]
    var_sol = solve_ivp(var_fun, t_span_var, y0_var, rtol=1e-7, atol=1e-8,
                        dense_output=True)
    var_envelope_solver = TrapEnvelope(var_fun, t_span_var, y0_var, T_guess=None,
                                       T=T_small/T_large, rtol=1e-1, atol=atol)
    var_env = var_envelope_solver.solve()

    eig,_ = np.linalg.eig(np.reshape(var_sol['y'][2:,-1],(2,2)))
    var_eig,_ = np.linalg.eig(np.reshape(var_env['y'][2:,-1],(2,2)))

    print('Eigenvalues of Phi:', eig)
    print('Eigenvalues of Phi computed with envelope:', var_eig)

    plt.plot(t_span_var,[0,0],'b')
    plt.plot(var_sol['t'],var_sol['y'][2],'k')
    plt.plot(var_env['t'],var_env['y'][2],'mo')
    plt.show()


if __name__ == '__main__':
    #variational_linear()
    #variational_vdp()
    variational_vdp_one_freq()
