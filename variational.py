
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from polimi.envel import TrapEnvelope, BEEnvelope, EnvelopeInterp
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

    plt.figure()
    plt.plot(sol['t'],sol['y'][0],'k',label='y_1')
    plt.plot(env['t'],env['y'][0],'ro',label='y_1 env')
    #plt.plot(sol['t'],sol['y'][1],'m',label='y_2')
    #plt.plot(env['t'],env['y'][1],'go',label='y_2 env')
    plt.legend(loc='best')
    #plt.show()

    atol = [1e-2,1e-2,1e-6,1e-6,1e-6,1e-6]
    var_sol = solve_ivp(var_fun, t_span_var, y0_var, rtol=1e-7, atol=1e-8,
                        dense_output=True)
    var_envelope_solver = TrapEnvelope(var_fun, t_span_var, y0_var, T_guess=None,
                                       T=T_small/T_large, rtol=1e-1, atol=atol)
    var_env = var_envelope_solver.solve()

    plt.figure()
    plt.plot(t_span_var,[0,0],'b')
    plt.plot(var_sol['t'],var_sol['y'][2],'k')
    plt.plot(var_env['t'],var_env['y'][2],'mo')
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
    var_sol = solve_ivp(var_fun, t_span_var, y0_var, rtol=1e-7, atol=1e-8)
    var_envelope_solver = TrapEnvelope(var_fun, t_span_var, y0_var, T_guess=None,
                                       T=T_small/T_large, rtol=1e-1, atol=atol)
    var_env = var_envelope_solver.solve()

    plt.subplot(1,2,1)
    plt.plot(var_sol['t'],var_sol['y'][0],'k')
    plt.plot(var_env['t'],var_env['y'][0],'ro')
    plt.subplot(1,2,2)
    plt.plot(t_span_var,[0,0],'b')
    plt.plot(var_sol['t'],var_sol['y'][2],'k')
    plt.plot(var_env['t'],var_env['y'][2],'ro')
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

    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.plot(var_sol['t'],var_sol['y'][i*2],'k')
        plt.plot(var_env['t'],var_env['y'][i*2],'ro')
    plt.show()


def many_cycles():
    epsilon = 1e-3
    A = [1]
    T_small = 2*np.pi
    T_large = 10*T_small
    T = [T_large]
    fun = lambda t,y: vdp(t,y,epsilon,A,T)
    jac = lambda t,y: vdp_jac(t,y,epsilon)
    rtol = 1e-8
    atol = 1e-10

    Y0 = np.array([[0,1],[1,0],[0,-1],[-1,0]])
    cycles = []

    for y0 in Y0:
        tran = solve_ivp(fun, [0,20*T_large], y0, method='BDF', jac=jac, rtol=rtol, atol=atol)
        #plt.plot(tran['t'],tran['y'][0],'k')
        #plt.show()
        cycle = solve_ivp(fun, tran['t'][-1]+np.array([0,T_large]),
                          tran['y'][:,-1], method='BDF', jac=jac, rtol=rtol, atol=atol)
        print('{} -> {}'.format(cycle['y'][:,0],cycle['y'][:,-1]))
        cycles.append(cycle)

    col = 'krgb'
    for i,cycle in enumerate(cycles):
        plt.subplot(1,2,1)
        plt.plot(cycle['t'],cycle['y'][0],col[i])
        plt.subplot(1,2,2)
        plt.plot(Y0[i,0],Y0[i,1],col[i]+'o')
        plt.plot(cycle['y'][0],cycle['y'][1],col[i],label='IC_{}'.format(i))
    plt.legend(loc='best')
    plt.show()


def variational_system_reduced(jac, t, y, interp, T):
    N = int(np.sqrt(len(y)))
    phi = np.reshape(y,(N,N))
    J = jac(t,interp(t))
    return T * np.matmul(J,phi).flatten()


def hybrid():
    epsilon = 1e-3
    A = [10,1]
    T = [4,400]
    T_large = max(T)
    T_small = min(T)
    t_span_var = [0,1]

    fun = lambda t,y: T_large * vdp(t*T_large,y,epsilon,A,T)
    jac = lambda t,y: vdp_jac(t,y,epsilon)

    y0 = np.array([-5.8133754 ,  0.13476983])

    sol = solve_ivp(fun, t_span_var, y0, rtol=1e-8, atol=1e-8)
    solver = TrapEnvelope(fun, t_span_var, y0, T_guess=None,
                          T=T_small/T_large, rtol=1e-1, atol=1e-2)
    env = solver.solve()
    interp = EnvelopeInterp(fun, env, T_small/T_large)

    y0_var = np.eye(2).flatten()
    var_fun = lambda t,y: variational_system_reduced(jac, t, y, interp, T_large)
    var_sol = solve_ivp(var_fun, [0,0.05], y0_var, rtol=1e-6, atol=1e-8)
    var_envelope_solver = TrapEnvelope(var_fun, [0,0.05], y0_var, T_guess=None,
                                       T=2*np.pi/T_large, rtol=1e-1, atol=1e-2)
    var_env = var_envelope_solver.solve()

    print('Total integrated time =', interp.total_integrated_time)
    plt.subplot(1,2,1)
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(env['t'],env['y'][0],'ro')
    for t in range(1,10):
        y = interp(t*0.1+0.003)
        plt.plot(t*0.1+0.003,y[0],'gx')
    plt.subplot(1,2,2)
    plt.plot(var_sol['t'],var_sol['y'][0],'k')
    plt.plot(var_env['t'],var_env['y'][0],'ro')
    plt.show()

if __name__ == '__main__':
    #variational_vdp()
    #variational_linear()
    variational_vdp_one_freq()
    #many_cycles()
    #hybrid()
