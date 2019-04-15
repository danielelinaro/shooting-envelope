
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


def main():
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
    #var_envelope_solver = BEEnvelope(var_fun, t_span_var, y0_var, T_guess=None,
    #                                 T=T_small/T_large, rtol=1e-1, atol=atol)
    var_env = var_envelope_solver.solve()

    plt.plot(t_span_var,[0,0],'b')
    plt.plot(var_sol['t'],var_sol['y'][2],'k')
    plt.plot(var_env['t'],var_env['y'][2],'mo')
    plt.show()

    
if __name__ == '__main__':
    main()
