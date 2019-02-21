#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from polimi.systems import vdp, vdp_jac, y0min
from polimi.envelope import RK45Envelope, _envelope_system, _one_period
import sys

def autonomous_vdp():
    epsilon = 0.001
    A = [0.,0.]
    T = [1.,1.]
    y0 = [2e-3,0]
    reltol = 1e-8
    abstol = 1e-10*np.ones(len(y0))
    T_exact = 2*np.pi
    T_guess = 0.9 * T_exact

    fun = lambda t,y: vdp(t,y,epsilon,A,T)

    tend = 5000
    print('Integrating the full system...')
    full = solve_ivp(fun, [0,tend], y0, method='RK45', atol=abstol, rtol=reltol)

    env_fun_1 = lambda t,y: _envelope_system(t, y, fun, T_exact, method='RK45', atol=abstol, rtol=reltol)
    print('Integrating the first envelope function...')
    var_step_1 = solve_ivp(env_fun_1, [0,tend], y0, method='RK45', atol=abstol, rtol=reltol)

    env_fun_2 = lambda t,y: _one_period(t, y, fun, T_guess, method='RK45', full_output=False, atol=abstol, rtol=reltol)
    print('Integrating the second envelope function...')
    var_step_2 = solve_ivp(env_fun_2, [0,tend], y0, method='RK45', atol=abstol, rtol=reltol)

    print('Integrating the envelope...')
    #envelope = solve_ivp(fun, [0,tend], y0, method=Envelope, T_guess=T_guess, atol=abstol, rtol=reltol)
    envelope = solve_ivp(fun, [0,tend], y0, method=RK45Envelope, T_guess=T_guess, atol=abstol, rtol=reltol)
    
    plt.figure()
    plt.plot(full['t'],full['y'][0],'k')
    plt.plot(var_step_1['t'],var_step_1['y'][0],'go-',lw=4)
    plt.plot(var_step_2['t'],var_step_2['y'][0],'ms-',lw=2)
    plt.plot(envelope['t'],envelope['y'][0],'r^-',lw=1)
    for t0,y0 in zip(envelope['t'],envelope['y'].transpose()):
        sol = solve_ivp(fun, [t0,t0+T_exact], y0, method='RK45', atol=abstol, rtol=reltol)
        plt.plot(sol['t'],sol['y'][0],'m')
    plt.show()


def main():
    if len(sys.argv) == 1 or argv[1] == 'autonomous':
        autonomous_vdp()

if __name__ == '__main__':
    main()
    
