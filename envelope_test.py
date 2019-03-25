#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from polimi.systems import vdp, vdp_jac, vdp_extrema, vdp_auto
from polimi.envelope import RK45Envelope, BDFEnvelope, _envelope_system, _one_period
import sys
import os

def autonomous_vdp():
    epsilon = 0.001
    A = [0.,0.]
    T = [1.,1.]
    y0 = [2e-3,0]
    
    rtol = {'fun': 1e-8, 'env': 1e-4}
    atol = {'fun': 1e-10*np.ones(len(y0)), 'env': 1e-6}
    T_exact = 2*np.pi
    T_guess = 0.9 * T_exact

    fun = lambda t,y: vdp(t,y,epsilon,A,T)
    jac = lambda t,y: vdp_jac(t,y,epsilon)

    method = 'RK45'
    tend = 10000
    print('Integrating the full system...')
    if method == 'BDF':
        full = solve_ivp(fun, [0,tend+2*T_exact], y0, method, jac=jac, atol=atol['fun'], rtol=rtol['fun'])
    else:
        full = solve_ivp(fun, [0,tend+2*T_exact], y0, method, atol=atol['fun'], rtol=rtol['fun'])

    env_fun_1 = lambda t,y: _envelope_system(t, y, fun, T_exact, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
    print('Integrating the first envelope function...')
    var_step_1 = solve_ivp(env_fun_1, [0,tend], y0, method='BDF', atol=atol['env'], rtol=rtol['env'])

    env_fun_2 = lambda t,y: _one_period(t, y, fun, T_guess, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
    print('Integrating the second envelope function...')
    var_step_2 = solve_ivp(env_fun_2, [0,tend], y0, method='BDF', atol=atol['env'], rtol=rtol['env'])

    print('Integrating the envelope with Runge-Kutta 4,5...')
    rk = solve_ivp(fun, [0,tend], y0, method=RK45Envelope, T_guess=T_guess,
                   rtol=rtol['env'], atol=atol['env'],
                   fun_method='RK45', fun_rtol=rtol['fun'], fun_atol=atol['fun'])

    print('Integrating the envelope with BDF...')
    bdf = solve_ivp(fun, [0,tend], y0, method=BDFEnvelope, T_guess=T_guess,
                    rtol=rtol['env'], atol=atol['env'],
                    fun_method='RK45', fun_rtol=rtol['fun'], fun_atol=atol['fun'])
    
    plt.figure()
    plt.plot(full['t'],full['y'][0],'k',label='Full integration (%s)'%method)
    plt.plot(var_step_1['t'],var_step_1['y'][0],'go-',lw=2,label='Var. step (fixed T)')
    plt.plot(var_step_2['t'],var_step_2['y'][0],'ms-',lw=2,label='Var. step (estimated T)')
    plt.plot(rk['t'],rk['y'][0],'r^-',lw=2,label='RK45')
    for t0,y0 in zip(rk['t'],rk['y'].transpose()):
        sol = solve_ivp(fun, [t0,t0+T_exact], y0, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
        plt.plot(sol['t'],sol['y'][0],'r')
    plt.plot(bdf['t'],bdf['y'][0],'cv-',lw=2,label='BDF')
    for t0,y0 in zip(bdf['t'],bdf['y'].transpose()):
        sol = solve_ivp(fun, [t0,t0+T_exact], y0, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
        plt.plot(sol['t'],sol['y'][0],'c')
    plt.xlabel('Time (s)')
    plt.ylabel('x')
    plt.legend(loc='best')
    plt.show()

def forced_vdp():
    epsilon = 0.001
    T_exact = 10.
    T_guess = 0.9 * T_exact
    rtol = {'fun': 1e-8, 'env': 1e-3}
    atol = {'fun': 1e-10, 'env': 1e-3}

    polar_forcing = True
    method = 'RK45'

    T = [T_exact,1000.]
    A = [1.,5]
    #T = [T_exact]
    #A = [10]

    if not polar_forcing:
        y0 = [2e-3,0]
        fun = lambda t,y: vdp(t,y,epsilon,A,T)
        jac = lambda t,y: vdp_jac(t,y,epsilon)
        event_fun = lambda t,y: vdp_extrema(t,y,epsilon,A,T,0)
        event_fun.direction = -1  # detect maxima (derivative goes from positive to negative)
    else:
        y0 = [2e-3,0]
        for i in range(len(A)):
            y0.append(1.)
            y0.append(0.)
        fun = lambda t,y: vdp_auto(t,y,epsilon,A,T)
        method = 'RK45'

    t0 = 0
    ttran = 2000
    if ttran > 0:
        print('Integrating the full system (transient)...')
        if method == 'BDF':
            tran = solve_ivp(fun, [t0,ttran], y0, method='BDF', jac=jac, atol=atol['fun'],
                             rtol=rtol['fun'], events=event_fun)
        else:
            tran = solve_ivp(fun, [t0,ttran], y0, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
        t0 = tran['t'][-1]
        y0 = tran['y'][:,-1]
        #plt.plot(tran['t'],tran['y'][0],'k')
        #plt.plot(tran['t'],tran['y'][2],'r')
        #plt.plot(tran['t'],tran['y'][4],'g')
        #plt.show()

    print('t0 =',t0)
    print('y0 =',y0)

    print('Integrating the full system...')
    tend = 3000
    if method == 'BDF':
        full = solve_ivp(fun, [t0,tend], y0, method='BDF', jac=jac, atol=atol['fun'],
                         rtol=rtol['fun'], events=event_fun, dense_output=True)
    else:
        full = solve_ivp(fun, [t0,tend], y0, method='RK45', atol=atol['fun'], rtol=rtol['fun'])

    env_fun = lambda t,y: _envelope_system(t, y, fun, T_exact, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
    print('Integrating the envelope at variable step...')
    var_step = solve_ivp(env_fun, [t0,tend], y0, method='BDF', atol=atol['env'], rtol=rtol['env'])

    print('Integrating the envelope with BDF...')
    bdf = solve_ivp(fun, [t0,tend], y0, method=BDFEnvelope, T_guess=T_guess,
                    rtol=rtol['env'], atol=atol['env'], dTtol=5e-3,
                    fun_method='RK45', fun_rtol=rtol['fun'], fun_atol=atol['fun'])

    plt.figure()
    for i in range(2):
        if i == 0:
            ax = plt.subplot(2,1,i+1)
        else:
            plt.subplot(2,1,i+1,sharex=ax)

        plt.plot(full['t'],full['y'][i],'k',label='Full integration (%s)'%method)
        if method == 'BDF':
            plt.plot(full['t_events'][0],full['sol'](full['t_events'][0])[i],'gx')

        plt.plot(var_step['t'],var_step['y'][i],'go-',lw=2,label='Var. step (fixed T)')
        #for t0,y0 in zip(var_step['t'],var_step['y'].transpose()):
        #    sol = solve_ivp(fun, [t0,t0+T_exact], y0, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
        #    plt.plot(sol['t'],sol['y'][i],'g')

        plt.plot(bdf['t'],bdf['y'][i],'cv-',lw=2,label='BDF')
        for t0,y0 in zip(bdf['t'],bdf['y'].transpose()):
            sol = solve_ivp(fun, [t0,t0+T_exact], y0, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
            plt.plot(sol['t'],sol['y'][i],'c')

        if i == 1:
            plt.xlabel('Time (s)')
            plt.ylabel('y')
            plt.legend(loc='best')
        else:
            plt.ylabel('x')
    plt.show()


def main():
    if len(sys.argv) == 1:
        forced_vdp()
    elif sys.argv[1] == 'autonomous':
        autonomous_vdp()
    elif sys.argv[1] == 'forced':
        forced_vdp()
    else:
        print('{}: unknown method: `{}`.'.format(os.path.basename(sys.argv[0]),sys.argv[1]))

if __name__ == '__main__':
    main()
    
