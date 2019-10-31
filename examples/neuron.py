
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import argparse as arg

from polimi.systems import Neuron7, Neuron4
from polimi.envelope import BEEnvelope, TrapEnvelope
from polimi.shooting import EnvelopeShooting


progname = os.path.basename(sys.argv[0])


def system_7(imposed_paths):
    if imposed_paths:
        kwargs = {'Cac': 0.15,
                  'Nac': 5.85,
                  'd': 1}
        Ca0 = 0
        Na0 = kwargs['Nac']
    else:
        kwargs = {}
        Ca0 = 0
        Na0 = 5.85

    neuron = Neuron7(imposed_paths, **kwargs)
    v0 = -80
    n0,m0,h0,s0 = neuron.compute_ss(v0)
    #      v, n, m, h, s, ca, na
    y0 = [v0,n0,m0,h0,s0,Ca0,Na0]
    print('y0 = ', y0)
    print('ydot = ', neuron(0,y0))
    atol = 1e-4
    rtol = 1e-6
    tend = 5000
    sol = solve_ivp(neuron, [0,tend], y0, atol=atol, rtol=rtol)
    
    fig,ax = plt.subplots(2,2)

    ax[0,0].plot(sol['t'],sol['y'][0],'k')
    ax[0,0].set_xlabel('Time')
    ax[0,0].set_ylabel('V (mV)')

    ax[0,1].plot(sol['t'], sol['y'][1], 'k', label='n')
    ax[0,1].plot(sol['t'], sol['y'][2], 'r', label='m')
    ax[0,1].plot(sol['t'], sol['y'][3], 'g', label='h')
    ax[0,1].plot(sol['t'], sol['y'][4], 'b', label='s')
    ax[0,1].set_xlabel('Time')
    ax[0,1].legend(loc='best')

    ax[1,0].plot(sol['t'],sol['y'][5],'k',label='Ca')
    ax[1,0].plot(sol['t'],sol['y'][6],'r',label='Na')
    ax[1,0].set_xlabel('Time')
    ax[1,0].legend(loc='best')

    ax[1,1].plot(sol['y'][5],sol['y'][6],'k')
    ax[1,1].plot(sol['y'][5,0],sol['y'][6,0],'go',markerfacecolor='w',markersize=4)
    ax[1,1].plot(sol['y'][5,-1],sol['y'][6,-1],'ro',markerfacecolor='w',markersize=4)
    ax[1,1].set_xlabel('Ca')
    ax[1,1].set_ylabel('Na')

    plt.show()

    
def system_4(imposed_paths):
    if imposed_paths:
        kwargs = {'epsilon': 0.002,
                  'Cac': 0.15,
                  'Nac': 5.85,
                  'd': 0.1}
        Ca0 = 0
        Na0 = kwargs['Nac']
    else:
        kwargs = {'epsilon': 0.002}
        Ca0 = 0
        Na0 = 5.85

    neuron = Neuron4(imposed_paths, **kwargs)
    
    v0 = -80
    n0,_,_,_ = neuron.compute_ss(v0)
    #      v, n, ca, na
    y0 = [v0,n0,Ca0,Na0]
    print('y0 = ', y0)
    print('ydot = ', neuron(0,y0))
    atol = 1e-6
    rtol = 1e-8
    tend = 5000
    sol = solve_ivp(neuron, [0,tend], y0, atol=atol, rtol=rtol)
    pks,props = find_peaks(sol['y'][0])
    isi = np.diff(sol['t'][pks])
    isi_var = (np.abs(isi[1:] - isi[:-1]) / isi[:-1]) * 100

    fig,ax = plt.subplots(2, 2, figsize=(8,7), sharex=True)

    ax[0,0].plot(sol['t'], sol['y'][0], 'k', lw=0.5)
    ax[0,0].plot(sol['t'][pks], sol['y'][0,pks], 'ro', \
                 markerfacecolor='w', markersize=2, lw=0.5)
    ax[0,0].set_xlabel('Time')
    ax[0,0].set_ylabel('V (mV)')

    ax[0,1].plot(sol['t'], sol['y'][1], 'k', lw=0.5)
    ax[0,1].set_xlabel('Time')
    ax[0,1].set_ylabel('n')

    #ax[1,0].plot(sol['t'], sol['y'][2], 'k', label='Ca', lw=1)
    #ax[1,0].plot(sol['t'], sol['y'][3], 'r', label='Na', lw=1)
    ax[1,0].plot(sol['t'][pks][1:], isi, 'k-o', label='ISI', \
                 markerfacecolor='w', markersize=3, lw=0.75)
    ax[1,0].plot(sol['t'][pks][2:], isi_var, 'r-o', label='ISI variation', \
                 markerfacecolor='w', markersize=3, lw=0.75)
    ax[1,0].set_xlabel('Time')
    ax[1,0].legend(loc='best')

    ax[1,1].plot(sol['y'][2], sol['y'][3], 'k')
    ax[1,1].plot(sol['y'][2,0], sol['y'][3,0], 'go', \
                 markerfacecolor='w', markersize=4)
    ax[1,1].plot(sol['y'][2,-1], sol['y'][3,-1], 'ro', \
                 markerfacecolor='w', markersize=4)
    ax[1,1].set_xlabel('Ca')
    ax[1,1].set_ylabel('Na')

    plt.show()


def envelope(imposed_paths):
    if imposed_paths:
        kwargs = {'epsilon': 0.002,
                  'Cac': 0.15,
                  'Nac': 5.85,
                  'd': 0.1}
        Ca0 = 0
        Na0 = kwargs['Nac']
    else:
        kwargs = {'epsilon': 0.002}
        Ca0 = 0
        Na0 = 5.85

    neuron = Neuron4(imposed_paths, **kwargs)

    atol = 1e-6
    rtol = 1e-8

    y0 = np.array([4.51773484, 0.0356291 , 0.03012965, 4.94827389])
    if y0 is None:
        v0 = -80
        n0,_,_,_ = neuron.compute_ss(v0)
        #      v, n, ca, na
        y0 = [v0,n0,Ca0,Na0]
        tend = 7000
        sol = solve_ivp(neuron, [0,tend], y0, atol=atol, rtol=rtol)
        t = sol['t']
        v = sol['y'][0]
        plt.ion()
        plt.plot(t, v, 'k', lw=0.5)
        plt.show()
        idx, = np.where((t > 3000) & (t < 3500))
        jdx = np.where(v[idx] > 0)[0][0]
        y0 = sol['y'][:,idx[jdx]]

    print('  y0 =', y0)
    print('ydot =', neuron(0, y0))
    env_rtol = 1e-1
    env_atol = 1e-1
    t_span = [0,200]

    be_env_solver = BEEnvelope(neuron, t_span, y0, T=None, T_guess=30, \
                               vars_to_use=[0,1], dT_tol=0.15, \
                               env_rtol=env_rtol, env_atol=env_atol, \
                               rtol=rtol, atol=atol)
    be_env_sol = be_env_solver.solve()

    trap_env_solver = TrapEnvelope(neuron, t_span, y0, T=None, T_guess=30, \
                                   vars_to_use=[0,1], \
                                   env_rtol=env_rtol, env_atol=env_atol, \
                                   rtol=rtol, atol=atol)
    trap_env_sol = trap_env_solver.solve()

    sol = solve_ivp(neuron, t_span, y0, rtol=rtol, atol=atol)

    fig,(ax1,ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(sol['t'], sol['y'][0], 'k', lw=0.75)
    ax1.plot(be_env_sol['t'], be_env_sol['y'][0], 'ro-', lw=1)
    ax1.plot(trap_env_sol['t'], trap_env_sol['y'][0], 'gs-', lw=1)
    ax1.set_ylabel(r'$V$ (mV)')
    ax2.plot(sol['t'], sol['y'][2], 'k', lw=0.75)
    ax2.plot(be_env_sol['t'], be_env_sol['y'][2], 'ro-', lw=1)
    ax2.plot(trap_env_sol['t'], trap_env_sol['y'][2], 'gs-', lw=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'$Ca$')
    plt.show()



cmds = {'system-4': system_4, 'system-7': system_7, 'envelope': envelope}


cmd_descriptions = {'system-4': 'integrate the 4D neuron model', \
                    'system-7': 'integrate the 7D neuron model', \
                    'envelope': 'compute the envelope of the 4D neuron model'}


def list_commands():
    print('\nThe following are accepted commands:')
    nch = 0
    for cmd in cmds:
        if len(cmd) > nch:
            nch = len(cmd)
    fmt = '\t{:<%ds} {}' % (nch + 5)
    for i,cmd in enumerate(cmds):
        print(fmt.format(cmd,cmd_descriptions[cmd]))


def usage():
    print('usage: {} [--imposed-paths] command'.format(progname))
    list_commands()


if __name__ == '__main__':

    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    if sys.argv[1] in ('-h', '--help', 'help'):
        usage()
        sys.exit(0)

    if sys.argv[1] == '--imposed-paths':
        imposed_paths = True
        if len(sys.argv) != 3:
            usage()
            sys.exit(1)
        cmd = sys.argv[2]
    else:
        imposed_paths = False
        cmd = sys.argv[1]

    if not cmd in cmds:
        print('{}: {}: unknown command.'.format(progname, cmd))
        list_commands()
        sys.exit(1)

    cmds[cmd](imposed_paths)
