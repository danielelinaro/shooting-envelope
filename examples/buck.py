import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import argparse as arg

from polimi.switching import Buck, solve_ivp_switch
from polimi.envelope import BEEnvelope, TrapEnvelope
from polimi.shooting import Shooting, EnvelopeShooting

# for saving data
pack = lambda t,y: np.concatenate((np.reshape(t,(len(t),1)),y.transpose()),axis=1)

progname = os.path.basename(sys.argv[0])


F0 = 100
def Vin(t, Vin0=20, dVin=1, F=F0):
    return Vin0 + dVin * np.sin(2*np.pi*F*t)


def system():
    T = 50e-6
    Vref = 10
    kp = 0.1
    ki = 10
    R = 6

    t0 = 0
    t_end = 2000*T
    t_span = np.array([t0, t_end])

    y0 = np.array([Vin(0),1,0])

    fun_rtol = 1e-8
    fun_atol = 1e-8

    buck = Buck(0, T=T, Vin=Vin, Vref=Vref, kp=kp, ki=ki, R=R, clock_phase=0)

    print('Vector field index at the beginning of the integration: %d.' % buck.vector_field_index)
    sol = solve_ivp_switch(buck, t_span, y0, \
                           method='BDF', jac=buck.jac, \
                           rtol=fun_rtol, atol=fun_atol)
    print('Vector field index at the end of the integration: %d.' % buck.vector_field_index)

    show_manifold = True
    if show_manifold:
        n_rows = 4
    else:
        n_rows = 3

    fig,ax = plt.subplots(n_rows, 1, sharex=True, figsize=(6,6))

    ax[0].plot(sol['t']*1e6, sol['y'][0], 'k', lw=1)
    ax[0].set_ylabel(r'$V_C$ (V)')

    ax[1].plot(sol['t']*1e6, sol['y'][1], 'k', lw=1)
    ax[1].set_ylabel(r'$I_L$ (A)')

    ax[2].plot(sol['t']*1e6, sol['y'][2], 'k', lw=1)
    ax[2].set_ylabel(r'$\int V_O$ $(\mathrm{V}\cdot\mathrm{s})$')
    ax[2].set_xlim(t_span*1e6)

    if show_manifold:
        t = np.arange(sol['t'][0], sol['t'][-1], T/1000)
        ramp = (t % T) / T
        manifold = kp * (sol['y'][0] - Vref) + ki * sol['y'][2]
        manifold[manifold < 1e-3] = 1e-3
        manifold[manifold > 1 - 1e-3] = 1 - 1e-3
        ax[3].plot(t*1e6, ramp, 'm', lw=1, label=r'$V_{ramp}$')
        ax[3].plot(sol['t']*1e6, manifold, 'g', lw=1, label='Manifold')
        ax[3].plot([0, sol['t'][-1]*1e6], [0,0], 'b')
        ax[3].set_xlabel(r'Time ($\mu$s)')
        ax[3].legend(loc='best')
    else:
        ax[2].set_xlabel(r'Time ($\mu$s)')
    plt.show()


def envelope():
    T = 50e-6
    Vref = 10
    kp = 0.1
    ki = 10
    R = 6

    y0 = np.array([Vin(0),1,0])

    fun_rtol = 1e-10
    fun_atol = 1e-12

    buck = Buck(0, T=T, Vin=Vin, Vref=Vref, kp=kp, ki=ki, R=R, clock_phase=0)

    t_span = [0, 1/F0]
    t_tran = 10*T
    if t_tran > 0:
        sol = solve_ivp_switch(buck, [0,t_tran], y0, \
                               method='BDF', jac=buck.jac, \
                               rtol=fun_rtol, atol=fun_atol)
        #plt.plot(sol['t']*1e6,sol['y'][0],'k')
        #plt.plot(sol['t']*1e6,sol['y'][1],'r')
        #plt.show()
        t_span += sol['t'][-1]
        y0 = sol['y'][:,-1]

    print('t_span =', t_span)
    print('y0 =', y0)
    print('index =', buck.vector_field_index)

    print('-' * 81)
    be_solver = BEEnvelope(buck, t_span, y0, max_step=50, \
                           T_guess=None, T=T, vars_to_use=[0,1], \
                           env_rtol=1e-2, env_atol=1e-3, \
                           solver=solve_ivp_switch, \
                           jac=buck.jac, method='BDF', \
                           rtol=fun_rtol, atol=fun_atol)
    sol_be = be_solver.solve()
    print('-' * 81)
    trap_solver = TrapEnvelope(buck, t_span, y0, max_step=50, \
                               T_guess=None, T=T, vars_to_use=[0,1], \
                               env_rtol=1e-2, env_atol=1e-3, \
                               solver=solve_ivp_switch, \
                               jac=buck.jac, method='BDF', \
                               rtol=fun_rtol, atol=fun_atol)
    sol_trap = trap_solver.solve()
    print('-' * 81)

    sys.stdout.write('Integrating the original system... ')
    sys.stdout.flush()
    sol = solve_ivp_switch(buck, t_span, y0, method='BDF',
                           jac=buck.jac, rtol=fun_rtol, atol=fun_atol)
    sys.stdout.write('done.\n')

    labels = [r'$V_C$ (V)', r'$I_L$ (A)']
    fig,ax = plt.subplots(2,1,sharex=True)
    for i in range(2):
        ax[i].plot(sol['t']*1e6, sol['y'][i], 'k', lw=1)
        ax[i].plot(sol_be['t']*1e6, sol_be['y'][i], 'ro-', ms=3)
        ax[i].plot(sol_trap['t']*1e6, sol_trap['y'][i], 'go-', ms=3)
        ax[i].set_ylabel(labels[i])
    ax[1].set_xlabel(r'Time ($\mu$s)')
    ax[1].set_xlim(t_span*1e6)
    plt.show()


def variational_integration():
    T = 50e-6
    Vref = 10
    kp = 0.1
    ki = 10
    R = 6

    y0 = np.array([0,0,0])

    fun_rtol = 1e-10
    fun_atol = 1e-12

    buck = Buck(0, T=T, Vin=Vin, Vref=Vref, kp=kp, ki=ki, R=R, clock_phase=0)
    N = buck.n_dim
    
    fun_rtol = 1e-10
    fun_atol = 1e-12

    t_tran = 10*T

    if t_tran > 0:
        y0 = np.array([Vin(0),1,0])
        print('Vector field index at the beginning of the integration: %d.' % buck.vector_field_index)
        sol = solve_ivp_switch(buck, [0,t_tran], y0, \
                               method='BDF', jac=buck.jac, \
                               rtol=fun_rtol, atol=fun_atol)
        print('Vector field index at the end of the integration: %d.' % buck.vector_field_index)
        #fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
        #ax1.plot(sol['t']*1e6,sol['y'][0],'k')
        #ax1.set_ylabel(r'$V_C$ (V)')
        #ax2.plot(sol['t']*1e6,sol['y'][1],'r')
        #ax2.set_xlabel(r'Time ($\mu$s)')
        #ax2.set_ylabel(r'$I_L$ (A)')
        #plt.show()
        y0 = sol['y'][:,-1]
    else:
        y0 = np.array([10.01785173, 1.79660146, 0.04963066])

    print('y0 =', y0)
    T_large = 1/F0
    buck.with_variational = True
    buck.variational_T = T_large

    t_span_var = [0,1]
    y0_var = np.concatenate((y0,np.eye(N).flatten()))

    sol = solve_ivp_switch(buck, t_span_var, y0_var, method='BDF', rtol=fun_rtol, atol=fun_atol)

    w,v = np.linalg.eig(np.reshape(sol['y'][N:,-1],(N,N)))
    print('eigenvalues:')
    print('   ' + ' %14.5e' * N % tuple(w))
    print('eigenvectors:')
    for i in range(N):
        print('   ' + ' %14.5e' * N % tuple(v[i,:]))

    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter()
    formatter.set_powerlimits((-3,4))

    labels = [r'$V_C$ (V)', r'$I_L$ (A)', r'$\int V_o$ $(\mathrm{V}\cdot\mathrm{s})$']
    fig,ax = plt.subplots(3,4,sharex=True,figsize=(9,5))
    for i in range(3):
        ax[i,0].plot(sol['t'],sol['y'][i],'k',lw=1)
        ax[i,0].set_ylabel(labels[i])
        ax[i,0].set_xlim([0,1])
        for j in range(3):
            k = i*3 + j
            ax[i,j+1].plot(sol['t'],sol['y'][k+3],'k',lw=1,label='Python')
            ax[i,j+1].set_ylabel(r'$\Phi_{%d,%d}$' % (i+1,j+1))
            ax[i,j+1].set_xlim([0,1])
            ax[i,j+1].yaxis.set_major_formatter(formatter)
            pos = list(ax[i,j+1].get_position().bounds)
            pos[0] = 0.375 + j*0.21
            pos[2] = 0.125
            pos[3] *= 0.9
            ax[i,j+1].set_position(pos)
            ax[2,j+1].set_xlabel('Normalized time')
    ax[2,0].set_xlabel('Normalized time')

    plt.show()
    return v


def variational_envelope():
    T = 50e-6
    Vref = 10
    kp = 0.1
    ki = 10
    R = 6

    y0 = np.array([Vin(0),1,0])

    fun_rtol = 1e-6
    fun_atol = 1e-8

    buck = Buck(0, T=T, Vin=Vin, Vref=Vref, kp=kp, ki=ki, R=R, clock_phase=0)
    N = buck.n_dim

    t_tran = 10*T

    if t_tran > 0:
        y0 = np.array([Vin(0),1,0])
        print('Vector field index at the beginning of the integration: %d.' % buck.vector_field_index)
        sol = solve_ivp_switch(buck, [0,t_tran], y0, \
                               method='BDF', jac=buck.jac, \
                               rtol=fun_rtol, atol=fun_atol)
        print('Vector field index at the end of the integration: %d.' % buck.vector_field_index)
        #fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
        #ax1.plot(sol['t']*1e6,sol['y'][0],'k')
        #ax1.set_ylabel(r'$V_C$ (V)')
        #ax2.plot(sol['t']*1e6,sol['y'][1],'r')
        #ax2.set_xlabel(r'Time ($\mu$s)')
        #ax2.set_ylabel(r'$I_L$ (A)')
        #plt.show()
        y0 = sol['y'][:,-1]
    else:
        y0 = np.array([10.01785173, 1.79660146, 0.04963066])

    print('y0 =', y0)
    T_large = 1/F0
    T_small = T
    buck.with_variational = True
    buck.variational_T = T_large

    t_span_var = [0,1]
    y0_var = np.concatenate((y0,np.eye(len(y0)).flatten()))

    now = time.time()
    sol = solve_ivp_switch(buck, t_span_var, y0_var, method='BDF', rtol=fun_rtol, atol=fun_atol)
    elapsed_conv = time.time() - now

    rtol = 1e-2
    atol = 1e-4

    be_var_solver = BEEnvelope(buck, [0,T_large], y0, T_guess=None, T=T_small, \
                               env_rtol=rtol, env_atol=atol, max_step=500, vars_to_use=[0,1], \
                               is_variational=True, T_var_guess=None, T_var=None, \
                               var_rtol=rtol, var_atol=atol, solver=solve_ivp_switch, \
                               rtol=fun_rtol, atol=fun_atol, method='BDF')
    trap_var_solver = TrapEnvelope(buck, [0,T_large], y0, T_guess=None, T=T_small, \
                                   env_rtol=rtol, env_atol=atol, max_step=500, vars_to_use=[0,1], \
                                   is_variational=True, T_var_guess=None, T_var=None, \
                                   var_rtol=rtol, var_atol=atol, solver=solve_ivp_switch, \
                                   rtol=fun_rtol, atol=fun_atol, method='BDF')
    print('-' * 100)
    now = time.time()
    var_sol_be = be_var_solver.solve()
    elapsed_be = time.time() - now
    print('-' * 100)
    now = time.time()
    var_sol_trap = trap_var_solver.solve()
    elapsed_trap = time.time() - now
    print('-' * 100)

    eig,_ = np.linalg.eig(np.reshape(sol['y'][N:,-1],(N,N)))
    print('         correct eigenvalues:', eig)
    eig,_ = np.linalg.eig(np.reshape(var_sol_be['y'][N:,-1],(N,N)))
    print('  BE approximate eigenvalues:', eig)
    eig,_ = np.linalg.eig(np.reshape(var_sol_trap['y'][N:,-1],(N,N)))
    print('TRAP approximate eigenvalues:', eig)

    print('Elapsed time:')
    print('  conventional: {:7.3f} sec.'.format(elapsed_conv))
    print('            BE: {:7.3f} sec.'.format(elapsed_be))
    print('          TRAP: {:7.3f} sec.'.format(elapsed_trap))

    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter()
    formatter.set_powerlimits((-3,4))

    labels = [r'$V_C$ (V)', r'$I_L$ (A)', r'$\int V_o$ $(\mathrm{V}\cdot\mathrm{s})$']
    fig,ax = plt.subplots(3,4,sharex=True,figsize=(9,5))
    for i in range(3):
        ax[i,0].plot(sol['t'],sol['y'][i],'k',lw=1)
        ax[i,0].plot(var_sol_be['t'],var_sol_be['y'][i],'rs-',ms=3)
        ax[i,0].plot(var_sol_trap['t'],var_sol_trap['y'][i],'go-',ms=3)
        ax[i,0].set_ylabel(labels[i])
        ax[i,0].set_xlim([0,1])
        for j in range(3):
            k = i*3 + j
            ax[i,j+1].plot(sol['t'],sol['y'][k+3],'k',lw=1,label='Python')
            ax[i,j+1].plot(var_sol_be['t'],var_sol_be['y'][k+3],'rs',ms=3)
            ax[i,j+1].plot(var_sol_trap['t'],var_sol_trap['y'][k+3],'go',ms=3)
            ax[i,j+1].set_ylabel(r'$\Phi_{%d,%d}$' % (i+1,j+1))
            ax[i,j+1].set_xlim([0,1])
            ax[i,j+1].yaxis.set_major_formatter(formatter)
            pos = list(ax[i,j+1].get_position().bounds)
            pos[0] = 0.375 + j*0.21
            pos[2] = 0.125
            pos[3] *= 0.9
            ax[i,j+1].set_position(pos)
            ax[2,j+1].set_xlabel('Normalized time')
    ax[2,0].set_xlabel('Normalized time')

    plt.show()


def shooting():
    T = 50e-7
    Vref = 10
    kp = 0.1
    ki = 10
    R = 6

    fun_rtol = 1e-6
    fun_atol = 1e-8

    buck = Buck(0, T=T, Vin=Vin, Vref=Vref, kp=kp, ki=ki, R=R, clock_phase=0)
    N = buck.n_dim

    y0_guess = np.array([Vin(0),1,0])

    t_tran = 10*T

    if t_tran > 0:
        tran = solve_ivp_switch(buck, [0,t_tran], y0_guess, method='BDF', \
                                jac=buck.jac, rtol=fun_rtol, atol=fun_atol)
        y0_guess = tran['y'][:,-1]
        #fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
        #ax1.plot(tran['t']/T,tran['y'][0],'k')
        #ax1.set_ylabel(r'$V_C$ (V)')
        #ax2.plot(tran['t']/T,tran['y'][1],'k')
        #ax2.set_xlabel('No. of periods')
        #ax2.set_ylabel(r'$I_L$ (A)')
        #plt.show()

    T_large = 1/F0
    T_small = T

    estimate_T = False

    shoot = Shooting(buck, T_large, estimate_T, tol=1e-3, \
                     solver=solve_ivp_switch, \
                     rtol=fun_rtol, atol=fun_atol, \
                     method='BDF')

    now = time.time()
    sol_shoot = shoot.run(y0_guess)
    elapsed = time.time() - now
    print('Number of iterations: %d.' % sol_shoot['n_iter'])
    print('Elapsed time: %7.3f sec.' % elapsed)

    col = 'krgbcmy'
    lw = 0.8
    fig,ax = plt.subplots(3,1,sharex=True,figsize=(6,6))

    for i,integr in enumerate(sol_shoot['integrations']):

        y0 = integr['y'][:N,0]
        y0_var = np.concatenate((y0,np.eye(N).flatten()))

        for j in range(3):
            #ax[j].plot(sol['t'],sol['y'][j],col[i],lw=lw,label='Iter #%d' % (i+1))
            ax[j].plot(integr['t'],integr['y'][j],col[i],lw=lw,label='Iter #%d' % (i+1))
    ax[2].set_xlabel('Normalized time')
    ax[0].legend(loc='best')
    ax[0].set_ylabel(r'$V_C$ (V)')
    ax[1].set_ylabel(r'$I_L$ (A)')
    ax[2].set_ylabel(r'$\int V_o$ $(\mathrm{V}\cdot\mathrm{s})$')
    plt.savefig('buck_shooting.pdf')
    plt.show()


def shooting_envelope():
    T = 50e-7
    Vref = 10
    kp = 0.1
    ki = 10
    R = 6

    fun_rtol = 1e-6
    fun_atol = 1e-8

    buck = Buck(0, T=T, Vin=Vin, Vref=Vref, kp=kp, ki=ki, R=R, clock_phase=0)
    N = buck.n_dim

    y0_guess = np.array([Vin(0),1,0])

    t_tran = 10*T

    if t_tran > 0:
        tran = solve_ivp_switch(buck, [0,t_tran], y0_guess, method='BDF', \
                                jac=buck.jac, rtol=fun_rtol, atol=fun_atol)
        y0_guess = tran['y'][:,-1]
        #fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
        #ax1.plot(tran['t']/T,tran['y'][0],'k')
        #ax1.set_ylabel(r'$V_C$ (V)')
        #ax2.plot(tran['t']/T,tran['y'][1],'k')
        #ax2.set_xlabel('No. of periods')
        #ax2.set_ylabel(r'$I_L$ (A)')
        #plt.show()

    T_large = 1/F0
    T_small = T

    estimate_T = False

    shoot = EnvelopeShooting(buck, T_large, estimate_T, T_small, \
                             tol=1e-3, env_solver=TrapEnvelope, \
                             env_rtol=1e-3, env_atol=1e-4, \
                             env_max_step=150, env_vars_to_use=[0,1], \
                             var_rtol=1e-1, var_atol=1e-2, \
                             fun_solver=solve_ivp_switch, \
                             rtol=fun_rtol, atol=fun_atol, \
                             method='BDF', jac=buck.jac)
    now = time.time()
    sol_shoot = shoot.run(y0_guess)
    elapsed = time.time() - now
    print('Number of iterations: %d.' % sol_shoot['n_iter'])
    print('Elapsed time: %7.3f sec.' % elapsed)

    t_span_var = [0,1]
    buck.with_variational = True
    buck.variational_T = T_large

    col = 'krgbcmy'
    lw = 0.8
    fig,ax = plt.subplots(3,1,sharex=True,figsize=(6,6))

    for i,integr in enumerate(sol_shoot['integrations']):

        y0 = integr['y'][:N,0]
        y0_var = np.concatenate((y0,np.eye(N).flatten()))
        sol = solve_ivp_switch(buck, t_span_var, y0_var, method='BDF', rtol=fun_rtol, atol=fun_atol)

        for j in range(3):
            ax[j].plot(sol['t'],sol['y'][j],col[i],lw=lw,label='Iter #%d' % (i+1))
            ax[j].plot(integr['t'],integr['y'][j],col[i]+'o-',lw=1,ms=3)
    ax[2].set_xlabel('Normalized time')
    ax[0].legend(loc='best')
    ax[0].set_ylabel(r'$V_C$ (V)')
    ax[1].set_ylabel(r'$I_L$ (A)')
    ax[2].set_ylabel(r'$\int V_o$ $(\mathrm{V}\cdot\mathrm{s})$')
    plt.savefig('buck_shooting_envelope.pdf')
    plt.show()


cmds = {'system': system, \
        'envelope': envelope, \
        'variational': variational_integration, \
        'variational-envelope': variational_envelope, \
        'shooting': shooting, \
        'shooting-envelope': shooting_envelope}

cmd_descriptions = {'system': 'integrate the buck converter', \
                    'envelope': 'compute the envelope of the buck converter', \
                    'variational': 'integrate the buck converter and its variational part', \
                    'variational-envelope': 'compute the envelope of the buck converter with variational part', \
                    'shooting': 'perform a shooting analysis of the buck converter', \
                    'shooting-envelope': 'perform a shooting analysis of the buck converter using the envelope'}


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
    print('usage: {} command'.format(progname))
    list_commands()


if __name__ == '__main__':

    if len(sys.argv) != 2:
        usage()
        sys.exit(1)

    if sys.argv[1] in ('-h', '--help', 'help'):
        usage()
        sys.exit(0)

    cmd = sys.argv[1]

    if not cmd in cmds:
        print('{}: {}: unknown command.'.format(progname, cmd))
        list_commands()
        sys.exit(1)

    cmds[cmd]()

