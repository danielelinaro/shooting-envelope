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

# circuit parameters
T = 50e-6
Vref = 10
kp = 0.1
ki = 10
R = 6

F0 = 100
def Vin(t, Vin0=20, dVin=1, F=F0):
    return Vin0 + dVin * np.sin(2*np.pi*F*t)

# simulation parameters
fun = {'rtol': 1e-6, 'atol': 1e-8}
env = {'rtol': 1e-2, 'atol': 1e-4, 'max_step': 100, 'vars_to_use': [0,1]}
var = {'rtol': 1e-2, 'atol': 1e-4}


def init(t_tran=10*T, y0=np.array([Vin(0),1,0]), rtol=fun['rtol'], atol=fun['atol']):
    ckt = Buck(0, T=T, Vin=Vin, Vref=Vref, kp=kp, ki=ki, R=R, clock_phase=0)
    sol = solve_ivp_switch(ckt, [0,t_tran], y0, \
                           method='BDF', jac=ckt.jac, \
                           rtol=fun['rtol'], atol=fun['atol'])
    return ckt,sol


def print_state(y, msg=None):
    if msg is not None:
        print(msg)
    print('{:>10s} {:>13s}'.format('Variable','Value'))
    var_names = ['VC','IL','Int']
    for name,state in zip(var_names,y):
        print('{:>10s} = {:13.5e}'.format(name,state))


def tran(show_plot=True):
    ckt,tran = init()
    t0 = tran['t'][-1]
    y0 = tran['y'][:,-1]

    print_state(y0, 'Initial condition for transient analysis:')
    t_span = t0 + np.array([0, 10/F0])
    start = time.time()
    sol = solve_ivp_switch(ckt, t_span, y0, \
                           method='BDF', jac=ckt.jac, \
                           rtol=fun['rtol'], atol=fun['atol'])
    elapsed = time.time() - start
    print('Elapsed time: {:.2f} sec.'.format(elapsed))

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

    plt.savefig('buck_tran.pdf')

    if show_plot:
        plt.show()


def envelope(show_plot=True):
    ckt,tran = init()
    t0 = tran['t'][-1]
    y0 = tran['y'][:,-1]

    print_state(y0, 'Initial condition for envelope analysis:')

    t_span = t0 + np.array([0, 10/F0])

    env_solver = TrapEnvelope(ckt, t_span, y0, max_step=env['max_step'], \
                              T_guess=None, T=T, vars_to_use=env['vars_to_use'], \
                              env_rtol=env['rtol'], env_atol=env['atol'], \
                              solver=solve_ivp_switch, \
                              jac=ckt.jac, method='BDF', \
                              rtol=fun['rtol'], atol=fun['atol'])
    start = time.time()
    sol_env = env_solver.solve()
    elapsed = time.time() - start
    print('Elapsed time: {:.2f} sec.'.format(elapsed))

    for t0,y0 in zip(sol_env['t'],sol_env['y'].T):
        sol = solve_ivp_switch(ckt, [t0,t0+ckt.T], y0, method='BDF', \
                        jac=ckt.jac, rtol=fun['rtol'], atol=fun['atol'])
        try:
            envelope['t'] = np.append(envelope['t'], sol['t'])
            envelope['y'] = np.append(envelope['y'], sol['y'], axis=1)
        except:
            envelope = {key: sol[key] for key in ('t','y')}

    labels = [r'$V_C$ (V)', r'$I_L$ (A)']
    fig,ax = plt.subplots(2, 1, sharex=True, figsize=(6,4))
    for i in range(2):
        ax[i].plot(envelope['t']*1e6, envelope['y'][i], 'k', lw=1)
        ax[i].set_ylabel(labels[i])
    ax[1].set_xlabel(r'Time ($\mu$s)')
    ax[1].set_xlim(t_span*1e6)

    plt.savefig('buck_envelope.pdf')

    if show_plot:
        plt.show()


def variational(envelope, show_plot=True):

    if envelope:
        suffix = 'envelope'
    else:
        suffix = 'tran'

    ckt,tran = init(10*T)
    y0 = tran['y'][:,-1]

    print_state(y0, 'Initial condition for variational {} analysis:'.format(suffix))

    N = ckt.n_dim
    T_large = 1/F0
    T_small = ckt.T
    ckt.with_variational = True
    ckt.variational_T = T_large

    t_span_var = [0,1]
    y0_var = np.concatenate((y0,np.eye(N).flatten()))

    if envelope:
        env_solver = TrapEnvelope(ckt, [0,T_large], y0, T_guess=None, T=T_small, \
                                  env_rtol=env['rtol'], env_atol=env['atol'], \
                                  max_step=env['max_step'], vars_to_use=env['vars_to_use'], \
                                  is_variational=True, T_var_guess=None, T_var=None, \
                                  var_rtol=var['rtol'], var_atol=var['atol'], \
                                  solver=solve_ivp_switch, \
                                  rtol=fun['rtol'], atol=fun['atol'], method='BDF')
        now = time.time()
        sol = env_solver.solve()
    else:
        now = time.time()
        sol = solve_ivp_switch(ckt, t_span_var, y0_var, method='BDF', rtol=fun['rtol'], atol=fun['atol'])

    elapsed = time.time() - now
    print('Elapsed time: {:.2f} sec.'.format(elapsed))

    w,_ = np.linalg.eig(np.reshape(sol['y'][N:,-1],(N,N)))
    print('Eigenvalues:')
    for i in range(N):
        if np.imag(w[i]) < 0:
            sign = '-'
        else:
            sign = '+'
        print('   {:9.2e} {} j {:8.2e}'.format(np.real(w[i]),sign,np.abs(np.imag(w[i]))))

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

    plt.savefig('buck_variational_{}.pdf'.format(suffix))

    if show_plot:
        plt.show()


def shooting(envelope, show_plot=True):
    if envelope:
        suffix = 'envelope'
    else:
        suffix = 'tran'

    ckt,tran = init(10*T)
    y0 = tran['y'][:,-1]

    print_state(y0, 'Initial condition for shooting {} analysis:'.format(suffix))

    N = ckt.n_dim
    T_large = 1/F0
    T_small = ckt.T
    shoot_tol = 1e-3
    estimate_T = False

    if envelope:
        shoot = EnvelopeShooting(ckt, T_large, estimate_T, T_small, \
                                 tol=shoot_tol, env_solver=TrapEnvelope, \
                                 env_rtol=env['rtol'], env_atol=env['atol'], \
                                 env_max_step=env['max_step'], \
                                 env_vars_to_use=env['vars_to_use'], \
                                 var_rtol=var['rtol'], var_atol=var['atol'], \
                                 fun_solver=solve_ivp_switch, \
                                 rtol=fun['rtol'], atol=fun['atol'], \
                                 method='BDF', jac=ckt.jac)
    else:
        shoot = Shooting(ckt, T_large, estimate_T, tol=shoot_tol, \
                         solver=solve_ivp_switch, \
                         rtol=fun['rtol'], atol=fun['atol'], \
                         method='BDF')

    now = time.time()
    sol_shoot = shoot.run(y0)
    elapsed = time.time() - now
    print('Number of iterations: %d.' % sol_shoot['n_iter'])
    print('Elapsed time: %7.3f sec.' % elapsed)

    lw = 0.8
    fig,ax = plt.subplots(3,1,sharex=True,figsize=(6,6))

    for i,integr in enumerate(sol_shoot['integrations']):
        y0 = integr['y'][:N,0]
        for j in range(3):
            ax[j].plot(integr['t'], integr['y'][j], lw=lw, label='Iter #%d' % (i+1))
    ax[2].set_xlabel('Normalized time')
    ax[0].legend(loc='best')
    ax[0].set_ylabel(r'$V_C$ (V)')
    ax[1].set_ylabel(r'$I_L$ (A)')
    ax[2].set_ylabel(r'$\int V_o$ $(\mathrm{V}\cdot\mathrm{s})$')

    plt.savefig('buck_shooting_{}.pdf'.format(suffix))

    if show_plot:
        plt.show()


def run_all():
    tran(show_plot=False)
    envelope(show_plot=False)
    variational(envelope=False, show_plot=False)
    variational(envelope=True, show_plot=False)
    shooting(envelope=False, show_plot=False)
    shooting(envelope=True, show_plot=False)


cmds = {
    'tran': tran, \
    'envelope': envelope, \
    'variational': lambda: variational(False), \
    'variational-envelope': lambda: variational(True), \
    'shooting': lambda: shooting(False), \
    'shooting-envelope': lambda: shooting(True), \
    'all': run_all \
}

cmd_descriptions = {
    'tran': 'integrate the buck converter', \
    'envelope': 'compute the envelope of the buck converter', \
    'variational': 'integrate the buck converter and its variational part', \
    'variational-envelope': 'compute the envelope of the buck converter with variational part', \
    'shooting': 'perform a shooting analysis of the buck converter', \
    'shooting-envelope': 'perform a shooting analysis of the buck converter using the envelope', \
    'all': 'run all examples without showing plots'
}


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

