import os
import sys
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import argparse as arg

from polimi.systems import ASK_OOK
from polimi.envelope import BEEnvelope, TrapEnvelope
from polimi.shooting import Shooting, EnvelopeShooting

# for saving data
pack = lambda t,y: np.concatenate((np.reshape(t,(len(t),1)),y.transpose()),axis=1)

progname = os.path.basename(sys.argv[0])

F1 = 1e6
F2 = 2e9
fun = {'atol': 1e-6, 'rtol': 1e-8}
env = {'atol': 1e-1, 'rtol': 1e-2, 'max_step': 50, 'vars_to_use': [2,3,5,7]}
var = {'atol': 1e-1, 'rtol': 1e-2}


def init(t_tran=0, atol=fun['atol'], rtol=fun['rtol']):
    from scipy.optimize import fsolve

    ckt = ASK_OOK()

    y0_guess = np.array([
        0.,      # ILtl
        0.0001,  # IL1
        0.00005, # IL2
        0,       # out
        0.8,     # l20
        0.8,     # l30
        0.8,     # d10
        0.8      # d20
    ])

    y0 = fsolve(lambda y: ckt(0,y), y0_guess)

    if t_tran > 0:
        sol = solve_ivp(ckt, [0,t_tran], y0, method='BDF', \
                        jac=ckt.jac, atol=atol, rtol=rtol)
        return ckt,sol

    return ckt,y0


def print_state(y, msg=None):
    if msg is not None:
        print(msg)
    print('{:>10s} {:>13s}'.format('Variable','Value'))
    var_names = ['ILtl', 'IL1', 'IL2', 'out', 'l20', 'l30', 'd10', 'd20']
    for name,state in zip(var_names,y):
        print('{:>10s} = {:13.5e}'.format(name,state))


def dump_data(outfile, **kwargs):
    data = {'F1': F1, 'F2': F2}
    for k,v in kwargs.items():
        data[k] = v
    pickle.dump(data, open(outfile, 'wb'))


def make_full_envelope_solution(ckt, sol_env, T):
    for t0,y0 in zip(sol_env['t'],sol_env['y'].T):
        if ckt.with_variational:
            sol = solve_ivp(ckt, [t0,t0+T], y0, method='BDF', \
                            rtol=fun['rtol'], atol=fun['atol'])
        else:
            sol = solve_ivp(ckt, [t0,t0+T], y0, method='BDF', \
                            jac=ckt.jac, rtol=fun['rtol'], atol=fun['atol'])
        try:
            envelope['t'] = np.append(envelope['t'], sol['t'])
            envelope['y'] = np.append(envelope['y'], sol['y'], axis=1)
        except:
            envelope = {key: sol[key] for key in ('t','y')}
    return envelope


def tran(show_plot=True):
    y0_correct = np.array([
        0.,          # ILtl
        0.000162828, # IL1
        0.000162828, # IL2
        0,           # out
        0.904516,    # l20
        0.904516,    # l30
        0.906144,    # d10
        0.902887     # d20
    ])

    ckt,y0 = init()

    print('Initial condition for transient analysis:')
    print('{:>10s} {:>13s} {:>13s}'.format('Variable','Computed','PAN'))
    var_names = ['ILtl', 'IL1', 'IL2', 'out', 'l20', 'l30', 'd10', 'd20']
    for name,py,pan in zip(var_names,y0,y0_correct):
        print('{:>10s} = {:13.5e} {:13.5e}'.format(name,py,pan))

    t_span = [0, ckt.T1]
    start = time.time()
    sol = solve_ivp(ckt, t_span, y0, method='BDF', \
                    jac=ckt.jac, atol=fun['atol'], rtol=fun['rtol'])
    elapsed = time.time() - start
    print('Elapsed time: {:.2f} sec.'.format(elapsed))
    dump_data('ASK_OOK_tran.pkl', sol=sol, t0=0, y0=y0, \
              elapsed_time=elapsed, sys_pars=fun, t_span=t_span)

    fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,figsize=(10,4))
    labels = ['i(Ltl)', 'i(L1)', 'i(L2)', \
              'v(out)', 'v(l20)', 'v(l30)', 'v(d10)', 'v(d20)']
    colors = 'krgb'
    for i in range(3):
        ax1.plot(sol['t']*1e9, sol['y'][i], colors[i], lw=1, label=labels[i])
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Current (A)')
    ax1.legend(loc='best')
    for i in range(4):
        ax2.plot(sol['t']*1e9, sol['y'][i+3], colors[i], lw=1, label=labels[i+3])
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Voltage (V)')
    ax2.legend(loc='best')

    plt.savefig('ASK_OOK_tran.pdf')

    if show_plot:
        plt.show()


def envelope(show_plot=True):

    ckt,y0 = init()

    print_state(y0, 'Initial condition for envelope analysis:')

    t_span = [0, ckt.T1]
    env_solver = TrapEnvelope(ckt, t_span, y0, max_step=50, \
                              T_guess=None, T=ckt.T2, vars_to_use=env['vars_to_use'], \
                              env_rtol=env['rtol'], env_atol=env['atol'], \
                              solver=solve_ivp, \
                              jac=ckt.jac, method='BDF', \
                              rtol=fun['rtol'], atol=fun['atol'])
    start = time.time()
    sol_env = env_solver.solve()
    elapsed = time.time() - start
    print('Elapsed time: {:.2f} sec.'.format(elapsed))

    full_sol_env = make_full_envelope_solution(ckt, sol_env, ckt.T2)

    dump_data('ASK_OOK_envelope.pkl', sol=sol_env, full_sol=full_sol_env, \
              t0=0, y0=y0, elapsed_time=elapsed, sys_pars=fun, \
              env_pars=env, t_span=t_span)

    fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,figsize=(10,4))
    labels = ['i(Ltl)', 'i(L1)', 'i(L2)', \
              'v(out)', 'v(l20)', 'v(l30)', 'v(d10)', 'v(d20)']
    colors = 'krgb'
    for i in range(2):
        ax1.plot(full_sol_env['t']*1e9, full_sol_env['y'][i], colors[i], lw=1, label=labels[i])
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Current (A)')
    ax1.legend(loc='best')
    for i in range(2):
        ax2.plot(full_sol_env['t']*1e9, full_sol_env['y'][i+3], colors[i], lw=1, label=labels[i+3])
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Voltage (V)')
    ax2.legend(loc='best')

    plt.savefig('ASK_OOK_envelope.pdf')

    if show_plot:
        plt.show()


def variational(envelope, show_plot=True):
    
    ckt,y0 = init()

    if envelope:
        suffix = 'envelope'
    else:
        suffix = 'tran'
    print_state(y0, 'Initial condition for variational {} analysis:'.format(suffix))

    N = ckt.n_dim
    T_large = ckt.T1
    T_small = ckt.T2
    ckt.with_variational = True
    ckt.variational_T = T_large

    t_span_var = [0,1]
    y0_var = np.concatenate((y0,np.eye(N).flatten()))

    pkl_file = 'ASK_OOK_variational_{}.pkl'.format(suffix)
    pdf_file = 'ASK_OOK_variational_{}.pdf'.format(suffix)

    if os.path.isfile(pkl_file):
        data = pickle.load(open(pkl_file, 'rb'))
        sol = data['sol']
        if 'full_sol' in data:
            full_sol = data['full_sol']
    else:
        if envelope:
            solver = TrapEnvelope(ckt, [0,T_large], y0, T_guess=None, T=T_small, \
                                  env_rtol=env['rtol'], env_atol=env['atol'], max_step=env['max_step'], \
                                  is_variational=True, T_var_guess=None, T_var=None, \
                                  var_rtol=var['rtol'], var_atol=var['atol'], solver=solve_ivp, \
                                  rtol=fun['rtol'], atol=fun['atol'], method='BDF')
            solver.verbose = False
            start = time.time()
            sol = solver.solve()
        else:
            start = time.time()
            sol = solve_ivp(ckt, t_span_var, y0_var, method='BDF', rtol=fun['rtol'], atol=fun['atol'])
        elapsed = time.time() - start
        print('Elapsed time: {:.2f} sec.'.format(elapsed))
        pickle.dump(sol, open(pkl_file, 'wb'))

    w,_ = np.linalg.eig(np.reshape(sol['y'][N:,-1],(N,N)))
    print('Eigenvalues:')
    for i in range(N):
        if np.imag(w[i]) < 0:
            sign = '-'
        else:
            sign = '+'
        print('   {:9.2e} {} j {:8.2e}'.format(np.real(w[i]),sign,np.abs(np.imag(w[i]))))

    if not os.path.isfile(pkl_file):
        if envelope:
            full_sol = make_full_envelope_solution(ckt, sol, T_small/T_large)
            dump_data('ASK_OOK_variational_envelope.pkl', sol=sol, full_sol=full_sol, \
                      t0=0, y0=y0, elapsed_time=elapsed, sys_pars=fun, \
                      env_pars=env, var_pars=var, T_large=T_large, T_small=T_small)
            sol = full_sol
        else:
            dump_data('ASK_OOK_variational_tran.pkl', sol=sol, \
                      t0=0, y0=y0, elapsed_time=elapsed, sys_pars=fun, \
                      T_large=T_large, T_small=T_small)
    elif envelope:
        sol = full_sol

    fig,ax = plt.subplots(9,8,sharex=True,figsize=(9,5))
    for i in range(9):
        for j in range(8):
            k = i*8 + j
            ax[i,j].plot(sol['t'], sol['y'][k], 'k', lw=0.5)
            ax[i,j].plot([0,1],[0,0],'--',color=[.6,.6,.6],lw=1)
            ax[i,j].set_xlim([0,1])
            ax[i,j].set_xticks([0,1])
            ax[i,j].set_xticklabels([])
            ax[i,j].set_yticks(ax[i,j].get_ylim())
            ax[i,j].set_yticklabels([])

    plt.savefig(pdf_file)

    if show_plot:
        plt.show()


def shooting(envelope, show_plot=True):

    ckt,y0 = init()

    print_state(y0, 'Initial condition for shooting analysis:')

    T_large = ckt.T1
    T_small = ckt.T2

    shoot_tol = 1e-2
    estimate_T = False

    if envelope:
        suffix = 'envelope'
        shoot = EnvelopeShooting(ckt, T_large, estimate_T, T_small, \
                                 tol=shoot_tol, env_solver=TrapEnvelope, \
                                 env_rtol=env['rtol'], env_atol=env['atol'], \
                                 env_max_step=env['max_step'], env_vars_to_use=env['vars_to_use'], \
                                 var_rtol=var['rtol'], var_atol=var['atol'], \
                                 fun_solver=solve_ivp, \
                                 rtol=fun['rtol'], atol=fun['atol'], \
                                 method='BDF', jac=ckt.jac)
        shoot.verbose = False
    else:
        suffix = 'tran'
        shoot = Shooting(ckt, T_large, estimate_T, tol=shoot_tol, \
                         solver=solve_ivp, rtol=fun['rtol'], atol=fun['atol'], \
                         method='BDF')

    pkl_file = 'ASK_OOK_shooting_{}.pkl'.format(suffix)
    pdf_file = 'ASK_OOK_shooting_{}.pdf'.format(suffix)

    if os.path.isfile(pkl_file):
        data = pickle.load(open(outfile,'rb'))
        sol = data['sol']
    else:
        now = time.time()
        sol = shoot.run(y0)
        elapsed = time.time() - now
        print('Number of iterations: {}.'.format(sol['n_iter']))
        print('Elapsed time: {:.2f} sec.'.format(elapsed))

    if envelope:
        ckt.with_variational = True
        full_sol = []
        for i in range(len(sol['integrations'])):
            sol['integrations'][i]['full_sol'] = make_full_envelope_solution(ckt, sol['integrations'][i], T_small/T_large)
            full_sol.append(sol['integrations'][i]['full_sol'])
        dump_data(pkl_file, sol=sol, full_sol=full_sol, \
                  t0=0, y0=y0, elapsed_time=elapsed, sys_pars=fun, \
                  env_pars=env, shoot_tol=shoot_tol, var_pars=var, \
                  T_large=T_large, T_small=T_small)
        sol['integrations'] = full_sol
    else:
        dump_data(pkl_file, sol=sol, \
                  t0=0, y0=y0, elapsed_time=elapsed, sys_pars=fun, \
                  shoot_tol=shoot_tol, T_large=T_large, T_small=T_small)

    jdx = [0,3]
    fig,ax = plt.subplots(1, len(jdx), sharex=True, figsize=(10,4))
    labels = ['i(Ltl)', 'i(L1)', 'i(L2)', \
              'v(out)', 'v(l20)', 'v(l30)', 'v(d10)', 'v(d20)']
    colors = 'krgbcmy'
    for i,sol in enumerate(sol['integrations']):
        for n,j in enumerate(jdx):
            ax[n].plot(sol['t'], sol['y'][j], colors[i], lw=1, label=labels[j])
            ax[n].set_xlabel('Time (ns)')
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')

    plt.savefig(pdf_file)

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
    'variational': lambda : variational(False), \
    'variational-envelope': lambda: variational(True), \
    'shooting': lambda: shooting(False), \
    'shooting-envelope': lambda: shooting(True), \
    'all': run_all
}

cmd_descriptions = {
    'tran': 'integrate the ASK/OOK RF modulator', \
    'envelope': 'compute the envelope of the ASK/OOK RF modulator', \
    'variational': 'integrate the ASK/OOK modulator and its variational part', \
    'variational-envelope': 'compute the envelope of the ASK/OOK modulator with variational part', \
    'shooting': 'perform a shooting analysis of the ASK/OOK RF modulator', \
    'shooting-envelope': 'perform a shooting analysis of the ASK/OOK RF modulator using the envelope', \
    'all': 'run all examples without showing plots'
}


def list_commands():
    print('\nThe following are accepted commands:')
    nch = 0
    for cmd in cmds:
        if len(cmd) > nch:
            nch = len(cmd)
    fmt = '   {:<%ds} {}' % (nch + 5)
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
        print("{}: '{}' is not an accepted command. See '{} --help'.".format(progname, cmd, progname))
        list_commands()
        sys.exit(1)

    cmds[cmd]()
