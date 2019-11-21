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


def init(t_tran=0, atol=1e-6, rtol=1e-8):
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


def system():
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

    t_end = 2 / F1
    ckt,sol = init(t_end)

    print('(FULL CIRCUIT) Initial condition:')
    print('{:>10s} {:>13s} {:>13s}'.format('Variable','Computed','PAN'))
    var_names = ['ILtl', 'IL1', 'IL2', 'out', 'l20', 'l30', 'd10', 'd20']
    for var,py,pan in zip(var_names,sol['y'][:,0],y0_correct):
        print('{:>10s} = {:13.5e} {:13.5e}'.format(var,py,pan))

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

    plt.show()


def envelope():

    t_tran = 100 / F2
    fun_atol = 1e-6
    fun_rtol = 1e-8
    ckt,sol = init(t_tran, fun_atol, fun_rtol)

    print('Initial condition for transient analysis:')
    print('{:>10s} {:>13s}'.format('Variable','Value'))
    var_names = ['ILtl', 'IL1', 'IL2', 'out', 'l20', 'l30', 'd10', 'd20']
    for var,ic in zip(var_names,sol['y'][:,0]):
        print('{:>10s} = {:13.5e}'.format(var,ic))

    fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,figsize=(10,4))
    labels = ['i(Ltl)', 'i(L1)', 'i(L2)', \
              'v(out)', 'v(l20)', 'v(l30)', 'v(d10)', 'v(d20)']
    colors = 'cmyk'
    for i in range(2):
        ax1.plot(sol['t']*1e9, sol['y'][i], colors[i], lw=1, label=labels[i] + ' tran')
    for i in range(2):
        ax2.plot(sol['t']*1e9, sol['y'][i+3], colors[i], lw=1, label=labels[i+3] + ' tran')

    y0 = sol['y'][:,-1]

    print('Initial condition for envelope analysis:')
    print('{:>10s} {:>13s}'.format('Variable','Value'))
    for var,ic in zip(var_names,y0):
        print('{:>10s} = {:13.5e}'.format(var,ic))

    t_span = t_tran + np.array([0, 100 * ckt.T2])
    vars_to_use = [2, 3, 5, 7]

    trap_solver = TrapEnvelope(ckt, t_span, y0, max_step=20, \
                               T_guess=None, T=ckt.T2, vars_to_use=vars_to_use, \
                               env_rtol=1e-1, env_atol=1e-2, \
                               solver=solve_ivp, \
                               jac=ckt.jac, method='BDF', \
                               rtol=fun_rtol, atol=fun_atol)
    sol_trap = trap_solver.solve()

    for t0,y0 in zip(sol_trap['t'],sol_trap['y'].T):
        sol = solve_ivp(ckt, [t0,t0+ckt.T2], y0, method='BDF', \
                        jac=ckt.jac, rtol=fun_rtol, atol=fun_atol)
        try:
            envelope['t'] = np.append(envelope['t'], sol['t'])
            envelope['y'] = np.append(envelope['y'], sol['y'], axis=1)
        except:
            envelope = {key: sol[key] for key in ('t','y')}

    colors = 'krgb'
    for i in range(2):
        ax1.plot(envelope['t']*1e9, envelope['y'][i], colors[i], lw=1, label=labels[i])
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Current (A)')
    ax1.legend(loc='best')
    for i in range(2):
        ax2.plot(envelope['t']*1e9, envelope['y'][i+3], colors[i], lw=1, label=labels[i+3])
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Voltage (V)')
    ax2.legend(loc='best')

    plt.show()


def shooting():
    t_tran = 100 / F2
    fun_atol = 1e-6
    fun_rtol = 1e-8
    ckt,sol = init(t_tran, fun_atol, fun_rtol)

    print('Initial condition for transient analysis:')
    print('{:>10s} {:>13s}'.format('Variable','Value'))
    var_names = ['ILtl', 'IL1', 'IL2', 'out', 'l20', 'l30', 'd10', 'd20']
    for var,ic in zip(var_names,sol['y'][:,0]):
        print('{:>10s} = {:13.5e}'.format(var,ic))

    y0_guess = sol['y'][:,-1]
    T_large = ckt.T1
    T_small = ckt.T2

    estimate_T = False

    shoot = Shooting(ckt, T_large, estimate_T, tol=1e-1, \
                     solver=solve_ivp, rtol=fun_rtol, atol=fun_atol, \
                     method='BDF')

    now = time.time()
    sol_shoot = shoot.run(y0_guess)
    elapsed = time.time() - now
    print('Number of iterations: %d.' % sol_shoot['n_iter'])
    print('Elapsed time: %7.3f sec.' % elapsed)

    import pickle
    pickle.dump(sol_shoot, open('shooting_ASK_OOK.pkl','wb'))

    return
    col = 'krgbcmy'
    lw = 0.8
    fig,ax = plt.subplots(3,1,sharex=True,figsize=(6,6))

    for i,integr in enumerate(sol_shoot['integrations']):

        y0 = integr['y'][:N,0]
        y0_var = np.concatenate((y0,np.eye(N).flatten()))

        for j in range(3):
            ax[j].plot(integr['t'],integr['y'][j],col[i],lw=lw,label='Iter #%d' % (i+1))
    ax[2].set_xlabel('Normalized time')
    ax[0].legend(loc='best')
    ax[0].set_ylabel(r'$V_C$ (V)')
    ax[1].set_ylabel(r'$I_L$ (A)')
    ax[2].set_ylabel(r'$\int V_o$ $(\mathrm{V}\cdot\mathrm{s})$')
    #plt.savefig('ask_ook_shooting.pdf')
    plt.show()


cmds = {'system': system, \
        'envelope': envelope, \
        #'variational': variational_integration, \
        #'variational-envelope': variational_envelope, \
        'shooting': shooting, \
        #'shooting-envelope': shooting_envelope
}

cmd_descriptions = {'system': 'integrate the ASK/OOK RF modulator', \
                    'envelope': 'compute the envelope of the ASK/OOK RF modulator', \
                    #'variational': 'integrate the buck converter and its variational part', \
                    #'variational-envelope': 'compute the envelope of the buck converter with variational part', \
                    'shooting': 'perform a shooting analysis of the ASK/OOK RF modulator', \
                    #'shooting-envelope': 'perform a shooting analysis of the buck converter using the envelope'
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
