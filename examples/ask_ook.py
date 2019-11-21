import os
import sys
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


def system():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.optimize import fsolve

    ckt = ASK_OOK()

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
    print('(FULL CIRCUIT) Initial condition:')
    print('{:>10s} {:>13s} {:>13s}'.format('Variable','Computed','PAN'))
    var_names = ['ILtl', 'IL1', 'IL2', 'out', 'l20', 'l30', 'd10', 'd20']
    for var,py,pan in zip(var_names,y0,y0_correct):
        print('{:>10s} = {:13.5e} {:13.5e}'.format(var,py,pan))

    tend = 100 * ckt.T2

    atol = 1e-6
    rtol = 1e-8

    comparison = False
    if comparison:
        elapsed = {}
        sys.stdout.write('Integrating using RK45... ')
        sys.stdout.flush()
        start = time.time()
        sol = solve_ivp(ckt, [0,tend], y0, method='RK45', atol=atol, rtol=rtol)
        elapsed['RK45'] = time.time() - start
        sys.stdout.write('done.')
        sys.stdout.write('Integrating using BDF... ')
        sys.stdout.flush()
        start = time.time()
        sol = solve_ivp(ckt, [0,tend], y0, method='BDF', jac=ckt.jac, atol=atol, rtol=rtol)
        elapsed['BDF'] = time.time() - start
        sys.stdout.write('done.')
        print('Elapsed times:')
        for k,v in elapsed.items():
            print('   {:>5s}: {:6.2f} sec.'.format(k,v))
    else:
        sol = solve_ivp(ckt, [0,tend], y0, method='BDF', jac=ckt.jac, atol=atol, rtol=rtol)

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
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
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
    vars_to_use = [2, 3, 5, 7]

    y0 = fsolve(lambda y: ckt(0,y), y0_guess)
    print('Initial condition for transient analysis:')
    print('{:>10s} {:>13s}'.format('Variable','Value'))
    var_names = ['ILtl', 'IL1', 'IL2', 'out', 'l20', 'l30', 'd10', 'd20']
    for var,ic in zip(var_names,y0):
        print('{:>10s} = {:13.5e}'.format(var,ic))

    t_tran = 100 * ckt.T2

    fun_atol = 1e-8
    fun_rtol = 1e-10

    sol = solve_ivp(ckt, [0,t_tran], y0, method='BDF', \
                    jac=ckt.jac, atol=fun_atol, rtol=fun_rtol)
    y0 = sol['y'][:,-1]

    fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,figsize=(10,4))
    labels = ['i(Ltl)', 'i(L1)', 'i(L2)', \
              'v(out)', 'v(l20)', 'v(l30)', 'v(d10)', 'v(d20)']
    colors = 'rgbk'
    for i in range(1):
        ax1.plot(sol['t']*1e9, sol['y'][i], colors[i], lw=1, label=labels[i] + ' tran')
    for i in range(1):
        ax2.plot(sol['t']*1e9, sol['y'][i+3], colors[i], lw=1, label=labels[i+3] + ' tran')

    print('Initial condition for envelope analysis:')
    print('{:>10s} {:>13s}'.format('Variable','Value'))
    for var,ic in zip(var_names,y0):
        print('{:>10s} = {:13.5e}'.format(var,ic))

    t_span = t_tran + np.array([0, 100 * ckt.T2])
    print('-' * 81)
    be_solver = BEEnvelope(ckt, t_span, y0, max_step=20, \
                           T_guess=None, T=ckt.T2, vars_to_use=vars_to_use, \
                           env_rtol=1e-1, env_atol=1e-2, \
                           solver=solve_ivp, \
                           jac=ckt.jac, method='BDF', \
                           rtol=fun_rtol, atol=fun_atol)
    sol_be = be_solver.solve()
    print('-' * 81)
    trap_solver = TrapEnvelope(ckt, t_span, y0, max_step=20, \
                               T_guess=None, T=ckt.T2, vars_to_use=vars_to_use, \
                               env_rtol=1e-1, env_atol=1e-2, \
                               solver=solve_ivp, \
                               jac=ckt.jac, method='BDF', \
                               rtol=fun_rtol, atol=fun_atol)
    sol_trap = trap_solver.solve()
    print('-' * 81)

    sys.stdout.write('Integrating the original system... ')
    sys.stdout.flush()
    sol = solve_ivp(ckt, t_span, y0, method='BDF',
                    jac=ckt.jac, rtol=fun_rtol, atol=fun_atol)
    sys.stdout.write('done.\n')

    colors = 'krgb'
    for i in range(1):
        ax1.plot(sol['t']*1e9, sol['y'][i], colors[i], lw=1, label=labels[i])
        ax1.plot(sol_be['t']*1e9, sol_be['y'][i], colors[i]+'o-', lw=1, \
                 label='BE', markersize=4, markerfacecolor='r')
        ax1.plot(sol_trap['t']*1e9, sol_trap['y'][i], colors[i]+'s-', lw=1, \
                 label='TRAP', markersize=4, markerfacecolor='g')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Current (A)')
    ax1.legend(loc='best')
    for i in range(1):
        ax2.plot(sol['t']*1e9, sol['y'][i+3], colors[i], lw=1, label=labels[i+3])
        ax2.plot(sol_be['t']*1e9, sol_be['y'][i+3], colors[i]+'o-', lw=1, \
                 label='BE', markersize=4, markerfacecolor='r')
        ax2.plot(sol_trap['t']*1e9, sol_trap['y'][i+3], colors[i]+'s-', lw=1, \
                 label='TRAP', markersize=4, markerfacecolor='g')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Voltage (V)')
    ax2.legend(loc='best')

    plt.show()


cmds = {'system': system, \
        'envelope': envelope, \
        #'variational': variational_integration, \
        #'variational-envelope': variational_envelope, \
        #'shooting': shooting, \
        #'shooting-envelope': shooting_envelope
}

cmd_descriptions = {'system': 'integrate the ASK/OOK RF modulator', \
                    'envelope': 'compute the envelope of the ASK/OOK RF modulator', \
                    #'variational': 'integrate the buck converter and its variational part', \
                    #'variational-envelope': 'compute the envelope of the buck converter with variational part', \
                    #'shooting': 'perform a shooting analysis of the buck converter', \
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
