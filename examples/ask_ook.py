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

    oscillator = ASK_OOK()

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

    y0 = fsolve(lambda y: oscillator._fun(0,y), y0_guess)
    print('(FULL CIRCUIT) Initial condition:')
    print('{:>10s} {:>13s} {:>13s}'.format('Variable','Computed','PAN'))
    var_names = ['ILtl', 'IL1', 'IL2', 'out', 'l20', 'l30', 'd10', 'd20']
    for var,py,pan in zip(var_names,y0,y0_correct):
        print('{:>10s} = {:13.5e} {:13.5e}'.format(var,py,pan))

    tend = 5 * oscillator.T2

    atol = 1e-6
    rtol = 1e-8

    sol = solve_ivp(oscillator, [0,tend], y0, method='RK45', atol=atol, rtol=rtol)

    fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,figsize=(10,4))
    ax1.plot(sol['t']*1e9, sol['y'][0], 'k', lw=1, label='i(Ltl)')
    ax1.plot(sol['t']*1e9, sol['y'][1], 'r', lw=1, label='i(L1)')
    ax1.plot(sol['t']*1e9, sol['y'][2], 'g', lw=1, label='i(L2)')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Current (A)')
    ax1.legend(loc='best')

    ax2.plot(sol['t']*1e9, sol['y'][3], 'k', lw=1, label='v(out)')
    ax2.plot(sol['t']*1e9, sol['y'][4], 'r', lw=1, label='v(l20)')
    ax2.plot(sol['t']*1e9, sol['y'][5], 'g', lw=1, label='v(l30)')
    ax2.plot(sol['t']*1e9, sol['y'][6], 'b', lw=1, label='v(d10)')
    ax2.plot(sol['t']*1e9, sol['y'][7], 'c', lw=1, label='v(d20)')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Voltage (V)')
    ax2.legend(loc='best')

    plt.show()


cmds = {'system': system, \
        #'envelope': envelope, \
        #'variational': variational_integration, \
        #'variational-envelope': variational_envelope, \
        #'shooting': shooting, \
        #'shooting-envelope': shooting_envelope
}

cmd_descriptions = {'system': 'integrate the ASK/OOK oscillator', \
                    #'envelope': 'compute the envelope of the buck converter', \
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
