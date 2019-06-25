
from sys import stdout
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from polimi.systems import boost, boost_jac
from polimi.envelope import BEEnvelope, TrapEnvelope


def envelope():
    T = 20e-6
    DC = 0.4
    fun = lambda t,y: boost(t, y, T, DC)
    jac = lambda t,y: boost_jac(t, y, T, DC)
    fun_rtol = 1e-10
    fun_atol = 1e-12

    if T == 20e-6 and DC <= 0.45:
        y0 = np.array([9.17375836, 1.00930474])
    else:
        y0 = None

    if y0 is None:
        sol = solve_ivp(fun, [0,1000*T], [10,1], method='BDF', jac=jac, rtol=fun_rtol, atol=fun_atol)
        y0 = sol['y'][:,-1]
        plt.plot(sol['t'],sol['y'][0],'k')
        plt.plot(sol['t'],sol['y'][1],'r')
        plt.show()

    print('y0 =', y0)

    t_span = [0, 500*T]
    be_solver = BEEnvelope(fun, t_span, y0, max_step=1000, \
                           T_guess=None, T=T, jac=jac, fun_method='BDF',\
                           rtol=1e-2, atol=1e-3, \
                           fun_rtol=fun_rtol, fun_atol=fun_atol)
    #trap_solver = TrapEnvelope(fun, t_span, y0, max_step=1000, \
    #                           T_guess=None, T=T, jac=jac, fun_method='BDF',\
    #                           rtol=1e-2, atol=1e-3, \
    #                           fun_rtol=fun_rtol, fun_atol=fun_atol)
    print('--------------------------------------------------------------------------')
    sol_be = be_solver.solve()
    print('--------------------------------------------------------------------------')
    #sol_trap = trap_solver.solve()
    #print('--------------------------------------------------------------------------')

    stdout.write('Integrating the original system... ')
    stdout.flush()
    sol = solve_ivp(fun, t_span, y0, method='BDF', jac=jac, rtol=fun_rtol, atol=fun_atol)
    stdout.write('done.\n')

    labels = [r'$V_C$ (V)', r'$I_L$ (A)']
    axes = []
    for i in range(2):
        if i == 0:
            ax = plt.subplot(2,1,i+1)
        else:
            plt.subplot(2,1,i+1,sharex=ax)
        plt.plot(sol['t'], sol['y'][i], 'k')
        plt.plot(sol_be['t'], sol_be['y'][i], 'ro-')
        #plt.plot(sol_trap['t'], sol_trap['y'][i], 'go-')
        plt.ylabel(labels[i])
    plt.xlabel('Time (s)')
    plt.show()


if __name__ == '__main__':
    envelope()
