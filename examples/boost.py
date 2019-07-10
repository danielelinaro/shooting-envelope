
from sys import stdout
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from polimi.envelope import BEEnvelope, TrapEnvelope
from polimi.switching import Boost, solve_ivp_switch


def system():
    T = 20e-6
    DC = 0.5
    ki = 1
    Vin = 5
    Vref = 5

    t0 = 0
    t_end = 50*T
    t_span = np.array([t0, t_end])

    y0 = np.array([Vin,1])

    fun_rtol = 1e-10
    fun_atol = 1e-12

    boost = Boost(0, T, DC, ki, Vin=Vin, Vref=Vref, clock_phase=0)

    print('Vector field index at the beginning of the first integration: %d.' % boost.vector_field_index)
    sol_a = solve_ivp_switch(boost, t_span, y0, \
                             method='BDF', jac=boost.J, \
                             rtol=fun_rtol, atol=fun_atol)
    print('Vector field index at the end of the first integration: %d.' % boost.vector_field_index)

    print('Vector field index at the beginning of the second integration: %d.' % boost.vector_field_index)
    sol_b = solve_ivp_switch(boost, sol_a['t'][-1]+t_span, sol_a['y'][:,-1], \
                             method='BDF', jac=boost.J, \
                             rtol=fun_rtol, atol=fun_atol)
    print('Vector field index at the end of the second integration: %d.' % boost.vector_field_index)

    ax = plt.subplot(2, 1, 1)
    plt.plot([0, sol_b['t'][-1]*1e6], [Vin,Vin], 'b')
    plt.plot(sol_a['t']*1e6, sol_a['y'][0], 'k')
    plt.plot(sol_b['t']*1e6, sol_b['y'][0], 'r')
    plt.ylabel(r'$V_C$ (V)')
    plt.subplot(2, 1, 2, sharex=ax)
    plt.plot(sol_a['t']*1e6, sol_a['y'][1], 'k')
    plt.plot(sol_b['t']*1e6, sol_b['y'][1], 'r')
    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel(r'$I_L$ (A)')
    plt.show()


def envelope():
    T = 20e-6
    DC = 0.5
    ki = 1
    Vin = 5
    Vref = 5

    boost = Boost(0, T, DC, ki, Vin=Vin, Vref=Vref, clock_phase=0)

    fun_rtol = 1e-10
    fun_atol = 1e-12

    y0 = np.array([Vin,1])
    t_tran = 50*T

    sol = solve_ivp_switch(boost, [0,t_tran], y0, \
                           method='BDF', jac=boost.J, \
                           rtol=fun_rtol, atol=fun_atol)
    #plt.plot(sol['t']*1e6,sol['y'][0],'k')
    #plt.plot(sol['t']*1e6,sol['y'][1],'r')
    #plt.show()

    t_span = sol['t'][-1] + np.array([0, 1000*T])
    y0 = sol['y'][:,-1]
    print('t_span =', t_span)
    print('y0 =', y0)
    print('index =', boost.vector_field_index)

    print('-' * 81)
    be_solver = BEEnvelope(boost, t_span, y0, max_step=1000, \
                           T_guess=None, T=T, jac=boost.J, \
                           fun_method=solve_ivp_switch, \
                           rtol=1e-2, atol=1e-3, \
                           fun_rtol=fun_rtol, fun_atol=fun_atol, \
                           method='BDF')
    sol_be = be_solver.solve()
    print('-' * 81)
    trap_solver = TrapEnvelope(boost, t_span, y0, max_step=1000, \
                               T_guess=None, T=T, jac=boost.J, \
                               fun_method=solve_ivp_switch, \
                               rtol=1e-2, atol=1e-3, \
                               fun_rtol=fun_rtol, fun_atol=fun_atol, \
                               method='BDF')
    sol_trap = trap_solver.solve()
    print('-' * 81)

    stdout.write('Integrating the original system... ')
    stdout.flush()
    sol = solve_ivp_switch(boost, t_span, y0, method='BDF',
                           jac=boost.J, rtol=fun_rtol, atol=fun_atol)
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
        plt.plot(sol_trap['t'], sol_trap['y'][i], 'go-')
        plt.ylabel(labels[i])
    plt.xlabel('Time (s)')
    plt.show()


def variational_integration():
    T = 20e-6
    DC = 0.5
    ki = 1
    Vin = 5
    Vref = 5

    boost = Boost(0, T, DC, ki, Vin=Vin, Vref=Vref, clock_phase=0)
    boost.vector_field_index = 0

    fun_rtol = 1e-10
    fun_atol = 1e-12

    y0 = np.array([Vin,1])
    t_tran = 50*T

    print('Vector field index at the beginning of the first integration: %d.' % boost.vector_field_index)
    sol = solve_ivp_switch(boost, [0,t_tran], y0, \
                           method='BDF', jac=boost.J, \
                           rtol=fun_rtol, atol=fun_atol)
    print('Vector field index at the end of the first integration: %d.' % boost.vector_field_index)
    #plt.plot(sol['t']*1e6,sol['y'][0],'k')
    #plt.plot(sol['t']*1e6,sol['y'][1],'r')
    #plt.show()

    y0 = sol['y'][:,-1]

    T_large = 100*T
    T_small = T
    boost.with_variational = True
    boost.variational_T = T_large

    t_span_var = [0,1]
    y0_var = np.concatenate((y0,np.eye(len(y0)).flatten()))

    sol = solve_ivp_switch(boost, t_span_var, y0_var, method='BDF', rtol=fun_rtol, atol=fun_atol)

    plt.subplot(1,2,1)
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(sol['t'],sol['y'][1],'r')
    plt.subplot(1,2,2)
    plt.plot(sol['t'],sol['y'][2],'k')
    plt.show()

    eig,_ = np.linalg.eig(np.reshape(sol['y'][2:,-1],(2,2)))
    print('         correct eigenvalues:', eig)


def variational():

    def variational_system(fun, jac, t, y, T):
        N = int((-1 + np.sqrt(1 + 4*len(y))) / 2)
        J = jac(t*T,y[:N])
        phi = np.reshape(y[N:N+N**2],(N,N))
        return np.concatenate((T * fun(t*T, y[:N]), \
                               T * np.matmul(J,phi).flatten()))

    T = 20e-6
    DC = 0.45
    T_large = 200*T
    T_small = T
    fun = lambda t,y: boost(t,y,T,DC)
    jac = lambda t,y: boost_jac(t,y,T,DC)
    var_fun = lambda t,y: variational_system(fun, jac, t, y, T_large)
    fun_rtol = 1e-10
    fun_atol = 1e-12

    t_span_var = [0,1]
    y0 = np.array([9.17375836, 1.00930474])
    y0_var = np.concatenate((y0,np.eye(len(y0)).flatten()))

    rtol = 1e-1
    atol = 1e-2
    be_var_solver = BEEnvelope(fun, [0,T_large], y0, T_guess=None, T=T_small, jac=jac, \
                               rtol=rtol, atol=atol, fun_method='BDF', max_step=1000, \
                               fun_rtol=fun_rtol, fun_atol=fun_atol, is_variational=True, \
                               T_var_guess=None, T_var=None, var_rtol=rtol, var_atol=atol)
    trap_var_solver = TrapEnvelope(fun, [0,T_large], y0, T_guess=None, T=T_small, jac=jac, \
                                   rtol=rtol, atol=atol, fun_method='BDF', max_step=100, \
                                   fun_rtol=fun_rtol, fun_atol=fun_atol, is_variational=True, \
                                   T_var_guess=None, T_var=None, var_rtol=rtol, var_atol=atol)
    print('-' * 100)
    var_sol_be = be_var_solver.solve()
    print('-' * 100)
    var_sol_trap = trap_var_solver.solve()
    print('-' * 100)

    sol = solve_ivp(var_fun, t_span_var, y0_var, method='BDF', rtol=fun_rtol, atol=fun_atol)

    eig,_ = np.linalg.eig(np.reshape(sol['y'][2:,-1],(2,2)))
    print('         correct eigenvalues:', eig)
    eig,_ = np.linalg.eig(np.reshape(var_sol_be['y'][2:,-1],(2,2)))
    print('  BE approximate eigenvalues:', eig)
    eig,_ = np.linalg.eig(np.reshape(var_sol_trap['y'][2:,-1],(2,2)))
    print('TRAP approximate eigenvalues:', eig)

    ax = plt.subplot(1,2,1)
    plt.plot(sol['t'],sol['y'][1],'k')
    plt.plot(var_sol_be['t'],var_sol_be['y'][1],'ro-')
    plt.plot(var_sol_trap['t'],var_sol_trap['y'][1],'go')
    plt.xlabel('Normalized time')
    plt.ylabel(r'$I_L$ (A)')
    plt.subplot(1,2,2,sharex=ax)
    plt.plot(t_span_var,[0,0],'b')
    plt.plot(sol['t'],sol['y'][2],'k')
    plt.plot(var_sol_be['t'],var_sol_be['y'][2],'ro')
    #for i in range(0,len(var_sol_be['var']['t']),3):
    #    plt.plot(var_sol_be['var']['t'][i:i+3],var_sol_be['var']['y'][0,i:i+3],'c.-')
    plt.plot(var_sol_trap['t'],var_sol_trap['y'][2],'go')
    #for i in range(0,len(var_sol_trap['var']['t']),3):
    #    plt.plot(var_sol_trap['var']['t'][i:i+3],var_sol_trap['var']['y'][0,i:i+3],'m.-')
    plt.xlabel('Normalized time')
    plt.ylabel(r'$J_{11}$')
    plt.show()


if __name__ == '__main__':
    #system()
    #envelope()
    variational_integration()
    #variational()
