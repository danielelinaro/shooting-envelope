
from sys import stdout
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from polimi.switching import Boost, solve_ivp_switch
from polimi.envelope import BEEnvelope, TrapEnvelope
from polimi.shooting import EnvelopeShooting

# for saving data
pack = lambda t,y: np.concatenate((np.reshape(t,(len(t),1)),y.transpose()),axis=1)


def system():
    T = 20e-6
    ki = 1
    Vin = 5
    Vref = 5

    t0 = 0
    t_end = 50*T
    t_span = np.array([t0, t_end])

    y0 = np.array([Vin,1])

    fun_rtol = 1e-10
    fun_atol = 1e-12

    boost = Boost(0, T=T, ki=ki, Vin=Vin, Vref=Vref, clock_phase=0)

    print('Vector field index at the beginning of the first integration: %d.' % boost.vector_field_index)
    sol_a = solve_ivp_switch(boost, t_span, y0, \
                             method='BDF', jac=boost.jac, \
                             rtol=fun_rtol, atol=fun_atol)
    print('Vector field index at the end of the first integration: %d.' % boost.vector_field_index)

    print('Vector field index at the beginning of the second integration: %d.' % boost.vector_field_index)
    sol_b = solve_ivp_switch(boost, sol_a['t'][-1]+t_span, sol_a['y'][:,-1], \
                             method='BDF', jac=boost.jac, \
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
    ki = 1
    Vin = 5
    Vref = 5

    boost = Boost(0, T=T, ki=ki, Vin=Vin, Vref=Vref, clock_phase=0)

    fun_rtol = 1e-10
    fun_atol = 1e-12

    y0 = np.array([Vin,1])
    t_tran = 50*T

    sol = solve_ivp_switch(boost, [0,t_tran], y0, \
                           method='BDF', jac=boost.jac, \
                           rtol=fun_rtol, atol=fun_atol)
    #plt.plot(sol['t']*1e6,sol['y'][0],'k')
    #plt.plot(sol['t']*1e6,sol['y'][1],'r')
    #plt.show()

    t_span = sol['t'][-1] + np.array([0, 100*T])
    y0 = sol['y'][:,-1]
    print('t_span =', t_span)
    print('y0 =', y0)
    print('index =', boost.vector_field_index)

    print('-' * 81)
    be_solver = BEEnvelope(boost, t_span, y0, max_step=1000, \
                           T_guess=None, T=T, jac=boost.jac, \
                           fun_method=solve_ivp_switch, \
                           rtol=1e-2, atol=1e-3, \
                           fun_rtol=fun_rtol, fun_atol=fun_atol, \
                           method='BDF')
    sol_be = be_solver.solve()
    print('-' * 81)
    trap_solver = TrapEnvelope(boost, t_span, y0, max_step=1000, \
                               T_guess=None, T=T, jac=boost.jac, \
                               fun_method=solve_ivp_switch, \
                               rtol=1e-2, atol=1e-3, \
                               fun_rtol=fun_rtol, fun_atol=fun_atol, \
                               method='BDF')
    sol_trap = trap_solver.solve()
    print('-' * 81)

    stdout.write('Integrating the original system... ')
    stdout.flush()
    sol = solve_ivp_switch(boost, t_span, y0, method='BDF',
                           jac=boost.jac, rtol=fun_rtol, atol=fun_atol)
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
    ki = 1
    Vin = 5
    Vref = 5

    boost = Boost(0, T=T, ki=ki, Vin=Vin, Vref=Vref, clock_phase=0)

    fun_rtol = 1e-10
    fun_atol = 1e-12

    y0 = np.array([Vin,1])
    t_tran = 100*T

    print('Vector field index at the beginning of the first integration: %d.' % boost.vector_field_index)
    sol = solve_ivp_switch(boost, [0,t_tran], y0, \
                           method='BDF', jac=boost.jac, \
                           rtol=fun_rtol, atol=fun_atol)
    print('Vector field index at the end of the first integration: %d.' % boost.vector_field_index)
    plt.figure()
    ax = plt.subplot(2,1,1)
    plt.plot(sol['t']*1e6,sol['y'][0],'k')
    plt.ylabel(r'$V_C$ (V)')
    plt.subplot(2,1,2,sharex=ax)
    plt.plot(sol['t']*1e6,sol['y'][1],'r')
    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel(r'$I_L$ (A)')
    plt.show()

    y0 = sol['y'][:,-1]

    T_large = 100*T
    boost.with_variational = True
    boost.variational_T = T_large

    t_span_var = [0,1]
    y0_var = np.concatenate((y0,np.eye(len(y0)).flatten()))

    sol = solve_ivp_switch(boost, t_span_var, y0_var, method='BDF', rtol=fun_rtol, atol=fun_atol)

    #np.savetxt('boost_variational.txt', pack(sol['t'],sol['y']), fmt='%.3e')

    eig,_ = np.linalg.eig(np.reshape(sol['y'][2:,-1],(2,2)))
    print('eigenvalues:', eig)

    plt.figure()
    ax = plt.subplot(2,2,1)
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.ylabel(r'$V_C$ (V)')
    plt.subplot(2,2,3,sharex=ax)
    plt.plot(sol['t'],sol['y'][1],'r')
    plt.xlabel('Normalized time')
    plt.ylabel(r'$I_L$ (A)')

    colors = ['g','b','m','y']
    plt.subplot(2,2,2,sharex=ax)
    for i in range(2):
        plt.plot(sol['t'],sol['y'][i+2],colors[i],label=r'$y_{}$'.format(i+1))
    plt.legend(loc='best')
    
    plt.subplot(2,2,4,sharex=ax)
    for i in range(2):
        plt.plot(sol['t'],sol['y'][i+4],colors[i+2],label=r'$y_{}$'.format(i+3))
    plt.legend(loc='best')
    plt.xlabel('Normalized time')
    plt.show()


def variational_envelope():
    T = 20e-6
    ki = 1
    Vin = 5
    Vref = 5

    boost = Boost(0, T=T, ki=ki, Vin=Vin, Vref=Vref, clock_phase=0)

    fun_rtol = 1e-10
    fun_atol = 1e-12

    y0 = np.array([Vin,1])
    t_tran = 50*T

    print('Vector field index at the beginning of the first integration: %d.' % boost.vector_field_index)
    sol = solve_ivp_switch(boost, [0,t_tran], y0, \
                           method='BDF', jac=boost.jac, \
                           rtol=fun_rtol, atol=fun_atol)
    print('Vector field index at the end of the first integration: %d.' % boost.vector_field_index)
    #plt.figure()
    #ax = plt.subplot(2,1,1)
    #plt.plot(sol['t']*1e6,sol['y'][0],'k')
    #plt.ylabel(r'$V_C$ (V)')
    #plt.subplot(2,1,2,sharex=ax)
    #plt.plot(sol['t']*1e6,sol['y'][1],'r')
    #plt.xlabel(r'Time ($\mu$s)')
    #plt.ylabel(r'$I_L$ (A)')
    #plt.show()

    y0 = sol['y'][:,-1]

    T_large = 100*T
    T_small = T
    boost.with_variational = True
    boost.variational_T = T_large

    t_span_var = [0,1]
    y0_var = np.concatenate((y0,np.eye(len(y0)).flatten()))

    sol = solve_ivp_switch(boost, t_span_var, y0_var, method='BDF', rtol=fun_rtol, atol=fun_atol)

    rtol = 1e-1
    atol = 1e-2
    be_var_solver = BEEnvelope(boost, [0,T_large], y0, T_guess=None, T=T_small, jac=boost.jac, \
                               rtol=rtol, atol=atol, fun_method=solve_ivp_switch, \
                               max_step=1000, fun_rtol=fun_rtol, fun_atol=fun_atol, \
                               is_variational=True, T_var_guess=None, T_var=None, \
                               var_rtol=rtol, var_atol=atol, method='BDF')
    trap_var_solver = TrapEnvelope(boost, [0,T_large], y0, T_guess=None, T=T_small, jac=boost.jac, \
                                   rtol=rtol, atol=atol, fun_method=solve_ivp_switch, \
                                   max_step=1000, fun_rtol=fun_rtol, fun_atol=fun_atol, \
                                   is_variational=True, T_var_guess=None, T_var=None, \
                                   var_rtol=rtol, var_atol=atol, method='BDF')
    print('-' * 100)
    var_sol_be = be_var_solver.solve()
    print('-' * 100)
    var_sol_trap = trap_var_solver.solve()
    print('-' * 100)

    eig,_ = np.linalg.eig(np.reshape(sol['y'][2:,-1],(2,2)))
    print('         correct eigenvalues:', eig)
    eig,_ = np.linalg.eig(np.reshape(var_sol_be['y'][2:,-1],(2,2)))
    print('  BE approximate eigenvalues:', eig)
    eig,_ = np.linalg.eig(np.reshape(var_sol_trap['y'][2:,-1],(2,2)))
    print('TRAP approximate eigenvalues:', eig)

    ax = plt.subplot(2,2,1)
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(var_sol_be['t'],var_sol_be['y'][0],'rs-')
    plt.plot(var_sol_trap['t'],var_sol_trap['y'][0],'go-')
    plt.ylabel(r'$V_C$ (V)')
    plt.subplot(2,2,3,sharex=ax)
    plt.plot(sol['t'],sol['y'][1],'k')
    plt.plot(var_sol_be['t'],var_sol_be['y'][1],'rs-')
    plt.plot(var_sol_trap['t'],var_sol_trap['y'][1],'go-')
    plt.xlabel('Normalized time')
    plt.ylabel(r'$I_L$ (A)')

    colors = ['c','b','m','y']
    plt.subplot(2,2,2,sharex=ax)
    for i in range(2):
        plt.plot(sol['t'],sol['y'][i+2],colors[i],label=r'$J_{1,%d}$' % (i+1))
        plt.plot(var_sol_be['t'],var_sol_be['y'][i+2],'rs')
        plt.plot(var_sol_trap['t'],var_sol_trap['y'][i+2],'go')
    plt.legend(loc='best')

    plt.subplot(2,2,4,sharex=ax)
    for i in range(2):
        plt.plot(sol['t'],sol['y'][i+4],colors[i+2],label=r'$J_{2,%d}$' % (i+1))
        plt.plot(var_sol_be['t'],var_sol_be['y'][i+4],'rs')
        plt.plot(var_sol_trap['t'],var_sol_trap['y'][i+4],'go')
    plt.legend(loc='best')
    plt.xlabel('Normalized time')

    plt.show()


def shooting():

    T = 20e-6
    ki = 1
    Vin = 5
    Vref = 5

    boost = Boost(0, T=T, ki=ki, Vin=Vin, Vref=Vref, clock_phase=0)

    fun_rtol = 1e-10
    fun_atol = 1e-12

    y0_guess = np.array([Vin,0])

    with_tran = False

    if with_tran:
        tran = solve_ivp_switch(boost, [0,50*T], y0_guess, method='BDF', \
                                jac=boost.jac, rtol=fun_rtol, atol=fun_atol)
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2,sharex=ax1)
        ax1.plot(tran['t']/T,tran['y'][0],'k')
        ax1.set_ylabel(r'$V_C$ (V)')
        ax2.plot(tran['t']/T,tran['y'][1],'k')
        ax2.set_xlabel('No. of periods')
        ax2.set_ylabel(r'$I_L$ (A)')

    T_large = 10*T
    T_small = T

    estimate_T = False

    shoot = EnvelopeShooting(boost, boost.n_dim, T_large, \
                             estimate_T, T_small, boost.jac, \
                             shooting_tol=1e-3, env_solver=BEEnvelope, \
                             env_rtol=1e-2, env_atol=1e-3, \
                             var_rtol=1e-1, var_atol=1e-2, \
                             fun_rtol=fun_rtol, fun_atol=fun_atol, \
                             env_fun_method=solve_ivp_switch, \
                             method='BDF')
    sol_shoot = shoot.run(y0_guess)
    print('Number of iterations: %d.' % sol_shoot['n_iter'])

    t_span_var = [0,1]
    boost.with_variational = True
    boost.variational_T = T_large

    col = 'krgbcmy'
    lw = 0.7
    fig = plt.figure(figsize=(12,7))

    for i,integr in enumerate(sol_shoot['integrations']):

        y0 = integr['y'][:2,0]
        y0_var = np.concatenate((y0,np.eye(2).flatten()))
        sol = solve_ivp_switch(boost, t_span_var, y0_var, method='BDF', rtol=fun_rtol, atol=fun_atol)

        ax1 = fig.add_subplot(3,2,1)
        ax1.plot(sol['t'],sol['y'][0],col[i],lw=lw,label='Iter #%d' % (i+1))
        ax1.plot(integr['t'],integr['y'][0],col[i]+'o-',lw=1,ms=3)

        ax2 = fig.add_subplot(3,2,2,sharex=ax1)
        ax2.plot(sol['t'],sol['y'][1],col[i],lw=lw)
        ax2.plot(integr['t'],integr['y'][1],col[i]+'o-',lw=1,ms=3)

        for j in range(3,7):
            ax = fig.add_subplot(3,2,j,sharex=ax1)
            ax.plot(sol['t'],sol['y'][j-1],col[i],lw=lw)
            ax.plot(integr['t'],integr['y'][j-1],col[i]+'o',lw=1,ms=3)
            ax.set_ylabel(r'$J_{%d,%d}$' % (int((j-1)/2),(j-1)%2+1))
            if j > 4:
                ax.set_xlabel('Normalized time')

    ax1.legend(loc='best')
    ax1.set_ylabel(r'$V_C$ (V)')
    ax2.set_ylabel(r'$I_L$ (A)')
    plt.savefig('boost_shooting.pdf')
    plt.show()


if __name__ == '__main__':
    #system()
    #envelope()
    #variational_integration()
    #variational_envelope()
    shooting()
