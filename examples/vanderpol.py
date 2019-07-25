
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from polimi.switching import VanderPol
from polimi.envelope import BEEnvelope, TrapEnvelope

def system():
    epsilon = 1e-3
    A = [10,1]
    T = [4,400]

    t0 = 0
    t_end = np.max(T)
    t_span = np.array([t0, t_end])

    y0 = np.array([-2,1])

    fun_rtol = 1e-10
    fun_atol = 1e-12

    vdp = VanderPol(epsilon, A, T)

    sol = solve_ivp(vdp, t_span, y0, method='BDF', \
                    jac=vdp.jac, rtol=fun_rtol, atol=fun_atol)

    ax = plt.subplot(2, 1, 1)
    plt.plot(sol['t'], sol['y'][0], 'k')
    plt.ylabel(r'$V_C$ (V)')
    plt.subplot(2, 1, 2, sharex=ax)
    plt.plot(sol['t'], sol['y'][1], 'k')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$I_L$ (A)')
    plt.show()


def envelope():
    epsilon = 1e-3
    A = [10,1]
    T = [4,400]

    vdp = VanderPol(epsilon, A, T)
    fun_rtol = 1e-10
    fun_atol = 1e-12

    t_tran = 0

    if t_tran > 0:
        sol = solve_ivp(vdp, [0, t_tran], [-2,1], method='BDF', \
                        jac=vdp.jac, rtol=fun_rtol, atol=fun_atol)
        y0 = sol['y'][:,-1]
    else:
        y0 = np.array([-5.84170838, 0.1623759])

    print('y0 =', y0)

    t0 = 0
    t_end = np.max(T)
    t_span = np.array([t0, t_end])

    env_rtol = 1e-1
    env_atol = 1e-2
    be_env_solver = BEEnvelope(vdp, t_span, y0, T=np.min(T), \
                               env_rtol=env_rtol, env_atol=env_atol, \
                               rtol=fun_rtol, atol=fun_atol, \
                               method='BDF', jac=vdp.jac)
    be_env_sol = be_env_solver.solve()

    trap_env_solver = TrapEnvelope(vdp, t_span, y0, T=np.min(T), \
                                   env_rtol=env_rtol, env_atol=env_atol, \
                                   rtol=fun_rtol, atol=fun_atol, \
                                   method='BDF', jac=vdp.jac)
    trap_env_sol = trap_env_solver.solve()

    sol = solve_ivp(vdp, t_span, y0, method='BDF', \
                    jac=vdp.jac, rtol=fun_rtol, atol=fun_atol)

    fig,(ax1,ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(sol['t'], sol['y'][0], 'k')
    ax1.plot(be_env_sol['t'], be_env_sol['y'][0], 'ro-')
    ax1.plot(trap_env_sol['t'], trap_env_sol['y'][0], 'gs-')
    ax1.set_ylabel(r'$V_C$ (V)')
    ax2.plot(sol['t'], sol['y'][1], 'k')
    ax2.plot(be_env_sol['t'], be_env_sol['y'][1], 'ro-')
    ax2.plot(trap_env_sol['t'], trap_env_sol['y'][1], 'gs-')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'$I_L$ (A)')
    plt.show()


def variational_envelope():
    from polimi import vdp, vdp_jac

    def variational_system(fun, jac, t, y, T):
        N = int((-1 + np.sqrt(1 + 4*len(y))) / 2)
        J = jac(t*T,y[:N])
        phi = np.reshape(y[N:N+N**2],(N,N))
        return np.concatenate((T * fun(t*T, y[:N]), \
                               T * (J @ phi).flatten()))

    epsilon = 1e-3
    A = [10,1]
    T = [4,400]
    T_large = max(T)
    T_small = min(T)
    T_small_guess = min(T) * 0.95
    fun = lambda t,y: vdp(t,y,epsilon,A,T)
    jac = lambda t,y: vdp_jac(t,y,epsilon)
    var_fun = lambda t,y: variational_system(fun, jac, t, y, T_large)

    t_span_var = [0,1]
    if A[0] == 10:
        y0 = np.array([-5.8133754, 0.13476983])
    elif A[0] == 1:
        y0 = np.array([9.32886314, 0.109778919])
    y0_var = np.concatenate((y0,np.eye(len(y0)).flatten()))

    sol = solve_ivp(var_fun, t_span_var, y0_var, rtol=1e-8, atol=1e-10, dense_output=True)

    rtol = 1e-1
    atol = 1e-2
    be_var_solver = BEEnvelope(fun, [0,T_large], y0, T_guess=None, T=T_small, jac=jac, \
                               rtol=rtol, atol=atol, is_variational=True, \
                               T_var_guess=2*np.pi*0.95, var_rtol=rtol, var_atol=atol)
    trap_var_solver = TrapEnvelope(fun, [0,T_large], y0, T_guess=None, T=T_small, jac=jac, \
                                   rtol=rtol, atol=atol, is_variational=True, \
                                   T_var_guess=2*np.pi*0.9, var_rtol=rtol, var_atol=atol)
    print('----------------------------------------------------------------------------------')
    var_sol_be = be_var_solver.solve()
    print('----------------------------------------------------------------------------------')
    var_sol_trap = trap_var_solver.solve()
    print('----------------------------------------------------------------------------------')

    eig,_ = np.linalg.eig(np.reshape(sol['y'][2:,-1],(2,2)))
    print('         correct eigenvalues:', eig)
    eig,_ = np.linalg.eig(np.reshape(var_sol_be['y'][2:,-1],(2,2)))
    print('  BE approximate eigenvalues:', eig)
    eig,_ = np.linalg.eig(np.reshape(var_sol_trap['y'][2:,-1],(2,2)))
    print('TRAP approximate eigenvalues:', eig)

    plt.subplot(1,2,1)
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(var_sol_be['t'],var_sol_be['y'][0],'ro')
    plt.plot(var_sol_trap['t'],var_sol_trap['y'][0],'go')
    plt.subplot(1,2,2)
    plt.plot(t_span_var,[0,0],'b')
    plt.plot(sol['t'],sol['y'][2],'k')
    plt.plot(var_sol_be['t'],var_sol_be['y'][2],'ro')
    for i in range(0,len(var_sol_be['var']['t']),3):
        plt.plot(var_sol_be['var']['t'][i:i+3],var_sol_be['var']['y'][0,i:i+3],'c.-')
    plt.plot(var_sol_trap['t'],var_sol_trap['y'][2],'go')
    for i in range(0,len(var_sol_trap['var']['t']),3):
        plt.plot(var_sol_trap['var']['t'][i:i+3],var_sol_trap['var']['y'][0,i:i+3],'m.-')
    plt.show()


if __name__ == '__main__':
    #system()
    envelope()
    #variational_envelope()
