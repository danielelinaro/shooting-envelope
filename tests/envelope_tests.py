
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from polimi.envelope import BEEnvelope, TrapEnvelope, VariationalEnvelope


# for saving data
pack = lambda t,y: np.concatenate((np.reshape(t,(len(t),1)),y.transpose()),axis=1)


def autonomous():
    from polimi.systems import vdp
    epsilon = 1e-3
    A = [0]
    T = [1]
    fun = lambda t,y: vdp(t,y,epsilon,A,T)
    t_span = [0,1000*2*np.pi]
    y0 = [2e-3,0]
    T_guess = 2*np.pi*0.9
    be_solver = BEEnvelope(fun, t_span, y0, T_guess, rtol=1e-3, atol=1e-6)
    trap_solver = TrapEnvelope(fun, t_span, y0, T_guess, rtol=1e-3, atol=1e-6)
    sol_be = be_solver.solve()
    sol_trap = trap_solver.solve()
    sol = solve_ivp(fun, [t_span[0],t_span[-1]], y0, method='BDF', rtol=1e-8, atol=1e-10)

    #np.savetxt('vdp_autonomous.txt', pack(sol['t'],sol['y']), fmt='%.3e')
    #np.savetxt('vdp_autonomous_envelope_BE.txt', pack(sol_be['t'],sol_be['y']), fmt='%.3e')
    #np.savetxt('vdp_autonomous_envelope_trap.txt', pack(sol_trap['t'],sol_trap['y']), fmt='%.3e')

    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(sol_be['t'],sol_be['y'][0],'ro-')
    plt.plot(sol_trap['t'],sol_trap['y'][0],'go-')
    plt.show()


def forced_polar():
    from polimi.systems import vdp_auto
    epsilon = 1e-3
    T_exact = 10
    T_guess = 0.9 * T_exact
    A = [5]
    T = [T_exact]
    rtol = {'fun': 1e-8, 'env': 1e-3}
    atol = {'fun': 1e-10, 'env': 1e-6}

    y0 = [2e-3,0]
    for i in range(len(A)):
        y0.append(1.)
        y0.append(0.)
    fun = lambda t,y: vdp_auto(t,y,epsilon,A,T)
    method = 'RK45'

    t0 = 0
    ttran = 200
    if ttran > 0:
        print('Integrating the full system (transient)...')
        tran = solve_ivp(fun, [t0,ttran], y0, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
        t0 = tran['t'][-1]
        y0 = tran['y'][:,-1]
        plt.plot(tran['t'],tran['y'][0],'k')
        plt.plot(tran['t'],tran['y'][2],'r')
        plt.show()

    print('t0 =',t0)
    print('y0 =',y0)

    t_span = [t0,t0+2000]
    be_solver = BEEnvelope(fun, t_span, y0, T_guess, rtol=rtol['env'], atol=atol['env'])
    trap_solver = TrapEnvelope(fun, t_span, y0, T_guess, rtol=rtol['env'], atol=atol['env'])
    sol_be = be_solver.solve()
    sol_trap = trap_solver.solve()
    sol = solve_ivp(fun, t_span, y0, method='BDF', rtol=1e-8, atol=1e-10)
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(sol_be['t'],sol_be['y'][0],'ro-')
    plt.plot(sol_trap['t'],sol_trap['y'][0],'go-')
    plt.show()


def forced():
    from polimi.systems import vdp
    epsilon = 1e-3
    T_exact = 10
    T_guess = 0.9 * T_exact
    A = [1,10]
    #A = [10,1]
    T = [T_exact,T_exact*100]
    rtol = {'fun': 1e-8, 'env': 1e-1}
    atol = {'fun': 1e-10, 'env': 1e-3}

    y0 = [2e-3,0]
    fun = lambda t,y: vdp(t,y,epsilon,A,T)
    method = 'RK45'

    t0 = 0
    ttran = 1000
    if ttran > 0:
        print('Integrating the full system (transient)...')
        tran = solve_ivp(fun, [t0,ttran], y0, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
        t0 = tran['t'][-1]
        y0 = tran['y'][:,-1]
        plt.plot(tran['t'],tran['y'][0],'k')
        plt.plot(tran['t'],tran['y'][1],'r')
        plt.show()

    print('t0 =',t0)
    print('y0 =',y0)

    t_span = [t0,t0+T[1]]
    be_solver = BEEnvelope(fun, t_span, y0, T_guess, T=T_exact, rtol=rtol['env'], atol=atol['env'])
    trap_solver = TrapEnvelope(fun, t_span, y0, T_guess, T=T_exact, rtol=rtol['env'], atol=atol['env'])
    sol_be = be_solver.solve()
    print('The number of integrated periods of the original system with BE is %d.' % be_solver.original_fun_period_eval)
    sol_trap = trap_solver.solve()
    print('The number of integrated periods of the original system with TRAP is %d.' % trap_solver.original_fun_period_eval)
    sol = solve_ivp(fun, t_span, y0, method='RK45', rtol=1e-8, atol=1e-10)

    np.savetxt('vdp_forced_T=[{},{}]_A=[{},{}].txt'.format(T[0],T[1],A[0],A[1]), \
               pack(sol['t'],sol['y']), fmt='%.3e')
    np.savetxt('vdp_forced_envelope_BE_T=[{},{}]_A=[{},{}].txt'.format(T[0],T[1],A[0],A[1]), \
               pack(sol_be['t'],sol_be['y']), fmt='%.3e')
    np.savetxt('vdp_forced_envelope_trap_T=[{},{}]_A=[{},{}].txt'.format(T[0],T[1],A[0],A[1]), \
               pack(sol_trap['t'],sol_trap['y']), fmt='%.3e')

    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(sol_be['t'],sol_be['y'][0],'ro-')
    plt.plot(sol_trap['t'],sol_trap['y'][0],'go-')
    plt.show()


def hr():
    from polimi.systems import hr
    b = 3
    I = 5
    fun = lambda t,y: hr(t,y,I,b)

    y0 = [0,1,0.1]
    t_tran = 100
    sol = solve_ivp(fun, [0,t_tran], y0, method='RK45', rtol=1e-8, atol=1e-10)
    y0 = sol['y'][:,-1]

    t_span = [0,5000]
    T_guess = 11

    be_solver = BEEnvelope(fun, t_span, y0, T_guess, rtol=1e-3, atol=1e-6,
                           fun_rtol=1e-8, fun_atol=1e-10)
    trap_solver = TrapEnvelope(fun, t_span, y0, T_guess, rtol=1e-3, atol=1e-6,
                               fun_rtol=1e-8, fun_atol=1e-10, vars_to_use=[0,1])
    sol_be = be_solver.solve()
    sol_trap = trap_solver.solve()
    sol = solve_ivp(fun, [t_span[0],t_span[-1]], y0, method='RK45', rtol=1e-8, atol=1e-10)

    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(sol_be['t'],sol_be['y'][0],'ro-')
    plt.plot(sol_trap['t'],sol_trap['y'][0],'go-')
    plt.plot(sol_trap['t'],sol_trap['T'],'ms-')
    plt.show()


def variational():
    from polimi.systems import vdp, vdp_jac

    def variational_system(fun, jac, t, y, T):
        N = int((-1 + np.sqrt(1 + 4*len(y))) / 2)
        J = jac(t,y[:N])
        phi = np.reshape(y[N:N+N**2],(N,N))
        return np.concatenate((T * fun(t*T, y[:N]), \
                               T * np.matmul(J,phi).flatten()))

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
    y0 = np.array([-5.8133754 ,  0.13476983])
    y0_var = np.concatenate((y0,np.eye(len(y0)).flatten()))

    sol = solve_ivp(var_fun, t_span_var, y0_var, rtol=1e-7, atol=1e-8, dense_output=True)

    be_solver = VariationalEnvelope(fun, jac, y0, T_large, T_small,
                                    rtol=1e-1, atol=1e-2,
                                    env_solver=BEEnvelope)
    trap_solver = VariationalEnvelope(fun, jac, y0, T_large, T_small,
                                      rtol=1e-1, atol=1e-2,
                                      env_solver=TrapEnvelope)
    sol_be = be_solver.solve()
    sol_trap = trap_solver.solve()

    eig,_ = np.linalg.eig(np.reshape(sol['y'][2:,-1],(2,2)))
    print('         correct eigenvalues:', eig)
    eig,_ = np.linalg.eig(np.reshape(sol_be['y'][2:,-1],(2,2)))
    print('  BE approximate eigenvalues:', eig)
    eig,_ = np.linalg.eig(np.reshape(sol_trap['y'][2:,-1],(2,2)))
    print('TRAP approximate eigenvalues:', eig)

    plt.subplot(1,2,1)
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(sol_be['t'],sol_be['y'][0],'ro')
    plt.plot(sol_trap['t'],sol_trap['y'][0],'go')
    plt.subplot(1,2,2)
    plt.plot(t_span_var,[0,0],'b')
    plt.plot(sol['t'],sol['y'][2],'k')
    plt.plot(sol_be['t'],sol_be['y'][2],'ro')
    plt.plot(sol_trap['t'],sol_trap['y'][2],'go')
    plt.show()


def main():
    #autonomous()
    #forced_polar()
    #forced()
    #hr()
    variational()


if __name__ == '__main__':
    main()
