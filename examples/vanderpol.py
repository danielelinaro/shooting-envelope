
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from polimi.envelope import BEEnvelope, TrapEnvelope


def main():
    from polimi import vdp, vdp_jac

    def variational_system(fun, jac, t, y, T):
        N = int((-1 + np.sqrt(1 + 4*len(y))) / 2)
        J = jac(t*T,y[:N])
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
    main()
