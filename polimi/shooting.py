
import numpy as np
from numpy.linalg import inv, solve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from systems import jacobian_finite_differences
import envel
import ipdb


def _extended_system(t, y, fun, jac, T, autonomous):
    if autonomous:
        N = int(np.max(np.roots([1,2,-len(y)])))
    else:        
        N = int(np.max(np.roots([1,1,-len(y)])))
    J = jac(t,y[:N])
    phi = np.reshape(y[N:N+N**2],(N,N))
    ydot = np.concatenate((T*fun(t*T,y[:N]),T*np.matmul(J,phi).flatten()))
    if autonomous:
        dxdt = y[-N:]
        ydot = np.concatenate((ydot,T*np.matmul(J,dxdt)+fun(t,y[:N])))
    return ydot


def shooting(fun, y0_guess, T_guess, autonomous, jac=None, max_iter=100, tol=1e-3, rtol=1e-5, atol=1e-7, do_plot=False):
    # original number of dimensions of the system
    N = len(y0_guess)
    # number of dimensions of the extended system
    N_ext = N**2 + N
    X = y0_guess
    if autonomous:
        N_ext += N
        X = np.append(X,T_guess)
    else:
        # the period is fixed if the system is non-autonomous
        T = T_guess
    if jac is None:
        import common
        global jac_factor
        jac_factor = None
        def jac(t,y):
            global jac_factor
            f = fun(t,y)
            J,jac_factor = common.num_jac(fun, t, y, f, atol, jac_factor, None)
            return J
    for i in range(max_iter):
        y0_ext = np.concatenate((X[:N],np.eye(N).flatten()))
        if autonomous:
            y0_ext = np.concatenate((y0_ext,np.zeros(N)))
            T = X[-1]
        sol = solve_ivp(lambda t,y: _extended_system(t,y,fun,jac,T,autonomous),
                        [0,1], y0_ext, atol=atol, rtol=rtol)
        r = np.array([x[-1]-x[0] for x in sol['y'][:N]])
        phi = np.reshape(sol['y'][N:N**2+N,-1],(N,N))
        if autonomous:
            b = sol['y'][-N:,-1]
            M = np.zeros((N+1,N+1))
            M[:N,:N] = phi - np.eye(N)
            M[:N,-1] = b
            M[-1,:N] = b
            r = np.append(r,0.)
        else:
            M = phi - np.eye(N)
        X_new = X - solve(M,r)
        if do_plot:
            if i == 0:
                fig,(ax1,ax2) = plt.subplots(1,2)
                ax1.plot(sol['t'],sol['y'][0,:],'r')
                ax2.plot(sol['y'][1,:],sol['y'][0,:],'r')
            else:
                if np.all(np.abs(X_new-X) < tol):
                    ax1.plot(sol['t'],sol['y'][0,:],'k')
                else:
                    ax1.plot(sol['t'],sol['y'][0,:])
        if np.all(np.abs(X_new-X) < tol):
            break
        X = X_new
    if do_plot:
        ax1.set_xlabel('Time')
        ax1.set_ylabel('x')
        ax2.plot(sol['y'][1,:],sol['y'][0,:],'k')
        ax2.set_xlabel('y')
        plt.show()

    if autonomous:
        return X[:N],X[-1],phi,i+1
    return X,phi,i+1


def shooting_envelope(fun, y0_guess, T_guess, small_T,
                      autonomous, fun_jac=None,
                      max_iter=100, shooting_tol=1e-3,
                      env_rtol=1e-1, env_atol=1e-3,
                      fun_rtol=1e-5, fun_atol=1e-7,
                      do_plot=False):
    # original number of dimensions of the system
    N = len(y0_guess)
    # number of dimensions of the extended system
    N_ext = N**2 + N
    X = y0_guess
    if autonomous:
        N_ext += N
        X = np.append(X,T_guess)
    else:
        # the period is fixed if the system is non-autonomous
        T = T_guess

    if fun_jac is None:
        import common
        global jac_factor
        jac_factor = None
        def fun_jac(t,y):
            global jac_factor
            f = fun(t,y)
            J,jac_factor = common.num_jac(fun, t, y, f, fun_atol, jac_factor, None)
            return J

    for i in range(max_iter):
        y0_ext = np.concatenate((X[:N],np.eye(N).flatten()))
        if autonomous:
            y0_ext = np.concatenate((y0_ext,np.zeros(N)))
            T = X[-1]
        trap_solver = envel.TrapEnvelope(lambda t,y: _extended_system(t,y,fun,fun_jac,T,autonomous),
                                         [0,1], y0_ext, None, T=small_T/T, rtol=env_rtol, atol=env_atol,
                                         fun_rtol=fun_rtol, fun_atol=fun_atol)
        sol = trap_solver.solve()
        r = np.array([x[-1]-x[0] for x in sol['y'][:N]])
        phi = np.reshape(sol['y'][N:N**2+N,-1],(N,N))
        if autonomous:
            b = sol['y'][-N:,-1]
            M = np.zeros((N+1,N+1))
            M[:N,:N] = phi - np.eye(N)
            M[:N,-1] = b
            M[-1,:N] = b
            r = np.append(r,0.)
        else:
            M = phi - np.eye(N)
        X_new = X - solve(M,r)
        if do_plot:
            if i == 0:
                fig,(ax1,ax2) = plt.subplots(1,2)
                ax1.plot(sol['t'],sol['y'][0,:],'r')
                ax2.plot(sol['y'][1,:],sol['y'][0,:],'r')
            else:
                if np.all(np.abs(X_new-X) < shooting_tol):
                    ax1.plot(sol['t'],sol['y'][0,:],'k')
                else:
                    ax1.plot(sol['t'],sol['y'][0,:])
        if np.all(np.abs(X_new-X) < shooting_tol):
            break
        X = X_new
    if do_plot:
        ax1.set_xlabel('Time')
        ax1.set_ylabel('x')
        ax2.plot(sol['y'][1,:],sol['y'][0,:],'k')
        ax2.set_xlabel('y')
        plt.show()

    if autonomous:
        return X[:N],X[-1],phi,i+1
    return X,phi,i+1


def autonomous(with_jac=True):
    import systems
    autonomous = True
    epsilon = 0.001
    T = [2*np.pi]
    A = [0]
    y0_guess = [-2,3]
    T_guess = 0.6*T[0]
    if with_jac:
        y0_opt,T,phi,n_iter = shooting(lambda t,y: systems.vdp(t,y,epsilon,A,T),
                                       y0_guess, T_guess, autonomous,
                                       lambda t,y: systems.vdp_jac(t,y,epsilon),
                                       rtol=1e-6, atol=1e-8, do_plot=True)
    else:
        y0_opt,T,phi,n_iter = shooting(lambda t,y: systems.vdp(t,y,epsilon,A,T),
                                       y0_guess, T_guess, autonomous, rtol=1e-6,
                                       atol=1e-8, do_plot=True)
    floquet_multi,_ = np.linalg.eig(phi)
    print('T = %g.' % T)
    print('eig(Phi) = (%f,%f).' % tuple(floquet_multi))
    print('Number of iterations: %d.' % n_iter)


def forced(with_jac=True):
    import systems
    autonomous = False
    epsilon = 0.001
    T = [10.]
    A = [1.2]
    y0_guess = [-1,2]
    if with_jac:
        y0_opt,phi,n_iter = shooting(lambda t,y: systems.vdp(t,y,epsilon,A,T),
                                     y0_guess, T[0], autonomous,
                                     lambda t,y: systems.vdp_jac(t,y,epsilon),
                                     rtol=1e-6, atol=1e-8, do_plot=True)
    else:
        y0_opt,phi,n_iter = shooting(lambda t,y: systems.vdp(t,y,epsilon,A,T),
                                     y0_guess, T[0], autonomous, rtol=1e-6,
                                     atol=1e-8, do_plot=True)
    print('Number of iterations: %d.' % n_iter)


def forced_two_frequencies(with_jac=True):
    import systems
    autonomous = False
    epsilon = 0.001
    T = [10.,200.]
    A = [1.2,1.2]
    # a point on the steady-state cycle
    pt_on_cycle = np.array([3.187493,-0.005534])
    y0_guess = [-2,0]
    if with_jac:
        y0_opt,phi,n_iter = shooting(lambda t,y: systems.vdp(t,y,epsilon,A,T),
                                     y0_guess, np.max(T), autonomous,
                                     lambda t,y: systems.vdp_jac(t,y,epsilon),
                                     rtol=1e-6, atol=1e-8, do_plot=True)
    else:
        y0_opt,phi,n_iter = shooting(lambda t,y: systems.vdp(t,y,epsilon,A,T),
                                     y0_guess, np.max(T), autonomous, jac=None,
                                     rtol=1e-6, atol=1e-8, do_plot=True)
    print('Number of iterations: %d.' % n_iter)


def forced_envelope():
    import systems
    import envel
    epsilon = 1e-3
    T = [10,1000]
    A = [10,1]
    rtol = {'fun': 1e-8, 'env': 1e-1}
    atol = {'fun': 1e-10, 'env': 1e-3}
    T_small = np.min(T)
    T_large = np.max(T)

    y0 = [2e-3,0]
    fun = lambda t,y: systems.vdp(t,y,epsilon,A,T)
    method = 'RK45'

    t0 = 0
    ttran = 500
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

    autonomous = False
    y0_guess = y0 * 0.9
    y0_opt,phi,n_iter = shooting_envelope(lambda t,y: systems.vdp(t,y,epsilon,A,T),
                                          y0_guess, T_large, T_small, autonomous,
                                          lambda t,y: systems.vdp_jac(t,y,epsilon),
                                          max_iter=100, shooting_tol=1e-3,
                                          env_rtol=1e-2, env_atol=1e-3,
                                          fun_rtol=1e-5, fun_atol=1e-7,
                                          do_plot=True)
    print('Number of iterations: %d.' % n_iter)


if __name__ == '__main__':
    #autonomous(False)
    #forced(False)
    #forced_two_frequencies(False)
    forced_envelope()

