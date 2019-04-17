
import numpy as np
from numpy.linalg import solve
from scipy.integrate import solve_ivp
from common import num_jac
import envel
import ipdb

class BaseShooting (object):

    def __init__(self, fun, N, T, estimate_T, jac=None, tol=1e-3, rtol=1e-5, atol=1e-7):
        # original number of dimensions of the system
        self.N = N
        self.fun = fun
        self.estimate_T = estimate_T
        # if estimate_T is true, self.T is the initial guess for the period
        self.T = T

        if jac is None:
            self.jac_factor = None
            def jac(t,y):
                f = self.fun(t,y)
                J,self.jac_factor = num_jac(self.fun, t, y, f, self.atol, self.jac_factor, None)
                return J
        self.jac = jac

        self.tol = tol
        self.rtol = rtol
        self.atol = atol


    def _extended_system(self, t, y):
        N = self.N
        T = self.T
        J = self.jac(t,y[:N])
        phi = np.reshape(y[N:N+N**2],(N,N))
        ydot = np.concatenate((T * self.fun(t * T, y[:N]), \
                               T * np.matmul(J,phi).flatten()))
        if self.estimate_T:
            ydot = np.concatenate((ydot, T * np.matmul(J,y[-N:]) + self.fun(t,y[:N])))
        return ydot


    def _integrate(self, y0):
        raise NotImplementedError


    def run(self, y0, max_iter=100, do_plot=False):
        if do_plot:
            import matplotlib.pyplot as plt
        N = self.N
        if len(y0) != N:
            raise Exception('y0 has wrong dimensions')
        if self.estimate_T:
            y0 = np.append(y0, self.T)
        for i in range(max_iter):
            print('BaseShooting.run({})>     y0 = {}.'.format(i+1,y0[:N]))
            y0_ext = np.concatenate((y0[:N],np.eye(N).flatten()))
            if self.estimate_T:
                self.T = y0[-1]
                y0_ext = np.concatenate((y0_ext,np.zeros(N)))
            sol = self._integrate(y0_ext)
            r = np.array([x[-1]-x[0] for x in sol['y'][:N]])
            phi = np.reshape(sol['y'][N:N**2+N,-1],(N,N))
            if self.estimate_T:
                b = sol['y'][-N:,-1]
                M = np.zeros((N+1,N+1))
                M[:N,:N] = phi - np.eye(N)
                M[:N,-1] = b
                M[-1,:N] = b
                r = np.append(r,0.)
            else:
                M = phi - np.eye(N)
            y0_new = y0 - solve(M,r)
            print('BaseShooting.run({})> y0_new = {}.'.format(i+1,y0_new[:N]))
            if do_plot:
                if i == 0:
                    fig,(ax1,ax2) = plt.subplots(1,2)
                    ax1.plot(sol['t'],sol['y'][0,:],'r')
                    ax2.plot(sol['y'][1,:],sol['y'][0,:],'r')
                else:
                    if np.all(np.abs(y0_new - y0) < self.tol):
                        ax1.plot(sol['t'],sol['y'][0,:],'k')
                    else:
                        ax1.plot(sol['t'],sol['y'][0,:])
            print('BaseShooting.run({})> error = {}.'.format(i+1,np.abs(y0_new - y0)))
            if np.all(np.abs(y0_new-y0) < self.tol):
                break
            y0 = y0_new

        if do_plot:
            ax1.set_xlabel('Time')
            ax1.set_ylabel('x')
            ax2.plot(sol['y'][1,:],sol['y'][0,:],'k')
            ax2.set_xlabel('y')

        sol = {'y0': y0_new[:N], 'phi': phi, 'n_iter': i+1,
               't': sol['t'], 'y': sol['y']}
        if self.estimate_T:
            sol['T'] = y0_new[-1]
        else:
            sol['T'] = self.T

        return sol


class Shooting (BaseShooting):

    def __init__(self, fun, N, T, estimate_T, jac=None, tol=1e-3, rtol=1e-5, atol=1e-7):
        super(Shooting, self).__init__(fun, N, T, estimate_T, jac, tol, rtol, atol)


    def _integrate(self, y0):
        return solve_ivp(self._extended_system, [0,1], y0, atol=self.atol, rtol=self.rtol)


class EnvelopeShooting (BaseShooting):

    def __init__(self, fun, N, T, estimate_T, small_T, jac=None, shooting_tol=1e-3,
                 env_rtol=1e-1, env_atol=1e-3, fun_rtol=1e-5, fun_atol=1e-7):
        super(EnvelopeShooting, self).__init__(fun, N, T, estimate_T,
                                               jac, shooting_tol, fun_rtol, fun_atol)
        self.small_T = small_T
        self.env_rtol = env_rtol
        self.env_atol = env_atol
        from envel import TrapEnvelope
        self.EnvSolver = TrapEnvelope


    def _integrate(self, y0):
        solver = self.EnvSolver(self._extended_system, [0,1], y0, None,
                                T=self.small_T/self.T, rtol=self.env_rtol,
                                atol=self.env_atol, fun_rtol=self.rtol,
                                fun_atol=self.atol)
        return solver.solve()


def normalized():
    import matplotlib.pyplot as plt
    from systems import vdp, vdp_jac
    epsilon = 1e-3
    A = [10,2]
    T = [10,200]
    fun = lambda t,y: vdp(t,y,epsilon,A,T)
    fun_norm = lambda t,y: np.max(T) * vdp(t*np.max(T), y, epsilon, A, T)
    t_span = [0,3*np.max(T)]
    y0 = [2e-3,0]
    tran = solve_ivp(fun, t_span, y0, rtol=1e-6, atol=1e-8)
    t0 = 0
    y0 = tran['y'][:,-1]
    t_span = [t0, t0+np.max(T)]
    sol = solve_ivp(fun, t_span, y0, rtol=1e-6, atol=1e-8)
    t_span = [0,1]
    sol_norm = solve_ivp(fun_norm, t_span, y0, rtol=1e-6, atol=1e-8)
    plt.plot(sol['t']/np.max(T),sol['y'][0],'k',label='Original')
    plt.plot(sol_norm['t'],sol_norm['y'][0],'r',label='Normalized')
    plt.legend(loc='best')
    plt.show()


def autonomous(with_jac=True):
    from systems import vdp, vdp_jac
    import matplotlib.pyplot as plt
    estimate_T = True
    epsilon = 1e-3
    A = [0]
    T = [2*np.pi]
    T_guess = 0.6*T[0]
    y0_guess = [-2,3]
    N = 2

    if with_jac:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         N, T_guess, estimate_T,
                         lambda t,y: vdp_jac(t,y,epsilon),
                         rtol=1e-6, atol=1e-8)
    else:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         N, T_guess, estimate_T,
                         rtol=1e-6, atol=1e-8)

    sol = shoot.run(y0_guess, do_plot=True)
    floquet_multi,_ = np.linalg.eig(sol['phi'])
    print('T = %g.' % sol['T'])
    print('eig(Phi) = (%f,%f).' % tuple(floquet_multi))
    print('Number of iterations: %d.' % sol['n_iter'])
    plt.show()


def forced(with_jac=True):
    from systems import vdp, vdp_jac
    import matplotlib.pyplot as plt
    estimate_T = False
    epsilon = 1e-3
    A = [1.2]
    T = [10.]
    y0_guess = [-1,2]
    N = 2

    if with_jac:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         N, T[0], estimate_T,
                         lambda t,y: vdp_jac(t,y,epsilon),
                         rtol=1e-6, atol=1e-8)
    else:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         N, T[0], estimate_T,
                         rtol=1e-6, atol=1e-8)

    sol = shoot.run(y0_guess, do_plot=True)
    print('Number of iterations: %d.' % sol['n_iter'])
    plt.show()


def forced_two_frequencies(with_jac=True):
    from systems import vdp, vdp_jac
    import matplotlib.pyplot as plt
    estimate_T = False
    epsilon = 1e-3
    A = [10,1]
    T = [4,400]
    y0_guess = [-2,0]
    N = 2

    if with_jac:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         N, np.max(T), estimate_T,
                         lambda t,y: vdp_jac(t,y,epsilon),
                         rtol=1e-8, atol=1e-10)
    else:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         N, np.max(T), estimate_T,
                         rtol=1e-8, atol=1e-10)

    sol = shoot.run(y0_guess, do_plot=False)
    print('Number of iterations: %d.' % sol['n_iter'])
    #plt.show()
    return sol


def forced_envelope():
    from systems import vdp, vdp_jac
    import matplotlib.pyplot as plt
    estimate_T = False
    epsilon = 1e-3
    A = [10,1]
    T = [4,400]
    T_small = np.min(T)
    T_large = np.max(T)
    y0_guess = [-2,0]
    N = 2

    shoot = EnvelopeShooting(lambda t,y: vdp(t,y,epsilon,A,T),
                             N, T_large, estimate_T, T_small,
                             lambda t,y: vdp_jac(t,y,epsilon),
                             shooting_tol=1e-3,
                             env_rtol=1e-1, env_atol=[1e-2,1e-2,1e-6,1e-6,1e-6,1e-6],
                             fun_rtol=1e-8, fun_atol=1e-10)

    sol = shoot.run(y0_guess, do_plot=False)
    print('Number of iterations: %d.' % sol['n_iter'])
    #plt.show()
    return sol


#def main():
#    import matplotlib.pyplot as plt
#    from envel import TrapEnvelope
#    from systems import vdp, vdp_jac
#
#    sol = forced_two_frequencies()
#    sol_env = forced_envelope()
#
#    plt.plot(sol['t'],sol['y'][2,:],'k')
#    plt.plot(sol_env['t'],sol_env['y'][2,:],'mo')
#    plt.show()


def main():
    from systems import vdp, vdp_jac
    import matplotlib.pyplot as plt
    estimate_T = False
    epsilon = 1e-3
    A = [1]
    T_small = 2*np.pi
    T_large = 100 * T_small
    T = [T_large]
    N = 2

    fun = lambda t,y: vdp(t,y,epsilon,A,T)
    jac = lambda t,y: vdp_jac(t,y,epsilon)

    tran = solve_ivp(fun, [0,20*T_large], [0,1], rtol=1e-6, atol=1e-8)
    y0 = tran['y'][:,-1]
    sol = solve_ivp(fun, [0,T_large], y0, rtol=1e-6, atol=1e-8)
    print('{} -> {}'.format(sol['y'][:,0],sol['y'][:,-1]))

    shoot = Shooting(fun, N, T_large, estimate_T, jac, tol=1e-3, rtol=1e-6, atol=1e-8)

    #plt.figure()
    sol = shoot.run(y0, max_iter=10, do_plot=True)
    print('Number of iterations: %d.' % sol['n_iter'])

    #env_shoot = EnvelopeShooting(lambda t,y: vdp(t,y,epsilon,A,T),
    #                               N, T_large, estimate_T, T_small,
    #                               lambda t,y: vdp_jac(t,y,epsilon),
    #                               shooting_tol=1e-3,
    #                               env_rtol=1e-1, env_atol=[1e-2,1e-2,1e-6,1e-6,1e-6,1e-6],
    #                               fun_rtol=1e-8, fun_atol=1e-10)

    #plt.figure()
    #env_sol = env_shoot.run(y0_guess, max_iter=10, do_plot=True)
    #print('Number of iterations: %d.' % env_sol['n_iter'])

    plt.show()

if __name__ == '__main__':
    #normalized()
    #autonomous()
    #forced()
    #forced_two_frequencies()
    #forced_envelope()
    main()

