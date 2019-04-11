
import numpy as np
from numpy.linalg import solve
from scipy.integrate import solve_ivp
from common import num_jac
import envel


class BaseShooting (object):

    def __init__(self, fun, y0_guess, T, estimate_T, jac=None, tol=1e-3, rtol=1e-5, atol=1e-7):
        # original number of dimensions of the system
        self.N = len(y0_guess)
        self.fun = fun
        self.X = y0_guess

        if estimate_T:
            self.X = np.append(self.X,T)
        else:
            # the period is given
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
        self.estimate_T = estimate_T


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


    def run(self, max_iter=100, do_plot=False):
        if do_plot:
            import matplotlib.pyplot as plt
        N = self.N
        X = self.X
        for i in range(max_iter):
            y0_ext = np.concatenate((X[:N],np.eye(N).flatten()))
            if self.estimate_T:
                y0_ext = np.concatenate((y0_ext,np.zeros(N)))
                self.T = X[-1]
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
            X_new = X - solve(M,r)
            if do_plot:
                if i == 0:
                    fig,(ax1,ax2) = plt.subplots(1,2)
                    ax1.plot(sol['t'],sol['y'][0,:],'r')
                    ax2.plot(sol['y'][1,:],sol['y'][0,:],'r')
                else:
                    if np.all(np.abs(X_new-X) < self.tol):
                        ax1.plot(sol['t'],sol['y'][0,:],'k')
                    else:
                        ax1.plot(sol['t'],sol['y'][0,:])
            if np.all(np.abs(X_new-X) < self.tol):
                break
            X = X_new

        if do_plot:
            ax1.set_xlabel('Time')
            ax1.set_ylabel('x')
            ax2.plot(sol['y'][1,:],sol['y'][0,:],'k')
            ax2.set_xlabel('y')

        sol = {'y0': X_new[:N], 'phi': phi, 'n_iter': i+1,
               't': sol['t'], 'y': sol['y']}
        if self.estimate_T:
            sol['T'] = X_new[-1]
        else:
            sol['T'] = self.T

        return sol


class Shooting (BaseShooting):

    def __init__(self, fun, y0_guess, T, estimate_T, jac=None, tol=1e-3, rtol=1e-5, atol=1e-7):
        super(Shooting, self).__init__(fun, y0_guess, T, estimate_T, jac, tol, rtol, atol)


    def _integrate(self, y0):
        return solve_ivp(self._extended_system, [0,1], y0, atol=self.atol, rtol=self.rtol)


class EnvelopeShooting (BaseShooting):

    def __init__(self, fun, y0_guess, T, estimate_T, small_T, jac=None, shooting_tol=1e-3,
                 env_rtol=1e-1, env_atol=1e-3, fun_rtol=1e-5, fun_atol=1e-7):
        super(EnvelopeShooting, self).__init__(fun, y0_guess, T, estimate_T,
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


def autonomous(with_jac=True):
    from systems import vdp, vdp_jac
    import matplotlib.pyplot as plt
    estimate_T = True
    epsilon = 1e-3
    T = [2*np.pi]
    A = [0]
    y0_guess = [-2,3]
    T_guess = 0.6*T[0]

    if with_jac:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         y0_guess, T_guess, estimate_T,
                         lambda t,y: vdp_jac(t,y,epsilon),
                         rtol=1e-6, atol=1e-8)
    else:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         y0_guess, T_guess, estimate_T,
                         rtol=1e-6, atol=1e-8)

    sol = shoot.run(do_plot=True)
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
    T = [10.]
    A = [1.2]
    y0_guess = [-1,2]
    if with_jac:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         y0_guess, T[0], estimate_T,
                         lambda t,y: vdp_jac(t,y,epsilon),
                         rtol=1e-6, atol=1e-8)
    else:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         y0_guess, T[0], estimate_T,
                         rtol=1e-6, atol=1e-8)

    sol = shoot.run(do_plot=True)
    print('Number of iterations: %d.' % sol['n_iter'])
    plt.show()


def forced_two_frequencies(with_jac=True):
    from systems import vdp, vdp_jac
    import matplotlib.pyplot as plt
    estimate_T = False
    epsilon = 1e-3
    T = [4,400]
    A = [10,1]
    y0_guess = [-2,0]

    if with_jac:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         y0_guess, np.max(T), estimate_T,
                         lambda t,y: vdp_jac(t,y,epsilon),
                         rtol=1e-8, atol=1e-10)
    else:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         y0_guess, np.max(T), estimate_T,
                         rtol=1e-8, atol=1e-10)

    sol = shoot.run(do_plot=False)
    print('Number of iterations: %d.' % sol['n_iter'])
    #plt.show()
    return sol


def forced_envelope():
    from systems import vdp, vdp_jac
    import matplotlib.pyplot as plt
    estimate_T = False
    epsilon = 1e-3
    T = [4,400]
    A = [10,1]
    y0_guess = [-2,0]
    T_small = np.min(T)
    T_large = np.max(T)

    shoot = EnvelopeShooting(lambda t,y: vdp(t,y,epsilon,A,T),
                             y0_guess, T_large, estimate_T, T_small,
                             lambda t,y: vdp_jac(t,y,epsilon),
                             shooting_tol=1e-3,
                             env_rtol=1e-3, env_atol=1e-5,
                             fun_rtol=1e-8, fun_atol=1e-10)

    sol = shoot.run(do_plot=False)
    print('Number of iterations: %d.' % sol['n_iter'])
    #plt.show()
    return sol


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #normalized()
    #autonomous()
    #forced()
    sol = forced_two_frequencies()
    sol_env = forced_envelope()
    plt.plot(sol['t'],sol['y'][0,:],'k')
    plt.plot(sol_env['t'],sol_env['y'][0,:],'ro')
    plt.show()

