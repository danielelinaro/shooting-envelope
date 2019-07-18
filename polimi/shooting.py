

__all__ = ['BaseShooting', 'Shooting', 'EnvelopeShooting']


import numpy as np
from numpy.linalg import solve
from scipy.integrate import solve_ivp
from . import envelope
#from envelope import common


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
                J,self.jac_factor = envelope.num_jac(self.fun, t, y, f, self.atol, self.jac_factor, None)
                return J
        self.jac = jac

        self.tol = tol
        self.rtol = rtol
        self.atol = atol

        self.integrations = []


    def _extended_system(self, t, y):
        N = self.N
        T = self.T
        J = self.jac(t,y[:N])
        phi = np.reshape(y[N:N+N**2],(N,N))
        ydot = np.concatenate((T * self.fun(t * T, y[:N]), \
                               T * (J @ phi).flatten()))
        if self.estimate_T:
            ydot = np.concatenate((ydot, T * (J @ y[-N:]) + self.fun(t,y[:N])))
        return ydot


    def _integrate(self, y0):
        raise NotImplementedError


    def run(self, y0, max_iter=100):
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
            self.integrations.append(sol)

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
            print('BaseShooting.run({})> error = {}.'.format(i+1,np.abs(y0_new - y0)))
            if np.all(np.abs(y0_new-y0) < self.tol):
                break
            y0 = y0_new

        sol = {'y0': y0_new[:N], 'phi': phi, 'n_iter': i+1,
               'integrations': self.integrations}

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

    def __init__(self, fun, N, T, estimate_T, small_T, jac, \
                 shooting_tol=1e-3, env_rtol=1e-1, env_atol=1e-3, \
                 var_rtol=1e-1, var_atol=1e-2, fun_rtol=1e-5, fun_atol=1e-7, \
                 env_solver=None, env_fun_method='RK45', **kwargs):
        super(EnvelopeShooting, self).__init__(fun, N, T, estimate_T, jac,
                                               shooting_tol, fun_rtol, fun_atol)
        self.T_large = T
        self.T_small = small_T
        self.env_rtol,self.env_atol = env_rtol,env_atol
        self.var_rtol,self.var_atol = var_rtol,var_atol

        self.env_solver = env_solver
        if self.env_solver is None:
            self.env_solver = envelope.TrapEnvelope

        self.env_fun_method = env_fun_method
        self.env_kwargs = kwargs


    def _integrate(self, y0):
        solver = self.env_solver(self.fun, [0,self.T_large], y0[:self.N], \
                                 T_guess=None, T=self.T_small, jac=self.jac, \
                                 rtol=self.env_rtol, atol=self.env_atol, \
                                 fun_method=self.env_fun_method, \
                                 fun_rtol=self.rtol, fun_atol=self.atol, \
                                 is_variational=True, T_var_guess=None, T_var=None, \
                                 var_rtol=self.var_rtol, var_atol=self.var_atol, \
                                 **self.env_kwargs)
        return solver.solve()
