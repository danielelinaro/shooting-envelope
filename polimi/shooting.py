

__all__ = ['BaseShooting', 'Shooting', 'EnvelopeShooting']


import numpy as np
from numpy.linalg import solve
from scipy.integrate import solve_ivp
from . import envelope


class BaseShooting (object):

    def __init__(self, system, T, estimate_T, tol=1e-3, solver=solve_ivp, **kwargs):
        # original number of dimensions of the system
        self.system = system
        self.estimate_T = estimate_T
        # if estimate_T is true, self.T is the initial guess for the period
        self.T = T
        self.tol = tol
        self.solver = solver
        self.solver_kwargs = kwargs
        self.integrations = []


    def _integrate(self, y0):
        raise NotImplementedError


    def run(self, y0, max_iter=100):
        N = self.system.n_dim
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
            self.system.variational_T = self.T
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

    def __init__(self, system, T, estimate_T, tol=1e-3, solver=solve_ivp, **kwargs):
        super(Shooting, self).__init__(system, T, estimate_T, tol, solver, **kwargs)


    def _integrate(self, y0):
        with_variational = self.system.with_variational
        self.system.with_variational = True
        sol = self.solver(self.system, [0,1], y0, **self.solver_kwargs)
        self.system.with_variational = with_variational
        return sol


class EnvelopeShooting (BaseShooting):

    def __init__(self, system, T, estimate_T, small_T, \
                 tol=1e-3, env_rtol=1e-1, env_atol=1e-3, \
                 env_max_step=1000, env_vars_to_use=[], \
                 T_var=None, T_var_guess=None, \
                 var_rtol=1e-1, var_atol=1e-2, \
                 env_solver=envelope.TrapEnvelope,
                 fun_solver=solve_ivp, **kwargs):
        super(EnvelopeShooting, self).__init__(system, T, estimate_T, tol, fun_solver, **kwargs)
        self.T_large = T
        self.T_small = small_T
        self.T_var = T_var
        self.T_var_guess = T_var_guess
        self.env_rtol,self.env_atol = env_rtol,env_atol
        self.env_max_step = env_max_step
        self.env_vars_to_use = env_vars_to_use
        self.var_rtol,self.var_atol = var_rtol,var_atol
        self.env_solver = env_solver
        self.fun_solver = fun_solver
        self.fun_kwargs = kwargs


    def _integrate(self, y0):
        with_variational = self.system.with_variational
        self.system.with_variational = True
        solver = self.env_solver(self.system, [0,self.T_large], y0[:self.system.n_dim], \
                                 T_guess=None, T=self.T_small, \
                                 max_step=self.env_max_step, vars_to_use=self.env_vars_to_use, \
                                 env_rtol=self.env_rtol, env_atol=self.env_atol, \
                                 is_variational=True, T_var_guess=self.T_var_guess, \
                                 T_var=self.T_var, var_rtol=self.var_rtol, \
                                 var_atol=self.var_atol, solver=self.fun_solver, \
                                 **self.fun_kwargs)
        sol = solver.solve()
        self.system.with_variational = with_variational
        return sol
