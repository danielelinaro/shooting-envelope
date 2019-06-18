
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, newton_krylov
from scipy.interpolate import interp1d


DEBUG = True
VERBOSE_DEBUG = False


__all__ = ['EnvelopeSolver', 'BEEnvelope', 'TrapEnvelope', 'VariationalEnvelope', 'EnvelopeInterp']


class EnvelopeInterp (object):

    def __init__(self, fun, sol, T):
        self.fun = fun
        self.sol = sol
        self.T = T
        self.t = None
        self.y = None
        self.total_integrated_time = 0

    def __call__(self, t):
        if t == self.t:
            return self.y
        t0 = np.floor(t/self.T) * self.T
        if np.any(np.abs(self.sol['t'] - t0) < self.T/2):
            idx = np.argmin(np.abs(self.sol['t'] - t0))
            y0 = self.sol['y'][:,idx]
        else:
            start = np.where(self.sol['t'] < t0)[0][-1]
            idx = np.arange(start, start+2)
            y0 = interp1d(self.sol['t'][idx],self.sol['y'][:,idx])(t0)
        sol = solve_ivp(self.fun, [t0,t], y0, method='RK45', rtol=1e-8, atol=1e-8)
        self.total_integrated_time += t-t0
        self.t = t
        self.y = sol['y'][:,-1]
        return self.y


class EnvelopeSolver (object):

    SUCCESS = 0
    DT_TOO_LARGE = 1
    LTE_TOO_LARGE = 2
    

    def __init__(self, fun, t_span, y0, T_guess, T=None, max_step=1000,
                 fun_rtol=1e-6, fun_atol=1e-8, dTtol=1e-2, rtol=1e-3, atol=1e-6,
                 jac=None, jac_sparsity=None, vectorized=False, fun_method='RK45',
                 vars_to_use=[], is_variational=False):
        self.dTtol = dTtol
        self.max_step = np.floor(max_step)
        self.rtol, self.atol = rtol, atol
        self.n_dim = len(y0)
        self.original_fun_rtol, self.original_fun_atol = fun_rtol, fun_atol
        self.original_fun = fun
        self.original_jac = jac
        self.original_jac_sparsity = jac_sparsity
        self.original_fun_vectorized = vectorized
        self.original_fun_method = fun_method
        self.original_fun_period_eval = 0
        self.t_span = t_span
        self.y0 = np.array(y0)
        self.t = np.array([t_span[0]])
        self.y = np.zeros((self.n_dim,1))
        self.y[:,0] = self.y0
        self.f = np.zeros((self.n_dim,1))
        if len(vars_to_use) == 0:
            self.vars_to_use = np.arange(self.n_dim)
        else:
            self.vars_to_use = vars_to_use
        if DEBUG:
            print('Variables to use: {}.'.format(self.vars_to_use))
        self.is_variational = is_variational
        if is_variational:
            self.mono_mat = []
            # the number of dimensions of the original system, i.e. the one
            # without the variational part added
            self.N = int((-1 + np.sqrt(1 + 4*self.n_dim)) / 2)
            if DEBUG:
                print('The number of dimensions of the original system is %d.' % self.N)
        if T is not None:
            self.estimate_T = False
            self.T = T
            self.f[:,0] = self._envelope_fun(self.t[0],self.y[:,0])
        elif T_guess is not None:
            self.estimate_T = True
            self.f[:,0] = self._envelope_fun(self.t[0],self.y[:,0],T_guess)
            self.T = self.T_new
        else:
            raise Exception('T_guess and T cannot both be None')
        self.H = self.T
        self.period = [self.T]
        if DEBUG:
            print('EnvelopeSolver.__init__(%.3f)> T = %.6f' % (self.t_span[0],self.T))


    def solve(self):
        while self.t[-1] < self.t_span[1]:
            # make one step
            flag = self._step()
            # check the result of the step
            if flag == EnvelopeSolver.SUCCESS:
                # the step was successful (LTE and variation in period below threshold
                self.t = np.append(self.t,self.t_next)
                self.y = np.append(self.y,np.reshape(self.y_next,(self.n_dim,1)),axis=1)
                self.f = np.append(self.f,np.reshape(self.f_next,(self.n_dim,1)),axis=1)
                if self.estimate_T:
                    self.T = self.T_new
                self.period.append(self.T)
                msg = 'OK'
            elif flag == EnvelopeSolver.DT_TOO_LARGE:
                # the variation in period was too large
                if self.H > 2*self.T:
                    # reduce the step if this is greater than twice the oscillation period
                    self.H_new = np.floor(np.round(self.H/self.T)/2) * self.T
                    msg = 'T/2'
                else:
                    # otherwise simply move the integration one period forward
                    self._one_period_step()
                    msg = '1T'
            elif flag == EnvelopeSolver.LTE_TOO_LARGE:
                # the LTE was above threshold: _step has already changed the value of H_new
                msg = 'LTE'

            if self.H_new < self.T:
                self._one_period_step()
                msg += ' 1T'
            self.H = self.H_new

            if DEBUG:
                print('EnvelopeSolver.solve(%.3f)> T = %f, H = %f - %s' % (self.t[-1],self.T,self.H,msg))

        idx, = np.where(self.t <= self.t_span[1] + self.T/2)
        sol = {'t': self.t[idx], 'y': self.y[:,idx],
               'T': np.array([self.period[i] for i in idx]),
               'period_eval': self.original_fun_period_eval}
        if self.is_variational:
            sol['M'] = [self.mono_mat[i] for i in idx[:-1]]
            for i,mat in enumerate(sol['M']):
                sol['y'][2:,i+1] = np.dot(np.reshape(sol['y'][2:,i],(self.N,self.N)), mat).flatten()

        return sol


    def _step(self):
        raise NotImplementedError


    def _one_period_step(self):
        self._envelope_fun(self.t[-1],self.y[:,-1])
        self.t = np.append(self.t,self.t_new)
        self.y = np.append(self.y,np.reshape(self.y_new,(self.n_dim,1)),axis=1)
        if self.is_variational:
            M = np.reshape(self.y_new[self.N:],(self.N,self.N)).copy()
            self.mono_mat.append(M)
            self.y[self.N:,-1] = np.eye(self.N).flatten()
        self.f = np.append(self.f,np.reshape(self._envelope_fun(self.t[-1],self.y[:,-1]),(self.n_dim,1)),axis=1)
        if self.estimate_T:
            self.T = self.T_new
        self.H_new = self.T
        self.H = self.T
        self.period.append(self.T)

        
    def _envelope_fun(self,t,y,T_guess=None):
        if not self.estimate_T:
            if self.original_fun_method == 'BDF':
                sol = solve_ivp(self.original_fun,[t,t+self.T],y,
                                self.original_fun_method,jac=self.original_jac,
                                jac_sparsity=self.original_jac_sparsity,
                                vectorized=self.original_fun_vectorized,
                                rtol=self.original_fun_rtol,
                                atol=self.original_fun_atol)
            else:
                sol = solve_ivp(self.original_fun,[t,t+self.T],y,
                                self.original_fun_method,
                                vectorized=self.original_fun_vectorized,
                                rtol=self.original_fun_rtol,
                                atol=self.original_fun_atol)
            self.t_new = t + self.T
            self.y_new = sol['y'][:,-1]
            if VERBOSE_DEBUG:
                print('EnvelopeSolver._envelope_fun(%.3f)> y = (%.6f,%.6f) T = %.6f.' % (t,self.y_new[0],self.y_new[1],self.T))
            # return the "vector field" of the envelope
            self.original_fun_period_eval += 1
            return 1./self.T * (sol['y'][:,-1] - y)

        if T_guess is None:
            T_guess = self.T
        # find the equation of the plane containing y and
        # orthogonal to fun(t,y)
        f = self.original_fun(t,y)
        w = f[self.vars_to_use] / np.linalg.norm(f[self.vars_to_use])
        b = -np.dot(w, y[self.vars_to_use])

        events_fun = lambda t,y: np.dot(w, y[self.vars_to_use]) + b
        events_fun.direction = 1

        if self.original_fun_method == 'BDF':
            sol = solve_ivp(self.original_fun,[t,t+1.5*T_guess],y,
                            self.original_fun_method,jac=self.original_jac,
                            jac_sparsity=self.original_jac_sparsity,
                            vectorized=self.original_fun_vectorized,
                            events=events_fun,dense_output=True,
                            rtol=self.original_fun_rtol,atol=self.original_fun_atol)
        else:
            sol = solve_ivp(self.original_fun,[t,t+1.5*T_guess],y,
                            self.original_fun_method,vectorized=self.original_fun_vectorized,
                            events=events_fun,dense_output=True,
                            rtol=self.original_fun_rtol,atol=self.original_fun_atol)

        T = sol['t_events'][0][1] - t

        try:
            self.T_new = T
        except:
            self.T_new = T_guess
            if DEBUG:
                print('EnvelopeSolver._envelope_fun(%.3f)> T = T_guess = %.6f.' % (t,self.T_new))
        self.t_new = t + self.T_new
        self.y_new = sol['sol'](self.t_new)
        if VERBOSE_DEBUG:
            print('EnvelopeSolver._envelope_fun(%.3f)> y = (%.4f,%.4f) T = %.6f.' % (t,self.y_new[0],self.y_new[1],self.T_new))
        self.original_fun_period_eval += 1.5
        # return the "vector field" of the envelope
        return 1./self.T_new * (self.y_new - sol['sol'](t))


class BEEnvelope (EnvelopeSolver):
    def __init__(self, fun, t_span, y0, T_guess, T=None, max_step=1000,
                 fun_rtol=1e-6, fun_atol=1e-8, dTtol=1e-2, rtol=1e-3, atol=1e-6,
                 jac=None, jac_sparsity=None, vectorized=False, fun_method='RK45',
                 vars_to_use=[], is_variational=False):
        super(BEEnvelope, self).__init__(fun, t_span, y0, T_guess, T, max_step,
                                         fun_rtol, fun_atol, dTtol, rtol, atol,
                                         jac, jac_sparsity, vectorized, fun_method,
                                         vars_to_use, is_variational)


    def _step(self):
        H = self.H

        t_cur = self.t[-1]
        y_cur = self.y[:,-1]
        f_cur = self.f[:,-1]

        if t_cur + H > self.t_span[1]:
            H = self.T * np.max((1,np.floor((self.t_span[1] - t_cur) / self.T)))
        t_next = t_cur + H

        # step size in units of period
        n_periods = int(np.round(H / self.T))

        # which variables to consider when deciding whether to accept or not the
        # step based on the LTE
        vars_to_use = np.arange(self.n_dim)
        if self.is_variational:
            # monodromy matrix
            M = np.reshape(self.y_new[self.N:],(self.N,self.N)).copy()
            vars_to_use = self.vars_to_use

        if n_periods == 1:
            # the step is equal to the period: we don't need to solve the implicit system
            y_next = self.y_new
        else:
            # estimate the next value by extrapolation using explicit Euler
            y_extrap = y_cur + H * f_cur
            # correct the estimate using implicit Euler
            #y_next = fsolve(lambda Y: Y - y_cur - H * self._envelope_fun(t_next,Y), y_extrap, xtol=1e-1)
            if self.is_variational and hasattr(self, '_reduced_fun'):
                y_next = np.zeros(self.n_dim)
                y_next[:self.N] = newton_krylov(lambda Y: Y - y_cur[:self.N] - \
                                                H * self._reduced_fun(t_next,Y), \
                                                y_extrap[:self.N], f_tol=1e-3)
            else:
                y_next = newton_krylov(lambda Y: Y - y_cur - H * self._envelope_fun(t_next,Y), \
                                       y_extrap, f_tol=1e-3)

        if self.estimate_T and np.abs(self.T - self.T_new) > self.dTtol:
            return EnvelopeSolver.DT_TOO_LARGE

        if self.is_variational:
            y_next[self.N:] = np.eye(self.N).flatten()

        # the value of the derivative at the new point
        f_next = self._envelope_fun(t_next,y_next)

        scale = self.atol + self.rtol * np.abs(y_next)
        # compute the local truncation error
        coeff = np.abs(f_next * (f_next - f_cur) / (y_next - y_cur))
        coeff[coeff == 0] = np.min(coeff[coeff > 0])
        lte = H**2 / 2 * coeff

        if self.estimate_T:
            T = self.T_new
        else:
            T = self.T

        # compute the new value of H as the maximum value that allows having an LTE below threshold
        self.H_new = np.min((self.max_step,np.floor(np.min(np.sqrt(2*scale[vars_to_use]/coeff[vars_to_use])) / T))) * T

        if np.any(lte[vars_to_use] > scale[vars_to_use]):
            return EnvelopeSolver.LTE_TOO_LARGE

        if self.is_variational:
            self.mono_mat.append(np.linalg.matrix_power(M,n_periods))

        self.t_next = t_next
        self.y_next = y_next
        self.f_next = f_next

        return EnvelopeSolver.SUCCESS


class TrapEnvelope (EnvelopeSolver):
    def __init__(self, fun, t_span, y0, T_guess, T=None, max_step=1000,
                 fun_rtol=1e-6, fun_atol=1e-8, dTtol=1e-2, rtol=1e-3, atol=1e-6,
                 jac=None, jac_sparsity=None, vectorized=False, fun_method='RK45',
                 vars_to_use=[], is_variational=False):
        super(TrapEnvelope, self).__init__(fun, t_span, y0, T_guess, T, max_step,
                                           fun_rtol, fun_atol, dTtol, rtol, atol,
                                           jac, jac_sparsity, vectorized, fun_method,
                                           vars_to_use, is_variational)
        self.df_cur = np.zeros(self.n_dim)


    def _one_period_step(self):
        super(TrapEnvelope, self)._one_period_step()
        self.df_cur = (self.f[:,-1] - self.f[:,-2]) / (self.y[:,-1] - self.y[:,-2])


    def _step(self):
        H = self.H

        t_cur = self.t[-1]
        y_cur = self.y[:,-1]
        f_cur = self.f[:,-1]
        df_cur = self.df_cur

        if t_cur + H > self.t_span[1]:
            H = self.T * np.max((1,np.floor((self.t_span[1] - t_cur) / self.T)))
        t_next = t_cur + H

        # step size in units of period
        n_periods = int(np.round(H / self.T))

        # which variables to consider when deciding whether to accept or not the
        # step based on the LTE
        vars_to_use = np.arange(self.n_dim)
        if self.is_variational:
            # monodromy matrix
            M = np.reshape(self.y_new[self.N:],(self.N,self.N)).copy()
            vars_to_use = self.vars_to_use

        if n_periods == 1:
            # the step is equal to the period: we don't need to solve the implicit system
            y_next = self.y_new
        else:
            # estimate the next value by extrapolation using explicit Euler
            y_extrap = y_cur + H * f_cur
            # correct the estimate using the trapezoidal rule
            #y_next = fsolve(lambda Y: Y - y_cur - H/2 * (f_cur + self._envelope_fun(t_next,Y)), y_extrap, xtol=1e-1)
            if self.is_variational and hasattr(self, '_reduced_fun'):
                y_next = np.zeros(self.n_dim)
                y_next[:self.N] = newton_krylov(lambda Y: Y - y_cur[:self.N] - H/2 * \
                                                (f_cur[:self.N] + self._reduced_fun(t_next,Y)), \
                                                y_extrap[:self.N], f_tol=1e-3)
            else:
                y_next = newton_krylov(lambda Y: Y - y_cur - H/2 * \
                                       (f_cur + self._envelope_fun(t_next,Y)), \
                                       y_extrap, f_tol=1e-3)

        if self.estimate_T and np.abs(self.T - self.T_new) > self.dTtol:
            return EnvelopeSolver.DT_TOO_LARGE

        if self.is_variational:
            y_next[self.N:] = np.eye(self.N).flatten()

        # the value of the derivative at the new point
        f_next = self._envelope_fun(t_next,y_next)

        scale = self.atol + self.rtol * np.abs(y_next)
        # compute the local truncation error
        df_next = (f_next - f_cur) / (y_next - y_cur)
        d2f_next = (df_next - df_cur) / (y_next - y_cur)
        coeff = np.abs(f_next * (f_next*d2f_next + 2*(df_next**2)))
        coeff[coeff == 0] = np.min(coeff[coeff > 0])
        lte = (H**3)/12 * coeff

        if self.estimate_T:
            T = self.T_new
        else:
            T = self.T

        # compute the new value of H as the maximum value that allows having an LTE below threshold
        self.H_new = np.min((self.max_step,np.floor(np.min((12*scale[vars_to_use]/coeff[vars_to_use])**(1/3)) / T))) * T

        if np.any(lte[vars_to_use] > scale[vars_to_use]):
            return EnvelopeSolver.LTE_TOO_LARGE

        if self.is_variational:
            self.mono_mat.append(np.linalg.matrix_power(M,n_periods))

        self.t_next = t_next
        self.y_next = y_next
        self.f_next = f_next
        self.df_cur = df_next

        return EnvelopeSolver.SUCCESS


class VariationalEnvelope (object):

    def _variational_system(self, t, y):
        N = self.N
        T = self.T_large
        J = self.jac(t,y[:N])
        phi = np.reshape(y[N:N+N**2],(N,N))
        return np.concatenate((T * self.fun(t*T, y[:N]), \
                               T * np.matmul(J,phi).flatten()))


    def __init__(self, fun, jac, y0, T_large, T_small, t_span=[0,1], rtol=1e-1, atol=1e-2,
                 vars_to_use=[], env_solver=TrapEnvelope, **kwargs):

        try:
            if kwargs['is_variational']:
                print('Ignoring "is_variational" argument.')
            else:
                print('is_variational is set to False: this does not make sense. Ignoring it.')
        except:
            pass

        if T_small is None or T_small < 0:
            raise ValueError('T_small cannot be None or negative')

        if T_large is None or T_large < 0:
            raise ValueError('T_large cannot be None or negative')

        if 'T_guess' in kwargs:
            print('Ignoring "T_guess" argument.')

        self.N = len(y0)
        self.fun = fun
        self.jac = jac
        self.T_large = T_large

        y0_var = np.concatenate((y0,np.eye(self.N).flatten()))

        if np.isscalar(atol):
            atol += np.zeros(self.N)
        if len(atol) == self.N:
            atol = np.concatenate((atol,1e-2+np.zeros(self.N**2)))

        if len(vars_to_use) == 0 or vars_to_use is None:
            vars_to_use = np.arange(self.N)

        self.variational_solver = env_solver(self._variational_system, t_span, y0_var,
                                             T_guess=None, T=T_small/T_large, jac=jac,
                                             rtol=rtol, atol=atol,
                                             vars_to_use=vars_to_use,
                                             is_variational=True,
                                             **kwargs)

        self.envelope_solver = EnvelopeSolver(lambda t,y: T_large*fun(t*T_large,y),
                                              t_span, y0, T_guess=None, T=T_small/T_large)
        self.variational_solver._reduced_fun = self.envelope_solver._envelope_fun


    def solve(self):
        return self.variational_solver.solve()
