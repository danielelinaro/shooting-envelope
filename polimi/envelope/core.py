
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, newton_krylov
from scipy.interpolate import interp1d


DEBUG = True
VERBOSE_DEBUG = False


NEWTON_KRYLOV_FTOL = 1e-3
FSOLVE_XTOL = 1e-1

__all__ = ['EnvelopeSolver', 'BEEnvelope', 'TrapEnvelope', 'EnvelopeInterp']



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
                 vars_to_use=[], is_variational=False, T_var=None,
                 var_rtol=1e-2, var_atol=1e-3):

        # number of dimensions of the system
        self.n_dim = len(y0)
        # tolerances for the computation of the envelope
        self.rtol, self.atol = rtol, atol
        # tolerances for the integration of the "fast" system
        self.original_fun_rtol, self.original_fun_atol = fun_rtol, fun_atol
        # tolerance on the variation of the estimated period
        self.dTtol = dTtol
        # max_step is in units of periods
        self.max_step = np.floor(max_step)

        # whether to compute the envelope of the variational system
        self.is_variational = is_variational
        
        if not self.is_variational:
            self.original_fun = fun
            self.original_jac = jac
            self.t_span = t_span
        else:
            if jac is None:
                raise Exception('jac cannot be None if is_variational is True')
            if T_var is None:
                raise Exception('T_var cannot be None if is_variational is True')
            if T is None:
                T_guess /= t_span[1]
            else:
                T /= t_span[1]
            self.T_large = t_span[1]
            self.T_var = T_var / self.T_large
            self.t_span = [0,1]
            self.original_fun = lambda t,y: self.T_large * fun(t*self.T_large, y)
            self.original_jac = lambda t,y: jac(t*self.T_large, y)
            self.mono_mat = []
            self.t_var = []
            self.y_var = []
            self.var_atol = var_atol
            self.var_rtol = var_rtol

        self.original_jac_sparsity = jac_sparsity
        self.original_fun_vectorized = vectorized
        self.original_fun_method = fun_method

        # how many period of the fast system have been integrated
        self.original_fun_period_eval = 0

        # initial condition
        self.y0 = np.array(y0)
        
        self.t = np.array([t_span[0]])
        self.y = np.zeros((self.n_dim,1))
        self.y[:,0] = self.y0
        self.f = np.zeros((self.n_dim,1))

        # indexes of the variables to use for the computation of the period
        if len(vars_to_use) == 0 or vars_to_use is None:
            self.vars_to_use = np.arange(self.n_dim)
        else:
            self.vars_to_use = vars_to_use

        if T is not None:
            self.estimate_T = False
            self.T = T
            self.f[:,0] = self._envelope_fun(self.t[0], self.y[:,0])
        elif T_guess is not None:
            self.estimate_T = True
            self.f[:,0] = self._envelope_fun(self.t[0], self.y[:,0], T_guess)
            self.T = self.T_new
            if self.is_variational:
                self.T_small = self.T_new
        else:
            raise Exception('T_guess and T cannot both be None')

        # initial integration step for the envelope
        self.H = self.T

        # period of the fast system during the envelope integration
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
            sol['y'] = np.zeros((self.n_dim+self.n_dim**2, len(idx)))
            sol['y'][:self.n_dim,:] = self.y[:,idx]
            sol['y'][self.n_dim:,0] = np.eye(self.n_dim).flatten()
            for i,mat in enumerate(sol['M']):
                sol['y'][self.n_dim:,i+1] = np.dot(np.reshape(sol['y'][self.n_dim:,i],\
                                                              (self.n_dim,self.n_dim)), mat).flatten()
            sol['var'] = {'t': np.array(self.t_var), \
                          'y': np.array([y.flatten() for y in self.y_var]).transpose()}

        return sol


    def _compute_y_next(self, y_cur, t_next, H, y_guess):
        raise NotImplementedError


    def _compute_LTE(self):
        raise NotImplementedError


    def _compute_variational_LTE(self, t0, t1, t2, y0, y1, y2):
        raise NotImplementedError


    def _compute_next_H(self, scale, coeff, T):
        raise NotImplementedError


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

        if self.is_variational:
            M,M_var = self._compute_monodromy_matrix(t_cur, y_cur)

        if n_periods == 1:
            # the step is equal to the period: we don't need to solve the implicit system
            y_next = self.y_new
        else:
            # estimate the next value by extrapolation using explicit Euler
            y_extrap = y_cur + H * f_cur
            # correct the estimate
            y_next = self._compute_y_next(y_cur, f_cur, t_next, H, y_extrap)

        if self.estimate_T and np.abs(self.T - self.T_new) > self.dTtol:
            return EnvelopeSolver.DT_TOO_LARGE

        if self.is_variational:
            n_periods_var = int(np.floor((t_next - t_cur) / self.T_var) + 1)
            t0 = t_cur
            t1 = t_cur + self.T_var
            t2 = t_cur + n_periods_var * self.T_var
            y0_var = np.eye(self.n_dim)
            for mat in self.mono_mat:
                y0_var = np.matmul(y0_var,mat)
            y1_var = np.matmul(y0_var,M_var)
            y2_var = y0_var.copy()
            for i in range(n_periods_var):
                y2_var = np.matmul(y2_var,M_var)
            if n_periods_var > 1:
                scale_var = self.var_atol + self.var_rtol * np.abs(y2_var.flatten())
                lte_var,coeff_var,H_new_var = self._compute_variational_LTE(t0, t1, t2,
                                                                            y0_var.flatten(),
                                                                            y1_var.flatten(),
                                                                            y2_var.flatten())

        # the value of the derivative at the new point
        f_next = self._envelope_fun(t_next,y_next)

        scale = self.atol + self.rtol * np.abs(y_next)
        # compute the local truncation error
        lte,coeff = self._compute_LTE(H, f_next, f_cur, y_next, y_cur)

        if self.estimate_T:
            T = self.T_new
        else:
            T = self.T

        # compute the new value of H as the maximum value that allows having an LTE below threshold
        self.H_new = self._compute_next_H(scale, coeff, T)
        if self.is_variational and n_periods_var > 1:
            self.H_new = np.min((self.H_new, np.floor(H_new_var / T) * T))

        if np.any(lte > scale) or \
           (self.is_variational and n_periods_var > 1 and np.any(lte_var > scale_var)):
            return EnvelopeSolver.LTE_TOO_LARGE

        self.t_next = t_next
        self.y_next = y_next
        self.f_next = f_next

        if self.is_variational:
            self.mono_mat.append(np.linalg.matrix_power(M,n_periods))
            self.t_var.append(t0)
            self.t_var.append(t1)
            self.t_var.append(t2)
            self.y_var.append(y0_var)
            self.y_var.append(y1_var)
            self.y_var.append(y2_var)

        return EnvelopeSolver.SUCCESS


    def _one_period_step(self):
        if self.is_variational:
            M,M_var = self._compute_monodromy_matrix(self.t[-1],self.y[:,-1])
            self.mono_mat.append(M)
            ### TODO: check the following code
            y0_var = np.eye(self.n_dim)
            for mat in self.mono_mat:
                y0_var = np.matmul(y0_var,mat)
            y1_var = np.matmul(y0_var,M_var)
            y2_var = y1_var.copy()
            self.t_var.append(self.t[-1])
            self.t_var.append(self.t[-1] + self.T_var)
            self.t_var.append(self.t[-1] + self.T_var)
            self.y_var.append(y0_var)
            self.y_var.append(y1_var)
            self.y_var.append(y2_var)
        self._envelope_fun(self.t[-1],self.y[:,-1])
        self.t = np.append(self.t,self.t_new)
        self.y = np.append(self.y,np.reshape(self.y_new,(self.n_dim,1)),axis=1)
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


    def _extended_system(self, t, y):
        return np.concatenate((self.original_fun(t, y[:self.n_dim]), \
                               self._variational_system(t, y)))


    def _variational_system(self, t, y):
        N = self.n_dim
        T = self.T_large
        J = self.original_jac(t,y[:N])
        phi = np.reshape(y[N:N+N**2],(N,N))
        return T * np.matmul(J,phi).flatten()


    def _compute_monodromy_matrix(self, t0, y0):
        t_stop = t0 + max([self.T, self.T_var])
        events_fun = lambda t,y: t - (t0 + min([self.T, self.T_var]))
        # compute the monodromy matrix by integrating the variational system
        sol = solve_ivp(self._extended_system, [t0,t_stop],
                        np.concatenate((y0,np.eye(self.n_dim).flatten())),
                        rtol=self.original_fun_rtol, atol=self.original_fun_atol,
                        events=events_fun, dense_output=True)
        y_ev = sol['sol'](sol['t_events'][0])
        if t_stop == t0 + self.T_var:
            # the variational system has a larger period than the original one
            M_var = np.reshape(sol['y'][self.n_dim:,-1],(self.n_dim,self.n_dim)).copy()
            M = np.reshape(y_ev[self.n_dim:],(self.n_dim,self.n_dim)).copy()
        else:
            # the variational system has a smaller period than the original one
            M = np.reshape(sol['y'][self.n_dim:,-1],(self.n_dim,self.n_dim)).copy()
            M_var = np.reshape(y_ev[self.n_dim:],(self.n_dim,self.n_dim)).copy()
        return M, M_var



class BEEnvelope (EnvelopeSolver):
    def __init__(self, fun, t_span, y0, T_guess, T=None, max_step=1000,
                 fun_rtol=1e-6, fun_atol=1e-8, dTtol=1e-2, rtol=1e-3, atol=1e-6,
                 jac=None, jac_sparsity=None, vectorized=False, fun_method='RK45',
                 vars_to_use=[], is_variational=False, T_var=None,
                 var_rtol=1e-2, var_atol=1e-3):
        super(BEEnvelope, self).__init__(fun, t_span, y0, T_guess, T, max_step,
                                         fun_rtol, fun_atol, dTtol, rtol, atol,
                                         jac, jac_sparsity, vectorized, fun_method,
                                         vars_to_use, is_variational, T_var,
                                         var_rtol, var_atol)

    def _compute_y_next(self, y_cur, f_cur, t_next, H, y_guess):
        #return fsolve(lambda Y: Y - y_cur - H * self._envelope_fun(t_next,Y), y_guess, xtol=1e-1)
        return newton_krylov(lambda Y: Y - y_cur - H * self._envelope_fun(t_next,Y), y_guess, f_tol=1e-3)


    def _compute_LTE(self, H, f_next, f_cur, y_next, y_cur):
        coeff = np.abs(f_next * (f_next - f_cur) / (y_next - y_cur))
        coeff[coeff == 0] = np.min(coeff[coeff > 0])
        lte = H**2 / 2 * coeff
        return lte, coeff


    def _compute_next_H(self, scale, coeff, T):
        return np.min((self.max_step,np.floor(np.min(np.sqrt(2*scale/coeff)) / T))) * T


    def _compute_variational_LTE(self, t0, t1, t2, y0, y1, y2):
        scale = self.var_atol + self.var_rtol * np.abs(y2)
        f1 = (y1 - y0) / (t1 - t0)
        f2 = (y2 - y1) / (t2 - t1)
        coeff = np.abs(f2 * (f2 - f1) / (y2 - y1))
        H = t2 - t1
        lte = H**2 / 2 * coeff
        H_new = np.floor(np.min(np.sqrt(2*scale/coeff)) / self.T_var) * self.T_var
        return lte,coeff,H_new



class TrapEnvelope (EnvelopeSolver):
    def __init__(self, fun, t_span, y0, T_guess, T=None, max_step=1000,
                 fun_rtol=1e-6, fun_atol=1e-8, dTtol=1e-2, rtol=1e-3, atol=1e-6,
                 jac=None, jac_sparsity=None, vectorized=False, fun_method='RK45',
                 vars_to_use=[], is_variational=False, T_var=None,
                 var_rtol=1e-2, var_atol=1e-3):
        super(TrapEnvelope, self).__init__(fun, t_span, y0, T_guess, T, max_step,
                                           fun_rtol, fun_atol, dTtol, rtol, atol,
                                           jac, jac_sparsity, vectorized, fun_method,
                                           vars_to_use, is_variational, T_var,
                                           var_rtol, var_atol)
        self.df_cur = np.zeros(self.n_dim)


    def _one_period_step(self):
        super(TrapEnvelope, self)._one_period_step()
        self.df_cur = (self.f[:,-1] - self.f[:,-2]) / (self.y[:,-1] - self.y[:,-2])


    def _compute_y_next(self, y_cur, f_cur, t_next, H, y_guess):
        #return fsolve(lambda Y: Y - y_cur - H/2 * (f_cur + self._envelope_fun(t_next,Y)), \
        #              y_guess, xtol=FSOLVE_XTOL)
        return newton_krylov(lambda Y: Y - y_cur - H/2 * (f_cur + self._envelope_fun(t_next,Y)), \
                             y_guess, f_tol=NEWTON_KRYLOV_FTOL)


    def _compute_LTE(self, H, f_next, f_cur, y_next, y_cur):
        df_next = (f_next - f_cur) / (y_next - y_cur)
        d2f_next = (df_next - self.df_cur) / (y_next - y_cur)
        coeff = np.abs(f_next * (f_next*d2f_next + 2*(df_next**2)))
        coeff[coeff == 0] = np.min(coeff[coeff > 0])
        lte = (H**3)/12 * coeff
        self.df_next = df_next
        return lte, coeff


    def _compute_next_H(self, scale, coeff, T):
        return np.min((self.max_step,np.floor(np.min((12*scale/coeff)**(1/3)) / T))) * T


    ### TODO: change this code to the expression for the trapezoidal LTE
    def _compute_variational_LTE(self, t0, t1, t2, y0, y1, y2):
        scale = self.var_atol + self.var_rtol * np.abs(y2)
        f1 = (y1 - y0) / (t1 - t0)
        f2 = (y2 - y1) / (t2 - t1)
        coeff = np.abs(f2 * (f2 - f1) / (y2 - y1))
        H = t2 - t1
        lte = H**2 / 2 * coeff
        H_new = np.floor(np.min(np.sqrt(2*scale/coeff)) / self.T_var) * self.T_var
        return lte, coeff, H_new


    def _step(self):
        flag = super(TrapEnvelope, self)._step()
        if flag == EnvelopeSolver.SUCCESS:
            self.df_cur = self.df_next
        return flag

