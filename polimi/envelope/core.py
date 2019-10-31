
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, newton_krylov
from scipy.interpolate import interp1d

from .. import switching

DEBUG = True
VERBOSE_DEBUG = False
if DEBUG:
    from .. import utils
    colors = utils.ColorFactory()

NEWTON_KRYLOV_FTOL = 1e-3
FSOLVE_XTOL = 1e-1

__all__ = ['EnvelopeSolver', 'BEEnvelope', 'TrapEnvelope']


class EnvelopeSolver (object):

    SUCCESS = 0
    DT_TOO_LARGE = 1
    LTE_TOO_LARGE = 2
    

    def __init__(self, system, t_span, y0, T_guess=None, T=None, max_step=1000,
                 dT_tol=1e-2, env_rtol=1e-3, env_atol=1e-6,
                 vars_to_use=[], is_variational=False, T_var=None,
                 T_var_guess=None, var_rtol=1e-2, var_atol=1e-3,
                 solver=solve_ivp, **kwargs):

        # the dynamical system
        self.system = system
        # tolerances for the computation of the envelope
        self.rtol, self.atol = env_rtol, env_atol
        # tolerance on the variation of the estimated period
        self.dT_tol = dT_tol
        # max_step is in units of periods
        self.max_step = np.floor(max_step)

        self.solver = solver
        self.solver_kwargs = kwargs

        # whether to compute the envelope of the variational system
        self.is_variational = is_variational

        # whether we are dealing with a switching system
        self.is_switching_system = isinstance(system, switching.SwitchingSystem)

        if not self.is_variational:
            self.t_span = t_span
            self.system.with_variational = False
        else:
            self.T_large = t_span[1]
            if T is None:
                T_guess /= t_span[1]
            else:
                T /= t_span[1]
            self.system.with_variational = True
            self.system.variational_T = self.T_large

        # how many period of the fast system have been integrated
        self.original_fun_period_eval = 0

        # initial condition
        self.y0 = np.array(y0)
        
        self.t = np.array([t_span[0]])
        self.y = np.zeros((self.system.n_dim,1))
        self.y[:,0] = self.y0
        self.f = np.zeros((self.system.n_dim,1))

        # indexes of the variables to use for the computation of the period
        if len(vars_to_use) == 0 or vars_to_use is None:
            self.vars_to_use = np.arange(self.system.n_dim)
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

        if self.is_variational:
            self.compute_variational_LTE = True
            if T_var is not None:
                self.estimate_T_var = False
                self.T_var = T_var / self.T_large
            elif T_var_guess is not None:
                self.estimate_T_var = True
                _,_,self.T_var = self._compute_monodromy_matrix(t_span[0], y0, T_var_guess/self.T_large)
            else:
                self.compute_variational_LTE = False
                self.estimate_T_var = False
                self.T_var = -1
                print('EnvelopeSolver.__init__(%.4e)> will not compute variational LTE.' % 0)
            self.t_span = [0,1]
            self.mono_mat = []
            self.t_var = []
            self.y_var = []
            self.var_atol = var_atol
            self.var_rtol = var_rtol

        # initial integration step for the envelope
        self.H = self.T

        # period of the fast system during the envelope integration
        self.period = [self.T]
        if DEBUG:
            if self.is_variational:
                print('EnvelopeSolver.__init__(%.4e)> T = %.3e T_var = %.3e' % (self.t_span[0],self.T,self.T_var))
            else:
                print('EnvelopeSolver.__init__(%.4e)> T = %.3e' % (self.t_span[0],self.T))


    def solve(self):
        n_dim = self.system.n_dim
        while self.t[-1] < self.t_span[1]:
            # make one step
            flag = self._step()
            # check the result of the step
            if flag == EnvelopeSolver.SUCCESS:
                # the step was successful (LTE and variation in period below threshold
                self.t = np.append(self.t,self.t_next)
                self.y = np.append(self.y,np.reshape(self.y_next,(n_dim,1)),axis=1)
                self.f = np.append(self.f,np.reshape(self.f_next,(n_dim,1)),axis=1)
                if self.estimate_T:
                    self.T = self.T_new
                self.period.append(self.T)
                if self.H_new == 0:
                    self.H_new = self.T
                msg = 'OK'
            elif flag == EnvelopeSolver.DT_TOO_LARGE:
                # the variation in period was too large
                if self.H > 2*self.T:
                    # reduce the step if this is greater than twice the oscillation period
                    self.H_new = np.floor(np.round(self.H/self.T)/2) * self.T
                    msg = 'DT T/2'
                else:
                    # otherwise simply move the integration one period forward
                    self._one_period_step()
                    msg = 'DT 1T'
            elif flag == EnvelopeSolver.LTE_TOO_LARGE:
                # the LTE was above threshold: _step has already changed the value of H_new
                msg = 'LTE'

            H = self.H
            if self.H_new < self.T and self.t[-1] < self.t_span[1]:
                self._one_period_step()
                msg += ' 1T'
            self.H = self.H_new

            if DEBUG:
                t_cur = self.t[-1]
                if msg != 'LTE':
                    H = np.diff(self.t[-2:])[0]
                N = round(H/self.T)
                color_fun = colors.green
                if msg == 'LTE':
                    color_fun = colors.red
                elif msg == 'LTE 1T':
                    color_fun = colors.yellow
                elif 'DT' in msg:
                    color_fun = colors.magenta
                H_str = color_fun('%.3e' % H)
                N_str = color_fun('%4d' % N)
                t_cur_str = color_fun('%.4e' % t_cur)
                msg = color_fun(msg)
                if self.is_variational:
                    print('EnvelopeSolver.solve(%s)> T = %.3e, T_var = %.3e, H = %s, N = %s - %s' % \
                          (t_cur_str, self.T, self.T_var, H_str, N_str, msg))
                else:
                    print('EnvelopeSolver.solve(%s)> T = %.3e, H = %s, N = %s - %s' % \
                          (t_cur_str, self.T, H_str, N_str, msg))

        idx, = np.where(self.t <= self.t_span[1] + self.T/2)
        sol = {'t': self.t[idx], 'y': self.y[:,idx],
               'T': np.array([self.period[i] for i in idx]),
               'period_eval': self.original_fun_period_eval}
        if self.is_variational:
            sol['M'] = [self.mono_mat[i] for i in idx[:-1]]
            sol['y'] = np.zeros((n_dim+n_dim**2, len(idx)))
            sol['y'][:n_dim,:] = self.y[:,idx]
            sol['y'][n_dim:,0] = np.eye(n_dim).flatten()
            for i,mat in enumerate(sol['M']):
                sol['y'][n_dim:,i+1] = (mat @ np.reshape(sol['y'][n_dim:,i],(n_dim,n_dim))).flatten()
            sol['var'] = {'t': np.array(self.t_var), \
                          'y': np.array([y.flatten() for y in self.y_var]).transpose()}

        return sol


    def _compute_y_next(self, y_cur, t_next, H, y_guess):
        raise NotImplementedError


    def _compute_LTE(self):
        raise NotImplementedError


    def _compute_variational_LTE(self, t_prev, t_cur, t_next, y_prev, y_cur, y_next):
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
            M,M_var,T_var = self._compute_monodromy_matrix(t_cur, y_cur)

        if n_periods == 1:
            # the step is equal to the period: we don't need to solve the implicit system
            y_next = self.y_new
        else:
            # estimate the next value by extrapolation using explicit Euler
            y_extrap = y_cur + H * f_cur
            # correct the estimate
            y_next = self._compute_y_next(y_cur, f_cur, t_next, H, y_extrap)

        if self.estimate_T and np.abs(self.T - self.T_new) / self.T > self.dT_tol:
            return EnvelopeSolver.DT_TOO_LARGE

        if self.is_variational and self.compute_variational_LTE:
            n_periods_var = int(np.floor((t_next - t_cur) / self.T_var) + 1)
            t0 = t_cur
            t1 = t_cur + self.T_var
            t2 = t_cur + n_periods_var * self.T_var
            y0_var = np.eye(self.system.n_dim)
            for mat in self.mono_mat:
                y0_var = mat @ y0_var
            y1_var = M_var @ y0_var
            y2_var = y0_var.copy()
            for i in range(n_periods_var):
                y2_var = M_var @ y2_var
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
        if t_next + self.H_new > self.t_span[1]:
            self.H_new = self.T * np.floor((self.t_span[1] - t_next) / self.T)

        if self.is_variational and self.compute_variational_LTE and n_periods_var > 1:
            self.H_new = np.min((self.H_new, np.floor(H_new_var / T) * T))

        if np.any(lte > scale) or \
           (self.is_variational and self.compute_variational_LTE and \
            n_periods_var > 1 and np.any(lte_var > scale_var)):
            return EnvelopeSolver.LTE_TOO_LARGE

        self.t_next = t_next
        self.y_next = y_next
        self.f_next = f_next

        if self.is_variational:
            self.mono_mat.append(np.linalg.matrix_power(M,n_periods))
            if self.compute_variational_LTE:
                self.t_var.append(t0)
                self.t_var.append(t1)
                self.t_var.append(t2)
                self.y_var.append(y0_var)
                self.y_var.append(y1_var)
                self.y_var.append(y2_var)
                self.T_var = T_var

        return EnvelopeSolver.SUCCESS


    def _one_period_step(self):
        n_dim = self.system.n_dim
        if self.is_variational:
            M,M_var,self.T_var = self._compute_monodromy_matrix(self.t[-1],self.y[:,-1])
            self.mono_mat.append(M)
            ### TODO: check the following code
            if self.compute_variational_LTE:
                y0_var = np.eye(n_dim)
                for mat in self.mono_mat:
                    y0_var = M_var @ y0_var
                y1_var = M_var @ y0_var
                y2_var = y1_var.copy()
                self.t_var.append(self.t[-1])
                self.t_var.append(self.t[-1] + self.T_var)
                self.t_var.append(self.t[-1] + self.T_var)
                self.y_var.append(y0_var)
                self.y_var.append(y1_var)
                self.y_var.append(y2_var)
        self._envelope_fun(self.t[-1],self.y[:,-1])
        self.t = np.append(self.t,self.t_new)
        self.y = np.append(self.y,np.reshape(self.y_new,(n_dim,1)),axis=1)
        self.f = np.append(self.f,np.reshape(self._envelope_fun(self.t[-1],self.y[:,-1]),(n_dim,1)),axis=1)
        if self.estimate_T:
            self.T = self.T_new
        self.H_new = self.T
        self.H = self.T
        self.period.append(self.T)

        
    def _envelope_fun(self,t,y,T_guess=None):
        with_variational = self.system.with_variational
        self.system.with_variational = False
        if not self.estimate_T:
            sol = self.solver(self.system, [t,t+self.T], y, **self.solver_kwargs)
            self.t_new = t + self.T
            self.y_new = sol['y'][:,-1]
            if VERBOSE_DEBUG:
                print('EnvelopeSolver._envelope_fun(%.4e)> y = (%.6f,%.6f) T = %.3e.' % (t,self.y_new[0],self.y_new[1],self.T))
            # return the "vector field" of the envelope
            self.original_fun_period_eval += 1
            self.system.with_variational = with_variational
            return 1./self.T * (sol['y'][:,-1] - y)

        if T_guess is None:
            T_guess = self.T
        # find the equation of the plane containing y and
        # orthogonal to fun(t,y)
        f = self.system(t,y)
        w = f[self.vars_to_use] / np.linalg.norm(f[self.vars_to_use])
        b = -np.dot(w, y[self.vars_to_use])

        events_fun = lambda t,y: np.dot(w, y[self.vars_to_use]) + b
        events_fun.direction = 1
        events_fun.terminal = 1

        sol = self.solver(self.system, [t,t+2*T_guess], y, \
                          events=events_fun, dense_output=True, \
                          **self.solver_kwargs)

        try:
            # find the first event that is not the initial time
            idx, = np.where(sol['t_events'][0] > t)
            T = sol['t_events'][0][idx[0]] - t
        except:
            T = T_guess
            if DEBUG:
                print('EnvelopeSolver._envelope_fun(%.4e)> T = T_guess = %.3e.' % (t,T))

        self.T_new = T
        self.t_new = t + self.T_new
        self.y_new = sol['sol'](self.t_new)
        if VERBOSE_DEBUG:
            print('EnvelopeSolver._envelope_fun(%.4e)> y = (%.4f,%.4f) T = %.3e.' % (t,self.y_new[0],self.y_new[1],self.T_new))
        self.original_fun_period_eval += 1
        self.system.with_variational = with_variational
        # return the "vector field" of the envelope
        return 1./self.T_new * (self.y_new - sol['sol'](t))


    def _variational_system(self, t, y):
        n_dim = self.system.n_dim
        T = self.T_large
        J = self.system.jac(t,y[:n_dim])
        phi = np.reshape(y[n_dim:n_dim+n_dim**2],(n_dim,n_dim))
        return np.concatenate((self.system(t, y[:n_dim]), \
                               T * (J @ phi).flatten()))


    def _compute_monodromy_matrix(self, t, y, T_var_guess=None):
        n_dim = self.system.n_dim
        y_ext = np.concatenate((y, np.eye(n_dim).flatten()))
        with_variational = self.system.with_variational
        self.system.with_variational = True
        solver_kwargs = self.solver_kwargs.copy()
        if 'jac' in solver_kwargs:
            solver_kwargs.pop('jac')

        if self.estimate_T_var:
            if T_var_guess is None:
                T_var_guess = self.T_var

            # find the equation of the plane containing y and
            # orthogonal to _variational_system(t,y)
            vars_to_use = np.arange(n_dim, n_dim + n_dim**2)
            f = self.system(t,y_ext)

            w = f[vars_to_use] / np.linalg.norm(f[vars_to_use])
            b = -np.dot(w, y_ext[vars_to_use])

            events_fun = [lambda tt,yy: np.dot(w, yy[vars_to_use]) + b, \
                          lambda tt,yy: tt - (t + self.T)]
            events_fun[0].direction = 1
            events_fun[1].direction = 0

            t_stop = t + max([self.T, 1.5*T_var_guess])

            sol = self.solver(self.system, [t,t_stop], y_ext, \
                              events=events_fun, dense_output=True,
                              **solver_kwargs)

            idx, = np.where(sol['t_events'][0] > t)
            idx = idx[0]
            T_var = sol['t_events'][0][idx] - t
            y_ev = sol['sol'](sol['t_events'][1])
            M = np.reshape(y_ev[n_dim:],(n_dim,n_dim)).copy(),
            y_ev = sol['sol'](sol['t_events'][0][idx])
            M_var = np.reshape(y_ev[n_dim:],(n_dim,n_dim)).copy(),
            self.system.with_variational = with_variational
            return M, M_var, T_var

        ### CHECK why don't I compute the variational LTE if estimate_T_var is True??
        if self.compute_variational_LTE:
            t_stop = t + max([self.T, self.T_var])
            events_fun = lambda tt,yy: tt - (t + min([self.T, self.T_var]))
            # compute the monodromy matrix by integrating the variational system
            sol = self.solver(self.system, [t,t_stop], y_ext, \
                              events=events_fun, dense_output=True,
                              **solver_kwargs)
            y_ev = sol['sol'](sol['t_events'][0])
            if t_stop == t + self.T_var:
                # the variational system has a larger period than the original one
                M_var = np.reshape(sol['y'][n_dim:,-1],(n_dim,n_dim)).copy()
                M = np.reshape(y_ev[n_dim:],(n_dim,n_dim)).copy()
            else:
                # the variational system has a smaller period than the original one
                M = np.reshape(sol['y'][n_dim:,-1],(n_dim,n_dim)).copy()
                M_var = np.reshape(y_ev[n_dim:],(n_dim,n_dim)).copy()
            self.system.with_variational = with_variational
            return M, M_var, self.T_var

        sol = self.solver(self.system, [t,t+self.T], y_ext, **solver_kwargs)
        M = np.reshape(sol['y'][n_dim:,-1],(n_dim,n_dim)).copy()
        self.system.with_variational = with_variational
        return M, None, -1



class BEEnvelope (EnvelopeSolver):
    def __init__(self, fun, t_span, y0, T_guess=None, T=None, max_step=1000,
                 dT_tol=1e-2, env_rtol=1e-3, env_atol=1e-6,
                 vars_to_use=[], is_variational=False, T_var=None,
                 T_var_guess=None, var_rtol=1e-2, var_atol=1e-3,
                 solver=solve_ivp, **kwargs):
        super(BEEnvelope, self).__init__(fun, t_span, y0, T_guess, T, max_step,
                                         dT_tol, env_rtol, env_atol,
                                         vars_to_use, is_variational, T_var,
                                         T_var_guess, var_rtol, var_atol,
                                         solver, **kwargs)

    def _compute_y_next(self, y_cur, f_cur, t_next, H, y_guess):
        #return fsolve(lambda Y: Y - y_cur - H * self._envelope_fun(t_next,Y), y_guess, xtol=FSOLVE_XTOL)
        return newton_krylov(lambda Y: Y - y_cur - H * self._envelope_fun(t_next,Y), y_guess, f_tol=NEWTON_KRYLOV_FTOL)


    def _compute_LTE(self, H, f_next, f_cur, y_next, y_cur):
        coeff = np.abs(f_next * (f_next - f_cur) / (y_next - y_cur))
        coeff[coeff == 0] = np.min(coeff[coeff > 0])
        lte = H**2 / 2 * coeff
        return lte, coeff


    def _compute_next_H(self, scale, coeff, T):
        return np.min((self.max_step,np.floor(np.min(np.sqrt(2*scale/coeff)) / T))) * T


    def _compute_variational_LTE(self, t_prev, t_cur, t_next, y_prev, y_cur, y_next):
        H = t_next - t_cur
        scale = self.var_atol + self.var_rtol * np.abs(y_next)
        f_cur = (y_cur - y_prev) / (t_cur - t_prev)
        f_next = (y_next - y_cur) / (t_next - t_cur)
        coeff = np.abs(f_next * (f_next - f_cur) / (y_next - y_cur))
        coeff[coeff == 0] = np.min(coeff[coeff > 0])
        lte = H**2 / 2 * coeff
        H_new = np.floor(np.min(np.sqrt(2*scale/coeff)) / self.T_var) * self.T_var
        return lte,coeff,H_new



class TrapEnvelope (EnvelopeSolver):
    def __init__(self, fun, t_span, y0, T_guess=None, T=None, max_step=1000,
                 dT_tol=1e-2, env_rtol=1e-3, env_atol=1e-6,
                 vars_to_use=[], is_variational=False, T_var=None,
                 T_var_guess=None, var_rtol=1e-2, var_atol=1e-3,
                 solver=solve_ivp, **kwargs):
        super(TrapEnvelope, self).__init__(fun, t_span, y0, T_guess, T, max_step,
                                           dT_tol, env_rtol, env_atol,
                                           vars_to_use, is_variational, T_var,
                                           T_var_guess, var_rtol, var_atol,
                                           solver, **kwargs)
        self.df_cur = np.zeros(self.system.n_dim)
        if self.is_variational:
            self.df_cur_var = np.zeros(self.system.n_dim**2)


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


    def _compute_variational_LTE(self, t_prev, t_cur, t_next, y_prev, y_cur, y_next):
        H = t_next - t_cur
        scale = self.var_atol + self.var_rtol * np.abs(y_next)
        f_cur = (y_cur - y_prev) / (t_cur - t_prev)
        f_next = (y_next - y_cur) / (t_next - t_cur)
        df_next = (f_next - f_cur) / (y_next - y_cur)
        d2f_next = (df_next - self.df_cur_var) / (y_next - y_cur)
        coeff = np.abs(f_next * (f_next*d2f_next + 2*(df_next**2)))
        coeff[coeff == 0] = np.min(coeff[coeff > 0])
        lte = H**3 / 12 * coeff
        H_new = np.floor(np.min((12*scale/coeff)**(1/3)) / self.T_var) * self.T_var
        self.df_next_var = df_next
        return lte, coeff, H_new


    def _step(self):
        flag = super(TrapEnvelope, self)._step()
        if flag == EnvelopeSolver.SUCCESS:
            self.df_cur = self.df_next
            if self.is_variational and hasattr(self, 'df_next_var'):
                self.df_cur_var = self.df_next_var
        return flag

