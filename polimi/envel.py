
import ipdb
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, newton_krylov
from scipy.interpolate import interp1d

DEBUG = True
VERBOSE_DEBUG = False

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
        print('Variables to use: {}.'.format(self.vars_to_use))
        self.is_variational = is_variational
        if is_variational:
            self.mono_mat = []
            # the number of dimensions of the original system, i.e. the one
            # without the variational part added
            self.N = int((-1 + np.sqrt(1 + 4*self.n_dim)) / 2)
            print('The number of dimensions of the original system is %d.' % self.N)
        if T is not None:
            self.estimate_T = False
            self.T = T
            self.f[:,0] = self._envelope_fun(self.t[0],self.y[:,0])
        else:
            self.estimate_T = True
            self.f[:,0] = self._envelope_fun(self.t[0],self.y[:,0],T_guess)
            self.T = self.T_new
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
        sol = {'t': self.t[idx], 'y': self.y[:,idx], 'T': np.array([self.period[i] for i in idx])}
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
        w = f/np.linalg.norm(f)
        b = -np.dot(w[self.vars_to_use],y[self.vars_to_use])

        events_fun = lambda t,y: np.dot(w[self.vars_to_use],y[self.vars_to_use]) + b
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

        # estimate the next value by extrapolation using explicit Euler
        y_extrap = y_cur + H * f_cur
        # correct the estimate using implicit Euler
        #y_next = fsolve(lambda Y: Y - y_cur - H * self._envelope_fun(t_next,Y), y_extrap, xtol=1e-1)
        y_next = newton_krylov(lambda Y: Y - y_cur - H * self._envelope_fun(t_next,Y), y_extrap, f_tol=1e-3)

        if self.estimate_T and np.abs(self.T - self.T_new) > self.dTtol:
            return EnvelopeSolver.DT_TOO_LARGE

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
        self.H_new = np.min((self.max_step,np.floor(np.min(np.sqrt(2*scale/coeff)) / T))) * T

        if np.any(lte > scale):
            return EnvelopeSolver.LTE_TOO_LARGE

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
            y_next = newton_krylov(lambda Y: Y - y_cur - H/2 * (f_cur + self._envelope_fun(t_next,Y)), y_extrap, f_tol=1e-3)

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


# for saving data
pack = lambda t,y: np.concatenate((np.reshape(t,(len(t),1)),y.transpose()),axis=1)


def autonomous():
    from systems import vdp
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

    import matplotlib.pyplot as plt
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(sol_be['t'],sol_be['y'][0],'ro-')
    plt.plot(sol_trap['t'],sol_trap['y'][0],'go-')
    plt.show()


def forced_polar():
    from systems import vdp_auto
    import matplotlib.pyplot as plt
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
    from systems import vdp
    import matplotlib.pyplot as plt
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
    from systems import hr
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

    import matplotlib.pyplot as plt
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(sol_be['t'],sol_be['y'][0],'ro-')
    plt.plot(sol_trap['t'],sol_trap['y'][0],'go-')
    plt.plot(sol_trap['t'],sol_trap['T'],'ms-')
    plt.show()


def variational():
    from systems import vdp, vdp_jac
    import matplotlib.pyplot as plt

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
    fun = lambda t,y: vdp(t,y,epsilon,A,T)
    jac = lambda t,y: vdp_jac(t,y,epsilon)
    var_fun = lambda t,y: variational_system(fun, jac, t, y, T_large)

    t_span_var = [0,1]
    y0 = np.array([-5.8133754 ,  0.13476983])
    y0_var = np.concatenate((y0,np.eye(len(y0)).flatten()))

    atol = [1e-2,1e-2,1e-6,1e-6,1e-6,1e-6]
    var_sol = solve_ivp(var_fun, t_span_var, y0_var, rtol=1e-7, atol=1e-8, dense_output=True)
    var_envelope_solver = TrapEnvelope(var_fun, t_span_var, y0_var, T_guess=None, T=T_small/T_large,
                                       max_step=100, jac=jac, rtol=1e-1, atol=atol,
                                       vars_to_use=[0,1], is_variational=True)
    var_env = var_envelope_solver.solve()

    eig_correct,_ = np.linalg.eig(np.reshape(var_sol['y'][2:,-1],(2,2)))
    eig_approx,_ = np.linalg.eig(np.reshape(var_env['y'][2:,-1],(2,2)))

    print('    correct eigenvalues:', eig_correct)
    print('approximate eigenvalues:', eig_approx)

    plt.subplot(1,2,1)
    plt.plot(var_sol['t'],var_sol['y'][0],'k')
    plt.plot(var_env['t'],var_env['y'][0],'ro')
    plt.subplot(1,2,2)
    plt.plot(t_span_var,[0,0],'b')
    plt.plot(var_sol['t'],var_sol['y'][2],'k')
    plt.plot(var_env['t'],var_env['y'][2],'ro')
    plt.show()


def main():
    #autonomous()
    #forced_polar()
    #forced()
    #hr()
    variational()


if __name__ == '__main__':
    main()
    
