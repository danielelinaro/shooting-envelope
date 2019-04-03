
import ipdb
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

DEBUG = True
VERBOSE_DEBUG = False

class EnvelopeSolver (object):

    SUCCESS = 0
    DT_TOO_LARGE = 1
    LTE_TOO_LARGE = 2
    
    def __init__(self, fun, t_span, y0, T_guess, T=None, max_step=1000,
                 fun_rtol=1e-6, fun_atol=1e-8, dTtol=1e-2, rtol=1e-3, atol=1e-6,
                 jac=None, jac_sparsity=None, vectorized=False, fun_method='RK45'):
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
        if T is not None:
            self.estimate_T = False
            self.T = T
            self.f[:,0] = self._envelope_fun(self.t[0],self.y[:,0])
        else:
            self.estimate_T = True
            self.f[:,0] = self._envelope_fun(self.t[0],self.y[:,0],T_guess)
            self.T = self.T_new
        self.H = self.T
        self.period = np.array([self.T])
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
                self.period = np.append(self.period,self.T)
            elif flag == EnvelopeSolver.DT_TOO_LARGE:
                # the variation in period was too large
                if self.H > 2*self.T:
                    # reduce the step if this is greater than twice the oscillation period
                    self.H_new = np.floor(np.round(self.H/self.T)/2) * self.T
                else:
                    # otherwise simply move the integration one period forward
                    self._one_period_step()
            elif flag == EnvelopeSolver.LTE_TOO_LARGE:
                # the LTE was above threshold: _step has already changed the value of H_new
                pass

            if self.H_new < self.T:
                self._one_period_step()
            self.H = self.H_new

            if DEBUG:
                print('EnvelopeSolver.solve(%.3f)> T = %f, H = %f' % (self.t[-1],self.T,self.H))
        return self.t,self.y


    def _one_period_step(self):
        self._envelope_fun(self.t[-1],self.y[:,-1])
        self.t = np.append(self.t,self.t_new)
        self.y = np.append(self.y,np.reshape(self.y_new,(self.n_dim,1)),axis=1)
        self.f = np.append(self.f,np.reshape(self._envelope_fun(self.t[-1],self.y[:,-1]),(self.n_dim,1)),axis=1)
        if self.estimate_T:
            self.T = self.T_new
        self.H_new = self.T
        self.H = self.T
        self.period = np.append(self.period,self.T)

        
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
                print('EnvelopeSolver._envelope_fun(%.3f)> y = (%.4f,%.4f) T = %.6f.' % (t,self.y_new[0],self.y_new[1],self.T))
            # return the "vector field" of the envelope
            self.original_fun_period_eval += 1
            return 1./self.T * (sol['y'][:,-1] - y)

        if T_guess is None:
            T_guess = self.T
        # find the equation of the plane containing y and
        # orthogonal to fun(t,y)
        f = self.original_fun(t,y)
        w = f/np.linalg.norm(f)
        b = -np.dot(w,y)
        # first integration without events, because event(t0) = 0
        # and the solver goes crazy
        if self.original_fun_method == 'BDF':
            sol_a = solve_ivp(self.original_fun,[t,t+0.5*T_guess],y,
                              self.original_fun_method,jac=self.original_jac,
                              jac_sparsity=self.original_jac_sparsity,
                              vectorized=self.original_fun_vectorized,
                              dense_output=True,rtol=self.original_fun_rtol,
                              atol=self.original_fun_atol)
            sol_b = solve_ivp(self.original_fun,[sol_a['t'][-1],t+2*T_guess],
                              sol_a['y'][:,-1],self.original_fun_method,jac=self.original_jac,
                              jac_sparsity=self.original_jac_sparsity,
                              vectorized=self.original_fun_vectorized,
                              events=lambda t,y: np.dot(w,y)+b,dense_output=True,
                              rtol=self.original_fun_rtol,atol=self.original_fun_atol)
        else:
            sol_a = solve_ivp(self.original_fun,[t,t+0.5*T_guess],y,
                              self.original_fun_method,vectorized=self.original_fun_vectorized,
                              dense_output=True,rtol=self.original_fun_rtol,
                              atol=self.original_fun_atol)
            sol_b = solve_ivp(self.original_fun,[sol_a['t'][-1],t+2*T_guess],
                              sol_a['y'][:,-1],self.original_fun_method,
                              vectorized=self.original_fun_vectorized,
                              events=lambda t,y: np.dot(w,y)+b,dense_output=True,
                              rtol=self.original_fun_rtol,atol=self.original_fun_atol)

        for t_ev in sol_b['t_events'][0]:
            x_ev = sol_b['sol'](t_ev)
            f = self.original_fun(t_ev,x_ev)
            # check whether the crossing of the plane is in
            # the same direction as the initial point
            if np.dot(w,f/np.linalg.norm(f))+b > 0:
                T = t_ev-t
                break
        try:
            self.T_new = T
        except:
            self.T_new = T_guess
            if DEBUG:
                print('EnvelopeSolver._envelope_fun(%.3f)> T = T_guess = %.6f.' % (t,self.T_new))
        self.t_new = t + self.T_new
        self.y_new = sol_b['sol'](self.t_new)
        if VERBOSE_DEBUG:
            print('EnvelopeSolver._envelope_fun(%.3f)> y = (%.4f,%.4f) T = %.6f.' % (t,self.y_new[0],self.y_new[1],self.T_new))
        self.original_fun_period_eval += 2
        # return the "vector field" of the envelope
        return 1./self.T_new * (self.y_new - sol_a['sol'](t))


class BEEnvelope (EnvelopeSolver):
    def __init__(self, fun, t_span, y0, T_guess, T=None, max_step=1000,
                 fun_rtol=1e-6, fun_atol=1e-8, dTtol=1e-2, rtol=1e-3, atol=1e-6,
                 jac=None, jac_sparsity=None, vectorized=False, fun_method='RK45'):
        super(BEEnvelope, self).__init__(fun, t_span, y0, T_guess, T, max_step,
                                         fun_rtol, fun_atol, dTtol, rtol, atol,
                                         jac, jac_sparsity, vectorized, fun_method)



    def _step(self):
        H = self.H
        if H == 0:
            ipdb.set_trace()
        t1 = self.t[-1]
        y1 = self.y[:,-1]
        f1 = self.f[:,-1]

        if t1+H > self.t_span[1]:
            H = self.T * np.max((1,np.floor((self.t_span[1] - t1) / self.T)))
        t2 = t1 + H

        # estimate the next value by extrapolation using explicit Euler
        y_extrap = y1 + H * f1
        # correct the estimate using implicit Euler
        y2 = fsolve(lambda Y: Y - y1 - H*self._envelope_fun(t2,Y), y_extrap, xtol=1e-3)

        if self.estimate_T and np.abs(self.T - self.T_new) > self.dTtol:
            return EnvelopeSolver.DT_TOO_LARGE

        # the value of the derivative at the new point
        f2 = self._envelope_fun(t2,y2)

        scale = self.atol + self.rtol * np.abs(y2)
        # compute the local truncation error
        coeff = np.abs(f2 * (f2 - f1) / (y2 - y1))
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

        self.t_next = t2
        self.y_next = y2
        self.f_next = f2

        return EnvelopeSolver.SUCCESS


class TrapEnvelope (EnvelopeSolver):
    def __init__(self, fun, t_span, y0, T_guess, T=None, max_step=1000,
                 fun_rtol=1e-6, fun_atol=1e-8, dTtol=1e-2, rtol=1e-3, atol=1e-6,
                 jac=None, jac_sparsity=None, vectorized=False, fun_method='RK45'):
        super(TrapEnvelope, self).__init__(fun, t_span, y0, T_guess, T, max_step,
                                           fun_rtol, fun_atol, dTtol, rtol, atol,
                                           jac, jac_sparsity, vectorized, fun_method)
        self.df_cur = np.zeros(self.n_dim)


    def _one_period_step(self):
        super(TrapEnvelope, self)._one_period_step()
        self.df_cur = (self.f[:,-1] - self.f[:,-2]) / (self.y[:,-1] - self.y[:,-2])


    def _step(self):
        H = self.H
        if H == 0:
            ipdb.set_trace()
        t_cur = self.t[-1]
        y_cur = self.y[:,-1]
        f_cur = self.f[:,-1]
        df_cur = self.df_cur

        if t_cur + H > self.t_span[1]:
            H = self.T * np.max((1,np.floor((self.t_span[1] - t_cur) / self.T)))
        t_next = t_cur + H

        # estimate the next value by extrapolation using explicit Euler
        y_extrap = y_cur + H * f_cur
        # correct the estimate using the trapezoidal rule
        y_next = fsolve(lambda Y: Y - y_cur - H/2 * (f_cur + self._envelope_fun(t_next,Y)), y_extrap, xtol=1e-3)

        if self.estimate_T and np.abs(self.T - self.T_new) > self.dTtol:
            return EnvelopeSolver.DT_TOO_LARGE

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
        self.H_new = np.min((self.max_step,np.floor(np.min((12*scale/coeff)**(1/3)) / T))) * T

        if np.any(lte > scale):
            return EnvelopeSolver.LTE_TOO_LARGE

        self.t_next = t_next
        self.y_next = y_next
        self.f_next = f_next
        self.df_cur = df_next

        return EnvelopeSolver.SUCCESS


def autonomous():
    from systems import vdp
    epsilon = 1e-3
    A = [0]
    T = [1]
    fun = lambda t,y: vdp(t,y,epsilon,A,T)
    t_span = [0,1000*2*np.pi]
    y0 = [2e-3,0]
    T_guess = 2*np.pi
    be_solver = BEEnvelope(fun, t_span, y0, T_guess, rtol=1e-3, atol=1e-6)
    trap_solver = TrapEnvelope(fun, t_span, y0, T_guess, rtol=1e-3, atol=1e-6)
    t_be,y_be = be_solver.solve()
    t_trap,y_trap = trap_solver.solve()
    sol = solve_ivp(fun, [t_span[0],t_span[-1]], y0, method='BDF', rtol=1e-8, atol=1e-10)
    import matplotlib.pyplot as plt
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(t_be,y_be[0],'ro-')
    plt.plot(t_trap,y_trap[0],'go-')
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
    t_be,y_be = be_solver.solve()
    t_trap,y_trap = trap_solver.solve()
    sol = solve_ivp(fun, t_span, y0, method='BDF', rtol=1e-8, atol=1e-10)
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(t_be,y_be[0],'ro-')
    plt.plot(t_trap,y_trap[0],'go-')
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
    t_be,y_be = be_solver.solve()
    print('The number of integrated periods of the original system with BE is %d.' % be_solver.original_fun_period_eval)
    t_trap,y_trap = trap_solver.solve()
    print('The number of integrated periods of the original system with TRAP is %d.' % trap_solver.original_fun_period_eval)
    sol = solve_ivp(fun, t_span, y0, method='RK45', rtol=1e-8, atol=1e-10)
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(t_be,y_be[0],'ro-')
    plt.plot(t_trap,y_trap[0],'go-')
    plt.show()


def main():
    #autonomous()
    #forced_polar()
    forced()


if __name__ == '__main__':
    main()
    
