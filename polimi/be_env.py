
import ipdb
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

DEBUG = True

class BEEnv (object):

    SUCCESS = 1
    FAILURE = 2
    
    def __init__(self, fun, t_span, y0, T_guess, max_step=np.inf,
                 fun_rtol=1e-6, fun_atol=1e-8, dTtol=1e-2,rtol=1e-3, atol=1e-6,
                 jac=None, jac_sparsity=None, vectorized=False, fun_method='RK45'):
        self.dTtol = dTtol
        self.max_step = max_step
        self.rtol, self.atol = rtol, atol
        self.n_dim = len(y0)
        self.original_fun_rtol, self.original_fun_atol = fun_rtol, fun_atol
        self.original_fun = fun
        self.original_jac = jac
        self.original_jac_sparsity = jac_sparsity
        self.original_fun_vectorized = vectorized
        self.original_fun_method = fun_method
        self.t_span = t_span
        self.y0 = np.array(y0)
        self._envelope_fun(t_span[0],y0,T_guess)
        self.T = self.T_new
        self.t = np.array([t_span[0]])
        self.y = np.zeros((self.n_dim,1))
        self.y[:,0] = self.y0
        self.H = self.T
        print('T =',self.T)

    def solve(self):
        while self.t[-1] < self.t_span[1]:
            if self._step() == BEEnv.SUCCESS:
                self.t = np.append(self.t,self.t_next)
                self.y = np.append(self.y,np.reshape(self.y_next,(self.n_dim,1)),axis=1)
            else:
                print('BEEnv.solve(%.3f)> error in self._step()' % self.t[-1])
                ipdb.set_trace()
        return self.t,self.y
        
    def _step(self):
        H = self.H
        t1 = self.t[-1]
        y1 = self.y[:,-1]
        y_p = y1 + H * self._envelope_fun(t1,y1)
        y_c = fsolve(lambda Y: Y - y1 - H*self._envelope_fun(t1+H,Y), y_p)
        t2 = t1 + H
        #if t2 > self.t_span[1]:
            #ipdb.set_trace()
            #t2 = self.t_span[1]
            #H = self.T * np.floor((self.t_span[1] - t1) / self.T)
        y2 = y_c
        t3 = self.t_new
        y3 = self.y_new
        dt12 = t2 - t1
        dy12 = (y2 - y1) / dt12
        dt23 = t3 - t2
        dy23 = (y3 - y2) / dt23
        d2y = (dy23 - dy12) / (self.T_new/2 + H/2)
        lte = H**2 / 2 * np.abs(d2y)
        scale = self.atol + self.rtol * np.abs(y_c)
        if np.all(lte < scale):
            self.t_next = t2
            self.y_next = y_c
            if self.H < 10*self.T:
                self.H *= 2
            return BEEnv.SUCCESS
        return BEEnv.FAILURE
    
    def _envelope_fun(self,t,y,T_guess=None):
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
            sol_a = solve_ivp(self.original_fun,[t,t+0.75*T_guess],y,
                              self.original_fun_method,jac=self.original_jac,
                              jac_sparsity=self.original_jac_sparsity,
                              vectorized=self.original_fun_vectorized,
                              dense_output=True,rtol=self.original_fun_rtol,
                              atol=self.original_fun_atol)
            sol_b = solve_ivp(self.original_fun,[sol_a['t'][-1],t+1.5*T_guess],
                              sol_a['y'][:,-1],self.original_fun_method,jac=self.original_jac,
                              jac_sparsity=self.original_jac_sparsity,
                              vectorized=self.original_fun_vectorized,
                              events=lambda t,y: np.dot(w,y)+b,dense_output=True,
                              rtol=self.original_fun_rtol,atol=self.original_fun_atol)
        else:
            sol_a = solve_ivp(self.original_fun,[t,t+0.75*T_guess],y,
                              self.original_fun_method,vectorized=self.original_fun_vectorized,
                              dense_output=True,rtol=self.original_fun_rtol,
                              atol=self.original_fun_atol)
            sol_b = solve_ivp(self.original_fun,[sol_a['t'][-1],t+1.5*T_guess],
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
                print('BEEnvelope._envelope_fun(%.3f)> T = T_guess = %.6f.' % (t,self.T_new))
        self.t_new = t + self.T_new
        self.y_new = sol_b['sol'](self.t_new)
        if DEBUG:
            print('BEEnv._envelope_fun(%.3f)> y = (%.4f,%.4f) T = %.6f.' % (t,self.y_new[0],self.y_new[1],self.T_new))
        # return the "vector field" of the envelope
        return 1./self.T_new * (self.y_new - sol_a['sol'](t))


def main():
    from systems import vdp
    epsilon = 1e-3
    A = [0]
    T = [1]
    fun = lambda t,y: vdp(t,y,epsilon,A,T)
    t_span = [0,5000*2*np.pi]
    y0 = [2e-3,0]
    T_guess = 2*np.pi
    solver = BEEnv(fun, t_span, y0, T_guess, rtol=1e-6, atol=1e-6)
    t,y = solver.solve()
    sol = solve_ivp(fun, t_span, y0, method='BDF', rtol=1e-8, atol=1e-10)
    import matplotlib.pyplot as plt
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(t,y[0],'ro-')
    plt.show()

if __name__ == '__main__':
    main()
    
