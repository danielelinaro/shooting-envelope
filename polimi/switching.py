
import numpy as np
from scipy.integrate import solve_ivp


__all__ = ['SwitchingSystem', 'Boost', 'solve_ivp_switch']



class SwitchingSystem (object):

    def __call__(self, t, y):
        raise NotImplementedError


    def J(self, t, y):
        raise NotImplementedError


    def handle_event(self, event_index):
        raise NotImplementedError


    @property
    def event_functions(self):
        raise NotImplementedError



class Boost (SwitchingSystem):

    def __init__(self, t0, T=20e-6, DC=0.5, ki=1.5, Vref=5, Vin=5, R=5, L=10e-6, C=47e-6, Rs=0):
        self.T = T
        self.DC = DC
        self.ki = ki
        self.Vref = Vref
        self.Vin = Vin
        self.R = R
        self.L = L
        self.C = C
        self.Rs = Rs
        self._make_matrixes()

        if (t0 % self.T) < (self.DC * self.T):
            self.flag = 1
        else:
            self.flag = 0

        self.n = int(t0 / self.T) + 1

        self._event_functions = [lambda t,y: Boost.manifold(self, t, y), \
                                 lambda t,y: Boost.clock(self, t, y)]
        self._event_functions[0].direction = 1
        for event_fun in self._event_functions:
            event_fun.terminal = 1


    def __call__(self, t, y):
        return self.flag * np.matmul(self.A1,y) + (1-self.flag) * np.matmul(self.A2,y) + self.B


    def J(self, t, y):
        return self.flag * self.A1 + (1-self.flag) * self.A2


    def handle_event(self, event_index, event_time):
        self.flag = event_index
        if event_index == 1:
            self.n += 1


    def clock(self, t, y):
        return t - self.n * self.T


    def manifold(self, t, y):
        return self.ki * y[1] - self.Vref


    @property
    def event_functions(self):
        return self._event_functions


    def _make_matrixes(self):
        self.A1 = np.array([ [-1/(self.R*self.C), 0],   [0, -self.Rs/self.L]    ])
        self.A2 = np.array([ [-1/(self.R*self.C), 1/self.C], [-1/self.L, -self.Rs/self.L] ])
        self.B  = np.array([0, self.Vin/self.L])



def solve_ivp_switch(sys, t_span, y0, **kwargs):

    for k in ('dense_output', 'events'):
        if k in kwargs:
            kwargs.pop(k)

    t_cur = t_span[0]
    t_end = t_span[1]
    y_cur = y0

    t = np.array([])
    y = np.array([[],[]])

    while t_cur < t_end:
        sol = solve_ivp(sys, [t_cur,t_end], y_cur,
                        events=sys.event_functions,
                        dense_output=True, **kwargs)
        t_next = np.inf
        for i,t_ev in enumerate(sol['t_events']):
            if len(t_ev) > 0 and t_ev[0] != t_cur and t_ev[0] < t_next:
                t_next = t_ev[0]
                ev_idx = i
        sys.handle_event(ev_idx, t_ev)
        y_next = sol['sol'](t_next)
        idx, = np.where(sol['t'] <= t_next)
        t = np.append(t, sol['t'][idx])
        t = np.append(t, t_next)
        y = np.append(y, sol['y'][:,idx], axis=1)
        y = np.append(y, np.array([y_next]).transpose(), axis=1)
        t_cur = t_next
        y_cur = y_next

    return {'t': t, 'y': y}
