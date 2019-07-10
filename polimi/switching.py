
import numpy as np
from scipy.integrate import solve_ivp


__all__ = ['SwitchingSystem', 'Boost', 'solve_ivp_switch']


def variational_system(fun, jac, t, y, T):
    N = int((-1 + np.sqrt(1 + 4*len(y))) / 2)
    J = jac(t*T,y[:N])
    phi = np.reshape(y[N:N+N**2],(N,N))
    return np.concatenate((T * fun(t*T, y[:N]), \
                           T * np.matmul(J,phi).flatten()))




class SwitchingSystem (object):

    def __init__(self, vector_field_index=0, with_variational=False, variational_T=None):
        self.vector_field_index = vector_field_index
        self.with_variational = with_variational
        self._variational_T = variational_T


    def __call__(self, t, y):
        if not self.with_variational:
            return self._fun(t,y)
        T = self.variational_T
        N = int((-1 + np.sqrt(1 + 4*len(y))) / 2)
        J = self.J(t * T, y[:N])
        phi = np.reshape(y[N:N+N**2], (N,N))
        return np.concatenate((T * self._fun(t*T, y[:N]), \
                               T * np.matmul(J,phi).flatten()))


    def _fun(self, t, y):
        raise NotImplementedError


    def J(self, t, y):
        raise NotImplementedError


    def handle_event(self, event_index):
        raise NotImplementedError


    @property
    def vector_field_index(self):
        return self._vector_field_index


    @vector_field_index.setter
    def vector_field_index(self, index):
        self._vector_field_index = index


    @property
    def with_variational(self):
        return self._with_variational


    @with_variational.setter
    def with_variational(self, value):
        if not isinstance(value, bool):
            raise ValueError('value must be of boolean type')
        self._with_variational = value


    @property
    def variational_T(self):
        return self._variational_T


    @variational_T.setter
    def variational_T(self, T):
        if T <= 0:
            raise ValueError('T must be greater than 0')
        self._variational_T = T


    @property
    def event_functions(self):
        raise NotImplementedError



class Boost (SwitchingSystem):

    def __init__(self, vector_field_index=0,
                 T=20e-6, DC=0.5, ki=1.5, Vref=5, Vin=5, R=5,
                 L=10e-6, C=47e-6, Rs=0, clock_phase=0,
                 with_variational=False, variational_T=1):

        if not with_variational:
            if not variational_T is None and variational_T != 1:
                print('with_variational is False, ignoring value of variational_T')
            variational_T = 1

        super(Boost, self).__init__(vector_field_index, with_variational, variational_T)

        self.T = T
        self.F = 1./T
        self.phi = clock_phase
        self.DC = DC
        self.ki = ki
        self.Vref = Vref
        self.Vin = Vin
        self.R = R
        self.L = L
        self.C = C
        self.Rs = Rs

        self._make_matrixes()

        self._event_functions = [lambda t,y: Boost.manifold(self, t, y), \
                                 lambda t,y: Boost.clock(self, t*self.variational_T, y)]
        for event_fun in self._event_functions:
            event_fun.direction = 1
            event_fun.terminal = 1


    def _fun(self, t, y):
        return np.matmul(self.A[self.vector_field_index],y) + self.B


    def J(self, t, y):
        return self.A[self.vector_field_index]


    def handle_event(self, event_index):
        self.vector_field_index = 1 - event_index


    def clock(self, t, y):
        return np.sin(2*np.pi*self.F*(t-self.phi))


    def manifold(self, t, y):
        return self.ki * y[1] - self.Vref


    @property
    def vector_field_index(self):
        return self._vector_field_index


    @vector_field_index.setter
    def vector_field_index(self, index):
        if not index in (0,1):
            raise ValueError('index must be either 0 or 1')
        self._vector_field_index = index


    @property
    def event_functions(self):
        return self._event_functions


    def _make_matrixes(self):
        self.A = [np.array([ [-1/(self.R*self.C), 0],   [0, -self.Rs/self.L]    ]), \
                  np.array([ [-1/(self.R*self.C), 1/self.C], [-1/self.L, -self.Rs/self.L] ])]
        self.B  = np.array([0, self.Vin/self.L])



def solve_ivp_switch(sys, t_span, y0, **kwargs):

    for k in ('dense_output', 'events'):
        if k in kwargs:
            kwargs.pop(k)

    t_cur = t_span[0]
    t_end = t_span[1]
    y_cur = y0

    t = np.array([])
    y = np.array([[] for _ in range(y0.shape[0])])

    event_functions = sys.event_functions.copy()
    event_functions.append(lambda t,y: t - t_end)
    event_functions[-1].terminal = 0
    event_functions[-1].direction = 1
    n_events = len(event_functions)

    while np.abs(t_cur - t_end) > 1e-10:
        sol = solve_ivp(sys, [t_cur,t_end*1.001], y_cur,
                        events=event_functions,
                        dense_output=True, **kwargs)
        t_next = np.inf
        ev_idx = None
        for i,t_ev in enumerate(sol['t_events']):
            if len(t_ev) > 0 and t_ev[-1] != t_cur and np.abs(t_ev[-1] - t_next) > 1e-10:
                t_next = t_ev[-1]
                ev_idx = i
        if ev_idx is None:
            t_next = sol['t'][-1]
            y_next = sol['y'][:,-1]
        else:
            if ev_idx < n_events-1:
                sys.handle_event(ev_idx)
            y_next = sol['sol'](t_next)
        idx, = np.where(sol['t'] < t_next)
        t = np.append(t, sol['t'][idx])
        t = np.append(t, t_next)
        y = np.append(y, sol['y'][:,idx], axis=1)
        y = np.append(y, np.array([y_next]).transpose(), axis=1)
        t_cur = t_next
        y_cur = y_next

    return {'t': t, 'y': y}
