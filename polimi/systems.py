
__all__ = ['DynamicalSystem', 'VanderPol', 'HindmarshRose']

import numpy as np


def jacobian_finite_differences(fun,t,y):
    n = len(y)
    J = np.zeros((n,n))
    ref = fun(t,y)
    eps = 1e-8
    for i in range(n):
        dy = np.zeros(n)
        dy[i] = eps
        pert = fun(t,y+dy)
        J[:,i] = (pert-ref)/eps
    return J


class DynamicalSystem (object):

    def __init__(self, n_dim, with_variational=False, variational_T=1):
        self.n_dim = n_dim
        self.with_variational = with_variational
        self.variational_T = variational_T


    def __call__(self, t, y):
        T = self.variational_T

        if not self.with_variational:
            return T * self._fun(t*T,y)

        N = self.n_dim
        phi = np.reshape(y[N:N+N**2], (N,N))
        J = self._J(t * T, y[:N])
        ydot = np.concatenate((T * self._fun(t*T, y[:N]), \
                               T * (J @ phi).flatten()))
        if len(y) == N * (2 + N):
            # we are estimating the period, too
            ydot = np.concatenate((ydot, T * (J @ y[-N:]) + self._fun(t,y[:N])))
        return ydot


    def jac(self, t, y):
        T = self.variational_T
        return T * self._J(t*T, y)


    def _fun(self, t, y):
        raise NotImplementedError


    def _J(self, t, y):
        raise NotImplementedError


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


class VanderPol (DynamicalSystem):

    def __init__(self, epsilon, A, T, with_variational=False, variational_T=1):
        super(VanderPol, self).__init__(2, with_variational, variational_T)

        self.epsilon = epsilon

        if np.isscalar(A):
            self.A = np.array([A])
        elif isinstance(A, list) or isinstance(A, tuple):
            self.A = np.array(A)
        else:
            self.A = A

        if np.isscalar(T):
            self.T = np.array([T])
        elif isinstance(T, list) or isinstance(T, tuple):
            self.T = np.array(T)
        else:
            self.T = T

        self.F = np.array([1./t for t in T])
        self.n_forcing = len(self.F)


    def _fun(self, t, y):
        ydot = np.array([
            y[1],
            self.epsilon*(1-y[0]**2)*y[1] - y[0]
        ])
        for i in range(self.n_forcing):
            ydot[1] += self.A[i] * np.cos(2 * np.pi * self.F[i] * t)
        return ydot


    def _J(self, t, y):
        return np.array([
            [0, 1],
            [-2 * self.epsilon * y[0] * y[1] - 1, self.epsilon * (1 - y[0] ** 2)]
        ])



class HindmarshRose (DynamicalSystem):

    def __init__(self, I, b, mu=0.01, s=4, x_rest=-1.6, with_variational=False, variational_T=1):
        super(HindmarshRose, self).__init__(3, with_variational, variational_T)
        self.I = I
        self.b = b
        self.mu = mu
        self.s = s
        self.x_rest = x_rest


    def _fun(self, t, y):
        return np.array([
            y[1] - y[0]**3 + self.b*y[0]**2 + self.I - y[2],
            1 - 5*y[0]**2 - y[1],
            self.mu * (self.s * (y[0] - self.x_rest) - y[2])
        ])


    def _J(self, t, y):
        return np.array([
            [-3*y[0]**2 + 2*self.b*y[0], 1, -1],
            [-10*y[0], -1, 0],
            [self.mu*self.s, 0, -self.mu]
        ])


#################################################################
# Unused stuff - START

def vdp_auto(t,y,epsilon,A,T):
    rho = 1
    N_forcing = len(A)
    N_eq = 2 + N_forcing*2
    ydot = np.zeros(N_eq)
    ydot[0] = y[1]
    ydot[1] = epsilon*(1-y[0]**2)*y[1] - y[0]
    for i in range(N_forcing):
        j = 2*(i+1)
        ydot[1] += A[i] * y[j]
        sum_of_squares = y[j]**2 + y[j+1]**2
        ydot[j+1] = 2*np.pi/T[i]*y[j] + y[j+1]*((rho-sum_of_squares)/(2*sum_of_squares))
        ydot[j] = (rho - sum_of_squares - 2*y[j+1]*ydot[j+1]) / (2*y[j])
    return ydot

def polar(t,y,rho,T):
    sum_of_squares = y[0]**2 + y[1]**2
    ydot = np.zeros(2)
    ydot[1] = 2*np.pi/T*y[0] + y[1]*((rho-sum_of_squares)/(2*sum_of_squares))
    ydot[0] = (rho - sum_of_squares - 2*y[1]*ydot[1]) / (2*y[0])
    return ydot

# Unused stuff - END
#################################################################


def hr_test():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    b = 3
    I = 5
    tend = 500
    hr = HindmarshRose(I, b)
    y0 = [0,1,0.1]
    atol = 1e-8
    rtol = 1e-10
    sol = solve_ivp(hr, [0,tend], y0, method='RK45', atol=atol, rtol=rtol)
    plt.plot(sol['t'],sol['y'][0],'r',label='RK45',)
    sol = solve_ivp(hr, [0,tend], y0, method='BDF', jac=hr.jac, atol=atol, rtol=rtol)
    plt.plot(sol['t'],sol['y'][0],'k',label='BDF')
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('x')
    plt.show()


if __name__ == '__main__':
    hr_test()
