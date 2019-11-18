
__all__ = ['DynamicalSystem', 'VanderPol', 'HindmarshRose', 'Neuron7', 'Neuron4']

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


class Neuron7 (DynamicalSystem):

    def __init__(self, imposed_paths, with_variational=False, variational_T=1, **kwargs):
        super(Neuron7, self).__init__(7, with_variational, variational_T)

        self._set_default_pars(**kwargs)

        for name,value in self.default_pars.items():
            try:
                setattr(self, name, kwargs[name])
            except:
                setattr(self, name, value)

        for name,value in kwargs.items():
            if not name in self.default_pars:
                print('parameter "{}" not among defaults.'.format(name))
                setattr(self, name, value)

        self.kNa3 = self.kNa ** 3
        self.phi_Nab = self.phi(self.Nab)

        self.n_inf = lambda v: self.x_inf(v, self.theta_n, self.sigma_n)
        self.m_inf = lambda v: self.x_inf(v, self.theta_m, self.sigma_m)
        self.h_inf = lambda v: self.x_inf(v, self.theta_h, self.sigma_h)
        self.s_inf = lambda v: self.x_inf(v, self.theta_s, self.sigma_s)
        self.tau_n = lambda v: self.tau_x(v, self.theta_n, self.sigma_n, self.t_n)
        self.tau_m = lambda v: self.tau_x(v, self.theta_m, self.sigma_m, self.t_m)
        self.tau_h = lambda v: self.tau_x(v, self.theta_h, self.sigma_h, self.t_h)

        self.imposed_paths = imposed_paths
        if self.imposed_paths:
            self._ca_na_dynamics = self._elliptic_ca_na
        else:
            self._ca_na_dynamics = self._regular_ca_na


    def compute_ss(self, v):
        n = self.n_inf(v)
        m = self.m_inf(v)
        h = self.h_inf(v)
        s = self.s_inf(v)
        return n,m,h,s


    def _set_default_pars(self, **kwargs):
        self.default_pars = {'C': 45, 'k': 1., 'rpump': 200, 'epsilon': 7e-4, 'alpha': 6.6e-5,
                             'gL': 3., 'gNa': 150., 'gK': 30., 'gsyn': 2.5, 'gCan': 4.,
                             'EL': -60., 'ENa': 85., 'EK': -75., 'Esyn': 0., 'ECan': 0.,
                             'theta_h': -30., 'theta_m': -36., 'theta_n': -30., 'theta_s': 15.,
                             'kCan': 0.9, 'sigma_h': 5., 'sigma_m': -8.5, 'sigma_n': -5.,
                             'sigma_s': -3., 'sigma_Can': -0.05, 't_h': 15., 't_m': 1.,
                             't_n': 30., 'tau_s': 15., 'kNa': 10, 'Nab': 5, 'kCa': 22.5,
                             'Cab': 0.05, 'kIP3': 1200}


    def phi(self, x):
        y = x**3
        return y / (y + self.kNa3)


    def x_inf(self, v, theta_x, sigma_x):
        return 1. / (1. + np.exp((v - theta_x) / sigma_x))


    def tau_x(self, v, theta_x, sigma_x, t_x):
        return t_x / np.cosh((v - theta_x) / (2*sigma_x))


    def iL(self, v):
        return self.gL * (v - self.EL)


    def iK(self, v, n):
        return self.gK * n**4 * (v - self.EK)


    def iNa(self, v, m, h):
        return self.gNa * m**3 * h * (v - self.ENa)


    def isyn(self, v, s):
        return self.gsyn * s * (v - self.Esyn)


    def iCan(self, v, ca):
        return self.gCan * (v - self.ECan) / (1 + np.exp((ca - self.kCan) / self.sigma_Can))


    def ipump(self, na):
        return self.rpump * (self.phi(na) - self.phi_Nab)


    def _regular_ca_na(self, ca, na, s, ican, ipump):
        cadot = self.epsilon * (self.kIP3 * s - self.kCa * (ca - self.Cab))
        nadot = - self.alpha * (ican + ipump)
        return cadot,nadot


    def _elliptic_ca_na(self, ca, na, s, ican, ipump):
        cadot = - self.epsilon * self.d * (na - self.Nac)
        nadot = self.epsilon / self.d * (ca - self.Cac)
        return cadot,nadot


    def _fun(self, t, y):
        # the state variables
        v = y[0]
        n = y[1]
        m = y[2]
        h = y[3]
        s = y[4]
        ca = y[5]
        na = y[6]

        # calcium current
        ican = self.iCan(v,ca)
        # sodium pump current
        ipump = self.ipump(na)

        # the derivatives
        vdot = - 1 / self.C * (self.iL(v) + self.iK(v,n) + self.iNa(v,m,h) + \
                               self.isyn(v,s) + ican + ipump)
        ndot = (self.n_inf(v) - n) / self.tau_n(v)
        mdot = (self.m_inf(v) - m) / self.tau_m(v)
        hdot = (self.h_inf(v) - h) / self.tau_h(v)
        sdot = ((1 - s) * self.n_inf(v) - self.k * s) / self.tau_s
        cadot,nadot = self._ca_na_dynamics(ca, na, s, ican, ipump)

        return np.array([vdot, ndot, mdot, hdot, sdot, cadot, nadot])



class Neuron4 (Neuron7):

    def __init__(self, imposed_paths, with_variational=False, variational_T=1, **kwargs):
        super(Neuron4, self).__init__(imposed_paths, with_variational, variational_T, **kwargs)
        self.n_dim = 4


    def _set_default_pars(self, **kwargs):
        super(Neuron4, self)._set_default_pars(**kwargs)
        # set the default values of parameters that are different
        # from the 7-variable model
        self.default_pars['gK'] = 15.
        self.default_pars['gCan'] = 10.
        self.default_pars['sigma_s'] = -8.
        self.default_pars['k'] = 10.
        self.default_pars['epsilon'] = 0.005
        self.default_pars['theta_s'] = 10.
        self.default_pars['kCan'] = 0.25
        self.default_pars['kCa'] = 60.
        self.default_pars['kIP3'] = 1700.
        self.default_pars['rpump'] = 1500.


    def _fun(self, t, y):
        # the state variables
        v = y[0]
        n = y[1]
        ca = y[2]
        na = y[3]

        # steady-state approximation for the other state variables
        # of the 7-variable model
        m = self.m_inf(v)
        h = 1 - 1.08 * n
        s_inf = self.s_inf(v)
        s = s_inf / (s_inf + self.k)

        # calcium current
        ican = self.iCan(v,ca)
        # sodium pump current
        ipump = self.ipump(na)

        # the derivatives
        vdot = - 1 / self.C * (self.iL(v) + self.iK(v,n) + self.iNa(v,m,h) + \
                               self.isyn(v,s) + ican + ipump)
        ndot = (self.n_inf(v) - n) / self.tau_n(v)
        cadot,nadot = self._ca_na_dynamics(ca, na, s, ican, ipump)
        return np.array([vdot, ndot, cadot, nadot])


def square_wave(t, A, T, DC, tr, tf):
    tau = t % T
    if tau < tr:
        return A * tau / tr
    if tau < DC * T:
        return A
    if tau < DC * T + tf:
        return A * (1 - (tau - DC*T)/tf)
    return 0


class ASK_OOK (DynamicalSystem):

    def __init__(self, with_variational=False, variational_T=1, **kwargs):
        super(ASK_OOK, self).__init__(8, with_variational, variational_T)

        self._set_default_pars(**kwargs)

        for name,value in self.default_pars.items():
            try:
                setattr(self, name, kwargs[name])
            except:
                setattr(self, name, value)

        for name,value in kwargs.items():
            if not name in self.default_pars:
                print('parameter "{}" not among defaults.'.format(name))
                setattr(self, name, value)


        self._Vg1 = lambda t: square_wave(t, self.VDD, self.T1, self.DC1, 10e-9, 10e-9)
        self._Vg2 = lambda t: square_wave(t, self.VDD, self.T2, self.DC2, 10e-12, 10e-12)


    def _set_default_pars(self, **kwargs):
        F1 = 1e6
        F2 = 2e9
        CR = 1e-15
        LR = 1 / (CR*(2*np.pi*F2)**2)
        self.default_pars = {'VDD': 1.8, 'Rd1': 10, 'Cm1': 100e-15, 'L1': 20e-9,
                             'C1': 0.5e-9, 'L2': 20e-9, 'Rd2': 10, 'Cm2': 100e-15,
                             'C2': 1e-9, 'Ctl': CR, 'Ltl': LR, 'Rl': 300, 'F1': F1,
                             'F2': F2, 'alpha': 1, 'beta': 0.025, 'KT': 2, 'VT': 1,
                             'T1': 1/F1, 'T2': 1/F2, 'DC1': 0.1, 'DC2': 0.5,
                             'IS': 1e-6, 'eta': 2, 'VTemp': 0.026}


    def _I_mos(self, Vgs, Vds):
        v = self.KT * (Vgs - self.VT)
        return 0.5 * self.beta * (v + np.log(np.exp(v) + np.exp(-v))) * np.tanh(self.alpha * Vds)


    def _I_diode(self, Vd):
        return self.IS * (np.exp(Vd/(self.eta * self.VTemp)) - 1)


    def _fun(self, t, y):
        # state variables
        ILtl = y[0]
        IL1  = y[1]
        IL2  = y[2]
        out  = y[3]
        l20  = y[4]
        l30  = y[5]
        d10  = y[6]
        d20  = y[7]

        # derivatives
        y_dot = np.zeros(self.n_dim)
        y_dot[0] = out / self.Ltl
        y_dot[1] = (d10 - l20 - IL1*self.Rd1) / self.L1
        y_dot[2] = (l20 - l30) / self.L2
        y_dot[3] = (-out * self.Rd2 + (d20 - l30 + IL2*self.Rd2 - ILtl*self.Rd2) * self.Rl) / \
                   (self.Ctl * self.Rd2 * self.Rl)
        y_dot[4] = (IL1 - IL2) / self.C1
        y_dot[5] = (-self.C2 * out * self.Rd2 + ((self.C2 + self.Ctl)*(d20 - l30) + \
                                                 (self.C2 + self.Ctl) * IL2 * self.Rd2 - self.C2 * ILtl * self.Rd2) * \
                    self.Rl) / (self.C2 * self.Ctl * self.Rd2 * self.Rl)
        y_dot[6] = - (self._I_diode(d10 - self.VDD) + IL1 - self._I_mos(self._Vg1(t), self.VDD - d10)) / self.Cm1
        y_dot[7] = - (d20 - l30 + self._I_mos(self._Vg2(t), d20) * self.Rd2) / (self.Cm2 * self.Rd2)

        return y_dot


def ask_ook_test():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.optimize import fsolve

    oscillator = ASK_OOK()

    y0_correct = np.array([
        0.,          # ILtl
        0.000162828, # IL1
        0.000162828, # IL2
        0,           # out
        0.904516,    # l20
        0.904516,    # l30
        0.906144,    # d10
        0.902887     # d20
    ])
    y0_guess = np.array([
        0.,      # ILtl
        0.0001,  # IL1
        0.00005, # IL2
        0,       # out
        0.8,     # l20
        0.8,     # l30
        0.8,     # d10
        0.8      # d20
    ])

    y0 = fsolve(lambda y: oscillator._fun(0,y), y0_guess)
    print('(FULL CIRCUIT) Initial condition:')
    print('{:>10s} {:>13s} {:>13s}'.format('Variable','Computed','PAN'))
    var_names = ['ILtl', 'IL1', 'IL2', 'out', 'l20', 'l30', 'd10', 'd20']
    for var,py,pan in zip(var_names,y0,y0_correct):
        print('{:>10s} = {:13.5e} {:13.5e}'.format(var,py,pan))

    tend = 5 * oscillator.T2

    atol = 1e-6
    rtol = 1e-8

    sol = solve_ivp(oscillator, [0,tend], y0, method='RK45', atol=atol, rtol=rtol)

    fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,figsize=(10,4))
    ax1.plot(sol['t']*1e9, sol['y'][0], 'k', lw=1, label='i(Ltl)')
    ax1.plot(sol['t']*1e9, sol['y'][1], 'r', lw=1, label='i(L1)')
    ax1.plot(sol['t']*1e9, sol['y'][2], 'g', lw=1, label='i(L2)')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Current (A)')
    ax1.legend(loc='best')

    ax2.plot(sol['t']*1e9, sol['y'][3], 'k', lw=1, label='v(out)')
    ax2.plot(sol['t']*1e9, sol['y'][4], 'r', lw=1, label='v(l20)')
    ax2.plot(sol['t']*1e9, sol['y'][5], 'g', lw=1, label='v(l30)')
    ax2.plot(sol['t']*1e9, sol['y'][6], 'b', lw=1, label='v(d10)')
    ax2.plot(sol['t']*1e9, sol['y'][7], 'c', lw=1, label='v(d20)')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Voltage (V)')
    ax2.legend(loc='best')

    plt.show()


class ASK_OOK_lower (ASK_OOK):

    def __init__(self, with_variational=False, variational_T=1, **kwargs):
        super(ASK_OOK_lower, self).__init__(with_variational, variational_T)
        self.n_dim = 5


    def _fun(self, t, y):
        # state variables
        ILtl = y[0]
        IL2  = y[1]
        out  = y[2]
        l30  = y[3]
        d20  = y[4]

        # derivatives
        y_dot = np.zeros(self.n_dim)
        y_dot[0] = out / self.Ltl
        y_dot[1] = (self.VDD - l30) / self.L2
        y_dot[2] = (-out*self.Rd2 + (d20 - l30 + IL2*self.Rd2 - ILtl*self.Rd2) * self.Rl) / \
                   (self.Ctl * self.Rd2 * self.Rl)
        y_dot[3] = (-self.C2 * out * self.Rd2 + ((self.C2 + self.Ctl) * (d20 - l30) + (self.C2 + self.Ctl) * \
                                                 IL2 * self.Rd2 - self.C2 * ILtl * self.Rd2) * self.Rl) / \
                                                 (self.C2 * self.Ctl * self.Rd2 * self.Rl)
        y_dot[4] = - (d20 - l30 + self._I_mos(self._Vg2(t),d20) * self.Rd2) / (self.Cm2 * self.Rd2)

        return y_dot


def ask_ook_lower_test():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.optimize import fsolve

    oscillator = ASK_OOK_lower()
    y0_correct = np.array([
        0.,             # ILtl
        0.000214755,    # IL2
        0.,             # out
        oscillator.VDD, # l30
        1.79785         # d20
    ])
    y0_guess = np.array([
        1e-3,  # ILtl
        1e-3,  # IL2
        10e-3, # out
        1.5,   # l30
        1.4    # d20
    ])

    y0 = fsolve(lambda y: oscillator._fun(0,y), y0_guess)
    print('(LOWER CIRCUIT) Initial condition:')
    print('{:>10s} {:>13s} {:>13s}'.format('Variable','Computed','PAN'))
    var_names = ['ILtl', 'IL2', 'out', 'l30', 'd20']
    for var,py,pan in zip(var_names,y0,y0_correct):
        print('{:>10s} = {:13.5e} {:13.5e}'.format(var,py,pan))

    tend = 2 * oscillator.T2

    atol = 1e-6
    rtol = 1e-8

    sol = solve_ivp(oscillator, [0,tend], y0, method='RK45', atol=atol, rtol=rtol)

    fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,figsize=(10,4))
    ax1.plot(sol['t']*1e9, sol['y'][0]*1e6, 'k', lw=1, label='i(Ltl)')
    ax2.plot(sol['t']*1e9, sol['y'][2], 'k', lw=1, label='v(out)')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('i(Ltl) (uA)')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('v(out) (V)')
    plt.show()


class ASK_OOK_upper (ASK_OOK):

    def __init__(self, with_variational=False, variational_T=1, **kwargs):
        super(ASK_OOK_upper, self).__init__(with_variational, variational_T)
        self.n_dim = 2


    def _fun(self, t, y):
        # state variables
        IL1 = y[0]
        d10  = y[1]

        # derivatives
        y_dot = np.zeros(self.n_dim)
        y_dot[0] = (d10 - self.Rd1 * IL1) / self.L1
        y_dot[1] = - (self._I_diode(d10 - self.VDD) + IL1 - self._I_mos(self._Vg1(t), self.VDD - d10)) / self.Cm1

        return y_dot


def ask_ook_upper_test():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.optimize import fsolve

    oscillator = ASK_OOK_upper()
    y0_correct = np.array([
        0.000215755, # IL1
        0.00215755   # d10
    ])
    y0_guess = np.array([
        1e-3, # IL1
        10e-3 # d10
    ])

    y0 = fsolve(lambda y: oscillator._fun(0,y), y0_guess)
    print('(UPPER CIRCUIT) Initial condition:')
    print('{:>10s} {:>13s} {:>13s}'.format('Variable','Computed','PAN'))
    var_names = ['IL1', 'd10']
    for var,py,pan in zip(var_names,y0,y0_correct):
        print('{:>10s} = {:13.5e} {:13.5e}'.format(var,py,pan))

    tend = 2 * oscillator.T1

    atol = 1e-10
    rtol = 1e-12

    sol = solve_ivp(oscillator, [0,tend], y0, method='RK45', atol=atol, rtol=rtol)

    fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,figsize=(10,4))
    ax1.plot(sol['t']*1e9, sol['y'][0]*1e3, 'k', lw=1, label='i(L1)')
    ax2.plot(sol['t']*1e9, sol['y'][1]*1e3, 'k', lw=1, label='v(d10)')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('i(L1) (mA)')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('v(d10) (mV)')
    plt.show()


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
    #hr_test()
    ask_ook_lower_test()
    ask_ook_upper_test()
    ask_ook_test()
    
