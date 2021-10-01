
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.common import num_jac
from polimi.systems import HindmarshRose, VanderPol, jacobian_finite_differences
from polimi.solvers import backward_euler, trapezoidal, backward_euler_var_step, trapezoidal_var_step

import ipdb

def hr():
    pars = {'b': 3, 'I': 6}
    atol = 1e-6
    rtol = 1e-8

    # I = 5
    #y0 = np.array([1.79567479, -4.12438334,  4.75890062])
    #T = 10.690191326420972
    # I = 6
    y0 = np.array([1.9402303,  -4.67468625,  5.31481002])
    T = 7.287559745299973
    n_steps = 1e4

    keys = ('bdf', 'be', 'trap')
    hr = {key: HindmarshRose(pars['I'], pars['b']) for key in keys}

    if y0 is None:
        t_end = 1000
        y0 = [0,1,0.1]
        x_max = lambda t,y: y[1] - y[0]**3 + pars['b']*y[0]**2 + pars['I'] - y[2]
        x_max.direction = -1
        x_max.terminal = False
        sol = solve_ivp(hr['bdf'], [0,t_end], y0, method='BDF', jac=hr['bdf'].jac, events=x_max,
                        atol=atol, rtol=rtol, dense_output=True)
        t_ev = np.array(sol['t_events'][0])
        y_ev = np.array([sol['sol'](t) for t in t_ev]).T
        T = t_ev[-1] - t_ev[-2]
        print('T = {}.'.format(T))
        y0 = y_ev[:,-2]
        print(y0)
        #plt.plot(sol['t'],sol['y'].T,lw=1)
        #plt.show()

    t_end = T * 1.05
    h = t_end / n_steps
    t_span = np.array([0,t_end])
    t_span_var = t_span / T

    sol = {}
    phi = {}
    eig = {}

    n_dim = hr['bdf'].n_dim
    y0_var = np.concatenate((y0, np.eye(n_dim).flatten()))
    hr['bdf'].with_variational = True
    hr['bdf'].variational_T = T

    sol['bdf'] = solve_ivp(hr['bdf'], t_span_var, y0_var, method='BDF', atol=atol, rtol=rtol)
    eig['bdf'] = np.array([sorted(np.linalg.eig(np.reshape(y[n_dim:], (n_dim,n_dim)))[0],
                                  reverse=True) for y in sol['bdf']['y'].T])

    sol['be'] = backward_euler(hr['be'], t_span, y0, h)
    phi = np.eye(n_dim)
    I = np.eye(n_dim)
    n_steps = len(sol['be']['t'])
    eig['be'] = []
    for i in range(1,n_steps):
        J = hr['be'].jac(sol['be']['t'][i], sol['be']['y'][:,i])
        tmp = I - h * J
        phi = np.linalg.inv(tmp) @ phi
        eig['be'].append(sorted(np.linalg.eig(phi)[0], reverse=True))
    eig['be'] = np.array(eig['be'])

    sol['trap'] = trapezoidal(hr['trap'], t_span, y0, h)
    phi = np.eye(n_dim)
    I = np.eye(n_dim)
    n_steps = len(sol['trap']['t'])
    eig['trap'] = []
    for i in range(1,n_steps):
        J = hr['trap'].jac(sol['trap']['t'][i], sol['trap']['y'][:,i])
        tmp = I - h * J
        phi = np.linalg.inv(tmp) @ phi
        eig['trap'].append(sorted(np.linalg.eig(phi)[0], reverse=True))
    eig['trap'] = np.array(eig['trap'])

    plt.plot(T+np.zeros(2), [0.85,2.2], '--', color=[.6,.6,.6], lw=1)
    plt.plot(sol['bdf']['t']*T, np.real(eig['bdf'][:,0]), 'k.-', lw=1, label='BDF')
    plt.plot(sol['be']['t'][1:], np.real(eig['be'][:,0]), 'r.-', lw=1, label='BE')
    plt.plot(sol['trap']['t'][1:], np.real(eig['trap'][:,0]), 'b.-', lw=1, label='TRAP')
    plt.legend(loc='best')

    col = {'bdf': 'k', 'be': 'r', 'trap': 'b'}
    plt.figure()
    for k,v in sol.items():
        plt.plot(v['t']*hr[k].variational_T, v['y'][0], col[k], lw=1, label=k)
    plt.legend(loc='best')
    plt.show()


def vdp():
    pars = {'eps': 0.001, 'A': [0], 'T': [4]}
    atol = 1e-6
    rtol = 1e-8

    #y0 = np.array([1.79567479, -4.12438334,  4.75890062])
    #T = 10.690191326420972
    y0 = None
    n_steps = 1e4

    if y0 is None:
        t_end = 1000
        y0 = [1,1]
        x_max = lambda t,y: y[1]
        x_max.direction = -1
        x_max.terminal = False
        vdp = VanderPol(pars['eps'], pars['A'], pars['T'])
        sol = solve_ivp(vdp, [0,t_end], y0, method='BDF', jac=vdp.jac, events=x_max,
                        atol=atol, rtol=rtol, dense_output=True)
        t_ev = np.array(sol['t_events'][0])
        y_ev = np.array([sol['sol'](t) for t in t_ev]).T
        T = t_ev[-1] - t_ev[-2]
        print('T = {}.'.format(T))
        y0 = y_ev[:,-2]
        #y0 = sol['y'][:,-1]
        plt.plot(sol['t'],sol['y'].T,lw=1)
        plt.show()

    keys = ('bdf','be_inv','be_dir')
    #sys.stdout.write('{:10s}'.format('# t_end'))
    #for k in keys:
    #    for i in range(3):
    #        suffix = '{}{}'.format(k,i+1)
    #        sys.stdout.write(' {:10s} {:10s}'.format('R('+suffix+')','I('+suffix+')'))
    #sys.stdout.write('\n')

    for coeff in np.linspace(1/2,1,20):
        vdp = {key: VanderPol(pars['eps'], pars['A'], pars['T']) for key in ('bdf','be')}
        t_end = T * coeff
        h = t_end / n_steps
        t_span = np.array([0,t_end])
        t_span_var = t_span / T

        sol = {}
        phi = {}
        eig = {}

        n_dim = vdp['bdf'].n_dim
        y0_var = np.concatenate((y0, np.eye(n_dim).flatten()))
        vdp['bdf'].with_variational = True
        vdp['bdf'].variational_T = T

        sol['bdf'] = solve_ivp(vdp['bdf'], t_span_var, y0_var, method='BDF', atol=atol, rtol=rtol)
        phi['bdf'] = np.reshape(sol['bdf']['y'][n_dim:,-1], (n_dim, n_dim))
        eig['bdf'],_ = np.linalg.eig(phi['bdf'])
        eig['bdf'] = sorted(eig['bdf'], reverse=True)
        thresh = np.max(np.linalg.inv(phi['bdf'])) * 5

        vdp['be'].with_variational = False
        vdp['be'].variational_T = 1
        sol['be'] = backward_euler(vdp['be'], t_span, y0, h)
        phi_inv = np.eye(n_dim)
        phi_dir = np.eye(n_dim)
        I = np.eye(n_dim)
        n_steps = len(sol['be']['t'])
        for i in range(1,n_steps):
            tmp = I - h * vdp['be'].jac(sol['be']['t'][i], sol['be']['y'][:,i])
            phi_inv = phi_inv @ tmp
            phi_dir = np.linalg.inv(tmp) @ phi_dir
            #if np.any(phi_inv @ tmp > thresh):
            #    ipdb.set_trace()

        phi['be_inv'] = np.linalg.inv(phi_inv)
        eig['be_inv'],_ = np.linalg.eig(phi['be_inv'])
        eig['be_inv'] = sorted(eig['be_inv'], reverse=True)
        phi['be_dir'] = phi_dir
        eig['be_dir'],_ = np.linalg.eig(phi['be_dir'])
        eig['be_dir'] = sorted(eig['be_dir'], reverse=True)

        sys.stdout.write('{:10.3e} '.format(t_end))
        for k in keys:
            for e in eig[k]:
                sys.stdout.write(' {:10.3e} {:10.3e}'.format(np.real(e), np.imag(e)))
        sys.stdout.write('\n')

        #col = {'bdf': 'k', 'be': 'r'}
        #plt.figure()
        #for k,v in sol.items():
        #    plt.plot(v['t']*hr[k].variational_T, v['y'][0], col[k], lw=1, label=k)
        #plt.legend(loc='best')
    #plt.show()


if __name__ == '__main__':
    hr()
    #vdp()
