#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from polimi.systems import vdp, vdp_jac, y0min
from polimi.envelope import Envelope, _envelope_system

epsilon = 0.001
A = [0.,0.]
T = [10.,100.]
T = [2*np.pi,100.]
y0 = [2e-3,0]
reltol = 1e-8
abstol = 1e-10*np.ones(len(y0))

print('Integrating transient...')
tran = solve_ivp(lambda t,y: vdp(t,y,epsilon,A,T),[0,10000], y0,
                 method='BDF', jac=lambda t,y: vdp_jac(t,y,epsilon),
                 atol=abstol, rtol=reltol)
y0 = tran['y'][:,-1]
#plt.plot(tran['t'],tran['y'][0],'k')
#plt.show()

tend = 3000
print('Integrating the full system...')
full = solve_ivp(lambda t,y: vdp(t,y,epsilon,A,T),[0,tend], y0,
                 method='BDF', jac=lambda t,y: vdp_jac(t,y,epsilon),
                 atol=abstol, rtol=reltol, events=y0min)
#full = solve_ivp(lambda t,y: vdp(t,y,epsilon,A,T),[0,tend], y0,
#                 method='RK45', atol=abstol, rtol=reltol)

T_min = np.min(T)
env_fun = lambda t,y: _envelope_system(t,y,lambda t,y: vdp(t,y,epsilon,A,T),
                                       lambda t,y: vdp_jac(t,y,epsilon),T_min,
                                       atol=abstol,rtol=reltol)
print('Integrating the envelope...')
envelope = solve_ivp(env_fun, [0,tend], y0, method=Envelope, period=T_min)

plt.figure()
plt.plot(full['t'],full['y'][0],'k')
plt.plot(envelope['t'],envelope['y'][0],'ro-')
for i in range(10):
    plt.plot(full['t_events'][0][i]+np.zeros(2),[-0.005,0.005],'b--')
plt.ion()
plt.show()
