import numpy as np
from scipy.integrate import solve_ivp

from .rk import RK23Envelope, RK45Envelope
from .bdf import BDFEnvelope
from .bdfenv import BDFEnv

#############################################################################
############################## HELPER FUNCTIONS #############################
#############################################################################


def _one_period(t,y,fun,T_guess,method='RK45',**options):
    # find the equation of the plane containing y and
    # orthogonal to fun(t,y)
    f = fun(t,y)
    w = f/np.linalg.norm(f)
    b = -np.dot(w,y)
    # first integration without events, because event(t0) = 0
    # and the solver goes crazy
    sol_a = solve_ivp(fun,[t,t+0.75*T_guess],y,method,dense_output=True,**options)
    sol_b = solve_ivp(fun,[sol_a['t'][-1],t+1.5*T_guess],sol_a['y'][:,-1],method,
                      events=lambda t,y: np.dot(w,y)+b,dense_output=True,**options)

    for t_ev in sol_b['t_events'][0]:
        x_ev = sol_b['sol'](t_ev)
        f = fun(t_ev,x_ev)
        # check whether the crossing of the plane is in
        # the same direction as the initial point
        if np.dot(w,f/np.linalg.norm(f))+b > 0:
            T = t_ev-t
            break

    # compute the "vector field" of the envelope
    G = 1./T * (sol_b['sol'](t+T) - sol_a['sol'](t))

    #print('_one_period> T({}) = {} s (guess = {} s).'.format(t,T,T_guess))
    
    return G


def _envelope_system(t,y,fun,T,method='RK45',**options):
    one_period = solve_ivp(fun,[0,T],y,method,**options)
    G = 1./T * (one_period['y'][:,-1] - one_period['y'][:,0])
    return G




