
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from polimi.envelope import BEEnvelope, TrapEnvelope
import pickle
import os

# for saving data
pack = lambda t,y: np.concatenate((np.reshape(t,(len(t),1)),y.transpose()),axis=1)


def autonomous():
    from polimi import VanderPol

    epsilon = 1e-3
    A = [0]
    T = [1]
    vdp = VanderPol(epsilon, A, T)

    t_span = [0,4000*2*np.pi]
    t_interval = [500,1000]
    t_interval = [9700,10200]
    y0 = np.array([2e-3, 1e-3])
    #y0 = np.array([1,0.5])

    fun_rtol = 1e-8
    fun_atol = 1e-10
    env_rtol = 1e-3
    env_atol = 1e-6

    T_guess = 2*np.pi*0.9

    be_solver = BEEnvelope(vdp, t_span, y0, T_guess=T_guess, \
                           env_rtol=env_rtol, env_atol=env_atol, \
                           solver=solve_ivp, rtol=fun_rtol, \
                           atol=fun_atol, method='BDF', jac=vdp.jac)
    trap_solver = TrapEnvelope(vdp, t_span, y0, T=2*np.pi, \
                               env_rtol=env_rtol, env_atol=env_atol, \
                               solver=solve_ivp, rtol=fun_rtol, \
                               atol=fun_atol, method='BDF', jac=vdp.jac)

    try:
        data = pickle.load(open('vdp.pkl', 'rb'))
        sol = data['sol']
        sol_be = data['sol_be']
        sol_trap = data['sol_trap']
        t_span = data['t_span']
        #t_interval = data['t_interval']
    except:
        print('-' * 81)
        sol_be = be_solver.solve()
        print('-' * 81)
        sol_trap = trap_solver.solve()
        print('-' * 81)
        sol = solve_ivp(vdp, t_span, y0, method='BDF', rtol=1e-8, atol=1e-10)
        data = {'sol': sol, 'sol_be': sol_be, 'sol_trap': sol_trap, \
                't_span': t_span, 't_interval': t_interval}
        pickle.dump(data, open('vdp.pkl', 'wb'))


    black = [0,0,0]
    grey = [.3,.3,.3]
    light_grey = [.7,.7,.7]

    fig = plt.figure(figsize=(3.5,5))
    ax1 = plt.axes([0.1,0.65,0.8,0.275])
    ax2 = plt.axes([0.1,0.3,0.8,0.275])
    ax3 = plt.axes([0.1,0.1,0.8,0.125])

    ms = 3
    ax1.plot(sol['t'], sol['y'][0], color=light_grey, linewidth=0.5)
    ax1.plot(sol_be['t'], sol_be['y'][0], 'o-', color=black, \
             linewidth=1, markerfacecolor='w', markersize=ms)
    ax1.plot(sol_trap['t'], sol_trap['y'][0], 's-', color=grey, \
             linewidth=1, markerfacecolor='w', markersize=ms)
    ax1.set_xlim(t_span)
    ax1.set_ylabel('x')

    idx, = np.where((sol['t'] > t_interval[0]) & (sol['t'] < t_interval[1]))
    ax2.plot(sol['t'][idx], sol['y'][0,idx], color=light_grey, linewidth=0.5)
    m = np.min(sol['y'][0,idx])
    M = np.max(sol['y'][0,idx])
    y_lim = 2 * np.array([m,M])
    ax1.plot(t_interval[0] + np.zeros(2), y_lim, 'k--', linewidth=1)
    ax1.plot(t_interval[1] + np.zeros(2), y_lim, 'k--', linewidth=1)
    ax1.plot(t_interval, y_lim[0]+np.zeros(2), 'k--', linewidth=1)
    ax1.plot(t_interval, y_lim[1]+np.zeros(2), 'k--', linewidth=1)

    idx, = np.where((sol_be['t'] > t_interval[0]) & (sol_be['t'] < t_interval[1]))
    idx = np.r_[idx[0]-1, idx, idx[-1]+1]
    ax2.plot(sol_be['t'][idx], sol_be['y'][0,idx], 'o-', color=black, \
             linewidth=1, markerfacecolor='w', markersize=ms+1)

    idx, = np.where((sol_trap['t'] > t_interval[0]) & (sol_trap['t'] < t_interval[1]))
    idx = np.r_[idx[0]-1, idx, idx[-1]+1]
    ax2.plot(sol_trap['t'][idx], sol_trap['y'][0,idx], 's-', color=grey, \
             linewidth=1, markerfacecolor='w', markersize=ms+1)
    ax2.set_xlim(t_interval)
    ax2.set_ylim([-0.5,0.5])
    ax2.set_yticks(np.arange(-0.4,0.5,0.2))
    ax2.set_ylabel('x')

    ax3.plot(sol_be['t'][1:], np.round(np.diff(sol_be['t'])/(2*np.pi)), 'o-', color=black, \
             linewidth=1, markerfacecolor='w', markersize=ms)
    ax3.plot(sol_trap['t'][1:], np.round(np.diff(sol_trap['t'])/(2*np.pi)), 's-', color=grey, \
             linewidth=1, markerfacecolor='w', markersize=ms)
    ax3.set_xlim(t_span)
    ax3.set_xlabel('Time')
    ax3.set_yticks(np.arange(0,510,100))
    ax3.set_ylabel('Envelope time-step')

    #plt.plot(sol['t'],sol['y'][0],'k')
    #plt.plot(sol_be['t'],sol_be['y'][0],'ro-')
    #plt.plot(sol_trap['t'],sol_trap['y'][0],'go-')
    plt.savefig('vdp_envelope.pdf')
    plt.show()


def HR():
    from polimi import HindmarshRose
    b = 3
    I = 5
    hr = HindmarshRose(I,b)

    y0 = [0,1,0.1]
    #y0 = np.array([-0.85477615, -3.03356705,  4.73029393])
    t_tran = 100
    sol = solve_ivp(hr, [0,t_tran], y0, method='RK45', rtol=1e-8, atol=1e-10)
    y0 = sol['y'][:,-1]

    t_span = [0,5000]
    T_guess = 11

    be_solver = BEEnvelope(hr, t_span, y0, T_guess=T_guess, \
                           max_step=500, integer_steps=False, \
                           env_rtol=1e-3, env_atol=1e-6, \
                           rtol=1e-8, atol=1e-10)
    trap_solver = TrapEnvelope(hr, t_span, y0, T_guess=T_guess, \
                               max_steps=500, integer_steps=True, \
                               env_rtol=1e-3, env_atol=1e-6, \
                               rtol=1e-8, atol=1e-10)

    print('-' * 81)
    sol_be = be_solver.solve()
    print('-' * 81)
    sol_trap = trap_solver.solve()
    print('-' * 81)
    sol = solve_ivp(hr, t_span, y0, method='RK45', rtol=1e-8, atol=1e-10)

    fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(10,6))
    ax1.plot(sol['t'],sol['y'][0],'k')
    for t0,y0,T in zip(sol_be['t'],sol_be['y'].T,sol_be['T']):
        period = solve_ivp(hr, [t0,t0+T], y0, method='RK45', rtol=1e-8, atol=1e-10)
        ax1.plot(period['t'], period['y'][0], color=[1,.6,.6], lw=1)
    for t0,y0,T in zip(sol_trap['t'],sol_trap['y'].T,sol_trap['T']):
        period = solve_ivp(hr, [t0,t0+T], y0, method='RK45', rtol=1e-8, atol=1e-10)
        ax1.plot(period['t'], period['y'][0], color=[.6,1,.6], lw=1)
    ax1.plot(sol_be['t'],sol_be['y'][0],'ro-')
    ax1.plot(sol_trap['t'],sol_trap['y'][0],'go-')
    ax1.plot(sol_be['t'],sol_be['T'],'ms-')
    ax1.set_xlabel('t')
    ax1.set_ylabel('y')

    idx, = np.where(sol['t'] > 1000)
    ax2.plot(sol['y'][0,idx],sol['y'][1,idx],'k')
    for t0,y0,T in zip(sol_be['t'],sol_be['y'].T,sol_be['T']):
        if t0 < 1000:
            continue
        period = solve_ivp(hr, [t0,t0+T], y0, method='RK45', rtol=1e-8, atol=1e-10)
        ax2.plot(period['y'][0], period['y'][1], color=[1,.6,.6], lw=1)
    idx, = np.where(sol_be['t'] > 1000)
    ax2.plot(sol_be['y'][0,idx],sol_be['y'][1,idx],'ro-')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    plt.show()


def main():
    import polimi.utils
    polimi.utils.set_rc_defaults()

    autonomous()
    HR()


if __name__ == '__main__':
    main()
