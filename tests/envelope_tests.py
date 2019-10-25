
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


def forced_polar():
    from polimi import vdp_auto
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
    sol_be = be_solver.solve()
    sol_trap = trap_solver.solve()
    sol = solve_ivp(fun, t_span, y0, method='BDF', rtol=1e-8, atol=1e-10)
    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(sol_be['t'],sol_be['y'][0],'ro-')
    plt.plot(sol_trap['t'],sol_trap['y'][0],'go-')
    plt.show()


def forced():
    from polimi import VanderPol

    epsilon = 1e-3
    T_exact = 10
    T_guess = 0.9 * T_exact
    A = [1,10]
    #A = [10,1]
    T = [T_exact,T_exact*100]
    rtol = {'fun': 1e-8, 'env': 1e-1}
    atol = {'fun': 1e-10, 'env': 1e-3}

    y0 = [2e-3,0]
    vdp = VanderPol(epsilon, A, T)
    method = 'RK45'

    t0 = 0
    ttran = 1000
    if ttran > 0:
        print('Integrating the full system (transient)...')
        tran = solve_ivp(vdp, [t0,ttran], y0, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
        t0 = tran['t'][-1]
        y0 = tran['y'][:,-1]
        plt.plot(tran['t'],tran['y'][0],'k')
        plt.plot(tran['t'],tran['y'][1],'r')
        plt.show()

    print('t0 =',t0)
    print('y0 =',y0)

    t_span = [t0,t0+T[1]]
    be_solver = BEEnvelope(vdp, t_span, y0, T=T_exact, \
                           env_rtol=rtol['env'], env_atol=atol['env'], \
                           rtol=rtol['fun'], atol=atol['fun'])
    trap_solver = TrapEnvelope(vdp, t_span, y0, T=T_exact, \
                               env_rtol=rtol['env'], env_atol=atol['env'], \
                               rtol=rtol['fun'], atol=atol['fun'])
    sol_be = be_solver.solve()
    print('The number of integrated periods of the original system with BE is %d.' % be_solver.original_fun_period_eval)
    sol_trap = trap_solver.solve()
    print('The number of integrated periods of the original system with TRAP is %d.' % trap_solver.original_fun_period_eval)
    sol = solve_ivp(vdp, t_span, y0, method='RK45', rtol=rtol['fun'], atol=atol['fun'])

    #np.savetxt('vdp_forced_T=[{},{}]_A=[{},{}].txt'.format(T[0],T[1],A[0],A[1]), \
    #           pack(sol['t'],sol['y']), fmt='%.3e')
    #np.savetxt('vdp_forced_envelope_BE_T=[{},{}]_A=[{},{}].txt'.format(T[0],T[1],A[0],A[1]), \
    #           pack(sol_be['t'],sol_be['y']), fmt='%.3e')
    #np.savetxt('vdp_forced_envelope_trap_T=[{},{}]_A=[{},{}].txt'.format(T[0],T[1],A[0],A[1]), \
    #           pack(sol_trap['t'],sol_trap['y']), fmt='%.3e')

    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(sol_be['t'],sol_be['y'][0],'ro-')
    plt.plot(sol_trap['t'],sol_trap['y'][0],'go-')
    plt.show()


def HR():
    from polimi import HindmarshRose
    b = 3
    I = 5
    hr = HindmarshRose(I,b)

    y0 = [0,1,0.1]
    t_tran = 100
    sol = solve_ivp(hr, [0,t_tran], y0, method='RK45', rtol=1e-8, atol=1e-10)
    y0 = sol['y'][:,-1]

    t_span = [0,5000]
    T_guess = 11

    be_solver = BEEnvelope(hr, t_span, y0, T_guess=T_guess, \
                           env_rtol=1e-3, env_atol=1e-6, \
                           rtol=1e-8, atol=1e-10)
    trap_solver = TrapEnvelope(hr, t_span, y0, T_guess=T_guess, \
                               env_rtol=1e-3, env_atol=1e-6, \
                               rtol=1e-8, atol=1e-10)

    print('-' * 81)
    sol_be = be_solver.solve()
    print('-' * 81)
    sol_trap = trap_solver.solve()
    print('-' * 81)
    sol = solve_ivp(hr, t_span, y0, method='RK45', rtol=1e-8, atol=1e-10)

    plt.plot(sol['t'],sol['y'][0],'k')
    plt.plot(sol_be['t'],sol_be['y'][0],'ro-')
    plt.plot(sol_trap['t'],sol_trap['y'][0],'go-')
    plt.plot(sol_trap['t'],sol_trap['T'],'ms-')
    plt.show()


def autonomous_vdp():
    from polimi import vdp, vdp_jac
    from polimi.envelope import RK45Envelope, BDFEnvelope, _envelope_system, _one_period

    epsilon = 0.001
    A = [0.,0.]
    T = [1.,1.]
    y0 = [2e-3,0]
    
    rtol = {'fun': 1e-8, 'env': 1e-4}
    atol = {'fun': 1e-10*np.ones(len(y0)), 'env': 1e-6}
    T_exact = 2*np.pi
    T_guess = 0.9 * T_exact

    fun = lambda t,y: vdp(t,y,epsilon,A,T)
    jac = lambda t,y: vdp_jac(t,y,epsilon)

    method = 'RK45'
    tend = 10000
    print('Integrating the full system...')
    if method == 'BDF':
        full = solve_ivp(fun, [0,tend+2*T_exact], y0, method, jac=jac, atol=atol['fun'], rtol=rtol['fun'])
    else:
        full = solve_ivp(fun, [0,tend+2*T_exact], y0, method, atol=atol['fun'], rtol=rtol['fun'])

    env_fun_1 = lambda t,y: _envelope_system(t, y, fun, T_exact, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
    print('Integrating the first envelope function...')
    var_step_1 = solve_ivp(env_fun_1, [0,tend], y0, method='BDF', atol=atol['env'], rtol=rtol['env'])

    env_fun_2 = lambda t,y: _one_period(t, y, fun, T_guess, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
    print('Integrating the second envelope function...')
    var_step_2 = solve_ivp(env_fun_2, [0,tend], y0, method='BDF', atol=atol['env'], rtol=rtol['env'])

    print('Integrating the envelope with Runge-Kutta 4,5...')
    rk = solve_ivp(fun, [0,tend], y0, method=RK45Envelope, T_guess=T_guess,
                   rtol=rtol['env'], atol=atol['env'],
                   fun_method='RK45', fun_rtol=rtol['fun'], fun_atol=atol['fun'])

    print('Integrating the envelope with BDF...')
    bdf = solve_ivp(fun, [0,tend], y0, method=BDFEnvelope, T_guess=T_guess,
                    rtol=rtol['env'], atol=atol['env'],
                    fun_method='RK45', fun_rtol=rtol['fun'], fun_atol=atol['fun'])
    
    plt.figure()
    plt.plot(full['t'],full['y'][0],'k',label='Full integration (%s)'%method)
    plt.plot(var_step_1['t'],var_step_1['y'][0],'go-',lw=2,label='Var. step (fixed T)')
    plt.plot(var_step_2['t'],var_step_2['y'][0],'ms-',lw=2,label='Var. step (estimated T)')
    plt.plot(rk['t'],rk['y'][0],'r^-',lw=2,label='RK45')
    for t0,y0 in zip(rk['t'],rk['y'].transpose()):
        sol = solve_ivp(fun, [t0,t0+T_exact], y0, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
        plt.plot(sol['t'],sol['y'][0],'r')
    plt.plot(bdf['t'],bdf['y'][0],'cv-',lw=2,label='BDF')
    for t0,y0 in zip(bdf['t'],bdf['y'].transpose()):
        sol = solve_ivp(fun, [t0,t0+T_exact], y0, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
        plt.plot(sol['t'],sol['y'][0],'c')
    plt.xlabel('Time (s)')
    plt.ylabel('x')
    plt.legend(loc='best')
    plt.show()


def forced_vdp():
    from polimi import vdp, vdp_jac, vdp_extrema, vdp_auto
    from polimi.envelope import RK45Envelope, BDFEnvelope, _envelope_system, _one_period

    epsilon = 0.001
    T_exact = 10.
    T_guess = 0.9 * T_exact
    rtol = {'fun': 1e-8, 'env': 1e-3}
    atol = {'fun': 1e-10, 'env': 1e-3}

    polar_forcing = True
    method = 'RK45'

    T = [T_exact,1000.]
    A = [1.,5]
    #T = [T_exact]
    #A = [10]

    if not polar_forcing:
        y0 = [2e-3,0]
        fun = lambda t,y: vdp(t,y,epsilon,A,T)
        jac = lambda t,y: vdp_jac(t,y,epsilon)
        event_fun = lambda t,y: vdp_extrema(t,y,epsilon,A,T,0)
        event_fun.direction = -1  # detect maxima (derivative goes from positive to negative)
    else:
        y0 = [2e-3,0]
        for i in range(len(A)):
            y0.append(1.)
            y0.append(0.)
        fun = lambda t,y: vdp_auto(t,y,epsilon,A,T)
        method = 'RK45'

    t0 = 0
    ttran = 2000
    if ttran > 0:
        print('Integrating the full system (transient)...')
        if method == 'BDF':
            tran = solve_ivp(fun, [t0,ttran], y0, method='BDF', jac=jac, atol=atol['fun'],
                             rtol=rtol['fun'], events=event_fun)
        else:
            tran = solve_ivp(fun, [t0,ttran], y0, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
        t0 = tran['t'][-1]
        y0 = tran['y'][:,-1]
        #plt.plot(tran['t'],tran['y'][0],'k')
        #plt.plot(tran['t'],tran['y'][2],'r')
        #plt.plot(tran['t'],tran['y'][4],'g')
        #plt.show()

    print('t0 =',t0)
    print('y0 =',y0)

    print('Integrating the full system...')
    tend = 3000
    if method == 'BDF':
        full = solve_ivp(fun, [t0,tend], y0, method='BDF', jac=jac, atol=atol['fun'],
                         rtol=rtol['fun'], events=event_fun, dense_output=True)
    else:
        full = solve_ivp(fun, [t0,tend], y0, method='RK45', atol=atol['fun'], rtol=rtol['fun'])

    env_fun = lambda t,y: _envelope_system(t, y, fun, T_exact, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
    print('Integrating the envelope at variable step...')
    var_step = solve_ivp(env_fun, [t0,tend], y0, method='BDF', atol=atol['env'], rtol=rtol['env'])

    print('Integrating the envelope with BDF...')
    bdf = solve_ivp(fun, [t0,tend], y0, method=BDFEnvelope, T_guess=T_guess,
                    rtol=rtol['env'], atol=atol['env'], dTtol=5e-3,
                    fun_method='RK45', fun_rtol=rtol['fun'], fun_atol=atol['fun'])

    plt.figure()
    for i in range(2):
        if i == 0:
            ax = plt.subplot(2,1,i+1)
        else:
            plt.subplot(2,1,i+1,sharex=ax)

        plt.plot(full['t'],full['y'][i],'k',label='Full integration (%s)'%method)
        if method == 'BDF':
            plt.plot(full['t_events'][0],full['sol'](full['t_events'][0])[i],'gx')

        plt.plot(var_step['t'],var_step['y'][i],'go-',lw=2,label='Var. step (fixed T)')
        #for t0,y0 in zip(var_step['t'],var_step['y'].transpose()):
        #    sol = solve_ivp(fun, [t0,t0+T_exact], y0, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
        #    plt.plot(sol['t'],sol['y'][i],'g')

        plt.plot(bdf['t'],bdf['y'][i],'cv-',lw=2,label='BDF')
        for t0,y0 in zip(bdf['t'],bdf['y'].transpose()):
            sol = solve_ivp(fun, [t0,t0+T_exact], y0, method='RK45', atol=atol['fun'], rtol=rtol['fun'])
            plt.plot(sol['t'],sol['y'][i],'c')

        if i == 1:
            plt.xlabel('Time (s)')
            plt.ylabel('y')
            plt.legend(loc='best')
        else:
            plt.ylabel('x')
    plt.show()


def main():
    import polimi.utils
    polimi.utils.set_rc_defaults()

    # do not use the first two examples, they are old
    #autonomous_vdp()
    #forced_vdp()

    autonomous()
    #forced_polar()
    #forced()
    #HR()


if __name__ == '__main__':
    main()
