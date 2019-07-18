
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from polimi.shooting import Shooting, EnvelopeShooting

def normalized():
    from polimi.systems import vdp, vdp_jac
    epsilon = 1e-3
    A = [10,2]
    T = [10,200]
    fun = lambda t,y: vdp(t,y,epsilon,A,T)
    fun_norm = lambda t,y: np.max(T) * vdp(t*np.max(T), y, epsilon, A, T)
    t_span = [0,3*np.max(T)]
    y0 = [2e-3,0]
    tran = solve_ivp(fun, t_span, y0, rtol=1e-6, atol=1e-8)
    t0 = 0
    y0 = tran['y'][:,-1]
    t_span = [t0, t0+np.max(T)]
    sol = solve_ivp(fun, t_span, y0, rtol=1e-6, atol=1e-8)
    t_span = [0,1]
    sol_norm = solve_ivp(fun_norm, t_span, y0, rtol=1e-6, atol=1e-8)
    plt.plot(sol['t']/np.max(T),sol['y'][0],'k',label='Original')
    plt.plot(sol_norm['t'],sol_norm['y'][0],'r',label='Normalized')
    plt.legend(loc='best')
    plt.show()


def plot_shooting_solution(integrations, ax=None, **kwargs):
    if ax is None:
        fig,(ax1,ax2) = plt.subplots(1,2)
    else:
        ax1,ax2 = ax
        fig = ax1.get_figure()
    ax1.plot(integrations[0]['t'], integrations[0]['y'][0], \
             color=[.8,.3,.3], label='Iter #1', **kwargs)
    ax2.plot(integrations[0]['y'][1], integrations[0]['y'][0], \
             color=[.8,.3,.3], **kwargs)
    for i,integr in enumerate(integrations):
        if i != 0 and i != len(integrations)-1:
            ax1.plot(integr['t'], integr['y'][0], color=[.6,.6,.6], lw=0.8, **kwargs)
            ax2.plot(integr['y'][1], integr['y'][0], color=[.6,.6,.6], lw=0.8, **kwargs)
    ax1.plot(integrations[-1]['t'], integrations[-1]['y'][0], \
             color=[0,0,0], lw=1.5, label='Iter #%d' % len(integrations), **kwargs)
    ax2.plot(integrations[-1]['y'][1], integrations[-1]['y'][0], color=[0,0,0], \
             lw=1.5, **kwargs)
    ax1.legend(loc='best')
    ax1.set_xlabel('Normalized time')
    ax1.set_ylabel(r'$y_1$')
    ax2.set_xlabel(r'$y_2$')
    return fig,(ax1,ax2)


def autonomous(with_jac=True):
    from polimi.systems import vdp, vdp_jac
    estimate_T = True
    epsilon = 1e-3
    A = [0]
    T = [2*np.pi]
    T_guess = 0.6*T[0]
    y0_guess = [-2,3]
    N = 2

    if with_jac:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         N, T_guess, estimate_T,
                         lambda t,y: vdp_jac(t,y,epsilon),
                         rtol=1e-6, atol=1e-8)
    else:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         N, T_guess, estimate_T,
                         rtol=1e-6, atol=1e-8)

    sol = shoot.run(y0_guess)
    floquet_multi,_ = np.linalg.eig(sol['phi'])
    print('T = %g.' % sol['T'])
    print('eig(Phi) = (%f,%f).' % tuple(floquet_multi))
    print('Number of iterations: %d.' % sol['n_iter'])
    plot_shooting_solution(sol['integrations'])
    plt.show()


def forced(with_jac=True):
    from polimi.systems import vdp, vdp_jac
    estimate_T = False
    epsilon = 1e-3
    A = [1.2]
    T = [10.]
    y0_guess = [-1,2]
    N = 2

    if with_jac:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         N, T[0], estimate_T,
                         lambda t,y: vdp_jac(t,y,epsilon),
                         rtol=1e-6, atol=1e-8)
    else:
        shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                         N, T[0], estimate_T,
                         rtol=1e-6, atol=1e-8)

    sol = shoot.run(y0_guess)
    print('Number of iterations: %d.' % sol['n_iter'])
    plot_shooting_solution(sol['integrations'])
    plt.show()


def forced_two_frequencies(A=[10,1], T=[4,400], y0_guess=[-2,0], do_plot=True):
    from polimi.systems import vdp, vdp_jac
    estimate_T = False
    epsilon = 1e-3
    N = 2

    shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                     N, np.max(T), estimate_T,
                     lambda t,y: vdp_jac(t,y,epsilon),
                     tol=1e-3, rtol=1e-8, atol=1e-10)

    sol = shoot.run(y0_guess)
    print('Number of iterations: %d.' % sol['n_iter'])
    if do_plot:
        plot_shooting_solution(sol['integrations'])
        plt.show()
    return sol


def forced_two_frequencies_envelope(A=[10,1], T=[4,400], y0_guess=[-2,0], do_plot=True):
    from polimi.systems import vdp, vdp_jac
    from polimi.envelope import BEEnvelope, TrapEnvelope
    estimate_T = False
    epsilon = 1e-3
    T_small = np.min(T)
    T_large = np.max(T)
    N = 2

    shoot = EnvelopeShooting(lambda t,y: vdp(t,y,epsilon,A,T),
                             N, T_large, estimate_T, T_small,
                             lambda t,y: vdp_jac(t,y,epsilon),
                             shooting_tol=1e-3, env_solver=BEEnvelope,
                             env_rtol=1e-2, env_atol=1e-3,
                             fun_rtol=1e-8, fun_atol=1e-10)

    sol = shoot.run(y0_guess)
    print('Number of iterations: %d.' % sol['n_iter'])
    if do_plot:
        plot_shooting_solution(sol['integrations'])
        plt.show()
    return sol


def forced_two_frequencies_comparison():
    import matplotlib
    matplotlib.rc('font', size=8)

    A = [10,1]
    T = [4,400]
    y0_guess = [-2,0]

    sol = forced_two_frequencies(A, T, y0_guess, False)
    sol_env = forced_two_frequencies_envelope(A, T, y0_guess, False)

    N = len(sol['integrations'])
    fig,ax = plt.subplots(N,2,figsize=(10,6))
    for i in range(N):
        ylim = [1.1*np.min((np.min(sol['integrations'][i]['y'][0]),np.min(sol_env['integrations'][i]['y'][0]))), \
                1.1*np.max((np.max(sol['integrations'][i]['y'][0]),np.max(sol_env['integrations'][i]['y'][0])))]
        ax[i,0].plot(sol['integrations'][i]['t'],sol['integrations'][i]['y'][0],'k',lw=1)
        ax[i,1].plot(sol['integrations'][i]['t'],sol['integrations'][i]['y'][2],'k',lw=1)
        ax[i,0].plot(sol_env['integrations'][i]['t'],sol_env['integrations'][i]['y'][0],'r.')
        ax[i,1].plot(sol_env['integrations'][i]['t'],sol_env['integrations'][i]['y'][2],'r.')
        ax[i,0].set_ylabel('x')
        ax[i,1].set_ylabel(r'$\Phi_{1,1}$')
        ax[i,1].text(0.7,0.7,'Iteration #%d' % (i+1))
        ax[i,0].set_xlim([0,1])
        ax[i,0].set_ylim([-10,12])
        ax[i,1].set_xlim([0,1])
        ax[i,1].set_ylim([-1.1,1.1])
    ax[N-1,0].set_xlabel('Normalized time')
    ax[N-1,1].set_xlabel('Normalized time')
    plt.savefig('shooting_envelope_%d_%d.pdf' % (T[0],T[1]))
    plt.show()


def main():
    #normalized()
    #autonomous()
    #forced()
    #forced_two_frequencies()
    #forced_two_frequencies_envelope()
    forced_two_frequencies_comparison()


if __name__ == '__main__':
    main()
