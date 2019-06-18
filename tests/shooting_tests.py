
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

    sol = shoot.run(y0_guess, do_plot=True)
    floquet_multi,_ = np.linalg.eig(sol['phi'])
    print('T = %g.' % sol['T'])
    print('eig(Phi) = (%f,%f).' % tuple(floquet_multi))
    print('Number of iterations: %d.' % sol['n_iter'])
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

    sol = shoot.run(y0_guess, do_plot=True)
    print('Number of iterations: %d.' % sol['n_iter'])
    plt.show()


def forced_two_frequencies(A=[10,1], T=[4,400], y0_guess=[-2,0], do_plot=True, ax=None):
    from polimi.systems import vdp, vdp_jac
    estimate_T = False
    epsilon = 1e-3
    N = 2

    shoot = Shooting(lambda t,y: vdp(t,y,epsilon,A,T),
                     N, np.max(T), estimate_T,
                     lambda t,y: vdp_jac(t,y,epsilon),
                     tol=1e-3, rtol=1e-8, atol=1e-10, ax=ax)

    sol = shoot.run(y0_guess, do_plot=do_plot)
    print('Number of iterations: %d.' % sol['n_iter'])
    if do_plot:
        plt.show()
    return sol


def forced_two_frequencies_envelope(A=[10,1], T=[4,400], y0_guess=[-2,0], do_plot=True, ax=None):
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
                             fun_rtol=1e-8, fun_atol=1e-10, ax=ax)

    sol = shoot.run(y0_guess, do_plot=do_plot)
    print('Number of iterations: %d.' % sol['n_iter'])
    if do_plot:
        plt.show()
    return sol


def forced_two_frequencies_comparison():
    import matplotlib
    matplotlib.rc('font', size=8)

    A = [10,1]
    T = [4,400]
    y0_guess = [-2,0]

    if max(T) == 400:
        N = 3
    elif max(T) == 4000:
        N = 2

    fig = plt.figure(figsize=[10,2*N])
    ax = [fig.add_subplot(N,2,i+1) for i in range(N*2)]

    sol = forced_two_frequencies(A, T, y0_guess, False, ax)
    sol_env = forced_two_frequencies_envelope(A, T, y0_guess, False, ax)

    for i in range(N):
        ax[i*2].set_ylabel('x')
        ax[i*2].set_xlim([0,1])
        ax[i*2].set_ylim([-10,12])
        ax[i*2+1].set_ylabel(r'$\Phi_{11}$')
        ax[i*2+1].set_xlim([0,1])
        ax[i*2+1].set_ylim([-1,1])
        ax[i*2+1].text(0.7,0.7,'Iteration #%d'%(i+1),fontsize=8)

    ax[-2].set_xlabel('Normalized time')
    ax[-1].set_xlabel('Normalized time')

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
