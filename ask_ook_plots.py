
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from polimi.systems import ASK_OOK
from polimi.utils import set_rc_defaults
from buck_analysis import compute_RMSE

FOLDER = 'ASK_OOK_SIMULATIONS'
grey = [.5,.5,.5]

set_rc_defaults()
plt.rc('font', family='Times New Roman', size=8)

def tran_env():
    ckt = ASK_OOK()
    tran = pickle.load(open(FOLDER + '/ASK_OOK_tran.pkl', 'rb'))
    envel = pickle.load(open(FOLDER + '/ASK_OOK_envelope.pkl', 'rb'))
    time = np.linspace(0, ckt.T1, 1000)
    carrier = np.array([ckt._Vg1(t) for t in time])

    fig = plt.figure(figsize=[8.5/2.54,3.5])
    ax = [
        plt.axes([0.15,0.65,0.825,0.325]),
        plt.axes([0.15,0.27,0.825,0.325]),
        plt.axes([0.15,0.12,0.825,0.1]),
        plt.axes([0.7,0.815,0.25,0.15]),
        plt.axes([0.7,0.435,0.25,0.15])
    ]

    ax[0].plot(tran['sol']['t']*1e6, tran['sol']['y'][3], color=grey, lw=0.5)
    ax[0].set_ylabel(r'$v_{out}$ tran (V)')

    ax[1].plot(envel['full_sol']['t']*1e6, envel['full_sol']['y'][3], 'r', lw=0.5)
    ax[1].plot(envel['sol']['t']*1e6, envel['sol']['y'][3], 'go', lw=0.5, \
               markerfacecolor='w', markersize=3)
    ax[1].set_ylabel(r'$v_{out}$ envel (V)')

    for i in range(2):
        ax[i].set_xticks(np.linspace(0,1,6))
        ax[i].set_yticks(np.linspace(-1,3,5))
        ax[i].set_xticklabels([])
        ax[i].set_xlim([-0.01,1.01])
        ax[i].set_ylim([-1.5,3.5])
    
    ax[2].set_xticks(np.linspace(0,1,6))
    ax[2].set_yticks(np.linspace(0,2,3))
    ax[2].plot(time*1e6, carrier, 'k', lw=1)
    ax[2].set_xlabel(r'Time ($\mu$s)')
    ax[2].set_xlim([-0.01,1.01])
    ax[2].set_ylim([-0.1,2.1])

    interval = [0.159e-6, 0.166e-6]
    
    idx, = np.where((tran['sol']['t'] > interval[0]) & (tran['sol']['t'] < interval[1]))
    ax[3].plot(tran['sol']['t'][idx]*1e6, tran['sol']['y'][3,idx], color=grey, lw=0.5)
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].autoscale(enable=True, tight=True)
    xl = ax[3].get_xlim()
    yl = ax[3].get_ylim()
    ax[0].plot(np.array([xl[0], xl[1], xl[1], xl[0], xl[0]]),
               np.array([yl[0], yl[0], yl[1], yl[1], yl[0]])*1.2,
               'k--', lw=0.5)
    

    idx, = np.where((envel['full_sol']['t'] > interval[0]) & (envel['full_sol']['t'] < interval[1]))
    ax[4].plot(envel['full_sol']['t'][idx]*1e6, envel['full_sol']['y'][3,idx], 'r', lw=0.5)
    idx, = np.where((envel['sol']['t'] > interval[0]) & (envel['sol']['t'] < interval[1]))
    ax[4].plot(envel['sol']['t'][idx]*1e6, envel['sol']['y'][3,idx], 'go', lw=0.5, \
               markerfacecolor='w', markersize=3)
    ax[4].plot([0.163, 0.164], [0,0], 'k', lw=1)
    ax[4].set_xticks([])
    ax[4].set_yticks([])
    ax[4].autoscale(enable=True, tight=True)
    xl = ax[4].get_xlim()
    yl = ax[4].get_ylim()
    ax[1].plot(np.array([xl[0], xl[1], xl[1], xl[0], xl[0]]),
               np.array([yl[0], yl[0], yl[1], yl[1], yl[0]])*1.2,
               'k--', lw=0.5)

    plt.savefig('ASK_OOK_tran_envel.pdf')
    #plt.show()


def shooting():
    tran = pickle.load(open(FOLDER + '/ASK_OOK_shooting_tran.pkl', 'rb'))
    envel = pickle.load(open(FOLDER + '/ASK_OOK_shooting_envelope.pkl', 'rb'))

    var_names = ['i_{L_{tl}}','i_{L_1}','i_{L_2}','v_o','v_{C_1}','v_{C_2}+v_o','v_{DD}-v_{C_{M_1}}','v_{C_{M_2}}']
    
    RMSE = compute_RMSE(FOLDER + '/ASK_OOK_shooting_tran.pkl', \
                        FOLDER + '/ASK_OOK_shooting_envelope.pkl', \
                        n_vars=8, do_plot=True, fig_name=FOLDER + '/ASK_OOK_shooting_envelope_RMSE.pdf', \
                        var_names=[r'$' + name + '$' for name in var_names])

    for var_name in var_names:
        sys.stdout.write('$' + var_name + '$ & '.format(var_name))
    sys.stdout.write('\\\\\n')
    for rmse in RMSE:
        expon = int(np.ceil(np.log10(1 / rmse)))
        sys.stdout.write('${:.2f} \\times 10^{{-{}}}$ & '.format(rmse*10**expon, expon))
    sys.stdout.write('\\\\\n')

    tran_mult,_ = np.linalg.eig(np.reshape(tran['sol']['integrations'][-1]['y'][8:,-1], (8,8)))
    envel_mult,_ = np.linalg.eig(envel['sol']['phi'])
    print('Floquet multipliers:')
    print('{:^11s} {:^11s}'.format('TRAN', 'ENVEL'))
    for i in range(8):
        print('{:11.3e} {:11.3e}'.format(np.real(tran_mult[i]), np.real(envel_mult[i])))

    offset = {'x': 0.125, 'y': 0.125}
    space = {'x': 0.05, 'y': 0.05}
    border = {'x': 0.01, 'y': 0.06}
    dx = (1 - offset['x'] - space['x'] - border['x']) / 2
    dy = (1 - offset['y'] - space['y'] - border['y']) / 2
    fig = plt.figure(figsize=[8.5/2.54,3])
    ax = [
        plt.axes([offset['x'], offset['y']+space['y']+dy, dx, dy]),
        plt.axes([offset['x']+space['x']+dx, offset['y']+space['y']+dy, dx, dy]),
        plt.axes([offset['x'], offset['y'], dx, dy]),
        plt.axes([offset['x']+space['x']+dx, offset['y'], dx, dy])
    ]

    ylim = [-1.5,3.5]
    for i in range(2):
        ax[i].plot(tran['sol']['integrations'][i]['t'], \
                   tran['sol']['integrations'][i]['y'][3], \
                   color=grey, lw=0.5)
        ax[i+2].plot(envel['sol']['integrations'][i]['full_sol']['t'], \
                     envel['sol']['integrations'][i]['full_sol']['y'][3], \
                     'r', lw=0.5)
        ax[i+2].plot(envel['sol']['integrations'][i]['t'], \
                     envel['sol']['integrations'][i]['y'][3], \
                     'go', lw=0.5, markerfacecolor='w', markersize=2)
        ax[i].plot([0.1,0.1], ylim, 'k--', lw=0.75)
        ax[i+2].plot([0.1,0.1], ylim, 'k--', lw=0.75)
        ax[i].set_xticks([0,0.5,1])
        ax[i].set_xticklabels([])
        ax[i+2].set_xticks([0,0.5,1])
        ax[i*2].set_yticks(np.arange(-1,3.1,1))
        ax[i*2+1].set_yticks(np.arange(-1,3.1,1))
        ax[i*2+1].set_yticklabels([])
        ax[i+2].set_xlabel('Time (us)')
        ax[i].set_title('Iteration {}'.format(i+1))
    ax[0].set_ylabel(r'$v_{out}$ tran (V)')
    ax[2].set_ylabel(r'$v_{out}$ envel (V)')

    for i in range(4):
        ax[i].set_xlim([-0.05,1.05])
        ax[i].set_ylim(ylim)

    plt.savefig('ASK_OOK_shooting.pdf')
    #plt.show()


def HB():
    data = np.loadtxt(FOLDER + '/Hb@Time.ascii')
    time = data[:,0] * 1e6
    vo = data[:,1]

    data = np.loadtxt(FOLDER + '/Hb.ascii')
    # frequency values
    freq = data[:,0]
    # complex coefficients of the Fourier series
    coeff = np.array([np.complex(re,im) for re,im in data[:,1:]])
    # modulus of the coefficients
    mod = np.array([np.linalg.norm(c) for c in coeff])
    
    fig = plt.figure(figsize=(8.5/2.54, 3))

    ax = [
        plt.axes([0.15,0.625,0.825,0.35]),
        plt.axes([0.15,0.15,0.825,0.35]),
        plt.axes([0.7,0.78,0.25,0.18]),
        plt.axes([0.2,0.175,0.25,0.19])
    ]

    ax[0].plot(time, vo, 'k', lw=0.5)
    ax[0].set_xlabel(r'$t$ [$\mu$s]')
    ax[0].set_ylabel(r'$v_o$ [V]')
    ax[0].set_xticks(np.linspace(0,1,6))
    ax[0].set_yticks(np.linspace(-1,3,5))
    ax[0].set_xticklabels([])
    ax[0].set_xlim([-0.01,1.01])
    ax[0].set_ylim([-1.5,3.75])
    
    interval = [0.159, 0.166]
    idx, = np.where((time > interval[0]) & (time < interval[1]))
    ax[2].plot(time[idx], vo[idx], color='k', lw=0.5)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].autoscale(enable=True, tight=True)
    xl = ax[2].get_xlim()
    yl = ax[2].get_ylim()
    ax[0].plot(np.array([xl[0], xl[1], xl[1], xl[0], xl[0]]),
               np.array([yl[0], yl[0], yl[1], yl[1], yl[0]]) * \
               np.array([1.2, 1.2, 0.8, 0.8, 1.2]),
               'r--', lw=0.5)

    ax[1].plot(freq, 20 * np.log10(mod), 'k', lw=0.5)
    ax[1].set_xscale('log')
    ax[1].set_xlabel(r'Frequency [Hz]')
    ax[1].set_ylabel(r'dB')
    
    interval = [2.5e8, 9e8]
    idx, = np.where((freq > interval[0]) & (freq < interval[1]))
    ax[3].plot(np.log10(freq[idx]), 20 * np.log10(mod[idx]), color='k', lw=0.5)
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].autoscale(enable=True, tight=True)
    xl = ax[3].get_xlim()
    yl = ax[3].get_ylim()
    ax[1].plot(np.array([interval[0], interval[1], interval[1], interval[0], interval[0]]),
               np.array([yl[0], yl[0], yl[1], yl[1], yl[0]]) * \
               np.array([1.05, 1.05, 0.8, 0.8, 1.05]),
               'r--', lw=0.5)

    plt.savefig('ASK_OOK_harmonic.pdf')
    #plt.show()


if __name__ == '__main__':
    #tran_env()
    shooting()
    #HB()

    
