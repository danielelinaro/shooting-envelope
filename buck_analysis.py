
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from polimi.utils import set_rc_defaults

def compute_RMSE(tran_file, envel_file, n_vars=2, do_plot=False, fig_name=None, var_names=None):
    tran = pickle.load(open(tran_file, 'rb'))
    envel = pickle.load(open(envel_file, 'rb'))

    sol = {'tran': tran['sol']['integrations'][-1], 'envel': envel['sol']['integrations'][-1]}
    time = {k: v['t'] for k,v in sol.items()}
    idx = []
    dst = []
    for t in time['envel']:
        idx.append(np.argmin(np.abs(time['tran'] - t)))
        dst.append(np.min(np.abs(time['tran'] - t)))
    print('Maximum distance between time instants: {:.4e}.'.format(np.max(dst)))
    print('Number of points for the calculation of the RMSE: {}.'.format(len(idx)))
    idx = np.array(idx)
    N = len(idx)
    x = np.arange(1,N+1)
    error = sol['tran']['y'][:n_vars,idx] - sol['envel']['y'][:n_vars]
    normalized_error = np.abs(error / sol['tran']['y'][:n_vars,idx])
    squared_error = error ** 2
    rmse = np.sqrt(np.array([np.mean(squared_error[:,:i+1], axis=1) for i in range(N)]))
    if do_plot:
        fig,ax = plt.subplots(n_vars, 1, figsize=(6, 3*n_vars))
        for i in range(n_vars):
            ax[i].plot(x, np.abs(error[i,:]), 'ko-', lw=1, markersize=3, markerfacecolor='w', label='Error')
            ax[i].plot(x, normalized_error[i,:], 'ro-', lw=1, markersize=3, markerfacecolor='w', label='Normalized error')
            ax[i].plot(x, rmse[:,i], 'bo-', lw=1, markersize=3, markerfacecolor='w', label='Cumulative RMSE')
            if x[-1] < 10:
                dx = 1
            elif x[-1] < 20:
                dx = 2
            elif x[-1] < 50:
                dx = 5
            else:
                dx = 10
            ax[i].set_xticks(np.arange(0,x[-1]+dx/2,dx))
            ax[i].set_yscale('log')
            if var_names is not None:
                ax[i].set_ylabel(var_names[i])
        ax[0].legend(loc='best')
        ax[-1].set_xlabel('Number of envelope samples')
        if fig_name is not None:
            plt.savefig(fig_name)
            plt.close(fig)
        else:
            plt.show()
    return rmse[-1,:]
    #return np.sqrt(np.mean((sol['tran']['y'][:n_vars,idx] - sol['envel']['y'][:n_vars]) ** 2, axis=1))



def main():
    folder = 'BUCK_SIMULATIONS/'
    files = {'tran': glob.glob(folder + 'buck_shooting_tran*T=*.pkl'), \
             'envel': glob.glob(folder + 'buck_shooting_envelope*T=*.pkl')}
    for k in files:
        files[k].sort()
    ratio = {k: [] for k in files}
    elapsed_time = {k: [] for k in files}
    y0 = {k: [] for k in files}
    error = {k: [] for k in files}
    multipliers = {k: [] for k in files}
    n_steps = {k: [] for k in files}

    for k,ff in files.items():
        for f in ff:
            data = pickle.load(open(f,'rb'))
            ratio[k].append(data['T_large']/data['T_small'])
            elapsed_time[k].append(data['elapsed_time'])
            y0[k].append(data['sol']['integrations'][-1]['y'][:3,0])
            error[k].append(np.abs(data['sol']['integrations'][-1]['y'][:3,0] - data['sol']['integrations'][-1]['y'][:3,-1]))
            multipliers[k].append(np.linalg.eig(data['sol']['phi'])[0])
            n_steps[k].append(len(data['sol']['integrations']))
        ratio[k] = np.array(ratio[k])
        elapsed_time[k] = np.array(elapsed_time[k])
        y0[k] = np.array(y0[k])
        error[k] = np.array(error[k])
        multipliers[k] = np.array(multipliers[k])
        n_steps[k] = np.array(n_steps[k])
        idx = np.argsort(ratio[k])
        ratio[k] = ratio[k][idx]
        elapsed_time[k] = elapsed_time[k][idx]
        y0[k] = y0[k][idx,:]
        error[k] = error[k][idx,:]
        multipliers[k] = multipliers[k][idx,:]
        n_steps[k] = n_steps[k][idx]

    n_files = len(files['tran'])
    RMSE = np.zeros((n_files, 2))
    for i in range(n_files):
        RMSE[i,:] = compute_RMSE(files['tran'][i], files['envel'][i], n_vars=2, do_plot=True, \
                                 fig_name=os.path.splitext(files['envel'][i][:-3])[0] + '_RMSE.pdf',
                                 var_names=[r'$v_C$',r'$i_L$'])

    for r,m_tr,m_en,n_tr,n_en,rmse in zip(ratio['tran'], multipliers['tran'], multipliers['envel'], n_steps['tran'], n_steps['envel'], RMSE):
        expon = [int(np.ceil(np.log10(1 / err))) for err in rmse]
        print('{:4.0f} & {:.4f} & {:.4f} & ${:.2f} \\times 10^{{-{}}}$ & ${:.2f} \\times 10^{{-{}}}$ & {}, {} \\\\'. \
              format(r, np.real(m_tr[0]), np.real(m_en[0]), rmse[0]*10**expon[0], expon[0], rmse[1]*10**expon[1], expon[1], n_tr, n_en))

    elapsed_time_ratio = elapsed_time['tran'] / elapsed_time['envel']
    dst = [np.linalg.norm(a[:2] - b[:2]) for a,b in zip(y0['tran'],y0['envel'])]

    N = len(elapsed_time_ratio)
    col = np.tile(np.linspace(0,1,N),[3,1]).T

    set_rc_defaults()
    plt.rc('font', family='Times New Roman', size=8)

    fig = plt.figure(figsize=(8.5/2.54,8.5/2.54))
    ax = [
        plt.axes([0.15,0.575,0.825,0.4]),
        plt.axes([0.2,0.125,0.3,0.3]),
        plt.axes([0.675,0.125,0.3,0.3])
    ]

    lim = np.array([50,2050])
    p = np.polyfit(ratio['tran'], elapsed_time_ratio, 1)
    ax[0].plot(lim, np.polyval(p, lim), 'k', lw=1)
    for i in range(N):
        ax[0].plot(ratio['tran'][i], elapsed_time_ratio[i], 'ko', lw=1, markerfacecolor=col[i])
    ax[0].set_xlim(lim + 50 * np.array([-1,1]))
    ax[0].set_xticks(np.arange(0,2100,250))
    ax[0].set_yticks(np.arange(0,11,2))
    ax[0].set_xlabel('Ratio between periods')
    ax[0].set_ylabel('Envelope speed-up')

    lim = [10.01,10.04]
    d = np.diff(lim)
    ax[1].plot(lim, lim, 'k--', lw=1)
    ax[1].set_xlim(lim + d * np.array([-0.05,0.05]))
    ax[1].set_ylim(lim + d * np.array([-0.05,0.05]))
    ax[1].set_xticks(lim)
    ax[1].set_yticks(lim)
    ax[1].set_xlabel(r'$V_C$ tran ($V$)')
    ax[1].set_ylabel(r'$V_C$ envel ($V$)')

    lim = [1.6, 2]
    d = np.diff(lim)
    ax[2].plot(lim, lim, 'k--', lw=1)
    ax[2].set_xlim(lim + d * np.array([-0.05,0.05]))
    ax[2].set_ylim(lim + d * np.array([-0.05,0.05]))
    ax[2].set_xticks(lim)
    ax[2].set_yticks(lim)
    ax[2].set_xlabel(r'$I_L$ tran ($A$)')
    ax[2].set_ylabel(r'$I_L$ envel ($A$)')

    for i in range(N):
        ax[1].plot(y0['tran'][i,0], y0['envel'][i,0], 'ko', lw=1, markerfacecolor=col[i], markersize=4)
        ax[2].plot(y0['tran'][i,1], y0['envel'][i,1], 'ko', lw=1, markerfacecolor=col[i], markersize=4)

    #plt.savefig('buck_analysis.pdf')

    plt.show()


if __name__ == '__main__':
    main()
