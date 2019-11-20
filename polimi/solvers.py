
__all__ = ['newton', 'forward_euler', 'backward_euler', 'trapezoidal', 'backward_euler_var_step', 'trapezoidal_var_step']

import numpy as np
from numpy.linalg import norm, inv, lstsq

def newton_1D(func, x0, fprime):
    x_cur = x0
    f_cur = func(x_cur)
    cnt = 0
    print('[{:02d}] {:10.3e} {:10.3e}'.format(cnt, x_cur, f_cur))
    while True:
        x_next = x_cur - f_cur / fprime(x_cur)
        f_next = func(x_next)
        cnt += 1
        print('[{:02d}] {:10.3e} {:10.3e}'.format(cnt, x_next, f_next))
        if np.abs(x_cur - x_next) < 1e-6 or np.abs(f_cur - f_next) < 1e-6:
            break
        x_cur = x_next
        f_cur = f_next
    return x_next,f_next


def newton(func, x0, fprime, xtol=1e-6, ftol=1e-6, max_step=100, full_output=False):
    x_cur = x0
    f_cur = func(x_cur)
    info = {'nfev': 1, 'njev': 0, 'nstep': 0}
    while info['nstep'] < max_step:
        x_next = x_cur - inv(fprime(x_cur)) @ f_cur
        f_next = func(x_next)
        info['nfev'] += 1
        info['njev'] += 1
        info['nstep'] += 1
        if np.max(np.abs(x_cur - x_next)) < xtol or np.max(np.abs(f_cur - f_next)) < ftol:
            break
        x_cur = x_next
        f_cur = f_next
    info['fval'] = f_next
    if full_output:
        return x_next,info
    return x_next


def forward_euler(sys, t_span, y0, h):
    n_dim = len(y0)
    t = np.arange(t_span[0],t_span[1],h)
    n_steps = len(t)
    y = np.zeros((n_dim,n_steps))
    y[:,0] = y0
    for i in range(1,n_steps):
        y[:,i] = y[:,i-1] + h * sys(t[i-1],y[:,i-1])
    return {'t': t, 'y': y}


def backward_euler(sys, t_span, y0, h):
    n_dim = len(y0)
    t = np.arange(t_span[0],t_span[1],h)
    n_steps = len(t)
    y = np.zeros((n_dim,n_steps))
    y[:,0] = y0
    I = np.eye(n_dim)
    J = lambda t,y: I - h * sys.jac(t,y)
    for i in range(1,n_steps):
        y[:,i] = newton(lambda Y: Y - y[:,i-1] - h * sys(t[i],Y), y[:,i-1], lambda Y: J(t[i],Y))
    return {'t': t, 'y': y}


def trapezoidal(sys, t_span, y0, h):
    n_dim = len(y0)
    t = np.arange(t_span[0],t_span[1],h)
    n_steps = len(t)
    y = np.zeros((n_dim,n_steps))
    y[:,0] = y0
    I = np.eye(n_dim)
    J = lambda t,y: I - h/2 * sys.jac(t,y)
    for i in range(1,n_steps):
        y[:,i] = newton(lambda Y: Y - y[:,i-1] - h/2 * (sys(t[i-1],y[:,i-1]) + sys(t[i],Y)), y[:,i-1], lambda Y: J(t[i],Y))
    return {'t': t, 'y': y}


def backward_euler_var_step(sys, t_span, y0, h0, hmax=np.inf, rtol=1e-3, atol=1e-6, exact=None, verbose=False):
    if np.isscalar(y0):
        n_dim = 1
    else:
        n_dim = len(y0)
    t = np.array([t_span[0]])
    y = np.zeros((n_dim,1))
    y[:,0] = y0
    dy = [sys(t_span[0],y0)]
    h = h0
    t_cur = t_span[0]
    y_cur = y[:,0]
    dy_cur = sys(t_cur,y_cur)
    if verbose:
        if exact is None:
            print('%13s %13s %13s %13s %13s %13s %13s %13s' % \
                  ('t_cur','h','y_cur','t_next','h_next','y_next','scale','LTE'))
        else:
            print('%13s %13s %13s %13s %13s %13s %13s %13s %13s' % \
                  ('t_cur','h','y_cur','t_next','h_next','y_next','y_exact','scale','LTE'))
    I = np.eye(n_dim)
    while t_cur < t_span[1]:
        t_next = t_cur + h
        if t_next > t_span[1]:
            t_next = t_span[1]
            h = t_next - t_cur
        y_next = newton(lambda Y: Y - y_cur - h * sys(t_next,Y), y_cur, lambda Y: I - h * sys.jac(t_next,Y))
        scale = rtol * np.abs(y_next) + atol
        dy_next = sys(t_next,y_next)
        coeff = np.abs(dy_next * (dy_next-dy_cur)/(y_next-y_cur))
        lte = (h**2)/2 * coeff
        h_new = np.min((hmax,np.min(0.9*np.sqrt(2*scale/coeff))))
        if verbose:
            if exact is None:
                print('%13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e' % \
                      (t_cur,h,y_cur,t_next,h_new,y_next,scale,lte), end='')
            else:
                print('%13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e' % \
                      (t_cur,h,y_cur,t_next,h_new,y_next,exact(t_next),scale,lte), end='')
        if np.all(lte <= scale):
            t = np.append(t,t_next)
            y = np.append(y,np.reshape(y_next,(n_dim,1)),axis=1)
            t_cur = t_next
            y_cur = y_next
            dy_cur = dy_next
            if verbose:
                print(' +')
        else:
            if verbose:
                print(' -')
        h = h_new
    return {'t': t, 'y': y}


def trapezoidal_var_step(sys, t_span, y0, h0, hmax=np.inf, rtol=1e-3, atol=1e-6, exact=None, verbose=False):
    if np.isscalar(y0):
        n_dim = 1
    else:
        n_dim = len(y0)
    t = np.array([t_span[0]])
    y = np.zeros((n_dim,1))
    y[:,0] = y0
    f = [sys(t_span[0],y0)]
    h = h0
    t_cur = t_span[0]
    y_cur = y[:,0]
    f_cur = sys(t_cur,y_cur)
    df_cur = np.zeros(n_dim)

    if verbose:
        if exact is None:
            print('%13s %13s %13s %13s %13s %13s %13s %13s' % \
                  ('t_cur','h','y_cur','t_next','h_next','y_next','scale','LTE'))
        else:
            print('%13s %13s %13s %13s %13s %13s %13s %13s %13s' % \
                  ('t_cur','h','y_cur','t_next','h_next','y_next','y_exact','scale','LTE'))
    I = np.eye(n_dim)
    while t_cur < t_span[1]:
        t_next = t_cur + h
        if t_next > t_span[1]:
            t_next = t_span[1]
            h = t_next - t_cur
        y_next = newton(lambda Y: Y - y_cur - h/2 * (sys(t_cur,y_cur) + sys(t_next,Y)), y_cur, lambda Y: I - h/2 * sys.jac(t_next,Y))
        scale = rtol * np.abs(y_next) + atol
        f_next = sys(t_next,y_next)
        df_next = (f_next-f_cur)/(y_next-y_cur)
        d2f_next = (df_next-df_cur)/(y_next-y_cur)
        coeff = np.abs(f_next * (f_next*d2f_next + 2*(df_next**2)))
        lte = (h**3)/12 * coeff
        h_new = np.min((hmax,np.min(0.9*(12*scale/coeff)**(1/3))))
        if verbose:
            if exact is None:
                print('%13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e' % \
                      (t_cur,h,y_cur,t_next,h_new,y_next,scale,lte), end='')
            else:
                print('%13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e' % \
                      (t_cur,h,y_cur,t_next,h_new,y_next,exact(t_next),scale,lte), end='')
        if np.all(lte <= scale):
            t = np.append(t,t_next)
            y = np.append(y,np.reshape(y_next,(n_dim,1)),axis=1)
            t_cur = t_next
            y_cur = y_next
            f_cur = f_next
            df_cur = df_next
            if verbose:
                print(' +')
        else:
            if verbose:
                print(' -')
        h = h_new
    return {'t': t, 'y': y}


def be_var_step_test():
    from scipy.integrate import solve_ivp
    
    if False:
        l = -1
        fun = lambda t,y: l*y
        Y = lambda t: np.exp(l*t)
        Yp = lambda t: l * np.exp(l*t)
        Ys = lambda t: l**2 * np.exp(l*t)
        t0 = 0
        y0 = Y(t0)
        tend = 1000
    elif False:
        f = 1
        w = 2*np.pi*f
        fun = lambda t,y: w * np.cos(w*t)
        Y = lambda t: np.sin(w*t)
        Yp = lambda t: w * np.cos(w*t)
        Ys = lambda t: -w**2 * np.sin(w*t)
        t0 = 0
        y0 = Y(t0)
        tend = 1/f
    elif False:
        l = -1
        f = 1
        w = 2*np.pi*f
        fun = lambda t,y: np.array([l*y[0], w*np.cos(w*t)])
        Y = lambda t: np.sin(w*t)
        t0 = 0
        y0 = [1,0]
        tend = 1/f
    else:
        from systems import vdp
        epsilon = 1e-3
        A = [5]
        f = [10]
        fun = lambda t,y: vdp(t,y,epsilon,A,f)
        t0 = 0
        y0 = [2e-3,0]
        tend = 10
        sol = solve_ivp(fun, [t0,tend], y0, method='BDF', rtol=1e-6, atol=1e-8, dense_output=True)
        Y = sol['sol']
        
    rtol = 1e-3
    atol = 1e-6
    h0 = 1e-5
    hmax = 1e3
    sol_be = backward_euler_var_step(fun, [t0,tend], y0, h0, hmax, rtol, atol, verbose=False)
    import matplotlib.pyplot as plt
    plt.plot(sol_be['t'],Y(sol_be['t'])[0],'k')
    plt.plot(sol_be['t'],sol_be['y'][0],'r.')
    #plt.plot(sol['t'],sol['y'][0],'ko-')
    #plt.plot(sol_be['t'],sol_be['y'][1],'r')
    plt.show()


def trap_var_step_test():
    from scipy.integrate import solve_ivp

    if False:
        l = -1
        fun = lambda t,y: l*y
        Y = lambda t: np.exp(l*t)
        Yp = lambda t: l * np.exp(l*t)
        Ys = lambda t: l**2 * np.exp(l*t)
        t0 = 0
        y0 = Y(t0)
        tend = 1000
    elif False:
        f = 1
        w = 2*np.pi*f
        fun = lambda t,y: w * np.cos(w*t)
        Y = lambda t: np.sin(w*t)
        Yp = lambda t: w * np.cos(w*t)
        Ys = lambda t: -w**2 * np.sin(w*t)
        t0 = 0
        y0 = Y(t0)
        tend = 1/f
    elif False:
        l = -1
        f = 1
        w = 2*np.pi*f
        fun = lambda t,y: np.array([l*y[0], w*np.cos(w*t)])
        Y = lambda t: np.sin(w*t)
        t0 = 0
        y0 = [1,0]
        tend = 1/f
    else:
        from systems import vdp
        epsilon = 1e-3
        A = [0]
        f = [10]
        fun = lambda t,y: vdp(t,y,epsilon,A,f)
        t0 = 0
        y0 = [2e-3,0]
        tend = 100
        sol = solve_ivp(fun, [t0,tend], y0, method='RK45', rtol=1e-6, atol=1e-8, dense_output=True)
        Y = sol['sol']

    rtol = 1e-3
    atol = 1e-6
    h0 = 1e-5
    hmax = 1000
    sol_trap = trapezoidal_var_step(fun, [t0,tend], y0, h0, hmax, rtol, atol, verbose=False)
    import matplotlib.pyplot as plt
    #plt.plot(sol_trap['t'],Y(sol_trap['t'])[0],'k.')
    plt.plot(sol['t'],sol['y'][0],'k.-')
    plt.plot(sol_trap['t'],sol_trap['y'][0],'r.-')
    plt.show()


def be_test():
    if False:
        l = -1
        fun = lambda t,y: l*y
        Y = lambda t: np.exp(l*t)
        Yp = lambda t: l * np.exp(l*t)
        Ys = lambda t: l**2 * np.exp(l*t)
    else:
        w = 2*np.pi*1
        fun = lambda t,y: w * np.cos(w*t)
        Y = lambda t: np.sin(w*t)
        Yp = lambda t: w * np.cos(w*t)
        Ys = lambda t: -w**2 * np.sin(w*t)
    t0 = 0
    y0 = np.array([Y(t0)])
    print('%13s %13s %13s %13s %13s %13s' % ('h','y','y_BE','error','LTE','scale'))
    rtol = 1e-6
    atol = 1e-8
    for n in range(1,11):
        h = 2**(-n)
        tend = t0 + h
        sol_be = backward_euler(fun, [t0,tend+h], y0, h)
        yend = sol_be['y'][0,-1]
        error = np.abs(Y(tend) - yend)
        #lte = (h**2)/2 * Ys(tend)
        #lte = (h**2)/2 * l**2 * Y(tend)
        #lte = (h**2)/2 * l**2 * yend
        coeff = np.abs(fun(tend,yend) * (fun(tend,yend)-fun(t0,y0))/(yend-y0))
        lte = (h**2)/2 * coeff
        scale = rtol * np.abs(yend) + atol
        print('%13.5e %13.5e %13.5e %13.5e %13.5e %13.5e' % (h,Y(tend),sol_be['y'][0,-1],error,lte,scale))
    return
    sol_be = backward_euler(fun, [t0,t0+10], y0, 1e-4)
    import matplotlib.pyplot as plt
    x = np.linspace(0,10,1000)
    y = Y(x)
    plt.plot(x,y,'k')
    plt.plot(sol_be['t'],sol_be['y'][0],'r')
    plt.show()


def trap_test():
    if True:
        l = -1
        fun = lambda t,y: l*y
        df0 = l
        Y = lambda t: np.exp(l*t)
        Yp = lambda t: l * np.exp(l*t)
        Ys = lambda t: l**2 * np.exp(l*t)
        Yt = lambda t: l**3 * np.exp(l*t)
    else:
        w = 2*np.pi*1
        fun = lambda t,y: w * np.cos(w*t)
        df0 = 0
        Y = lambda t: np.sin(w*t)
        Yp = lambda t: w * np.cos(w*t)
        Ys = lambda t: -w**2 * np.sin(w*t)
        Yt = lambda t: -w**3 * np.cos(w*t)
    t0 = 0
    y0 = np.array([Y(t0)])
    print('%13s %13s %13s %13s %13s %13s' % ('h','y','y_TRAP','error','LTE','scale'))
    rtol = 1e-6
    atol = 1e-8
    for n in range(1,11):
        h = 2**(-n)
        tend = t0 + h
        sol_trap = trapezoidal(fun, [t0,tend+h], y0, h)
        yend = sol_trap['y'][0,-1]
        error = np.abs(Y(tend) - yend)
        #lte = (h**3)/12 * np.abs(Yt(tend))
        f0 = fun(t0,y0)
        fend = fun(tend,yend)
        dfend = (fend-f0)/(yend-y0)
        d2fend = (dfend-df0)/(yend-y0)
        coeff = np.abs(fend * ((1+fend)*d2fend + dfend**2))
        lte = (h**3)/12 * coeff
        scale = rtol * np.abs(yend) + atol
        print('%13.5e %13.5e %13.5e %13.5e %13.5e %13.5e' % (h,Y(tend),sol_trap['y'][0,-1],error,lte,scale))

    sol_trap = trapezoidal(fun, [t0,t0+1], y0, 5e-4)
    import matplotlib.pyplot as plt
    x = np.linspace(0,1,1000)
    y = Y(x)
    plt.plot(x,y,'k')
    plt.plot(sol_trap['t'],sol_trap['y'][0],'r')
    plt.show()


def BDF(fun, t_span, y0, h, order):

    if order <= 0 or order > 6:
        raise Exception('order must be a value between 1 and 6')
    
    A = np.zeros((6,6))
    B = np.zeros((6,6))
    Bstar = np.zeros((6,6))
    A[:,0] = 1.
    B[0,0] = 1.
    B[1,:2] = np.array([3.,-1.])/2.
    B[2,:3] = np.array([23.,-16.,5.])/12.
    B[3,:4] = np.array([55.,-59.,37.,-9.])/24.
    B[4,:5] = np.array([1901.,-2774.,2616.,-1274.,251.])/720.
    B[5,:6] = np.array([4277.,-7923.,9982.,-7298.,2877.,-475.])/1440.
    Bstar[0,0] = 1.
    Bstar[1,:2] = np.array([1.,1.])/2.
    Bstar[2,:3] = np.array([5.,8.,-1.])/12.
    Bstar[3,:4] = np.array([9.,19.,-5.,1.])/24.
    Bstar[4,:5] = np.array([251.,646.,-264.,106.,-19.])/720.
    Bstar[5,:6] = np.array([475.,1427.,-798.,482.,-173.,27.])/1440.

    n_dim = len(y0)
    t = np.arange(t_span[0],t_span[1],h)
    n_steps = len(t)
    y = np.zeros((n_dim,n_steps))
    dydt = np.zeros((n_dim,n_steps))
    y[:,0] = y0
    dydt[:,0] = fun(t[0],y[:,0])
    max_step = h*20
    
    for i in range(1,order):
        k1 = fun(t[i-1],y[:,i-1])
        k2 = fun(t[i-1],y[:,i-1]+h/2*k1)
        k3 = fun(t[i-1],y[:,i-1]+h/2*k2)
        k4 = fun(t[i-1],y[:,i-1]+h*k3)
        y[:,i] = y[:,i-1] + h*(k1+2*k2+2*k3+k4)/6.
        dydt[:,i] = fun(t[i],y[:,i])

    rtol = 1e-3
    atol = 1e-6
    corr_tol = max(10 * np.finfo(float).eps / rtol, min(0.03, rtol ** 0.5))
    max_corr_iter = 5
    
    for i in range(order,n_steps):

        ### predictor ###
        y_p = np.zeros(n_dim)
        for j in range(order):
            y_p += A[order-1,j]*y[:,i-1-j] + h*B[order-1,j]*dydt[:,i-1-j]
        scale = atol + rtol * np.abs(y_p)
        
        #### corrector ###
        converged = False
        dy_norm_old = None
        
        for k in range(max_corr_iter):
            # correct the current value
            y_c = A[order-1,0]*y[:,i-1] + h*Bstar[order-1,0]*fun(t[i],y_p)
            for j in range(1,order):
                y_c += A[order-1,j]*y[:,i-j] + h*Bstar[order-1,j]*dydt[:,i-j]

            # have we reached convergence?
            dy = y_c - y_p
            dy_norm = norm(dy / scale)
            if dy_norm_old is None:
                rate = None
            else:
                rate = dy_norm / dy_norm_old

            if dy_norm == 0 or (rate is not None and rate / (1 - rate) * dy_norm < corr_tol):
                converged = True
                break

            y_p = y_c
            dy_norm_old = dy_norm

        if not converged:
            import ipdb
            ipdb.set_trace()
        elif h < max_step:
            h *= 2

        y[:,i] = y_c
        dydt[:,i] = fun(t[i],y[:,i])

    return {'t': t, 'y': y}


def AB(fun, t_span, y0, h, order):

    if order <= 0 or order > 6:
        raise Exception('order must be a value between 1 and 6')

    A = np.zeros((6,6))
    B = np.zeros((6,6))
    Bstar = np.zeros((6,6))
    A[:,0] = 1.
    B[0,0] = 1.
    B[1,:2] = np.array([3.,-1.])/2.
    B[2,:3] = np.array([23.,-16.,5.])/12.
    B[3,:4] = np.array([55.,-59.,37.,-9.])/24.
    B[4,:5] = np.array([1901.,-2774.,2616.,-1274.,251.])/720.
    B[5,:6] = np.array([4277.,-7923.,9982.,-7298.,2877.,-475.])/1440.

    n_dim = len(y0)
    t = np.arange(t_span[0],t_span[1],h)
    n_steps = len(t)
    y = np.zeros((n_dim,n_steps))
    dydt = np.zeros((n_dim,n_steps))
    y[:,0] = y0
    dydt[:,0] = fun(t[0],y[:,0])

    for i in range(1,n_steps):
        ### predictor ###
        y_p = np.zeros(n_dim)
        for j in range(np.min((i,order))):
            y_p += A[order-1,j]*y[:,i-1-j] + h*B[order-1,j]*dydt[:,i-1-j]
        #### corrector ###
        y_c = y_p
        y[:,i] = y_c
        dydt[:,i] = fun(t[i],y[:,i])
    
    return {'t': t, 'y': y}


def vanderpol():
    from systems import VanderPol
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    import time
    A = [0]
    T = [1]
    epsilon = 1e-3
    y0 = [2e-3,0]
    tend = 1000
    h = 0.05
    vdp = VanderPol(epsilon, A, T)
    sol = solve_ivp(vdp, [0,tend], y0, method='BDF', jac=vdp.jac, rtol=1e-10, atol=1e-8)
    #sol_fw = forward_euler(fun, [0,tend], y0, h/5)
    start = time.time()
    sol_bw = backward_euler(vdp, [0,tend], y0, h/5)
    elapsed = time.time() - start
    print('Elapsed time: {:.3f} sec.'.format(elapsed))
    #k_am = 3
    #sol_bdf = BDF(fun, [0,tend], y0, h, order=k_am)
    #k_ab = 3
    #sol_ab = AB(fun, [0,tend], y0, h, order=k_ab)
    plt.plot(sol['t'],sol['y'][0],'k',label='solve_ivp')
    #plt.plot(sol_fw['t'],sol_fw['y'][0],'b',label='FW')
    plt.plot(sol_bw['t'],sol_bw['y'][0],'r',label='BW')
    #plt.plot(sol_bdf['t'],sol_bdf['y'][0],'b',label='A-M(%d)'%k_am)
    #plt.plot(sol_ab['t'],sol_ab['y'][0],'m',label='A-B(%d)'%k_ab)
    plt.legend(loc='best')
    plt.show()


def bdf_test():
    l = -1
    fun = lambda t,y: l*y
    sol = lambda t: np.exp(l*t)
    y0 = np.array([1])
    order = 3
    print('%13s %13s %13s %13s' % ('h','y','error','error/h^%d'%order))
    for n in range(1,11):
        h = 2**(-n)
        tend = 2+h
        sol_bdf = BDF(fun, [0,tend], y0, h, order)
        error = sol(sol_bdf['t'][-1]) - sol_bdf['y'][0,-1]
        print('%13.5e %13.5e %13.5e %13.5e' % (h,sol_bdf['y'][0,-1],error,error/(h**order)))


def ab_test():
    l = -1
    fun = lambda t,y: l*y
    sol = lambda t: np.exp(l*t)
    y0 = np.array([1])
    print('%13s %13s %13s %13s' % ('h','y','error','error/h^2'))
    for n in range(1,11):
        h = 2**(-n)
        tend = 1+h
        sol_ab = AB(fun, [0,tend], y0, h, order=2)
        error = sol(sol_ab['t'][-1]) - sol_ab['y'][0,-1]
        print('%13.5e %13.5e %13.5e %13.5e' % (h,sol_ab['y'][0,-1],error,error/(h**1)))

    print('')
    for k in range(1,8):
        h = 2**(-k)
        n = 2**k - 1
        y = np.exp(-h)
        fold = -h
        for i in range(n):
            f = -h*y
            y += 1.5*f - 0.5*fold
            fold = f
        error = np.exp(-1.) - y
        errbyh = error/h**2
        print('%13.5e %13.5e %13.5e %13.5e' % (h,y,error,errbyh))


def main():
    import matplotlib.pyplot as plt
    equil = True
    if equil:
        l = -1
        fun = lambda t,y: l*y
        sol = lambda t: np.exp(l*t)
        y0 = np.array([1])
        tend = 10
        h = 0.05
    else:
        f = 1
        fun = lambda t,y: 2 * np.pi * f * np.cos(2 * np.pi * f * t)
        sol = lambda t: np.sin(2 * np.pi * f * t)
        y0 = np.array([0])
        tend = 5./f
        h = 0.001
    sol_fw = forward_euler(fun, [0,tend], y0, h)
    sol_bw = backward_euler(fun, [0,tend], y0, h)
    sol_bdf = bdf(fun, [0,tend], y0, 2*h, order=3)
    t = np.linspace(0,tend,1000)
    plt.plot(t,sol(t),'k',label='Solution')
    plt.plot(sol_fw['t'],sol_fw['y'][0],'b',label='FW')
    plt.plot(sol_bw['t'],sol_bw['y'][0],'r',label='BW')
    plt.plot(sol_bdf['t'],sol_bdf['y'][0],'m',label='BDF')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    #main()
    vanderpol()
    #ab_test()
    #bdf_test()
    #be_test()
    #be_var_step_test()
    #trap_test()
    #trap_var_step_test()
    #newton_test()
