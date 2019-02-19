def envelope_old(fun,t_span,y0,T,H=None,order=3,rtol=1e-8,atol=None):
    N = len(y0)
    if atol is None:
        atol = np.array([1e-10 for i in range(N)])
    if H is None:
        H = 10*T
    sol = {'t': np.array([]),
           'y': np.array([[] for i in range(N)]),
           'T': np.array([t_span[0]]),
           'Z': y0.copy().reshape(N,1)}
    t = t_span[0]
    g = np.array([[] for i in range(N)])
    while t < t_span[1]:
        y = solve_ivp(fun,[0,T],sol['Z'][:,-1],rtol=rtol,atol=atol)
        sol['t'] = np.concatenate((sol['t'],t+y['t']))
        sol['y'] = np.concatenate((sol['y'],y['y']),axis=1)
        g = np.concatenate((g,(1./T * (y['y'][:,-1] - y['y'][:,0])).reshape(N,1)),axis=1)
        # predictor
        z_p = np.zeros(N)
        for i in range(order):
            try:
                z_p += A[order-1,i]*sol['Z'][:,-1-i] + H*B[order-1,i]*g[:,-1-i]
            except:
                break
        y = solve_ivp(fun,[0,T],z_p,rtol=rtol,atol=atol)
        g_tmp = 1./T * (y['y'][:,-1] - y['y'][:,0])
        # corrector
        z_c = A[order-1,0]*sol['Z'][:,-1] + H*Bstar[order-1,0]*g_tmp
        #z_c = A[order-1,0]*sol['Z'][:,-1] + H*beta0star(order,T/H)*g_tmp
        for i in range(1,order):
            try:
                z_c += A[order-1,i]*sol['Z'][:,-1-i+1] + H*Bstar[order-1,i]*g[:,-1-i+1]
            except:
                break
        t += H
        sol['T'] = np.append(sol['T'],t)
        sol['Z'] = np.concatenate((sol['Z'],z_c.reshape(N,1)),axis=1)
    return sol
