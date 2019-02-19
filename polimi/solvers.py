
import numpy as np

def forward_euler(fun, t_span, y0, h):
    n_dim = len(y0)
    t = np.arange(t_span[0],t_span[1],h)
    n_steps = len(t)
    y = np.zeros((n_dim,n_steps))
    y[:,0] = y0
    for i in range(1,n_steps):
        y[:,i] = y[:,i-1] + h*fun(t[i-1],y[:,i-1])
    return {'t': t, 'y': y}

def backward_euler(fun, t_span, y0, h):
    from scipy.optimize import newton
    n_dim = len(y0)
    t = np.arange(t_span[0],t_span[1],h)
    n_steps = len(t)
    y = np.zeros((n_dim,n_steps))
    y[:,0] = y0
    for i in range(1,n_steps):
        try:
            y_next = newton(lambda Y: Y-y[:,i-1]-h*fun(t[i],Y), y[:,i-1], maxiter=5000)
        except:
            import ipdb
            ipdb.set_trace()
        y[:,i] = y[:,i-1] + h*fun(t[i],y_next)
    return {'t': t, 'y': y}

def bdf(fun, t_span, y0, h, order):

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
    
    for i in range(1,order):
        k1 = fun(t[i-1],y[:,i-1])
        k2 = fun(t[i-1],y[:,i-1]+h/2*k1)
        k3 = fun(t[i-1],y[:,i-1]+h/2*k2)
        k4 = fun(t[i-1],y[:,i-1]+h*k3)
        y[:,i] = y[:,i-1] + h*(k1+2*k2+2*k3+k4)/6.
        dydt[:,i] = fun(t[i],y[:,i])
        
    for i in range(order,n_steps):
        ### predictor ###
        y_p = np.zeros(n_dim)
        for j in range(order):
            y_p += A[order-1,j]*y[:,i-1-j] + h*B[order-1,j]*dydt[:,i-1-j]
        ### corrector ###
        y_c = A[order-1,0]*y[:,i-1] + h*Bstar[order-1,0]*fun(t[i],y_p)
        for j in range(1,order):
            y_c += A[order-1,j]*y[:,i-j] + h*Bstar[order-1,j]*dydt[:,i-j]
        y[:,i] = y_c
        dydt[:,i] = fun(t[i],y[:,i])
    
    return {'t': t, 'y': y}
