import numpy as np

data = np.loadtxt('ecoli.csv', delimiter=',')
t_experiment = data[:,0]
N_experiment = data[:,1]

def error(p):
    r = p[0]
    T = 1200     # cell can divide after T sec
    t_max = 5*T  # 5 generations in experiment
    t = np.linspace(0, t_max, len(t_experiment))
    dt = (t[1] - t[0])
    N = np.zeros(t.size)

    N[0] = 100
    for n in range(0, len(t)-1, 1):
        N[n+1] = N[n] + r*dt*N[n]

    e = np.sqrt(np.sum((N - N_experiment)**2))/N[0]  # error measure
    e = abs(N[-1] - N_experiment[-1])/N[0]
    print 'r=', r, 'e=',e
    return e

from scipy.optimize import minimize

p = minimize(error, [0.0006], tol=1E-5)
print p
