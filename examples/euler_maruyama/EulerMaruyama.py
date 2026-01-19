import numpy as np
import matplotlib.pyplot as plt

class EulerMaruyama:

    def __init__(self, dt, T, nk, v0):
        self.dt = dt  # Time step.
        self.T = T # Total time.
        self.nt = int(T / dt)  # Number of time steps.
        self.sqrtdt = np.sqrt(dt)
        self.nk = nk
        self.etas = np.random.randn((self.nt, self.nk))

    def forward(self, params):
        x = np.zeros((self.nk, self.nt))
        x[:, 0]=self.v0
        for i in range(nt - 1):
            x[ :, i + 1] = x[:, i] + mu*x[:, i]*dt  + \
            sigma*x[:, i] * sqrtdt * self.etas[:, i]
        return x
        
    def backward(self, params, r):
        x = np.zeros((self.nk, self.nt))
        x[:, -1]=r[:,-1]
        for i in  range(nt - 2, 0):
            x[ :, i] = x[:, i+1] + mu*x[:, i+1]*dt  + \
            sigma*x[:, i+1] * sqrtdt * self.etas[:, i+1]+r[i]
        return x
    
    def forward_lin(self, params0, dparams):
        x0 = np.zeros(self.nt)
        dx = np.zeros(self.nt)
        x0[0]=x0
        dx[0]=dx0
        for i in range(nt - 1):
            x0[ i + 1] = x0[i] + mu0*x0[i]*dt  + \
            sigma0*x0[i] * sqrtdt * np.random.randn()
            dx[ i + 1] = dx[i] + dmu*x0[i]*dt  + mu*dx[i]*dt + \
            dsigma*x0[i] * sqrtdt * np.random.randn() + \
            sigma0*dx[i] * sqrtdt * np.random.randn()
        return dx
    
    def adjoint(self, params0, dx):
        x0 = np.zeros(self.nt)
        dparams
        x0[0]=x0
        dx[0]=dx0
        for i in range(nt - 1):
            x0[ i + 1] = x0[i] + mu*x0[i]*dt  + \
            sigma*x0[i] * sqrtdt * np.random.randn()
            dmu += dx[i+1]*x0[i]*dt 
            dsigma+= dx[i+1]*x0[i] * sqrtdt * np.random.randn()
        return dparams

sigma = 0.1  # Standard deviation.
mu = 0.1  # Mean.
epsilon = 0.01
nviz = 10
k = 0.9

dt = .001  # Time step.
T = 10.  # Total time.
nt = int(T / dt)  # Number of time steps.
t = np.linspace(0., T, nt)  # Vector of times.
x = np.zeros((nviz, nt))
y = np.zeros((nviz, nt))
sqrtdt = np.sqrt(dt)

x[:,0]=1
y[:,0]=100
for i in range(nt - 1):
    x[:, i + 1] = x[:, i] + mu*x[:, i]*dt  + \
        sigma*x[:, i] * sqrtdt * np.random.randn(nviz)
    y[:, i + 1] = y[:, i]*np.exp(epsilon*dt* (k*x[:, i]-1))

fig,(ax0, ax1) = plt.subplots(2)
for iviz in range(nviz):
    ax0.plot(t,x[iviz,:])
    ax1.plot(t,y[iviz,:])

plt.savefig("euler_maruyama")