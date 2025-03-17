import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class TemporalPointProcess:

    def __init__(self, events, T = 1., mu = 1., alpha = 1., beta = 1.):
        '''
        - events:   [ nevents ]
        - T:        horizon of the events
        '''
        self.N      = len(events)  
        self.events = events
        self.P      = np.zeros([self.N, self.N]) # diag elements are P_0j, off-diag elements are P_ij 
        self.mu     = mu
        self.alpha  = alpha
        self.beta   = beta
        self.T      = T

    def kernel(self, interval):
        '''interval is a scalar'''
        a = self.alpha * self.beta * np.exp( - self.beta * interval)
        return a
    
    def lam(self, event):
        mask        = self.events < event
        intervals   = event - self.events[mask] # [ N(t) ]
        a = self.mu + self.kernel(intervals).sum()
        return a

    def E_step(self):
        '''update the triggering matrix'''
        for i in range(self.N):
            base = self.lam(self.events[i])
            for j in range(i):
                self.P[j, i] = self.kernel(self.events[i] - self.events[j]) / base # u_ji
            self.P[i, i] = self.mu / base # u_0i
        return None

    def M_step(self):
        '''update the parameters'''
        m0          = np.triu(self.P, k = 1).sum()
        self.mu     = m0 / self.T
        m1          = np.diag(self.P).sum()
        self.alpha  = m1 / self.N
        # TODO: how to update beta?
        return self.mu, self.alpha

    def EM(self, niter):
        mu_, alpha_ = [], []
        for i in tqdm(range(niter)):
            self.E_step()
            a, b = self.M_step()
            mu_.append(a)
            alpha_.append(b)
        plt.plot(mu_, label = 'Update trajectopry for ' + r'$\mu$')
        plt.plot(alpha_, label = 'Update trajectopry for ' + r'$\alpha$')
        plt.legend()
        plt.show()
        print(f'Fittted Results: Mu = {self.mu}, Alpha = {self.alpha}')
        return None
    
if __name__ == '__main__':

    events = np.random.uniform(0, 1, 20)
    events.sort()

    kwds = {
        'alpha':    1e-1,
        'mu':       1e-1,
        'beta':     1e-1,
        'T':        1
    }

    tpp = TemporalPointProcess(events, **kwds)
    tpp.EM(100)