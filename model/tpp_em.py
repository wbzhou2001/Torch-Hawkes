import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import os

class TemporalPointProcess_:

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
        if mask.sum() == 0:
            return self.mu
        else:
            return self.mu + self.kernel(intervals).sum()

    def E_step(self):
        '''update the triggering matrix'''
        for i in range(self.N):
            base = self.lam(self.events[i])
            for j in range(i):
                self.P[j, i] = self.kernel(self.events[i] - self.events[j]) / base # u_ji, influence from j to i
            self.P[i, i] = self.mu / base # u_0i
        return self.P

    def M_step(self):
        '''update the parameters'''
        m0          = np.diag(self.P).sum()
        self.mu     = m0 / self.T
        m1          = np.triu(self.P, k = 1).sum()
        self.alpha  = m1 / self.N
        # TODO: how to update beta?
        return self.mu, self.alpha, self.beta

    def EM(self, niter, gif = False):
        folder = 'cache/figs'
        os.makedirs(folder, exist_ok=True)

        mu_, alpha_, beta_, P_ = [self.mu], [self.alpha], [self.beta], [self.P]
        for i in tqdm(range(niter)):
            P = self.E_step()
            a, b, c = self.M_step()
            mu_.append(a)
            alpha_.append(b)
            beta_.append(c)

            if gif:
                P_.append(P)
                plt.imshow(np.tril(P, k = 1))
                plt.title(f'Iteration {i}')
                plt.colorbar()
                plt.savefig(folder + f'/iter_{i}.png')
                plt.close()
        
        if gif:
            filenames = [ folder + f'/iter_{i}.png' for i in range(niter) ]
            with imageio.get_writer('triggering_matrix.gif', duration=0.1) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)

        plt.plot(mu_, label = 'Update trajectopry for ' + r'$\mu$')
        plt.plot(alpha_, label = 'Update trajectopry for ' + r'$\alpha$')
        plt.plot(beta_, label = 'Update trajectopry for ' + r'$\beta$')
        # make an animation of P evolution...
        plt.legend()
        plt.show()
        print(f'Fitted Results: Mu = {self.mu}, Alpha = {self.alpha}, Beta = {self.beta}.')
        return None

if __name__ == '__main__':

    # NOTE: this demo requires Woody's point process simulator to be imported.
    kwds = {
        'alpha':    5e-1,
        'mu':       1e-0,
        'beta':     1., # fixed parameter
        'T':        300.
    }
    mu = 1
    alpha = 0.8
    
    # generate events
    T      = [0., kwds['T']]
    beta   = kwds['beta']
    kernel = ExpKernel(beta=beta, alpha = alpha)
    lam    = HawkesLam(mu, kernel, maximum=1e+3)
    pp     = TemporalPointProcess(lam)
    points, sizes = pp.generate(
        T=T, batch_size=1, verbose=False)
    events = points.reshape(-1) # 1d array
    
    # EM algorithm
    tpp = TemporalPointProcess_(events, **kwds)
    tpp.EM(30, gif = True)
