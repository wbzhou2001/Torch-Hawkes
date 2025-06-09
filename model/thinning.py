import numpy as np
from tqdm import tqdm

class StandardThinning:
    '''
    Standard Implementation of the Thinning Algorithm
    '''
    def __init__(self, S, T, mu, beta, alpha):
        
        if len(S) == 0: # temporal only
            self.S = np.array([]).reshape(0, 2)     # [ nspace, 2 ]
        else:
            self.S = S  # [ nspace, 2 ]
        self.T     = T  # [ 2 ]
        self.mu    = mu
        self.beta  = beta
        self.alpha = alpha

    def simulate(self, lam_bar = None, verbose = True):

        # init
        if lam_bar is None:
            lam_bar = self.mu      # default initialization
        pbar = tqdm(total=100, desc="Simulation progress") if verbose else None
        retained, lams = [], []
        t = 0
        while t < self.T[1]:

            # simulate new point
            t = t + np.random.exponential(scale=1/lam_bar)
            s = np.random.uniform(*self.S.T)
            x = np.concatenate([np.array([t]), s])  # [ nspace + 1 ]

            if verbose: # progress bar update
                perc = ((t - self.T[0]) * 100 / np.diff(self.T)).astype(int).item()
                if perc - pbar.n >= 1:
                    pbar.update(perc - pbar.n)
                    pbar.refresh()

            # compute lambda
            dx  = np.linalg.norm(x - np.array(retained), ord=2, axis=1) if len(retained) > 0 else None # [ len_xs, nspace + 1 ] 
            lam = self.mu_(x) + self.kernel(dx).sum() # scalar
            lams.append(lam)

            if lam > lam_bar:   # fail
                raise KeyError('lambda exceed lam_bar!')   
            
            if np.random.uniform(0, 1) * lam_bar <= lam: # accept
                retained.append(x)

        retained = np.array(retained)     # [ sample_size (=0), nspace + 1 ]
        retained = retained[retained[:, 0] <= self.T[1]] if len(retained) > 0 else retained # remove the extra point
        print(f'Maximum observed lambda: {np.max(lams) : .2f}')
        return retained # [ num_sample, nsapce + 1 ]
                
    def mu_(self, x):
        '''
        Args:
        - [ nspace + 1 ]
        '''
        return self.mu

    def kernel(self, dx = None):
        '''
        Args:
        - dx:   [ batch_size ] distance
        '''
        if dx is None:
            return np.array([0.])
        else:
            return self.alpha * self.beta * np.exp( - self.beta * dx)
        
import numpy as np
from tqdm import tqdm

class FastThinning:
    '''
    Faster thinning algorithm with batched processing
    '''
    def __init__(self, S, T, mu, beta, alpha):
        
        if len(S) == 0: # temporal only
            self.S = np.array([]).reshape(0, 2)     # [ nspace, 2 ]
        else:
            self.S = S  # [ nspace, 2 ]
        self.T     = T  # [ 2 ]
        self.mu    = mu
        self.beta  = beta
        self.alpha = alpha

    def simulate(self, lam_bar = None, verbose = True):

        # init
        if lam_bar is None:
            lam_bar = self.mu      # default initialization
        else:
            assert lam_bar >= self.mu, 'lambda must be no less than the baserate' 

        retained, lams = [], []
        X_ = np.concatenate([self.T[None, :], self.S], axis = 0)    # [ nspace + 1, 2 ]
        N = np.random.poisson(lam = lam_bar * np.diff(X_).prod())
        if N > 0:      # no points generated, end thinning
            pbar = tqdm(total=100, desc="Simulation progress") if verbose else None
            candidate = np.random.uniform(*X_.T, size=[N, len(self.S) + 1])    # [ N, nspace + 1 ]
            candidate = candidate[candidate[:, 0].argsort()]   # same as above
            for x in candidate:
                dx  = np.linalg.norm(x - np.array(retained), ord=2, axis=1) if len(retained) > 0 else None # [ len_xs, nspace + 1 ] 
                lam = self.mu_(x) + self.kernel(dx).sum() # scalar
                lams.append(lam)

                if lam > lam_bar:
                    raise KeyError('lambda exceed lam_bar!')   
                
                if np.random.uniform(0, 1) * lam_bar <= lam:
                    retained.append(x)

                # progress bar
                if verbose:
                    perc = ((x[0] - self.T[0]) * 100 / np.diff(self.T)).astype(int).item()
                    if perc - pbar.n >= 1:
                        pbar.update(perc - pbar.n)
                        pbar.refresh()
        retained = np.array(retained)     # [ sample_size (=0), nspace + 1 ]
        print(f'Maximum observed lambda: {np.max(lams) if len(lams) > 0 else None: .2f}')
        return retained
                
    def mu_(self, x):
        '''
        Args:
        - [ nspace + 1 ]
        '''
        return self.mu

    def kernel(self, dx = None):
        '''
        Args:
        - dx:   [ batch_size ] distance
        '''
        if dx is None:
            return np.array([0.])
        else:
            return self.alpha * self.beta * np.exp( - self.beta * dx)
        
if __name__ == "__main__":
    
    kwds = {
        'S': np.array([[0, 1], [0, 1]]),  # Example space intervals
        'T': np.array([0., 1.]),
        'mu':   1.,
        'beta': 1.,
        'alpha':1.
    }

    sim = StandardThinning(**kwds)
    sim.simulate(lam_bar = 100000., verbose=True)
    sim = FastThinning(**kwds)
    sim.simulate(lam_bar = 100000., verbose=True)
