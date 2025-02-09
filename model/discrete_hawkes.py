import torch
import numpy as np
import arrow
import os

class DiscreteExponentialKernel_(torch.nn.Module):

    def __init__(self, obs, alpha = np.eye(99), beta = np.ones(1)):
        '''
        Args:
        - obs:      [ n_time, n_space ] torch long
        - alpha:    [ n_space, n_space ] np
        - beta:     [ 1 ] np
        '''
        super().__init__()
        self.obs = obs
        self.alpha  = torch.nn.Parameter(torch.tensor(alpha).float())       # [ n_space ] torch
        self.beta   = torch.nn.Parameter(torch.tensor(beta).float())        # [ 1 ] torch

    def forward(self, tp, sp, t, s):
        '''
        Args:
        - four [ batch_size ] torch
        Returns:
        - val:  [ batch_size ] torch
        '''
        obs_   = self.obs[tp, sp]                               # [ batch_size ] torch  
        temp   = torch.eye(self.obs.shape[1]) * self.alpha      # [ n_space, n_space ] torch
        alpha_ = temp[sp, s]                                    # [ batch_size ] torch  
        val    = alpha_ * obs_ * self.beta * torch.exp( - self.beta * torch.abs(t - tp))
        return val
    
    def update_obs(self, obs):
        '''
        Args:
        - [ n_time, n_space ]
        '''
        self.obs = obs
    
    @staticmethod
    def top_k_neighbors(k):
        # TODO: select the top k nearst neightbor to be included in initialized alpha
        raise NotImplementedError

class DiscreteHawkes(torch.nn.Module):
    '''
    Reference:
    https://arxiv.org/abs/2109.09711
    '''
    def __init__(self, obs, mu, kernel_kwargs):
        '''
        Args:
        - obs:      [ n_time, n_space ] np
        - mu:       [ n_space ] np, base rate
        '''
        # init
        super().__init__()
        self.n_time, self.n_space = obs.shape
        self.obs    = torch.tensor(obs).long()                      # [ n_time, n_space ] torch
        self.kernel = DiscreteExponentialKernel_(obs, **kernel_kwargs)
        self.mu_    = torch.nn.Parameter(torch.tensor(mu).float())  # [ n_space ] torch

    def lam(self, t, s):
        """
        Conditional intensity function lambda at x
        Args:
        - two   [ batch_size ] torch
        Return:
        - lam:  [ batch_size ] torch
        """
        # TODO: speed up algorithm by avoiding mask...
        if len(t) == 0 or len(s) == 0:
            return 0.
        batch_size  = len(t)
        max_t   = torch.max(t).long().item()    # int
        tp      = torch.arange(torch.max(t)).unsqueeze(0).unsqueeze(0).repeat(batch_size, self.n_space, 1)  # [ batch_size, n_space, max_t ] torch
        sp      = torch.arange(self.n_space).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, max_t)        # [ batch_size, n_space, max_t ] torch
        # Step 1: Pad all redundant future times
        mask = tp < t.unsqueeze(1).unsqueeze(1).repeat(1, self.n_space, max_t)      # [ batch_size, M, max_t ] torch
        # Step 2: Transform to batch inputs
        tp_, sp_ = tp.reshape(-1), sp.reshape(-1)                                   # both [ ext_batch_size ] torch
        t_ = t.unsqueeze(1).unsqueeze(1).repeat(1, self.n_space, max_t).reshape(-1) # [ ext_batch_size ] torch
        s_ = s.unsqueeze(1).unsqueeze(1).repeat(1, self.n_space, max_t).reshape(-1) # [ ext_batch_size ] torch
        # Step 3: Feed to kernel
        val_batch   = self.kernel(tp_, sp_, t_, s_)                                 # [ extended batch size ] torch
        val         = val_batch.reshape(mask.shape) * mask                          # [ batch size, M, max_t ] torch, also padded for redundancy
        lam         = self.mu(s) + val.sum(-1).sum(-1)  # [ batch_size ] torch
        lam         = torch.clamp(lam, min = 1e-5)      # prevent negative and too small lambda, which would cause trouble
        return lam

    def mu(self, s):
        ''' [ batch_size ] torch '''
        return self.mu_[s]  # [ batch_size ] torch

    def loglik(self):
        '''
        Returns:
        - loglik:   scalar
        - lam:      [ n_time, n_space ] torch
        '''
        t = torch.arange(self.n_time)       # [ n_time ] torch
        s = torch.arange(self.n_space)      # [ n_space ] torch
        t_grid, s_grid = torch.meshgrid(t, s, indexing='ij')
        t_grid, s_grid = t_grid.reshape(-1), s_grid.reshape(-1) # [ n_time * n_space ] torch
        lam     = self.lam(t_grid, s_grid)  # [ n_time * n_space ] torch
        obs_    = self.obs[t_grid, s_grid]  # [ n_time * n_space ] torch
        loglik  = - lam.sum() + (obs_ * torch.log(lam)).sum() 
        return loglik, lam.reshape(self.n_time, self.n_space)

    def fit(self, num_epochs, lr, save_folder, patience = 50):
        self.train()
        optimizer = torch.optim.Adadelta(self.parameters(), lr=lr)
        losses = []
        for iter in range(num_epochs):
            optimizer.zero_grad()
            loglik, _ = self.loglik()
            loss         = - loglik
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if iter % (num_epochs // 10) == 0:
                print(f'Time: {arrow.now()}\t Epoch: {iter :5d}\t Loss: {losses[-1] :.2f}')

            if np.isnan(loss.item()):
                raise ValueError('NaN Encountered!')
        
            if len(losses) - np.where(np.min(losses) == losses)[0][0] > patience:
                print(f'Training not improved in {patience} rounds, early stopped!')
                break

        os.makedirs(save_folder, exist_ok=True)
        torch.save(self.state_dict(), save_folder + '/state_dict.pth')
        print(f'Trained model has been saved!')

    def load(self, save_folder):
        self.eval()
        state_dict = torch.load(save_folder + '/state_dict.pth')
        self.load_state_dict(state_dict)
        print('Found trained model and loaded!')

    def simulate_step(self, history = None):
        '''
        Simulate for next step based on provided history (only next step because lam is history dependent and needs manual concatenation)
        More precisely "predicting" since it is using the expectation of the poisson distribution
        Args:
        - history:  [ n_hist, n_space ] np, default (None) set to be self.obs
        Returns:
        - out:      [ n_space ] np
        '''
        horizon = 1
        with torch.no_grad():
            obs_ = self.obs.clone()
            self.obs = torch.tensor(history).long() if history is not None else self.obs  
            self.kernel.update_obs(self.obs)
            t_next = torch.ones(self.n_space).unsqueeze(-1) * torch.arange(len(self.obs), len(self.obs) + horizon)     # [ n_space, horizon ] torch
            s_next = torch.arange(self.n_space).unsqueeze(-1).repeat(1, horizon)                                       # [ n_space, horizon ] torch 
            t_next_ = t_next.reshape(-1).long()                                                                        # [ n_space * horizon ] torch 
            s_next_ = s_next.reshape(-1).long()                                                                        # [ n_space * horizon ] torch  
            lam     = self.lam(t_next_, s_next_)                                                                       # [ n_space * horizon ] torch
            lam     = lam.reshape(t_next.shape) 
            self.obs = obs_.clone()
            self.kernel.update_obs(self.obs)
        out = lam.numpy().reshape(-1)
        return out
    
    def simulate_horizon(self, history = None, horizon = 1):
        '''
        Args:
        - history:  [ n_hist, n_space ] np, default (None) set to be self.obs
        - horizon:  scalar
        Returns:
        - outs:     [ horizon, n_space ] np
        '''
        outs = []
        if history is None:
            history = self.obs.numpy()          # [ n_time, n_space ] np
        for i in range(horizon):
            out = self.simulate_step(history)   # [ n_space ] np
            history = np.concatenate([history, out[None, :]], 0)    # [ ++, n_space ] np
            outs.append(out)
        outs = np.stack(outs, 0)                # [ horizon, n_space ] np
        return outs
    
    def simulate_insample_step(self, D):
        '''
        One step ahead forecast (used for insampled evaluation)
        - D:    [ n_time, n_space ] np
        Returns:
        - y:    [ n_time, n_space ] np
        '''
        y = np.stack([ self.simulate_step(D[:i, :], horizon = 1).reshape(-1) for i in range(1, D.shape[0] + 1) ], 0)   # [ ntime, nspace ]
        return y
    

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    Dtr = np.clip(np.random.randn(100, 10), a_min = 0., a_max=None)
    Dte = np.clip(np.random.randn(100, 10), a_min = 0., a_max=None)

    pp_kwargs = {
        'obs':      Dtr,
        'kernel_kwargs':    {
            'alpha':    3e-0 * np.eye(Dtr.shape[-1]),
            'beta':     1e-1
        },
        'mu':       1e-1 * np.ones(Dtr.shape[-1])
    }

    pp_fit_kwargs = {
        'num_epochs':   1000,
        'lr':           1e-3,
        'save_folder':  'cache/discrete_hawkes'  
    }

    pp = DiscreteHawkes(**pp_kwargs)

    # pp.fit(**pp_fit_kwargs)
    # pp.load(pp_fit_kwargs['save_folder'])

    y = pp.simulate_horizon(history=Dte, horizon=10).reshape(-1)
    plt.bar(np.arange(len(y)), y)
    plt.show()