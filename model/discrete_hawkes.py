import torch
import numpy as np
import arrow
import os

class DiscreteExponentialKernel(torch.nn.Module):

    def __init__(self, obs, alpha = np.eye(99), beta = np.ones(1)):
        '''
        Args:
        - obs:      [ n_time, n_space ] torch long
        - alpha:    [ n_space, n_space ] np
        - beta:     [ 1 ] np
        '''
        super().__init__()
        self.obs    = obs 
        self.alpha  = torch.nn.Parameter(torch.tensor(alpha).float())   # [ n_space ] torch
        self.beta   = torch.nn.Parameter(torch.tensor(beta).float())    # [ 1 ] torch

    def forward(self, tp, sp, t, s):
        '''
        Args:
        - four [ batch_size ] torch
        Returns:
        - val:  [ batch_size ] torch
        '''
        obs_   = self.obs[tp, sp]   # [ batch_size ] torch  
        alpha_ = self.alpha[sp, s]  # [ batch_size ] torch  
        val    = alpha_ * obs_ * self.beta * torch.exp( - self.beta * torch.abs(t - tp))
        return val
    
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
        self.kernel = DiscreteExponentialKernel(self.obs, **kernel_kwargs)
        self.mu_    = torch.nn.Parameter(torch.tensor(mu).float())  # [ n_space ] torch

    def lam(self, t, s):
        """
        Conditional intensity function lambda at x
        Args:
        - two   [ batch_size ] torch
        Return:
        - lam:  [ batch_size ] torch
        """
        # TODO: spped up algorithm by avoiding mask...
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
        lam         = torch.nn.functional.relu(lam)     # prevent negative lam
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

    def fit(self, num_epochs, lr, save_folder):
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
        os.makedirs(save_folder, exist_ok=True)
        torch.save(self.state_dict(), save_folder + '/state_dict.pth')
        print(f'Trained model has been saved!')
        self.eval()

    def load(self, save_folder):
        state_dict = torch.load(save_folder + '/state_dict.pth')
        self.load_state_dict(state_dict)
        print('Found trained model and loaded!')

    def simulate(self, history = None, horizon = 1):
        '''
        Simulate for next some horizon of time steps based on provided history
        Args:
        - history:  [ n_hist, n_space ] np, default (None) set to be self.obs
        - horizon:  scalar
        Returns:
        - data:     [ n_space, horizon ] np
        '''
        with torch.no_grad():
            obs_ = self.obs.clone()
            self.obs = torch.tensor(history).long() if history is not None else self.obs  
            t_next = torch.ones(self.n_space).unsqueeze(-1) * torch.arange(len(self.obs), len(self.obs) + horizon)     # [ n_space, horizon ] torch
            s_next = torch.arange(self.n_space).unsqueeze(-1).repeat(1, horizon)                                       # [ n_space, horizon ] torch 
            t_next_ = t_next.reshape(-1).long()
            s_next_ = s_next.reshape(-1).long()
            lam     = self.lam(t_next_, s_next_)                                                                       # [ n_space * horizon ] torch
            lam     = lam.reshape(t_next.shape) 
            self.obs = obs_.clone()
        return lam.numpy()
    

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    Dtr = np.clip(np.random.randn(100, 10), a_min = 0., a_max=None)
    Dte = np.clip(np.random.randn(100, 10), a_min = 0., a_max=None)

    pp_kwargs = {
        'obs':      Dtr,
        'kernel_kwargs':    {
            'alpha':    1e-1 * np.eye(Dtr.shape[-1]),
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
    pp.load(pp_fit_kwargs['save_folder'])

    y = pp.simulate(history=Dte, horizon=1).reshape(-1)
    plt.bar(np.arange(len(y)), y)
    plt.show()