import torch
import numpy as np
from abc import abstractmethod
import torch.optim as optim
import arrow
import itertools
from torch.utils.data import TensorDataset, DataLoader
import os
from utils.lambda_scheduler import LambdaScheduler

class BaseSpatioTemporalHawkes(torch.nn.Module):

    # @abstractmethod
    def __init__(self, T, S, data_dim, int_res=100):
        """
        Args:
        - T:             time horizon. e.g. [0, 1]
        - S:             bounded space for marks. e.g. a two dimensional box region [[0, 1], [0, 1]]
        - data_dim:      dimension of input data
        - numerical_int: numerical integral flag
        - int_res:       numerical integral resolution
        """
        super().__init__()
        # configuration
        self.data_dim      = data_dim
        self.T             = T # time horizon. e.g. (0, 1)
        self.S             = S # bounded space for marks. e.g. a two dimensional box region [[0, 1], [0, 1]]
        assert len(S) + 1 == self.data_dim, f"Expected len(S) + 1 == data_dim, got {len(S)} + 1 not equal to {self.data_dim}"
        assert len(np.array(S).shape) == 2, f"Expected S to be of depth 2, got {len(np.array(S).shape)}."

        # numerical likelihood integral preparation
        if int_res is not None:
            self.int_res  = int_res
            self.tt       = torch.FloatTensor(np.linspace(self.T[0], self.T[1], int_res))  # [ in_res ]
            self.ss       = [ np.linspace(S_k[0], S_k[1], int_res) for S_k in self.S ]     # [ data_dim - 1, in_res ]
            # spatio-temporal coordinates that need to be evaluated
            self.t_coords = torch.ones((int_res ** (data_dim - 1), 1))                     # [ int_res^(data_dim - 1), 1 ]
            self.s_coords = torch.FloatTensor(np.array(list(itertools.product(*self.ss)))) # [ int_res^(data_dim - 1), data_dim - 1 ]
            # unit volumn
            self.unit_vol = np.prod([ S_k[1] - S_k[0] for S_k in self.S ] + [ self.T[1] - self.T[0] ]) / (self.int_res) ** self.data_dim

    def numerical_integral(self, x):
        """
        return conditional intensity evaluation at grid points, the numerical 
        integral can be further calculated by summing up these evaluations and 
        scaling by the unit volumn.

        Args:
        - x         : [ batch_size, seq_len, data_dim ] torch
        Returns:
        - integral  : [ batch_size, int_res, int_res^(data_dim - 1) ] torch
        """
        batch_size, seq_len, _ = x.shape
        integral = []
        for t in self.tt:
            # all possible points at time t (x_t) 
            t_coord = self.t_coords * t
            xt      = torch.cat([t_coord, self.s_coords], 1) # [ int_res^(data_dim - 1), data_dim ] 
            xt      = xt\
                .unsqueeze_(0)\
                .repeat(batch_size, 1, 1)\
                .reshape(-1, self.data_dim)                  # [ batch_size * int_res^(data_dim - 1), data_dim ]
            # history points before time t (H_t)
            mask = ((x[:, :, 0].clone() <= t) * (x[:, :, 0].clone() > 0))\
                .unsqueeze_(-1)\
                .repeat(1, 1, self.data_dim)                 # [ batch_size, seq_len, data_dim ]
            ht   = x * mask                                  # [ batch_size, seq_len, data_dim ]
            ht   = ht\
                .unsqueeze_(1)\
                .repeat(1, self.int_res ** (self.data_dim - 1), 1, 1)\
                .reshape(-1, seq_len, self.data_dim)         # [ batch_size * int_res^(data_dim - 1), seq_len, data_dim ]
            # lambda and integral 
            lams = torch.nn.functional.softplus(self.cond_lambda(xt, ht))\
                .reshape(batch_size, -1)                     # [ batch_size, int_res^(data_dim - 1) ]
            integral.append(lams)                            
        # NOTE: second dimension is time, third dimension is mark space
        integral = torch.stack(integral, 1)                  # [ batch_size, int_res, int_res^(data_dim - 1) ]
        return integral
    
    def cond_lambda(self, xi, hti):
        """
        return conditional intensity given x
        Args:
        - xi:   current i-th point       [ batch_size, data_dim ] torch
        - hti:  history points before ti [ batch_size, seq_len, data_dim ] torch
        Return:
        - lami: i-th lambda              [ batch_size ] torch
        """
        mu = self.mu(xi, hti)           # [ batch_size ] torch
        # if length of the history is zero
        if hti.size()[0] == 0 or hti.size()[1] == 0:
            return mu
        # otherwise treat zero in the time (the first) dimension as invalid points
        batch_size, seq_len, _ = hti.shape
        mask = hti[:, :, 0].clone() > 0                                          # [ batch_size, seq_len ]
        _xi  = xi.unsqueeze(1).repeat(1, seq_len, 1).reshape(-1, self.data_dim)  # [ batch_size * seq_len, data_dim ]
        _hti = hti.reshape(-1, self.data_dim)                                    # [ batch_size * seq_len, data_dim ]
        K    = self.kernel(_xi, _hti).reshape(batch_size, seq_len)               # [ batch_size, seq_len ]
        K    = K * mask                                                          # [ batch_size, seq_len ]
        lami = K.sum(1) + mu                                                     # [ batch_size ]
        return lami

    def log_likelihood(self, x):
        """
        return log-likelihood given sequence x
        Args:
        - x:      input points sequence [ batch_size, seq_len, data_dim ]
        Return:
        - lams:   sequence of lambda    [ batch_size, seq_len ]
        - loglik: log-likelihood        [ batch_size ]
        """
        _, seq_len, _ = x.shape
        lams     = [
            torch.nn.functional.softplus(self.cond_lambda(
                x[:, i, :].clone(), 
                x[:, :i, :].clone())) + 1e-5
            for i in range(seq_len) ]           # TODO: check if softplus and +1e-5 is necessary?                               
        lams     = torch.stack(lams, dim=1)     # [ batch_size, seq_len ]
        # log-likelihood
        mask     = x[:, :, 0] > 0               # [ batch_size, seq_len ]
        sumlog   = torch.log(lams) * mask       # [ batch_size, seq_len ]
        integral = self.numerical_integral(x)   # [ batch_size, int_res, int_res^(data_dim - 1) ]
        loglik = sumlog.sum(1) - integral.sum(-1).sum(-1) * self.unit_vol # [ batch_size ]
        return lams, loglik

    @abstractmethod
    def mu(self, *args):
        """
        return base intensity
        """
        raise NotImplementedError()

    # @abstractmethod
    def forward(self, x):
        '''[ batch_size, seq_len, data_dim ] torch'''
        return self.log_likelihood(x)   # return conditional intensities and corresponding log-likelihood
    
    def fit(self, data, batch_size, num_epochs, lr, save_folder):
        '''
        Args:
        - data:         [ batch_size, seq_len, data_dim ] or [ seq_len, data_dim ], numpy or torch
        - save_folder:  save folder of model dict
        Returns:
        - model's state dict
        - model's training log
        '''
        self.train()
        data            = data.reshape(1, *data.shape) if len(data.shape) == 2 else data # [ batch_size, seq_len, data_dim ]
        data            = torch.tensor(data).float() if isinstance(data, np.ndarray) else data
        traindataset    = TensorDataset(data)
        trainloader     = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

        opt = optim.Adadelta(self.parameters(), lr=lr)
        losses = []
        for i in range(num_epochs):
            epoch_losses = []
            for data in trainloader:    # [ batch_size, seq_len, data_dim ] torch
                opt.zero_grad()
                x_  = data[0]           # [ batch_size, seq_len, data_dim ] torch
                _, loglik = self(x_)
                loss      = - loglik.mean()
                loss.backward()
                opt.step()
                epoch_losses.append(loss.item())
            losses.append(np.mean(epoch_losses))

            if i % (num_epochs // 10) == 0:
                print(f"[{arrow.now()}] Epoch : {i} \t Loss : {losses[-1]:.5e}")

        os.makedirs(save_folder, exist_ok=True)
        torch.save(self.state_dict(), save_folder + '/state_dict.pth')
        np.save(save_folder + '/losses.npy', np.array(losses))
        print('Model has been trained!')
        self.eval()

    def load(self, save_folder):
        self.load_state_dict(torch.load(save_folder + '/state_dict.pth'))

    def simulate(self, data, t_start, t_end, ls_kwargs, verbose = False):
        '''
        Simulating sequences using the Thinning algorithm (Lewis and Ogata)
        Args:
        - data:         [ hist_len >= 0, data_dim ] np, history data
        - t_start:      starting simulation time
        - t_end:        ending simulation time
        - ls_kwargs:    kwargs passed to LambdaScheduler class
        Returns:
        - sim_traj:     [ seq_len, data_dim ] np, simulated trajectory
        - x:
        '''
        assert len(data.shape) == 2, f'Expected data of dimension [ hist_len, data_dim ], got {data.shape}'
        data    = torch.tensor(data).float()            # [ hist_len, data_dim ] torch
        ls      = LambdaScheduler(**ls_kwargs)
        space   = [[t_start, t_end], *self.S]           # [ data_dim, 2 ] list
        volume  = np.prod([s[1] - s[0] for s in space]) 

        # loop until no error is found
        with torch.no_grad():
            flag = True
            while flag:
                try:
                    # STEP 1: simulate points in the observation space, from an "optimistic" homogeneous poisson process
                    N       = np.random.poisson(size = 1, lam = ls() * volume)  # [ 1 ] np
                    homo_points = [ np.random.uniform(s[0], s[1], N) for s in space ]
                    homo_points = np.array(homo_points).transpose()             # [ simulated_count, data_dim ] np
                    homo_points = homo_points[homo_points[:, 0].argsort()]      # [ simulated_count, data_dim ] np
                    homo_points = torch.FloatTensor(homo_points)                # [ simulated_count, data_dim ] torch
                    
                    # STEP 2: thinning algorithm and/or rejection sampling
                    retained_points = data.reshape(1, *data.shape) # [ 1, hist_len, data_dim ] torch
                    for x in homo_points: # [ data_dim ] torch
                        x   = x.reshape(1, *x.shape) # [ 1, data_dim ] torch
                        lam = self.cond_lambda(x, retained_points) # scalar torch
                        D = np.random.uniform()
                        # maximum value exceeded, thinning algorithm fails.
                        if lam > ls():
                            ls.step(False) # maximum value exceeded!
                            raise NotImplementedError
                        # retain point
                        if lam >= D * ls():
                            ls.step(True)
                            retained_points = torch.cat([retained_points, x.reshape(1, *x.shape)], dim = 1) # [ 1, hist_len++, data_dim ] torch
                    flag = False
                except NotImplementedError:
                    if verbose:
                        print('Thinning algorithm fails! Resarting...')

            sim_traj = retained_points.squeeze(0)[data.shape[0]:]   # [ seq_len, data_dim ] torch
            sim_traj = sim_traj.numpy()                             # [ seq_len, data_dim ] np
        return sim_traj                                             # [ seq_len >= 0, data_dim ] np


class ExponentialDecayingKernel(torch.nn.Module):
    """
    Exponential Decaying Kernel for Spatio-Temporal Hawkes
    """
    def __init__(self, alpha, beta):
        """
        Arg:
        - beta: decaying rate
        """
        super(ExponentialDecayingKernel, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(1).float() * alpha)
        self.beta = torch.nn.Parameter(torch.tensor(1).float() * beta)
    
    def forward(self, x, y):
        """
        Args:
        - x:    [ batch_size, data_dim ] torch
        - y:    [ batch_size, data_dim ] torch
        Returns:
        - [ batch_size ] torch
        """
        return self.alpha * self.beta * torch.exp(- self.beta * torch.abs(x - y).sum(-1))


class SpatioTemporalExponentialHawkes(BaseSpatioTemporalHawkes):

    def __init__(self, base_kwargs, kernel_kwargs):
        '''
        Args:
        - mu:   [  ]
        '''
        super().__init__(**base_kwargs)
        self.mu_    = torch.nn.Linear(base_kwargs['data_dim'], 1)
        self.kernel = ExponentialDecayingKernel(**kernel_kwargs)

    def mu(self, xi, *args):
        '''
        Args:
        - xi:   [ batch_size, data_dim ] torch
        Returns:
        - [ batch_size ] torch
        '''
        return self.mu_(xi).reshape(-1)