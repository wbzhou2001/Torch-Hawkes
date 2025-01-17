import numpy as np
import torch
import torch.optim as optim
import arrow
import os
from abc import abstractmethod
from torch.utils.data import TensorDataset, DataLoader
from utils.lambda_scheduler import LambdaScheduler

class BaseMultivariateHawkes(torch.nn.Module):
    '''
    Hawkes model with discrete space, continuous time
    '''
    def __init__(self, T, n_space, int_res = 100):
        '''
        Args:
        - T:        list of starting time and ending time, e.g. [ 0., 30. ]
        - n_space:  int, dimension of spatial, e.g. 99
        '''
        super().__init__()
        self.T =        T 
        self.n_space =  n_space
        self.int_res =  int_res

        # # model params
        # self._mu        = torch.nn.Parameter(torch.ones(S_dim, dtype=float))        # [ number of spatial nodes ]
        # self.kernel     = ExponentialKernel(self.locs)                              # [ batch size, 2 ] and [ batch size, 2 ] -> [ batch size ]

    def fit(self, data, batch_size, num_epochs, lr, save_folder):
        '''
        Args:
        - data:         [ seq_len, data_dim ], numpy or torch
        - save_folder:  save folder of model dict
        Returns:
        - model's state dict
        - model's training log
        '''
        # TODO: extend to batch version
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
                x_  = data[0][0]        # [ seq_len, data_dim ] torch
                loglik      = self(x_)
                loss        = - loglik.mean()
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

    def lam(self, x, h):
        '''
        Args:
        - x   : [ batch size, 2 ] torch or torch scalar
        - h   : [ batch size, history length, 2 ] torch
        Returns:
        - lam : [ batch size ] torch  
        '''
        shape   = h.shape 
        if shape[0] == 0 or shape[1] == 0:
            return torch.tensor(0.) # scalar
        # Step 1: get baserate
        mu      = self.mu(x)                        # [ batch_size ]
        # Step 2: get kernel term
        x_ext   = x.unsqueeze(1).repeat(1, shape[1], 1)                                # [ batch size, history length, 2 ] torch
        x_batch, h_batch = x_ext.reshape(-1, shape[-1]), h.reshape(-1, shape[-1])   # both [ mod batch size, 2 ] torch
        val     = self.kernel(x_batch, h_batch)     # [ mod batch_size ]
        val     = val.reshape(shape[0], shape[1])   # [ batch size, history length ]
        # mask all paddings
        mask    = h[:, :, 0] != 0.                  # [ batch size, history length ] torch 
        val     = val * mask                        # [ batch size, history length ] torch 
        lam     = mu + val.sum(-1)                  # [ batch size ] torch
        lam     = torch.relu(lam)                   # [ batch_size ] torch
        return lam                                  # [ batch size ] torch
    
    def simulate(self, data, t_start, t_end, ls_kwargs = {}, verbose = False):
        '''
        Using efficient multivariate hawkes thinning algorithm described in
        https://www.math.fsu.edu/~ychen/research/multiHawkes.pdf

        Args:
        - data:         [ seq_len >= 0, data_dim ] np, history data 
        - t_start:      starting simulation time
        - t_end:        ending simulation time
        - ls_kwargs:    kwargs passed to LambdaScheduler class
        Returns:
        - sim_traj:     [ seq_len, data_dim ] numpy, simulated trajectory
        - x:
        '''
        # TODO: verbose method
        assert len(data.shape) == 2, f'Expected data of dimension [ seq_len, data_dim ], got {data.shape}'
        data    = torch.tensor(data).float()    # [ seq_len, data_dim ] torch
        ls      = LambdaScheduler(**ls_kwargs)
        retained_points = [x for x in data]

        t = t_start
        while t < t_end:
            s_grids = torch.arange(self.n_space)                    # [ n_space ] torch
            ts      = torch.ones(self.n_space) * t                  # [ n_space ] torch
            xs      = torch.stack([ts, s_grids], -1)                # [ n_space, 2 ] torch 
            hs      = torch.stack(retained_points, 0).unsqueeze(0).repeat(self.n_space, 1, 1)    # [ n_space, seq_len, data_dim ] torch
            lams    = self.lam(xs, hs)                              # [ n_space ] torch
            lam_bar = ls()    # torch scalar
            
            if lam_bar < lams.sum():
                ls.step(False)
                continue
            
            u = torch.rand(1) # torch scalar
            w = - torch.log(u) / lam_bar
            t = t + w
            D = torch.rand(1) # torch scalar

            try: # TODO: index 0 is out of bounds for dimension 0 with size 0 (index = torch.where(mask)[0][0].item() + 1)
                if D * lam_bar <= lams.sum():
                    mask  = D * lam_bar <= lams.cumsum(-1)      # [ n_space ] torch
                    index = torch.where(mask)[0][0].item()      # int 
                    x     = torch.tensor([t, index])            # [ 2 ] torch
                    retained_points.append(x)
            except:
                pass

        retained_points = retained_points[len(data):]       # remove all history values

        try:
            if t > t_end:
                sim_traj = torch.stack(retained_points[:-1], 0) # [ seq_len, 2 ] torch
            else:
                sim_traj = torch.stack(retained_points, 0)      # [ seq_len, 2 ] torch
        except: # zero length sim_traj encountered
            sim_traj = torch.tensor([]).reshape(0, 2)
        sim_traj = sim_traj.detach().numpy() # [ seq_len, 2 ] numpy
        return sim_traj
    
    def loglik(self, data):
        '''
        Args:
        - data      : [ number of datapoints, 2 ] torch
        Returns:
        - loglik    : torch float
        '''
        points, histories = [], []
        for i in range(len(data)):
            point, history = data[i], data[:i] # [ 2 ] and [ history length, 2 ] torch
            # padding value is 0.
            history = torch.nn.functional.pad(history, pad=(0, 0, 0, len(data)-len(history)), value=0.)   # [ max history length, 2 ]
            points.append(point)
            histories.append(history)
        points    = torch.stack(points, 0)      # [ number of datapoints, 2 ]
        histories = torch.stack(histories, 0)   # [ number of datapoints, max history length, 2 ]
        lams      = self.lam(points, histories) # [ number of datapoints ] torch

        # see Reinhart's review of point process Eq.(8)
        term1 = torch.log(lams).sum()                               # scalar torch
        term2 = self.integral_term(data).sum() * self.T[-1] / self.int_res   # scalar torch
        loglik = term1 - term2
        return loglik                           # scalar torch

    def integral_term(self, data,
                      T = None, int_res = None):
        '''
        Args:
        - T                 : list of starting time and ending time, e.g. [ 0., 30. ], default uses the same as the training data
        - data              : [ number of datapoints, 2 ] np or torch
        Returns:
        - lams              : [ number of time grids, number of spatial nodes ] torch
        '''
        if T is None:
            T = self.T
        if int_res is None:
            int_res = self.int_res
        data = torch.tensor(data).float() if isinstance(data, np.ndarray) else data # [ number of datapoints, 2 ] torch

        t_grids = torch.linspace(T[0], T[1], int_res)    # [ int_res ]  torch
        s_grids = torch.arange(self.n_space)        # [ number of spatial nodes ] torch

        points, histories = [], []
        for t in t_grids:
            mask    = data[:, 0] < t
            history = data[mask]
            history = torch.nn.functional.pad(history, pad=(0, 0, 0, len(data)-len(history)), value=0.)   # [ max history length, 2 ]
            for s in s_grids:
                point = torch.tensor([t, s]) # [ 2 ] torch
                points.append(point)
                histories.append(history)
        points    = torch.stack(points, 0)       # [ spatial * temporal grids, 2 ]
        histories = torch.stack(histories, 0)    # [ spatial * temporal grids, max history length, 2 ]
        lams = self.lam(points, histories)
        lams = lams.reshape(len(t_grids), len(s_grids)) # [ number of time grids, number of spatial nodes ] torch
        return lams # [ number of time grids, number of spatial nodes ] torch
    
    # @abstractmethod
    def forward(self, x):
        '''[ batch_size, seq_len, data_dim ] torch'''
        return self.loglik(x)   # return conditional intensities and corresponding log-likelihood
    
    @abstractmethod
    def mu(self, *args):
        """
        return base intensity
        """
        raise NotImplementedError()
    
class ExponentialMultivariateKernel(torch.nn.Module):

    def __init__(self, alpha, beta):
        '''
        Args:
        - alpha:    [ n_space, n_space ] np
        - beta:     [ 1 ] np
        '''
        super().__init__()
        self.alpha  = torch.nn.Parameter(torch.tensor(alpha).float())   # [ n_space ] torch
        self.beta   = torch.nn.Parameter(torch.tensor(beta).float())    # [ 1 ] torch

    def forward(self, x, xp):
        '''
        Args:
        - four [ batch_size ] torch
        Returns:
        - val:  [ batch_size ] torch
        '''
        alpha_ = self.alpha[xp[:, 1].long(), x[:, 1].long()]  # [ batch_size ] torch  
        val    = alpha_ * self.beta * torch.exp( - self.beta * torch.abs(x[:, 0] - xp[:, 0]))
        return val
    
    @staticmethod
    def top_k_neighbors(k):
        # TODO: select the top k nearst neightbor to be included in initialized alpha
        raise NotImplementedError()
        
    
class ExponentialMultivariateHawkes(BaseMultivariateHawkes):

    def __init__(self, base_kwargs, kernel_kwargs, mu):
        '''
        Args:
        - mu:   [ n_space ] np
        '''
        super().__init__(**base_kwargs)
        self.kernel = ExponentialMultivariateKernel(**kernel_kwargs)
        self.mu_    = torch.nn.Parameter(torch.tensor(mu).float())                # [ n_space ] torch

    def mu(self, x):
        '''
        Args:
        - x:    [ batch_size, 2 ] torch
        Returns:
        - [ batch_size ] torch
        '''
        return self.mu_[x[:, 1].long()]  # [ batch_size ] torch 
