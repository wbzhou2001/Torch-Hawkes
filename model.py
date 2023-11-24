import matplotlib.pyplot as plt
import numpy as np

class DiscreteHawkes(object):
    
    def __init__(self, beta, mu_config : dict):
        self.beta = beta
        self.mu = mu_config['mu']
        self.dynamic = mu_config['dynamic']
        self.linear = mu_config['linear']
        self.quadratic = mu_config['quadratic']
        self.a1 = mu_config['a1']
        self.a2 = mu_config['a2']

    def Mu(self, _t):
        if self.dynamic == False:
            return self.mu
        elif self.dynamic == True and self.linear == True and self.quadratic == False:
            return max(self.mu + self.a1 * _t, 0)
        elif self.dynamic == True and self.linear == False and self.quadratic == True:
            return max(self.mu + self.a2 * _t ** 2, 0)
        elif self.dynamic == True and self.linear == True and self.quadratic == True:
            return max(self.mu + self.a1 * _t + self.a2 * _t ** 2, 0)
        else:
            raise NotImplementedError

    def kernel(self, _t, traj):
        T = np.ones(_t) * _t
        Tp = np.arange(_t)
        return np.sum(self.beta * np.array(traj[:_t]) * np.exp( - self.beta * (T - Tp)))
    
    def simulate(self, t = 10, plot = False):
        lam = []
        traj = []
        for _t in range(t):
            if _t == 0:
                lam.append(self.Mu(_t))
                traj.append(np.random.poisson(lam[-1]))
            else:
                lam.append(self.Mu(_t) + self.kernel(_t, traj))
                traj.append(np.random.poisson(lam[-1]))
        if plot == True:
            plt.bar(np.arange(t), traj, label = 'Trajectory', color = 'grey', alpha = 0.8)
            plt.plot(lam, label = 'Intensity Rate', c = 'orange', lw = 4)
            plt.fill_between(np.arange(t), lam, color = 'orange', alpha = 0.5)
            plt.xlabel('Timestep')
            plt.ylabel('Counts')
            plt.title(f'Beta = {self.beta}')
            plt.grid(True)
            plt.legend()
            plt.show()
        return lam, traj
    
    def generate(self, prev_traj, t = 10, plot = False):
        lam = []
        traj = prev_traj.copy()
        for _t in range(len(traj)):
            if _t == 0:
                lam.append(self.Mu(_t))
            else:
                lam.append(self.Mu(_t) + self.kernel(_t, traj))
        for _t in range(t):
            lam.append(self.Mu(len(traj)) + self.kernel(len(traj), traj))
            traj.append(np.random.poisson(lam[-1]))
        if plot == True:
            # Previous Trajectory
            plt.bar(np.arange(len(prev_traj)), traj[:len(prev_traj)], label = 'Previous Trajectory', color = 'grey', alpha = 0.8)
            plt.plot(np.arange(len(prev_traj)), lam[:len(prev_traj)], label = 'Previous Intensity Rate', c = 'orange', lw = 4)
            plt.fill_between(np.arange(len(prev_traj)), lam[:len(prev_traj)], color = 'orange', alpha = 0.5)
            # Generated Trajectory
            plt.bar(np.arange(len(prev_traj), (len(prev_traj) + t)), traj[len(prev_traj):], label = 'Generated Trajectory', color = 'red', alpha = 0.8)
            plt.plot(np.arange(len(prev_traj) - 1, (len(prev_traj) + t)), lam[(len(prev_traj) - 1):], label = 'Generated Intensity Rate', c = 'red', lw = 4)
            plt.fill_between(np.arange(len(prev_traj) - 1, (len(prev_traj) + t)), lam[(len(prev_traj) - 1):], color = 'red', alpha = 0.5)
            # Settings
            plt.xlabel('Timestep')
            plt.ylabel('Counts')
            plt.title(f'Beta = {self.beta}')
            plt.grid(True)
            plt.legend()
            plt.show()
        return lam, traj
    
    def plot_mu(self, t):
        plt.plot([self.Mu(_t) for _t in range(t)], color = 'blue', lw = 4)
        plt.fill_between(np.arange(t), [self.Mu(_t) for _t in range(t)], color = 'blue', alpha = 0.3)
        plt.xlabel('Timestep')
        plt.ylabel('Value')
        plt.title('Shape of Base Rate Mu')
        plt.grid(True)
        plt.show()
        return None