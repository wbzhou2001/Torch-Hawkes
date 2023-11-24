import matplotlib.pyplot as plt
import numpy as np
from model import DiscreteHawkes

def main():
    beta            = 0.6               # beta
    _scale          = 0.1               # tool variable to adjust Mu 
    mu_config = {
        'mu'        : 0 * _scale,       # mu
        'dynamic'   : True,             # constant mu?
        'linear'    : True,             # include linear term?
        'quadratic' : True,             # include quad term?
        'a1'        : - 0.2 * _scale,   # linear coef
        'a2'        : 0.003 * _scale    # quadratic coef
    }
    sim_len = 150
    gen_len = 50

    model = DiscreteHawkes(beta = beta, mu_config = mu_config)
    lam, traj = model.simulate(t = sim_len, plot = True)
    lam, traj = model.generate(traj, t = gen_len, plot = True)
    model.plot_mu(200)

if __name__ == '__main__':
    main()