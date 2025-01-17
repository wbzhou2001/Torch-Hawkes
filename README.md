[![PyPI](https://img.shields.io/pypi/v/torch-hawkes?label=pypi)](https://pypi.org/project/torch-hawkes)
[![PyPI-Downloads](https://img.shields.io/pepy/dt/torch-hawkes)](https://pepy.tech/project/torch-hawkes)

# Torch Hawkes

All-in one PyTorch-based implementations of classical Hawkes process algorithms, featuring:
- Three available model variants:
  - Discrete Hawkes process (discrete time, discrete space)
  - Multivariate Hawkes process (continuous time, discrete space)
  - Spatio-Temporal Hawkes process (continuous time, continuous sapce)
- Flexible choice of self-excitation kernels
- Parameter fitting using maximum likelihood (MLE) and expectation-maximization (EM) algorithms
- Sampling using the thinning algorithm
  - Offers $\lambda$-scheduler for speed-up 

# Visualizations

<div style="display: flex; align-items: center; justify-content: space-around;">
    <img src="/img/fig1.png" alt="fig1" style="width:30%;"/>
    <img src="/img/fig2.png" alt="fig2" style="width:30%;"/>
    <img src="/img/fig3.png" alt="fig3" style="width:30%;"/>
</div>

# References:
1. [Hierarchical Conformal Prediction for Spatio-Temporal Point Processes](https://arxiv.org/abs/2411.12193)
2. [Neural Spectral Marked Point Processes](https://iclr.cc/virtual/2022/poster/6311)

<!---
# Discrete Hawkes Generator

A dicrete Hawkes process is characterized by its discrete time intensity rate $\lambda_t$, $t \in \mathbb{N}_+$, 

$$\lambda_t = \mu(t) + \sum_{t' < t} \beta e^{- \beta (t - t')}.$$

## Usage
- Initialized model with ${\tt beta}$ and ${\tt mu \textunderscore config}$, which describes the value of the parameters of the discrete Hawkes process (see equation above).
- ${\tt simulate()}$ : Simulate a trajectory of event occurance of length $\tt t$. A list of intensity rates ${\tt lam}$ is returned as the output. One can also set parameter ${\tt plot = True}$ to automatically visualize the simulated trajectory.
- ${\tt generate()}$ : Given previous trajectory ${\tt prev \textunderscore traj}$, generate future trajectory of length ${\tt t}$.
- ${\tt plot \textunderscore mu()}$ : Plot the base intensity rate $\mu(t)$ on $[0, {\tt t}]$.

---!>
