[![PyPI](https://img.shields.io/pypi/v/sklearn_ensemble_cv?label=pypi)](https://pypi.org/project/)
[![PyPI-Downloads](https://img.shields.io/pepy/dt/sklearn_ensemble_cv)](https://pepy.tech/project/)

# Hawkes Process

All-in one PyTorch-based implementations of classical Hawkes process algorithms, featuring:
- Offers three model variants:
  - Discrete Hawkes process (discrete time, discrete space)
  - Multivariate Hawkes process (continuous time, discrete space)
  - Spatio-Temporal Hawkes process (continuous time, continuous sapce)
- Flexible choice self-excitation kernel specification
- Parameter fitting using Maximum likelihood (MLE) and Expectation-Maximization (EM) algorithms
- Sampling using the Thinning algorithm

# References:
1. [Uncertainty Conformal Prediction for Spatio-Temporal Point Processes](https://arxiv.org/abs/2411.12193)
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

Demos for the plots are shown below.

![fig1](/img/fig1.png) 

![fig1](/img/fig2.png)

![fig1](/img/fig3.png)
---!>
