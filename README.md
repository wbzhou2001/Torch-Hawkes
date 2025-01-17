# Hawkes-Process

Pytorch-based Hawkes Process all-in-one implementation. Featuring:
- Offers both discrete and continuous Hawkes process models
- Flexible choice self-excitation kernel specification
- Parameter fitting using Maximum likelihood (MLE) and Expectation-Maximization (EM) algorithms
- Sampling using thinning algorithm

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
