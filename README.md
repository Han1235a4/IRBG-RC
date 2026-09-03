# Reshaping Input-to-Reservoir Topologies in Reservoir Computing via Bipartite Graph Generation

![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
![Hardware](https://img.shields.io/badge/GPU-RTX%203090%20Ti-green.svg)

## Overview
This repository contains the dataset and source code for the paper **"Reshaping Input-to-Reservoir Topologies in Reservoir Computing via Bipartite Graph Generation"**. It provides the implementation of the proposed bipartite graph generation method (IRBG) for optimizing reservoir topologies, applied to the prediction of complex chaotic time series.

## Hardware & Environment
* **CPU:** Intel Core i9-10980XE
* **GPU:** NVIDIA RTX 3090 Ti
* **Python:** 3.7

## Datasets

The models and experiments are evaluated on the following classic and high-dimensional chaotic dynamical systems. 
> **Note:** The references and detailed descriptions for the four datasets used in our experiments have been formally cited and elaborated in the main manuscript.
### 1. Lorenz System
The 3D Lorenz system is governed by the following differential equations:

$$
\begin{cases}
\dot{x} = 10(y - x) \\
\dot{y} = x(28 - z) - y \\
\dot{z} = xy - \frac{8}{3}z
\end{cases}
$$

### 2. 4D Hyperchaotic System
The 4D hyperchaotic system is defined as:

$$
\begin{cases}
\dot{x} = ax - yz + w \\
\dot{y} = xz - by \\
\dot{z} = xy - cz \\
\dot{w} = -y + d
\end{cases}
$$

**Parameters & Initial Conditions:**
* **Parameters:** $a = 8$, $b = 40$, $c = 15$, $d = -0.1$
* **Initial State:** $[x_0, y_0, z_0, w_0]^T = [10, 1, 10, 1]^T$


## RC Variants and Hyperparameter Search Spaces

### 1. Leaky-RC

The Leaky-RC model is defined as:

## Leaky-RC

The Leaky-RC model is defined as:

$$
\begin{cases}
x(t) = (1-\alpha)x(t-1) + \alpha\tanh\left(W_{bg}^{T}W_{in}u(t) + Wx(t-1)+\gamma\mathbf{1}\right)
\end{cases}
$$


## Deep RC

For a Deep RC with $L$ reservoir layers, the state update is defined as:

$$
\begin{cases}
x^{(1)}(t) = (1-\alpha_1)x^{(1)}(t-1) + \alpha_1\tanh\left(W_{bg}^{(1)T}W_{in}^{(1)}u(t) + W^{(1)}x^{(1)}(t-1)+\gamma\mathbf{1}\right), \\
x^{(l)}(t) = (1-\alpha_l)x^{(l)}(t-1) + \alpha_l\tanh\left( W_p^{(l-1)}x^{(l-1)}(t) + W^{(l)}x^{(l)}(t-1)\right), \quad l=2,\ldots,L.
\end{cases}
$$


## ES^2N

The ES$$^2$$N model is defined as:

$$
\begin{cases}
x(t) = \beta\tanh\left(\rho Wx(t-1) + W_{bg}^{T}W_{in}u(t) + \gamma\mathbf{1}\right) + (1-\beta)Ox(t-1)
\end{cases}
$$


## MCI-ESN

The MCI-ESN model is defined as:

$$
\begin{cases}
x^{(1)}(t) = \tanh\left(W^{(1)}x^{(1)}(t-1) + W_{bg}^{(1)T}W_{in}^{(1)}u(t) + W_{12}x^{(2)}(t-1)+\gamma\mathbf{1}_1\right), \\
x^{(2)}(t) = \tanh\left(W^{(2)}x^{(2)}(t-1) + W_{bg}^{(2)T}W_{in}^{(2)}u(t) + W_{21}x^{(1)}(t-1)+\gamma\mathbf{1}_2\right).
\end{cases}
$$


## Hyperparameter Search Spaces

| Model | Reservoir size | Depth $L$ | Spectral radius $\rho$ | Input scaling | Leak rate | $\beta$ | Bias scaling | $P$ |
|---|---|---|---|---|---|---|---|---|
| **Leaky-RC** | $\{100,200,300,400,500\}$ | -- | $\{0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5\}$ | $\{0.001,0.01,0.1,1.0\}$ | $\{0.001,0.01,0.1,1.0\}$ | -- | -- | $\{-1,...,-0.1,0,0.1,...,1\}$ |
| **Deep RCc**  | $\{100,200,300,400,500\}$ | $\{2,4,5,10\}$ | $\{0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5\}$ | $\{0.001,0.01,0.1,1.0\}$ | $\{0.001,0.01,0.1,1.0\}$ | -- | -- | $\{-1,...,-0.1,0,0.1,...,1\}$ |
|**ES<sup>2</sup>N** | $\{100,200,300,400,500\}$ | -- | $\{0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5\}$ | $\{1.0,0.1,0.01,0.001\}$ | -- | $\{0.001,0.01,0.1,1.0\}$ | $\{1.0,0.1,0.01,0.001\}$ |$\{-1,...,-0.1,0,0.1,...,1\}$ |
| **MCI-ESN** | $\{100,200,300,400,500\}$ | -- | $\{0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5\}$ | $\{0.001,0.01,0.1,1.0\}$ | -- | -- | -- | $\{-1,...,-0.1,0,0.1,...,1\}$ |

