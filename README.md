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

The reservoir state is updated as

\[
\mathbf{x}(t)
=
(1-\alpha)\mathbf{x}(t-1)
+
\alpha
\tanh\left(
W_{bg}^{T}W_{\mathrm{in}}\mathbf{u}(t)
+
W_r\mathbf{x}(t-1)
\right),
\]

where \(\alpha\) denotes the leak rate, \(W_{\mathrm{in}}\) is the input-weight matrix, \(W_{bg}\) is the bi-adjacency matrix introduced by the proposed IRBG rewiring method, and \(W_r\) is the reservoir recurrent weight matrix.

The hyperparameter search space is

\[
\begin{aligned}
N &\in \{100,200,300,400,500\},\\
\rho &\in \{0.5,0.7,0.9,1.1,1.3,1.5\},\\
\alpha_{\mathrm{in}} &\in \{0.001,0.01,0.1,1.0\},\\
\alpha &\in \{0.001,0.01,0.1,1.0\},\\
P &\in \{-1,-0.9,\ldots,-0.1,0,0.1,\ldots,0.9,1\}.
\end{aligned}
\]

---

### 2. Deep RC

For the input-to-all Deep RC architecture, the state update of the first reservoir layer is

\[
\mathbf{x}^{(1)}(t)
=
(1-\alpha_1)\mathbf{x}^{(1)}(t-1)
+
\alpha_1
\tanh\left(
W_{bg}^{(1)T}
W_{\mathrm{in}}^{(1)}
\mathbf{u}(t)
+
W_r^{(1)}\mathbf{x}^{(1)}(t-1)
\right).
\]

For the \(l\)-th layer, \(l=2,\ldots,L\),

\[
\begin{aligned}
\mathbf{x}^{(l)}(t)
={}&
(1-\alpha_l)\mathbf{x}^{(l)}(t-1)\\
&+
\alpha_l
\tanh\left(
W_{bg}^{(l)T}
W_{\mathrm{in}}^{(l)}
\mathbf{u}(t)
+
W_p^{(l-1)}\mathbf{x}^{(l-1)}(t)
+
W_r^{(l)}\mathbf{x}^{(l)}(t-1)
\right),
\end{aligned}
\]

where \(L\) is the number of reservoir layers, \(W_p^{(l-1)}\) denotes the connection matrix from the \((l-1)\)-th to the \(l\)-th reservoir layer, and \(W_{bg}^{(l)}\) is applied only to the external input-to-reservoir connections.

The hyperparameter search space is

\[
\begin{aligned}
N &\in \{100,200,300,400,500\},\\
L &\in \{2,4,5,10\},\\
\rho &\in \{0.5,0.7,0.9,1.1,1.3,1.5\},\\
\alpha_{\mathrm{in}} &\in \{0.001,0.01,0.1,1.0\},\\
\alpha &\in \{0.001,0.01,0.1,1.0\},\\
P &\in \{-1,-0.9,\ldots,-0.1,0,0.1,\ldots,0.9,1\}.
\end{aligned}
\]

Here, \(N\) denotes the number of reservoir neurons per layer.

---

### 3. ES\(^{2}\)N

The state update of ES\(^{2}\)N is

\[
\mathbf{x}(t)
=
\beta
\tanh\left(
\rho W_r\mathbf{x}(t-1)
+
W_{bg}^{T}W_{\mathrm{in}}\mathbf{u}(t)
+
\mathbf{b}
\right)
+
(1-\beta)O\mathbf{x}(t-1),
\]

where \(O\) is an orthogonal matrix, \(\beta\) is the proximity parameter, and \(\mathbf{b}\) denotes the reservoir bias vector.

The hyperparameter search space is

\[
\begin{aligned}
N &\in \{100,200,300,400,500\},\\
\rho &\in \{0.5,0.7,0.9,1.1,1.3,1.5\},\\
\alpha_{\mathrm{in}} &\in \{1.0,0.1,0.01,0.001\},\\
\beta &\in \{0.001,0.01,0.1,1.0\},\\
\alpha_{\mathrm{bias}} &\in \{1.0,0.1,0.01,0.001\},\\
P &\in \{-1,-0.9,\ldots,-0.1,0,0.1,\ldots,0.9,1\}.
\end{aligned}
\]

Here, \(\alpha_{\mathrm{bias}}\) denotes the scaling factor of the reservoir bias vector.

---

### 4. MCI-ESN

MCI-ESN consists of two interacting reservoir modules. The state updates can be written as

\[
\mathbf{x}^{(1)}(t)
=
\tanh\left(
W_r^{(1)}\mathbf{x}^{(1)}(t-1)
+
W_{bg}^{(1)T}W_{\mathrm{in}}^{(1)}\mathbf{u}(t)
+
W_{12}\mathbf{x}^{(2)}(t-1)
\right),
\]

and

\[
\mathbf{x}^{(2)}(t)
=
\tanh\left(
W_r^{(2)}\mathbf{x}^{(2)}(t-1)
+
W_{bg}^{(2)T}W_{\mathrm{in}}^{(2)}\mathbf{u}(t)
+
W_{21}\mathbf{x}^{(1)}(t-1)
\right),
\]

where \(W_{12}\) and \(W_{21}\) denote the two interaction connections between the two reservoir modules. The proposed \(W_{bg}\) matrices are applied only to the external input-to-reservoir connections.

The hyperparameter search space is

\[
\begin{aligned}
N &\in \{100,200,300,400,500\},\\
\rho &\in \{0.5,0.7,0.9,1.1,1.3,1.5\},\\
\alpha_{\mathrm{in}} &\in \{0.001,0.01,0.1,1.0\},\\
P &\in \{-1,-0.9,\ldots,-0.1,0,0.1,\ldots,0.9,1\}.
\end{aligned}
\]

Here, \(N\) denotes the number of neurons in each reservoir module.

