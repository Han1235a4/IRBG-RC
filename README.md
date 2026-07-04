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
> **Note:** The references and detailed descriptions for the four datasets (**Hadcet** and **PEMS-BAY**) used in our experiments have been formally cited and elaborated in the main manuscript.
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

