<img width="196" height="85" alt="image" src="https://github.com/user-attachments/assets/3125cad7-9c8f-4b91-9373-dfe7eda2e2c8" /><img width="196" height="85" alt="image" src="https://github.com/user-attachments/assets/2569bb05-4e6b-4b45-a4dd-c1b1c01cd1c6" /># IRBG-RC
The dataset and code for the paper entitled "Reshaping Input-to-Reservoir Topologies in Reservoir Computing via Bipartite Graph Generation"
CPU i9 10980Xe
GPU 3090Ti, 
python 3.7
\\ Datasets
Lorenz system
$$\dot{x} = 10(y - x)$$
$$\dot{y} = x(28 - z) - y$$
$$\dot{z} = xy - \frac{8}{3}z$$
\\
4D hyperchaotic
$$\dot{x} = ax - yz + w$$
$$\dot{y} = xz - by$$
$$\dot{z} = xy - cz$$
$$\dot{w} = -y + d$$
,where $a = 8, b = 40, c = 15, d = -0.1$ and $[x_0, y_0, z_0, w_0]^T = [10, 1, 10, 1]^T$
