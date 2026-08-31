# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 16:43:37 2025

@author: 1235a4
"""
import networkx as nx
import numpy as np
from scipy.linalg import eigvalsh # 用于计算拉普拉斯矩阵的特征值
from scipy.stats import rankdata
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr

from typing import Tuple

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# def calculate_lle_rosenstein(data, tau=1, window=50, max_iter=100, plot=False):
#     """
#     计算多维时间序列的最大 Lyapunov 指数 (Rosenstein 算法变体)
#     包含边界检查修复，防止 crash。
    
#     参数:
#         data: (T, N) 数组，T是时间步，N是特征维度
#         tau: 采样时间间隔 (默认为1，如果是连续系统需乘以 dt)
#         window (Theiler window): 寻找最近邻时，排除时间上相邻点的窗口大小
#         max_iter: 追踪邻居发散的最大步数 (k_max)
#         plot: 是否绘图显示 log(divergence) 曲线
        
#     返回:
#         lle: 估计的最大 Lyapunov 指数
#     """
#     T, N = data.shape
    
#     # --- 1. 最近邻搜索 (Nearest Neighbor Search) ---
#     # 使用 KDTree 或 Brute Force 在 N 维空间寻找距离
#     # 我们多查几个邻居(n_neighbors=20)，以便在排除掉时间窗口内的点后还能找到邻居
#     nbrs = NearestNeighbors(n_neighbors=20, algorithm='auto', metric='euclidean').fit(data)
#     distances, indices = nbrs.kneighbors(data)
    
#     # --- 2. 追踪发散 (Track Divergence) ---
#     divergence = np.zeros(max_iter)
#     counts = np.zeros(max_iter)
    
#     valid_pairs = 0
    
#     # 只需要遍历到 T - max_iter，因为剩下的点没有足够的未来去追踪
#     for i in range(T - max_iter):
        
#         # 寻找 i 的最近邻 j，且满足时间间隔 |i-j| > window (Theiler 窗)
#         found_neighbor = False
#         nearest_idx = -1
#         initial_dist = 0
        
#         # 遍历 i 的候选邻居列表
#         for k_idx, neighbor_idx in enumerate(indices[i]):
#             if np.abs(i - neighbor_idx) > window:
#                 nearest_idx = neighbor_idx
#                 initial_dist = distances[i][k_idx]
#                 found_neighbor = True
#                 break
        
#         # [修复 1]: 如果没找到合法的邻居，或者初始点重合(dist=0)，跳过
#         if not found_neighbor or initial_dist < 1e-12:
#             continue

#         # [修复 2 - 关键]: 检查邻居 nearest_idx 是否太靠近数据末尾
#         # 如果邻居后面没有 max_iter 这么长的数据，无法进行广播计算，必须跳过
#         if nearest_idx + max_iter > T:
#             continue
            
#         valid_pairs += 1
        
#         # 向量化计算: 获取两条轨迹片段
#         # shape 都是 (max_iter, N)
#         traj_i = data[i : i+max_iter]
#         traj_j = data[nearest_idx : nearest_idx+max_iter]
        
#         # 计算对应时刻的欧氏距离
#         dists = np.linalg.norm(traj_i - traj_j, axis=1)
        
#         # 累加对数距离 (加一个小量防止 log(0))
#         divergence += np.log(dists + 1e-12)
#         counts += 1

#     if valid_pairs == 0:
#         print("错误: 未找到任何有效的邻居对，请检查 window 设置或数据长度。")
#         return 0.0

#     # --- 3. 计算平均对数发散度 ---
#     # 避免除以0
#     avg_divergence = divergence / (counts + 1e-10)
#     time_axis = np.arange(max_iter) * tau
    
#     # --- 4. 线性拟合 (Estimation) ---
#     # 这里自动选取中间段进行拟合，实际科研中建议看图手动调整范围
#     fit_start = int(max_iter * 0.1) # 去掉最开始的过渡期
#     fit_end = int(max_iter * 0.5)   # 通常混沌发散在前半段最明显，后面会饱和
    
#     coeffs = np.polyfit(time_axis[fit_start:fit_end], avg_divergence[fit_start:fit_end], 1)
#     lle = coeffs[0]
    
#     if plot:
#         plt.figure(figsize=(8, 5))
#         plt.plot(time_axis, avg_divergence, label='Average Log Divergence', linewidth=2)
        
#         # 绘制拟合线
#         fit_y = np.polyval(coeffs, time_axis[fit_start:fit_end])
#         plt.plot(time_axis[fit_start:fit_end], fit_y, 'r--', lw=2, label=f'Linear Fit (LLE={lle:.4f})')
        
#         # 标记拟合区域
#         plt.axvspan(time_axis[fit_start], time_axis[fit_end], color='yellow', alpha=0.1, label='Fit Range')
        
#         plt.xlabel('Evolution Time (k * tau)')
#         plt.ylabel('<ln(divergence)>')
#         plt.title(f'Largest Lyapunov Exponent Estimation\n(Valid Pairs: {valid_pairs})')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.show()
        
#     return lle




def calculate_lle(traj_x, traj_y, dt=1.0, saturation_threshold=2,plot=False):
    """
    专门针对有界数据 (如 [-1, 1]) 的 Lyapunov 指数计算。
    自动检测饱和，避免拟合到平坦区域。

    参数:
        traj_x, traj_y: 轨迹数据
        saturation_threshold: 距离阈值。
                              对于 [-1, 1] 的数据，建议设为 0.1 到 0.2 之间。
                              一旦距离超过这个值，就认为不再是"微小扰动"，停止拟合。
    """
    # --- 1. 数据准备 ---
    traj_x = np.array(traj_x, dtype=float)
    traj_y = np.array(traj_y, dtype=float)
    
    if traj_x.ndim == 1:
        traj_x = traj_x[:, np.newaxis]
        traj_y = traj_y[:, np.newaxis]

    # --- 2. 计算距离 ---
    delta = traj_x - traj_y
    distances = np.linalg.norm(delta, axis=1)
    
    # 过滤掉距离为 0 的点 (避免 log(0))
    # 同时也只保留距离小于饱和阈值的点用于拟合！
    # valid_mask: 既要大于0，又要还没有"撞墙"(饱和)
    valid_mask = (distances > 1e-12) 
    
    # 如果有效点太少，说明扰动一开始就太大了，或者已经饱和了
    # if np.sum(valid_mask) < 5:
    #     print("错误: 有效线性区太短，无法拟合。可能是初始扰动过大或阈值设置过小。")
    #     return 0.0

    # --- 3. 准备拟合数据 ---
    time_axis = np.arange(len(distances)) * dt
    log_dist = np.full(len(distances), np.nan)
    
    # 只计算未饱和区域的 log
    log_dist[valid_mask] = np.log(distances[valid_mask])
    
    # 提取用于拟合的 x 和 y
    x_fit = time_axis[valid_mask]
    y_fit = log_dist[valid_mask]
    
    # --- 4. 再次截断 (可选但推荐) ---
    # 通常指数分离发生在最开始。即使距离没到阈值，如果时间太长也不对了。
    # 这里强制只取前 50% 的时间窗口内的有效点，防止后期虽然距离小但实际上是收敛的情况
    limit_idx = int(len(distances) * 0.5)
    mask_time = x_fit < (limit_idx * dt)
    
    x_fit = x_fit[mask_time]
    y_fit = y_fit[mask_time]
    
    if len(x_fit) < 2:
        return 0.0

    # --- 5. 线性拟合 ---
    coeffs = np.polyfit(x_fit, y_fit, 1)
    lambda_1 = coeffs[0]
    
    # --- 6. 绘图 (带有物理意义的辅助线) ---
    if plot:
        plt.figure(figsize=(10, 6))
        
        # 画出完整距离曲线 (灰色背景)
        full_log_dist = np.log(distances + 1e-16)
        plt.plot(time_axis, full_log_dist, color='gray', alpha=0.3, label='Full Evolution')
        
        # 画出拟合段 (蓝色)
        plt.plot(x_fit, y_fit, 'b-', linewidth=2, label='Linear Region (Fit)')
        
        # 画拟合直线 (红色虚线)
        plt.plot(x_fit, np.polyval(coeffs, x_fit), 'r--', label=f'Slope = {lambda_1:.4f}')
        
        # 画饱和阈值线
        plt.axhline(np.log(saturation_threshold), color='orange', linestyle=':', label='Saturation Threshold')
        
        plt.xlabel('Time')
        plt.ylabel('ln(Distance)')
        plt.title(f'LLE Calculation (Bounded Data [-1, 1])\nThreshold={saturation_threshold}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    return lambda_1




# A. 谱/能量辅助函数 (用于 Q-LC)
def calculate_laplacian_energy(G):
    """计算图的拉普拉斯能量：拉普拉斯矩阵特征值绝对值之和"""
    # 确保图非空
    if not G or G.number_of_nodes() == 0:
        return 0.0
    L = nx.laplacian_matrix(G).todense()
    # 使用 scipy.linalg.eigvalsh (更快且适用于对称矩阵) 求解特征值
    eigenvalues = eigvalsh(L)
    return np.sum(np.abs(eigenvalues))

# B. 引力辅助函数
def get_node_properties(G):
    """获取节点质量 (k_i) 和最短路径距离矩阵 (d)"""
    mass = dict(G.degree())
    # 计算所有节点对的最短路径距离
    d_matrix_dict = dict(nx.shortest_path_length(G))
    return mass, d_matrix_dict

# C. 引力中心性 (GC)
def calculate_gc(G, mass, d_matrix_dict, alpha=1.0):
    """计算 Gravity Centrality (GC): GC_i = sum(m_i * m_j / d(i, j)^alpha)"""
    gc_scores = {}
    nodes = list(G.nodes())
    for i in nodes:
        gc_i = 0
        m_i = mass[i]
        for j in nodes:
            if i == j: continue
            
            # 获取距离，如果不可达，距离为无穷大 (np.inf)
            dist = d_matrix_dict.get(i, {}).get(j, np.inf)
            
            if dist != np.inf and dist > 0:
                m_j = mass[j]
                # 引力项
                gc_i += (m_i * m_j) / (dist ** alpha)
        gc_scores[i] = gc_i
    return gc_scores

# D. 引力紧密度中心性 (GCC)
def calculate_gcc(G, mass, d_matrix_dict, alpha=1.0):
    """计算 Gravity Closeness Centrality (GCC): GCC_i = 1 / sum(m_j / d(i, j)^alpha)"""
    gcc_scores = {}
    nodes = list(G.nodes())
    for i in nodes:
        S_i = 0
        for j in nodes:
            if i == j: continue
            
            dist = d_matrix_dict.get(i, {}).get(j, np.inf)
            
            if dist != np.inf and dist > 0:
                m_j = mass[j]
                # 引力衰减项
                S_i += m_j / (dist ** alpha)
        
        # GCC = 1 / S_i
        gcc_scores[i] = 1.0 / S_i if S_i > 0 else 0.0
    return gcc_scores

# --- 3. 集成中心性计算函数 ---
#第一个参数加权邻接矩阵  第二个参数 使用哪一个中心性指标
import numpy as np
import networkx as nx
def get_node_properties(G, weight='weight'):
    """
    考虑权重的预处理函数
    """
    # 1. 计算质量 (Mass)：使用点强度 (Strength) 代替单纯的度 (Degree)
    # 如果 weight=None，则退化为普通度数
    mass = dict(G.degree(weight=weight))
    
    # 2. 计算距离矩阵 (Distance Matrix)：使用 Dijkstra 算法
    # 注意：如果权重代表“相似度”或“强度”，计算最短路径前应取倒数
    # 这里假设 G 中的 weight 已经代表了“距离/代价”
    d_iter = nx.all_pairs_dijkstra_path_length(G, weight=weight)
    d_matrix_dict = dict(d_iter)
    
    return mass, d_matrix_dict

def calculate_node_centrality_from_matrix(adj_matrix_np, metric_index=0, f_beta=1.0, f_p=1.0, f_type='tanh'):
    """
    扩展后的节点中心性计算函数
    :param f_beta: tanh 函数的增益系数
    :param f_p: 幂函数的指数 (若使用 power 类型)
    :param f_type: 非线性函数类型 'tanh' 或 'power'
    """
    
    # === 步骤 1: 矩阵到 NetworkX 图的转换 ===
    # 注意：对于特征向量类算法，DiGraph 会保留矩阵的非对称权重
    G = nx.from_numpy_array(adj_matrix_np, create_using=nx.DiGraph)
    NODE_LIST = list(G.nodes())
    N = len(NODE_LIST)

    # --- 扩展指标映射表 ---
    metrics_map = {
        0: "weight_sum",
        1: "Degree Centrality", 2: "Eigenvector Centrality", 3: "PageRank",
        4: "Betweenness Centrality", 5: "Closeness Centrality", 6: "Coreness",
        7: "Clustered-local-degree (CLD)", 8: "Two-way Random Walk (RW - 框架)",
        9: "Quasi-Laplacian Centrality (Q-LC)", 
        10: "Gravity Centrality (GC, m=k, alpha=1)", 
        11: "Gravity Closeness Centrality (GCC, m=k, alpha=1)",
        12: "Path length sum",
        13: f"f-Eigenvector Centrality (f-EC, type={f_type})",
        14: "Katz Centrality"
    }
    # 除了4 5 10 11 12 之外，其他几个比较相似 其中 4 5 和 12 相似
    if metric_index not in metrics_map:
        # print(f"错误：无效的指标索引 {metric_index}。")
        return None

    # print(f"--- 正在计算: {metrics_map[metric_index]} ---")
    
    scores_dict = {}
    
    # --- 核心计算逻辑 ---
    
    # 1-6 经典指标
    if metric_index in [1, 2, 3, 4, 5, 6]:
        if metric_index == 1: scores_dict = nx.degree_centrality(G)
        elif metric_index == 2: scores_dict = nx.eigenvector_centrality(G, max_iter=10000, weight='weight',tol=1e-4)
        elif metric_index == 3: scores_dict = nx.pagerank(G, alpha=0.85, weight='weight')
        elif metric_index == 4: scores_dict = nx.betweenness_centrality(G, weight='weight')
        elif metric_index == 5: scores_dict = nx.closeness_centrality(G, distance='weight')
        elif metric_index == 6: scores_dict = nx.core_number(G) #本质上 不需要用它考虑权重
        
    elif metric_index == 7: # CLD
        degrees = dict(G.degree())
        clustering = nx.clustering(G)
        scores_dict = {node: degrees[node] * clustering[node] for node in G.nodes()}
        
    elif metric_index == 8: # RW (框架)
        A = adj_matrix_np
        A_sq = A @ A
        scores_dict = {node: A_sq[node, node] for node in NODE_LIST}
        
    elif metric_index == 9: # Q-LC
        G = nx.from_numpy_array(adj_matrix_np, create_using=nx.Graph) #9 只适用于 无向网络
        # 假设 calculate_laplacian_energy 已定义
        E_original = calculate_laplacian_energy(G)
        for node in G.nodes():
            G_removed = G.copy()
            G_removed.remove_node(node)
            E_removed = calculate_laplacian_energy(G_removed)
            scores_dict[node] = E_original - E_removed
            
    elif metric_index in [10, 11]: # 引力系列
        mass, d_matrix_dict = get_node_properties(G, weight='weight')
        if metric_index == 10: 
            scores_dict = calculate_gc(G, mass, d_matrix_dict, alpha=1.0)
        elif metric_index == 11:
            scores_dict = calculate_gcc(G, mass, d_matrix_dict, alpha=1.0)

    elif metric_index == 12: # Path length sum
        # 计算每个节点到所有其他节点的最短路径之和
        path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
        for node in G.nodes():
            scores_dict[node] = sum(path_lengths[node].values())

    elif metric_index == 13: # f-Eigenvector Centrality (核心实现)
        # 1. 初始值：归一化的全 1 向量
        x = np.ones(N) / np.sqrt(N)+10
        max_iter = 1000
        tol = 1e-9
        
        A = adj_matrix_np # 直接使用矩阵提高效率
        
            # 1. 定义非线性函数 f
        if f_type == 'tanh':
            # RC中最常用，范围 [-1, 1]
            func = np.tanh
        elif f_type == 'sigmoid':
            # 范围 [0, 1]
            func = lambda x: 1 / (1 + np.exp(-x))
        elif f_type == 'relu':
            # 范围 [0, inf)
            func = lambda x: np.maximum(0, x)
        elif f_type == 'abs':
            # 取绝对值，用于处理带符号网络
            func = np.abs
        elif f_type == 'linear':
            # 退化为普通特征向量中心性
            func = lambda x: x
        else:
            raise ValueError(f"Unknown f_type: {f_type}")
    
        # 2. 初始化状态向量 x
        # 注意：不能初始化为全0，因为对于 tanh(0)=0，全0是平凡解，系统动不起来
        np.random.seed(42) 
        x = np.random.uniform(-0.5, 0.5, N)
        
        # 3. 迭代求解不动点 x = f(c * A * x + bias)
        for i in range(max_iter):
            x_prev = x.copy()
            
            # 计算线性部分: Linear summation
            linear_part = 1 * np.dot(A, x) + 0
            
            # 应用非线性: Apply f
            x_new = func(linear_part)
            
            # 4. 检查收敛性 (L2 范数)
            diff = np.linalg.norm(x_new - x_prev)
            
            # 更新 x
            x = x_new
            if diff < tol:
              break
          
        

    elif metric_index==14:
        phi = max(nx.adjacency_spectrum(G, weight='weight')).real
        alpha_val = 1 / phi *0.9  # 稍微小于最大特征值的倒数
        
        centrality_vector= nx.katz_centrality(G, alpha=0.1, beta=1.0, normalized=True,max_iter=10000)

    # --- 结果处理 ---
    if metric_index == 0:
        centrality_vector = np.sum(adj_matrix_np, axis=1) # 行和
    elif  metric_index == 13:
        centrality_vector=x
    else:
        centrality_vector = np.array([scores_dict.get(node, 0.0) for node in NODE_LIST])
    
    return centrality_vector


# # 1. 计算 Q-LC (索引 9)
# qlc_vector = calculate_node_centrality_from_matrix(ADJ_MATRIX_NP, metric_index=9)
# if qlc_vector is not None:
#     print(f"\nQ-LC 向量 (索引 9):\n{qlc_vector}")


    
    
#随机种子
Seed=42
rng=np.random.RandomState(Seed)



def generate_rank_controlled_bipartite(f_A: np.ndarray, f_B: np.ndarray, rho_target: float) -> np.ndarray:
    """
    基于线性插值目标等级和贪婪算法，生成具有目标相关系数 rho_target 的二部图邻接矩阵 M。
    M[i, j] = 1 表示 A[i] 连接到 B[j]。

    Args:
        f_A (np.ndarray): 集合 A 的 N 个一维特征向量 (行标)。 输入的特征
        f_B (np.ndarray): 集合 B 的 N 个一维特征向量 (列标)。 网络的特征
        rho_target (float): 目标相关系数 [-1, 1]。

    Returns:
        np.ndarray: N x N 邻接矩阵 M (A 行, B 列)。
    """
    Seed=42
    rng=np.random.RandomState(Seed)
    N = len(f_A)
    if N != len(f_B):
        raise ValueError("集合 A 和 B 的特征向量长度必须相等 (N)。")
        
    rho_target = np.clip(rho_target, -1.0, 1.0)
    
    # --- 1. 计算等级 (Rank) ---
    # R_A/R_B: 1 为最小特征值，N 为最大特征值
    # 元素所在的序列等级
    R_A = rankdata(f_A).astype(int)
    R_B = rankdata(f_B).astype(int)
 
    #----2 计算最优的A 
    I_A = np.argsort(f_A) #f_A 中元素从小到大的原始索引
    I_B= np.argsort(f_B)
    M0 = np.zeros((N, N), dtype=int)
    M0[ I_A ,  I_B ] = 1
    R_A_O=np.dot(np.arange(N), M0)
    
    # --- 3. 计算最差的A 
    
    # R_A_anti: 异配对应等级 N + 1 - R_A
    I_B= np.argsort(-f_B)
    M0_a = np.zeros((N, N), dtype=int)
    M0_a[ I_A ,  I_B ] = 1
    R_A_anti = np.dot(np.arange(N), M0_a)
    
    # 权重 alpha = (1 + rho_target) / 2
    alpha = (1 + rho_target) / 2
    # --- 4.计算需要均匀排序的样本数

    population_indices = np.arange(N)
    q = 1.0 - (np.abs(rho_target) * (N - 1)) / N
    random_vals = rng.rand(N)
    perturb_mask = random_vals < q
    sample_indices = population_indices[perturb_mask]
    sample_indices = rng.permutation(sample_indices)
    m = len(sample_indices)

    if rho_target>0:

        R_target_A =np.dot(np.arange(N), M0)
        if rho_target<1 and m>1:
            R_target_A[np.sort(sample_indices)]=R_A_O[sample_indices]
    else:
        R_target_A =np.dot(np.arange(N), M0_a)
        if rho_target>-1 and m>1:
            R_target_A[np.sort(sample_indices)]=R_A_anti[sample_indices]

    
    M = np.zeros((N, N), dtype=int)
    # row_ind 是 A 的行索引 (0到N-1)，col_ind 是 B 的匹配列索引
    M[R_target_A,  np.arange(N)] = 1
    
    return M,R_target_A

# --- 示例验证 ---
def calculate_bipartite_assortativity(M: np.ndarray, f_A: np.ndarray, f_B: np.ndarray) -> float:
    """
    计算基于节点属性（特征值）的二部图同配系数 r。 
    对于加权网络就是皮尔逊相关系数

    Args:
        M (np.ndarray): N x N 的二部图邻接矩阵 (行A, 列B)。
        f_A (np.ndarray): 集合 A 的特征向量。
        f_B (np.ndarray): 集合 B 的特征向量。

    Returns:
        float: 同配系数 r (范围在 [-1, 1])。
    """
    
    vector_A=np.dot(f_A, M)
    vector_B=f_B
    
    pearson_r, pearson_p_value = pearsonr(vector_A, vector_B)
    
    # --- 2. 斯皮尔曼秩相关系数 (Spearman's rho_s) ---
    # 衡量单调关系 (即等级/排序的相关性)
    
    # spearmanr 返回两个值：rho_s值 和 p值
    spearman_rho, spearman_p_value = spearmanr(vector_A, vector_B)
    return pearson_r,spearman_rho

def generate_transfor(W_in=None,R_network=None,metr_index=0,corr=0):
    # 计算输入矩阵的能量
    F_in=np.linalg.norm(W_in, axis=1) #使用模来计算不同的能量
    noise = rng.uniform(-10e-10, 10e-10, size=F_in.shape[0])
    #计算储备池网络的能量 0 c传统的权重和     0:"weight_sum",
      # 1: "Degree Centrality", 2: "Eigenvector Centrality", 3: "PageRank",
      # 4: "Betweenness Centrality", 5: "Closeness Centrality", 6: "Coreness",
      # 7: "Clustered-local-degree (CLD)", 8: "Two-way Random Walk (RW - 框架)"
    F_res=calculate_node_centrality_from_matrix(R_network,metr_index)
    #计算转移矩阵，输出win
    M,R_target_A=generate_rank_controlled_bipartite(F_in+noise,F_res+noise,corr)
    pearson_r,spearman_rho=calculate_bipartite_assortativity(M,F_in+noise, F_res+noise)
    return np.dot(M.T,W_in),pearson_r,spearman_rho,R_target_A

# N_nodes = 5
# f_A = rng.rand(300, 1)[:,0] # A 的节点重要程度特征
# f_B = rng.rand(300, 1)[:,0]# B 的节点重要程度特征

# # 目标 1: 强同配 (rho_target = 0.9)
# for i in range(20):
    
#     M_homophily = generate_rank_controlled_bipartite(f_A, f_B, rho_target=np.linspace(-1, 1,20)[i])
#     print("--- 目标 rho = 0.9 (强同配) 的匹配矩阵 M ---")
#     M_homophily
#     print(calculate_bipartite_assortativity(M_homophily,f_A, f_B))

# # 目标 2: 中性/随机 (rho_target = 0.0)
# M_neutral = generate_rank_controlled_bipartite(f_A, f_B, rho_target=0.0)
# print("\n--- 目标 rho = 0.0 (中性) 的匹配矩阵 M ---")
# print(M_neutral)
# calculate_bipartite_assortativity(M_neutral,f_A, f_B)

# # 目标 3: 强异配 (rho_target = -0.9)
# M_heterophily = generate_rank_controlled_bipartite(f_A, f_B, rho_target=-1)
# print("\n--- 目标 rho = -0.9 (强异配) 的匹配矩阵 M ---")
# print(M_heterophily)
# calculate_bipartite_assortativity(M_heterophily,f_A, f_B)
def calculate_instantaneous_entropy(x_t: np.ndarray) -> float:
    """
    根据公式 (6) 计算在单个时间步 t 的瞬时状态熵 H(t)。
    
    这个函数实现了 Renyi's quadratic entropy 的一个高效估计器。

    Args:
        x_t (np.ndarray): 一个一维NumPy数组，代表在时间 t 的水库状态向量 (reservoir state)。
                          形状为 (N_R,)，其中 N_R 是水库神经元的数量。

    Returns:
        float: 在时间步 t 的瞬时熵 H(t)。
    """
    # N_R 是水库的大小（神经元数量）
    N_R = x_t.shape[0]
    if N_R == 0:
        return 0.0

    # --------------------------------------------------------------------------
    # 步骤 1: 计算高斯核的宽度 (kernel width)
    # 根据论文描述，核的尺寸(宽度)是通过将瞬时水库激活的标准差缩小0.3倍得到的。
    # --------------------------------------------------------------------------
    std_dev = np.std(x_t)
    # 避免标准差为0（如果所有激活值都相同）导致除零错误
    if std_dev == 0:
        return 0.0 
    kernel_width = 0.3 * std_dev  #论文里面给的

    # --------------------------------------------------------------------------
    # 步骤 2: 高效计算双重求和 (∑∑ K(...))
    # 我们使用NumPy的广播(broadcasting)功能来避免使用两个for循环，这会快得多。
    # --------------------------------------------------------------------------
    # 1. 将 x_t 扩展为列向量和行向量
    x_col = x_t[:, np.newaxis]  # 形状变为 (N_R, 1)
    x_row = x_t[np.newaxis, :]  # 形状变为 (1, N_R)
    
    # 2. 计算所有激活值两两之间的差值矩阵
    diff_matrix = x_col - x_row   # 形状为 (N_R, N_R)
    
    # 3. 对差值矩阵的每个元素应用高斯核函数 K(u) = exp(-u^2 / (2 * sigma^2))
    #    这里的 sigma 就是 kernel_width
    kernel_matrix = np.exp(-np.square(diff_matrix) / (2 * np.square(kernel_width)))
    
    # 4. 对整个核矩阵求和，即为双重求和的结果
    double_summation = np.sum(kernel_matrix)

    # --------------------------------------------------------------------------
    # 步骤 3: 根据公式 (6) 计算最终的 H(t)
    # --------------------------------------------------------------------------
    # 计算括号内的平均值
    argument = double_summation / (N_R ** 2)
    
    # 取负对数
    H_t = -np.log(argument)
    
    return H_t

#传统的富集程度度量指标
def  richness(R_state=None):
    # ASE 
    T = R_state.shape[0]
    if T == 0:
        return 0.0
        
    # 存储每个时间步的瞬时熵 H(t)
    h_values = []
    
    # 遍历每一个时间步的状态向量
    for t in range(T):
        x_t = R_state[t, :]
        h_t = calculate_instantaneous_entropy(x_t)
        h_values.append(h_t)
        
    # 计算所有 H(t) 的算术平均值
    ASE= np.mean(h_values) #越高富集程度越高 
    
    U, s, Vt = np.linalg.svd(R_state, full_matrices=False)
    LUD=np.sum(np.cumsum(s/np.sum(s))<0.9) #论文里面是0.9     越高富集程度越高 
    
    CN=np.max(s)/np.min(s) #条件数，越小越好  
    
    
    # return {"ASE":ASE,
    #         "LUD":LUD,
    #         "CN":CN}
    return ASE,LUD,CN



import numpy as np
import nolds
import pyentrp.entropy as ent
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from scipy.stats import linregress

def calc_correlation_dimension(data, emb_dim=None, tau=1, plot=True):
    """
    计算关联维数 (D2)。
    - 模式 A (直接计算): data 为多维数组 (N, D)，emb_dim=None。直接计算几何维数。
    - 模式 B (时间延迟嵌入): data 为一维序列 (N,)，指定 emb_dim > 1。先进行相空间重构，再计算。
    
    参数:
    data: 输入数据。可以是 1D 时间序列，也可以是 2D/ND 坐标点。
    emb_dim (int): 嵌入维数 (Embedding Dimension)。
                   如果为 None 或 1，则不进行嵌入，直接使用原始数据维度。
    tau (int): 延迟时间 (Time Delay)。仅在 emb_dim > 1 时有效。默认为 1。
    """
    
    data = np.array(data)
    
    # --- 1. 嵌入处理逻辑 (Phase Space Reconstruction) ---
    if emb_dim is not None and emb_dim > 1:
        # 如果要求嵌入，首先确保输入是一维序列
        if data.ndim > 1 and data.shape[1] > 1:
            print("警告: 检测到多维输入但请求了嵌入。将自动展平为 1D 序列进行处理。")
            data = data.flatten()
            
        N = len(data)
        # 计算可用的向量数量
        # 例子: 10个点, m=3, tau=1 -> 能生成 [0,1,2]...[7,8,9] 共 8个向量
        n_vectors = N - (emb_dim - 1) * tau
        
        if n_vectors <= 0:
            raise ValueError(f"数据太短，无法满足 m={emb_dim}, tau={tau} 的嵌入要求。")
            
        # 构建嵌入矩阵 (N_vectors, emb_dim)
        # 使用 numpy 的切片技巧快速构建
        embedded_data = np.zeros((n_vectors, emb_dim))
        for i in range(emb_dim):
            # 每一列是原序列向后移动 i*tau 的版本
            embedded_data[:, i] = data[i*tau : i*tau + n_vectors]
            
        data = embedded_data
        print(f"已执行时间延迟嵌入: m={emb_dim}, tau={tau}. 重构后形状: {data.shape}")
        
    else:
        # 不进行嵌入，处理原始形状
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        print(f"未使用嵌入 (直接计算). 输入数据形状: {data.shape}")

    # --- 2. 计算距离 (核心算法) ---
    # pdist 自动处理任意维度的欧氏距离
    dists = pdist(data, metric='euclidean')
    
    # 排除 0 距离
    dists = dists[dists > 0]
    
    # --- 3. 设定 r 范围 ---
    r_min, r_max = np.min(dists), np.max(dists)
    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), 50)
    
    # --- 4. 计算关联积分 C(r) ---
    C_r = []
    total_pairs = len(dists)
    
    for r in r_vals:
        count = np.sum(dists < r)
        C_r.append(count / total_pairs)
    
    C_r = np.array(C_r)
    
    # --- 5. 寻找线性标度区并拟合 ---
    # 排除无效值
    valid_mask = (C_r > 0) & (C_r < 1)
    log_r = np.log10(r_vals)[valid_mask]
    log_Cr = np.log10(C_r)[valid_mask]
    
    # 自动截取中间线性段 (C(r) 在 0.005 ~ 0.5 之间)
    fit_mask = (np.power(10, log_Cr) > 0.005) & (np.power(10, log_Cr) < 0.5)
    
    # 鲁棒性检查：如果点太少，回退到取中间段
    if np.sum(fit_mask) < 3:
        mid = len(log_r) // 2
        start, end = max(0, mid - 5), min(len(log_r), mid + 5)
        fit_mask = np.zeros(len(log_r), dtype=bool)
        fit_mask[start:end] = True
        print("提示: 自动寻找线性区未找到足够点，已回退到中心区域拟合。")

    x_fit = log_r[fit_mask]
    y_fit = log_Cr[fit_mask]
    
    if len(x_fit) < 2:
        print("错误: 无法找到足够的线性区域进行拟合。")
        return 0.0

    slope, intercept, _, _, _ = linregress(x_fit, y_fit)
    
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(log_r, log_Cr, 'o-', markersize=4, label='Data (log-log)', alpha=0.6)
        plt.plot(x_fit, slope*x_fit + intercept, 'r--', 
                 linewidth=2, label=f'Fit Line (Slope = {slope:.3f})')
        
        title_str = f'Correlation Dimension (D2 = {slope:.3f})\n'
        if emb_dim and emb_dim > 1:
            title_str += f'Method: Embedding (m={emb_dim}, tau={tau})'
        else:
            title_str += f'Method: Direct Input (Dim={data.shape[1]})'
            
        plt.xlabel('log10(r)')
        plt.ylabel('log10(C(r))')
        plt.title(title_str)
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.show()
        
    return slope


def hjorth_params(data):
    """
    计算时间序列的 Hjorth 参数
    
    返回 (tuple):
        activity:   信号的方差 (代表能量/功率)
        mobility:   信号频率的近似估计
        complexity: 信号波形的复杂度 (越接近1越像正弦波，越高越复杂/频带越宽)
    """
    x = np.array(data)
    
    # 1. 计算一阶差分 (x') 和 二阶差分 (x'')
    dx = np.diff(x)
    ddx = np.diff(dx)
    
    # 2. 计算方差 (注意: 使用 var 而不是 std，Hjorth 定义基于方差)
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    
    # 3. 计算三个指标
    activity = var_x
    
    # 防止除以0的保护
    if var_x == 0:
        mobility = 0
    else:
        mobility = np.sqrt(var_dx / var_x)
        
    if mobility == 0 or var_dx == 0:
        complexity = 0
    else:
        # Complexity = Mobility(dx) / Mobility(x)
        mobility_deriv = np.sqrt(var_ddx / var_dx)
        complexity = mobility_deriv / mobility
        
    return activity, mobility, complexity

def run_nonlinear_analysis(ts,emb_dim=1,tau=1,matrix_dim=3):
    # print("开始非线性指标计算...")
    
    # --- 2. 最大李雅普诺夫指数 (LMax) ---
    # 使用 nolds.lyap_r (Rosenstein 算法)
    try:
        lmax =nolds.lyap_e(ts, emb_dim=emb_dim, tau=tau,matrix_dim=matrix_dim)
        # print(f"最大李雅普诺夫指数 (LMax): {lmax:.4f}")
    except Exception as e:
        lmax = np.nan
        # print(f"LMax 计算出错: {e}")

    # --- 3. 相关维数 (D2) ---
    # 使用 nolds.corr_dim
    try:
        d2 = nolds.corr_dim(ts, emb_dim=emb_dim,lag=tau)
        # print(f"相关维数 (D2): {d2:.4f}")
    except Exception as e:
        d2 = np.nan
        # print(f"D2 计算出错: {e}")

    # --- 4. 多尺度熵 (MSE) - 使用 pyentrp ---
    # 注意：pyentrp 的参数是位置参数，不能写 m=...
    # 参数顺序: (序列, 嵌入维数m, 容限r)
    # try:
    #     m = 2
    #     r = 0.2 * np.std(ts)
    #     max_scale = 20
        
    #     # 调用 pyentrp 的 multiscale_entropy
    #     # 它返回一个列表，包含从尺度 1 到 max_scale 的熵值
    #     mse_values = ent.multiscale_entropy(ts, m, r, max_scale)
    #     # print("多尺度熵 (MSE) 计算完成。")
    # except Exception as e:
    #     mse_values = None
        # print(f"MSE 计算出错: {e}")

    # return lmax, d2, mse_values
    return d2


