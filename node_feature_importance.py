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
    #p=1-m/N
    population_indices = np.arange(N)
    if rho_target>=0:
        #p=1-m/N
        m=np.round((1-rho_target)*N)
        sample_indices = rng.choice(population_indices, size=m.astype(int), replace=False)
        R_target_A =np.dot(np.arange(N), M0)
        R_target_A[np.sort(sample_indices)]=R_A_O[sample_indices]
    else:
        #p=m/N-1
        m=np.round((1+rho_target)*N)
        sample_indices = rng.choice(population_indices, size=m.astype(int), replace=False)
        R_target_A =np.dot(np.arange(N), M0_a)
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
    #计算储备池网络的能量 0 c传统的权重和     0:"weight_sum",
      # 1: "Degree Centrality", 2: "Eigenvector Centrality", 3: "PageRank",
      # 4: "Betweenness Centrality", 5: "Closeness Centrality", 6: "Coreness",
      # 7: "Clustered-local-degree (CLD)", 8: "Two-way Random Walk (RW - 框架)"
    F_res=calculate_node_centrality_from_matrix(R_network,metr_index)
    #计算转移矩阵，输出win
    M,R_target_A=generate_rank_controlled_bipartite(F_in,F_res,corr)
    pearson_r,spearman_rho=calculate_bipartite_assortativity(M,F_in, F_res)
    return np.dot(M.T,W_in),pearson_r,spearman_rho,R_target_A







