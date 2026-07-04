
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:48:39 2024


@author: Administrator
"""


import torch
import numpy as np
import pickle
from itertools import combinations, permutations
import random
import copy
import itertools
from scipy import signal
from matplotlib import pyplot as plt
import networkx as nx
import pandas as pd
import os  
from statsmodels.tsa.api import VAR
import pyinform
from scipy.special import comb
from scipy.stats import wasserstein_distance
from sklearn import linear_model
from scipy.linalg import orth
from sklearn.decomposition import PCA
import math
from sklearn.cluster import KMeans,Birch
from sklearn.manifold import TSNE
from scipy.integrate import odeint
import matplotlib.colors as colors
from itertools import cycle
from statsmodels.tsa.stattools import grangercausalitytests
#from utils import tsne
from networkx.algorithms import bipartite
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel,RFECV,RFE
from sklearn.linear_model import Lasso,LassoCV,MultiTaskLassoCV,RidgeCV,Ridge,ElasticNetCV,orthogonal_mp,OrthogonalMatchingPursuit,LinearRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.utils import (as_float_array, check_array, check_X_y, safe_sqr,
                     safe_mask)
from sklearn.utils.extmath import safe_sparse_dot, row_norms
from scipy import special, stats
from scipy.sparse import issparse
from torch.nn.parameter import Parameter
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sys import path
import warnings
from scipy.stats import ks_2samp
import scipy.stats
from sklearn.exceptions import ConvergenceWarning
from statsmodels.tsa.stattools import adfuller
from matplotlib import cm
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
# if use cuda
Data_precision=7
use_cuda = torch.cuda.is_available()
use_cuda=False
np.random.seed(42)
#AMI 计算
def compute_ami(time_series, max_lag=100, bins=100):
    """
    计算时间序列 x(t) 与其不同延迟项的 AMI（平均互信息）
    
    :param time_series: 输入时间序列 (1D array)
    :param max_lag: 最大时间延迟
    :param bins: 计算概率分布的直方图分箱数
    :return: AMI 值列表
    """
    time_series=time_series.reshape(-1)
    ami_values = []
    eps=1e-10
    for tau in range(1, max_lag + 1):
         x_t = time_series[:-tau]  # 当前值
         x_tau = time_series[tau:]  # 延迟后的值
 
         # 计算 2D 直方图
         hist_2d, x_edges, y_edges = np.histogram2d(x_t, x_tau, bins=bins, density=True)
         joint_prob = hist_2d + eps  # 避免 log(0)
 
         # 计算边际概率
         p_x = np.sum(joint_prob, axis=1, keepdims=True)
         p_y = np.sum(joint_prob, axis=0, keepdims=True)
 
         # 计算互信息
         ami = np.sum(joint_prob * np.log(joint_prob / (p_x @ p_y)))
         ami_values.append(ami)
    # 绘制 AMI 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_lag + 1), ami_values, marker='o', linestyle='-')
    plt.xlabel("Time Lag (τ)")
    plt.ylabel("AMI Value")
    plt.title("Average Mutual Information (AMI) vs Time Lag")
    plt.grid()
    plt.show()
#计算相关系数矩阵。

def analyze_corr_det_numpy(data_matrix: np.ndarray, alpha: float = 0.05) -> (float, np.ndarray):
    """
    接收一个NumPy矩阵，计算其过滤后的相关矩阵及其行列式。

    Args:
        data_matrix (np.ndarray): 输入的时间序列矩阵，形状为 (n_samples, n_features)。
        alpha (float): 统计检验的显著性水平，通常为 0.05。

    Returns:
        tuple: (determinant, filtered_corr_matrix)
               返回一个浮点数（行列式）和一个NumPy数组（过滤后的相关矩阵）。
    """
    print("--- 开始计算 (使用 NumPy + SciPy) ---")
    
    # 1. 获取变量数量
    if data_matrix.ndim != 2:
        raise ValueError("输入数据必须是一个二维矩阵 (n_samples, n_features)。")
    n_samples, n_features = data_matrix.shape
    
    # 2. 初始化相关矩阵和p值矩阵
    corr_matrix = np.eye(n_features)  # 对角线为1
    p_value_matrix = np.zeros((n_features, n_features))
    
    print("步骤1：正在通过循环计算相关系数和p值...")
    # 3. 通过循环计算成对相关性和p值
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # 提取两列数据 (两个一维向量)
            col1 = data_matrix[:, i]
            col2 = data_matrix[:, j]
            
            # 使用 scipy.stats.pearsonr 计算
            corr, p_val = pearsonr(col1, col2)
            
            # 对称地填充两个矩阵
            corr_matrix[i, j] = corr_matrix[j, i] = corr
            p_value_matrix[i, j] = p_value_matrix[j, i] = p_val
            
    # 4. 基于p值过滤相关矩阵
    filtered_corr_matrix = corr_matrix.copy()
    filtered_corr_matrix[p_value_matrix >= alpha] = 0
    np.fill_diagonal(filtered_corr_matrix, 1) # 确保对角线始终为1
    
    print("\n步骤2：过滤完成。")
    
    # 5. 计算过滤后矩阵的行列式
    determinant = np.mean(np.abs(filtered_corr_matrix))
    
    print("步骤3：行列式计算完成。")
    
    return determinant, filtered_corr_matrix





def Loss_Prediciton(Prediction=None,Real_data=None,Method=0):
    Loss=0
    Loss_Final=0
    mask = np.not_equal(Real_data, 0)
    mask = mask.astype('float32')
    mask /= np.mean(mask)
    # for i in range(L_pre):
    #     #RMSE
    #     if Method==1:
    #         Loss[i]=np.average((Prediction[i,:]-Real_data[i,:])**2)
    #     #MAE
    #     elif Method==0:
    #         Loss[i]=np.abs(Prediction[i,:]-Real_data[i,:])
    #     #MAPE
    #     else:
    #         Loss[i]=np.average(np.abs((Prediction[i,:]-Real_data[i,:])/Real_data[i,:]))
    #RMSE
    if Method==1:
        Loss = np.square(np.subtract(Prediction, Real_data)).astype('float32')
        Loss = np.nan_to_num(Loss * mask)
        Loss_Final=np.sqrt(np.mean(Loss))
        # Loss_Final=np.sqrt(np.sum(Loss)/L_pre)
    #MAE
    elif Method==0:
        Loss = np.abs(np.subtract(Prediction, Real_data)).astype('float32')
        Loss = np.nan_to_num(Loss * mask)
        Loss_Final=np.mean(Loss)
        # Loss_Final=np.sum(Loss)/L_pre
    #MAPE
    else:
        Loss = np.abs(np.divide(np.subtract(Prediction, Real_data).astype('float32')
                                ,Real_data))
        Loss= np.nan_to_num(mask * Loss)
        Loss_Final=np.mean(Loss)*100
    return Loss_Final


#计算预计输出与目标延迟目标之间的person相关系数
def memory_capacity(Prediction=None, Output=None, length=1000):
    sumR_coor=0
#    for i in range(1,length+1):
#        sumR_coor=sumR_coor+np.corrcoef(Prediction[-i], Output[-i])[0,1]
    if Prediction.shape[1]==1:
        sumR_coor=(np.corrcoef(Prediction[:length].T, Output[:length].T)[0,1])**2
    else:
        # for i in range(1,length+1):
        #     sumR_coor=sumR_coor+(np.corrcoef(Prediction[-i,:], Output[-i,:])[0,1])**2
        # sumR_coor=sumR_coor*1.0/length
        sumR_coor=0
        for i in range(Prediction.shape[1]):
            sumR_coor=sumR_coor+(np.corrcoef(Prediction[:length,i], Output[-length:,i])[0,1])**2
        if np.isnan(sumR_coor):
            sumR_coor=0
        sumR_coor=sumR_coor*1.0/Prediction.shape[1]
    return sumR_coor
    

    
def Network_initial(network_index=None,network_size=300,Seed=None,density=0.2,Depth=10,P_SM=0.3,MC_configure=None):

    if network_index==0:
        #network_name is "regular":
        K_number=int(network_size*density/2)
        rg=nx.circulant_graph(network_size, list(range(1, K_number+1)))#规则网络
        R_initial=nx.adjacency_matrix(rg).toarray()
    elif network_index==1:
        #network_name is "WS":
        K_number=int(network_size*density)
        rg=nx.random_graphs.watts_strogatz_graph(network_size,K_number,p=density*P_SM/(1-density*(1-P_SM)),seed=Seed)#WS 随机网络
        if rg.number_of_edges()==0: 
            R_initial=np.zeros((network_size,network_size))
        else:
            R_initial=nx.adjacency_matrix(rg).toarray()
    elif network_index==2:
        #network_name is "ER"
        rg=nx.erdos_renyi_graph(network_size,density,directed=False,seed=Seed)#ER
        if rg.number_of_edges()==0: 
            R_initial=np.zeros((network_size,network_size))
        else:
            R_initial=nx.adjacency_matrix(rg).toarray()
    elif network_index==3:
        #network_name is "DCG":
        rg=nx.erdos_renyi_graph(network_size,density,directed=True)
        # nx.is_directed_acyclic_graph(G)
        R_initial=nx.adjacency_matrix(rg).toarray()    
    elif network_index==4:
        #network_name is "DAG":
        if MC_configure is not None:
            xx=np.append(0,np.cumsum(MC_configure['number']))
            for i in range(xx.shape[0]-1):
                Reject_index=1
                for j in range(0,xx.shape[0]-1):
                    if len(MC_configure[i+1])==np.sum(np.isin(MC_configure[i+1],MC_configure[j+1]+1)):
                        Reject_index=0
                if Reject_index==1 and (MC_configure[i+1]!=1).all():
                    print("fail to construct the DAN under current Memory commnity strcutrue configuration")                    
                    Reject_index=2
            if Reject_index !=2:
                R_initial_0=np.zeros((network_size,network_size))
                for i in range(xx.shape[0]-1):
                    for j in range(xx.shape[0]-1):
                        if len(MC_configure[i+1])==np.sum(np.isin(MC_configure[i+1]+1,MC_configure[j+1])):
                            R_initial_0[xx[i]:xx[i+1],xx[j]:xx[j+1]]=1
                R_initial= np.triu(R_initial_0,1)
            else:
                R_initial=None
            
        else:
            
            rg=nx.turan_graph(network_size,Depth)
            x=nx.adjacency_matrix(rg).toarray()
            R_initial= np.triu(x,1)
        Real_density=density*Depth*(network_size-1)/(network_size*(Depth-1))
        if Real_density>0 and density<Real_density:
            R_initial[np.random.rand(*R_initial.shape) <= (1.0-Real_density)] = 0 
        R_initial= np.triu(R_initial,1)  
    elif network_index==5:
        #network_name is "DLG":
        R_initial=np.eye(network_size,k=1)
    return np.round(R_initial,Data_precision)
#定义训练输出矩阵的方法。
# def traing_Wout(train_data,R_state,index=0,alpha=0.01):
#     W_out= np.zeros((R_state.shape[1], train_data.shape[1]))
#     if index==0:
#         W_out=np.dot(np.linalg.pinv(R_state),train_data)#Dr*N
#     elif index==1:
#         W_out=np.dot(np.dot(np.linalg.pinv(np.dot(R_state.T,R_state)
#         +alpha*np.identity(R_state.shape[1])),R_state.T),train_data)
#     return W_out


class GroupRidge(BaseEstimator, RegressorMixin):
    def __init__(self, x=1.0, Node_MC=None):
        self.alpha = x
        self.Node_MC = Node_MC  # 分组标签，长度应与特征数一致

    def fit(self, R_state, train_data):
        if self.Node_MC is None:
            raise ValueError("Node_MC (group labels for features) must be set.")

        # 计算每个特征对应的 alpha
        values, counts = np.unique(self.Node_MC, return_counts=True)
        C_0=counts/max(counts)
        C_1=(C_0-min(C_0))/(max(C_0)-min(C_0))*(self.alpha*0.9)+self.alpha*0.1
        group_size_dict = dict(zip(values, C_1))
        alphas = np.array([
            group_size_dict[g] / max(counts) * self.alpha for g in self.Node_MC
        ])
        A = np.diag(alphas)

        RtR = R_state.T @ R_state
        RtY = R_state.T @ train_data

        self.W_out_ = np.linalg.pinv(RtR + A) @ RtY
        return self

    def predict(self, R_state):
        return R_state @ self.W_out_

    def get_params(self, deep=True):
        return {"x": self.alpha, "Node_MC": self.Node_MC}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

def traing_Wout(train_data,R_state,index=0,alpha=0.01,Node_MC=None,W_out0=None):
    R_state=R_state.reshape(R_state.shape[0],-1)
    train_data=train_data.reshape(train_data.shape[0],-1)
    W_out= np.zeros((R_state.shape[1], train_data.shape[1]))
    if index==0:
        W_out=np.dot(np.linalg.pinv(R_state),train_data)#Dr*N
    # elif index==7:
    #     scores = np.zeros_like(Node_MC, dtype=float)
    #     values, counts = np.unique(Node_MC, return_counts=True)
    #     for g in values:
    #         idx = Node_MC == g
    #         scores[idx] = 1.0 / len(values) / idx.sum()

        
        
    #     W_out=np.dot(np.diag(1/scores),np.dot(np.linalg.pinv(R_state),train_data))#Dr*N
        
    elif index==6:
        # values, counts = np.unique(Node_MC, return_counts=True)
        # group_size_dict = dict(zip(values, counts))
        # alphas = np.array([group_size_dict[g]/max(counts)*0.01 for g in Node_MC])
        # A = np.diag(alphas)
        # # W_out=np.dot(np.linalg.pinv(np.dot(R_state.T,R_state)+A),train_data)
        # W_out=np.dot(np.dot(np.linalg.pinv(np.dot(R_state.T,R_state)
        #         +A),R_state.T),train_data)
        
        param_grid = {
            'x': [0.0001,0.001,0.01, 0.1, 1.0, 10.0],
            'Node_MC': [Node_MC]  # 固定分组不参与搜索，但 GridSearchCV 要有
        }
        
        model = GroupRidge()
        
        grid = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
        grid.fit(R_state, train_data);
        best_model = grid.best_estimator_
        W_out = best_model.W_out_
    elif index==1:
#        W_out=np.dot(np.dot(np.linalg.pinv(np.dot(R_state.T,R_state)
#        +alpha*np.identity(R_state.shape[1])),R_state.T),train_data)
        alphas = np.array([0.0001,0.001, 0.01, 0.1, 1.0, 10.0])
        base_cv1 = RidgeCV(alphas = alphas,fit_intercept=False)
        # base_cv = LassoCV(alphas = alphas,fit_intercept=False)
        base_cv1.fit(R_state,train_data)
        # print(base_cv.alpha_)
        W_out=(base_cv1.coef_).T
        # for X_i in range(train_data.shape[1]):
        #     base_cv1.fit(R_state,train_data[:, X_i])
        #     # print(base_cv.alpha_)
        #     W_out[:,X_i]=base_cv1.coef_
    # elif index==2:
    #     clf = Ridge(alpha=alpha)
    #     clf.fit(R_state,train_data)
    #     W_out=(clf.coef_).T
    elif index==2:
        anova_filter = SelectKBest(f_regression, k=int(R_state.shape[1]*alpha))#int(self.n_reservoir*0.8 ))#k_number)
        clf = Pipeline([
        ('feature_selection', anova_filter),
        ('Linearregression', linear_model.Ridge(alpha=1e-8, fit_intercept=False))
        ])
        for X_i in range(train_data.shape[1]):
            clf.fit(R_state,train_data[:, X_i])
            W_out[clf.named_steps['feature_selection'].get_support(),X_i]=clf.named_steps['Linearregression'].coef_
    elif index==3:         
        # alphas = 10**np.linspace(-10,10,100)
        # base_cv = RidgeCV(alphas = alphas,fit_intercept=False)
        # anova_filter=SelectFromModel(base_cv)
        # clf = Pipeline([
        # ('feature_selection', anova_filter),
        # ('Linearregression', linear_model.LinearRegression(fit_intercept=False))
        # ])
        # for X_i in range(train_data.shape[1]):
        #     clf.fit(R_state,train_data[:, X_i])
        #     W_out[clf.named_steps['feature_selection'].get_support(),X_i]=clf.named_steps['Linearregression'].coef_
        base=linear_model.LinearRegression(fit_intercept=False)
        # print(100)
        anova_filter = RFECV(base, step=1,cv=5,n_jobs=5)#int(self.n_reservoir*0.8 ))#k_number)
        # anova_filter = RFE(base, step=1)#int(self.n_reservoir*0.8 ))#k_number)
        clf = Pipeline([
        ('feature_selection', anova_filter),
        ('Linearregression', linear_model.LinearRegression(fit_intercept=False))
        ])
        for X_i in range(train_data.shape[1]):
            try:
                clf.fit(R_state,train_data[:, X_i])
                
            except np.linalg.LinAlgError:
                print("SVD did not converge, skipping...")

            else:
                W_out[clf.named_steps['feature_selection'].get_support(),X_i]=clf.named_steps['Linearregression'].coef_
    elif index==4:
        # alpha1=np.sqrt(alpha)
        alpha1=(alpha)
        K_number=int(Node_MC.shape[0]*(alpha1))
        input_size=train_data.shape[1]

        W_out=np.zeros((Node_MC.shape[0],input_size))
        # if W_out0 is None:
        #     W_out_score=np.dot(np.linalg.pinv(R_state),train_data)#Dr*N
        # else:
        #     W_out_score=W_out0
        for i_x in range(input_size):
            # Node_MC[Node_MC<0]=1
            num_selector = SelectKBest(score_func=f_regression, k='all')
            MC_score=np.zeros((Node_MC.shape[0],2))
            MC_score[:,0]=Node_MC
            num_selector.fit(R_state, train_data[:,i_x])
            MC_score[:,1]=num_selector.scores_
        
            df = pd.DataFrame(MC_score)
            cluster_sum=(np.round(df.groupby(0).size()*1.0/df.shape[0]*K_number))
            # cluster_sum=(np.round(df.groupby(0)[1].sum()*1.0/df[1].sum()*K_number))
            # values, counts = np.unique(Node_MC, return_counts=True)
            # cluster_sum_0=np.round(counts/np.sum(counts)*K_number)
            
            
            
            # 定义选择前 N 个的函数
            def select_top_n(group):
                n = int(cluster_sum[group.name])  # 获取当前类别需要的数量
                # print(n)
                if n==0:
                    return group.nlargest(0, 1)
                else:
                    return group.nlargest(n, 1)  # 选择得分前 N 个
            
            # 分组并应用选择函数
            initial_selected= df.groupby(0, group_keys=False).apply(select_top_n)
            num_selected = len(initial_selected)
            num_remaining = K_number - num_selected
            
            if  num_selected>0:
                final_selected = initial_selected
   
            else:
                
                if num_remaining > 0:
                    # 找出未被选中的行
                    remaining_df = df.drop(index=initial_selected.index)
                    additional_selected = remaining_df.nlargest(num_remaining, 1)
                    final_selected = pd.concat([initial_selected, additional_selected])
                else:
                    final_selected = initial_selected
            num_selected = len(final_selected)
            # reg=LinearRegression(fit_intercept=False).fit(R_state[:,final_selected.index],train_data[:,i_x])
            # W_out[final_selected.index,i_x]=reg.coef_
            W_out[final_selected.index,i_x]=np.dot(np.linalg.pinv(R_state[:,final_selected.index]),train_data[:,i_x])

            
    elif index==5:
   
            K_number=int(Node_MC.shape[0]*(alpha))
            input_size=train_data.shape[1]
   
            W_out=np.zeros((Node_MC.shape[0],input_size))
            for i_x in range(input_size):
                num_selector = SelectKBest(score_func=f_regression, k='all')
                MC_score=np.zeros((Node_MC.shape[0],2))
                MC_score[:,0]=Node_MC
                num_selector.fit(R_state, train_data[:,i_x])
                MC_score[:,1]=num_selector.scores_
            
                df = pd.DataFrame(MC_score)
                values, counts = np.unique(Node_MC, return_counts=True)
                cluster_sum=(np.round(df.groupby(0)[1].sum()/df[1].sum()*K_number))
                cluster_sum[:]=np.minimum(cluster_sum.values,counts)
            
                
                # 定义选择前 N 个的函数
                def process_group(group, x,y, n):
                    if n>1:
                        X_group = x[:,group.index] # 转置回来，每行为样本
                        # print(group)
                        model = LinearRegression(fit_intercept=False)
                        selector = RFE(model, n_features_to_select=n).fit(X_group, y)
                        selected_index=np.where(selector.support_)
                 
                        S_F=group.iloc[selected_index]
                    elif n==1:
                        S_F=group.nlargest(1, 1)
                    else:
                        S_F=group.nlargest(0,1)
                    return S_F
                
                # 分组并应用选择函数
                initial_selected= df.groupby(0, group_keys=False).apply(
                                lambda group_df: process_group(group_df, R_state,train_data, int(cluster_sum[group_df.name]))
                            )
                num_selected = len(initial_selected)
                num_remaining = K_number - num_selected
                
                if  num_selected>0:
                    final_selected = initial_selected
       
                else:
                    
                    if num_remaining > 0:
                        # 找出未被选中的行
                        remaining_df = df.drop(index=initial_selected.index)
                        additional_selected = remaining_df.nlargest(num_remaining, 1)
                        final_selected = pd.concat([initial_selected, additional_selected])
                    else:
                        final_selected = initial_selected
                num_selected = len(final_selected)
                reg=LinearRegression(fit_intercept=False).fit(R_state[:,final_selected.index],train_data[:,i_x])
                W_out[final_selected.index,i_x]=reg.coef_
                # anova_filter = SelectKBest(f_regression, k=int(num_selected*alpha1))#int(self.n_reservoir*0.8 ))#k_number)
                # clf = Pipeline([
                # ('feature_selection', anova_filter),
                # ('Linearregression', linear_model.LinearRegression(fit_intercept=False))
                # ])
                # clf.fit(R_state[:,final_selected.index],train_data[:, i_x])
                # W_out[final_selected.index[clf.named_steps['feature_selection'].get_support()],i_x]=clf.named_steps['Linearregression'].coef_
                
        
           
    return np.round(W_out,Data_precision)






#定义预测和目标之间的误差。 
def plot_figure(pred_test,test_output,number,Lt=1,dt=0.1,name=None,index=0,Etstander=0.5):
    '''
    index 1  plot the figure
    Lt the max predit time
    Et the judege indicator
    Etstander the stander which we can accept
    '''
    Et=np.linalg.norm(pred_test[:number]-test_output[:number],ord=2,axis=1)/np.linalg.norm(test_output[:number],ord=2,axis=1)
    # print(Et)
    t = np.arange(0, test_output[:number].shape[0]*dt,dt)
    # plt.plot(t)
    plt.show()
    t = np.arange(0, test_output.shape[0]*dt,dt)
    t=t/Lt    
    t=t[:number]
    # Etstander=0.5#the given error criterion 
    if np.where(Et>=Etstander)[0].shape[0]>0:
        Etindex=np.min(np.where(Et>=Etstander))
    else:
#        print("----------")
        Etindex=-1   
    if index==1:
        if test_output.shape[1]==1:
            ax = plt.plot()
            plt.plot(t,test_output[:number,0],color='blue',label='Actual')
            plt.plot(t,pred_test[:number,0],color='red',linestyle="dashed" ,label='Predict')
            plt.legend(fontsize='x-small',bbox_to_anchor=(1.05,1))
            plt.ylabel("x(t)")
            plt.title(name)
            if name is not None:
                plt.savefig(name,format='pdf',bbox_inches='tight')
            # plt.title(t[Etindex])
            plt.show()
        if test_output.shape[1]==2:
            ax1 = plt.subplot(211)
            plt.plot(t,test_output[:number,0],color='blue',label='Actual')
            plt.plot(t,pred_test[:number,0],color='red',linestyle="dashed",label='Predict')
            plt.legend(fontsize='x-small',bbox_to_anchor=(1.05,1))
            plt.ylabel("x(t)")
            plt.setp(ax1.get_xticklabels(), visible=False)
            
            
            # share x only
            ax2 = plt.subplot(212, sharex=ax1)
            plt.plot(t,test_output[:number,1],color='blue')
            plt.plot(t,pred_test[:number,1],color='red',linestyle="dashed")
            plt.ylabel("y(t)")
            # make these tick labels invisible
            
            plt.xlim(0,round(t[-1]))
            plt.xlabel("$\lambda_{max}t$")
            plt.title(name)
            #plt.savefig(name,format='pdf',bbox_inches='tight')
            # if name is not None:
            #     plt.savefig(name,format='pdf',bbox_inches='tight')
            # plt.title(t[Etindex])
            plt.show()
        if test_output.shape[1]>=3:
            ax1 = plt.subplot(311)
            plt.plot(t,test_output[:number,0],color='blue',label='Actual')
            plt.plot(t,pred_test[:number,0],color='red',linestyle="dashed",label='Predict')
            plt.legend(fontsize='x-small',bbox_to_anchor=(1.05,1))
            plt.ylabel("x(t)")
            plt.setp(ax1.get_xticklabels(), visible=False)
            
            
            # share x only
            ax2 = plt.subplot(312, sharex=ax1)
            plt.plot(t,test_output[:number,1],color='blue')
            plt.plot(t,pred_test[:number,1],color='red',linestyle="dashed")
            plt.ylabel("y(t)")
            # make these tick labels invisible
            plt.setp(ax2.get_xticklabels(), visible=False)
            
            # share x and y
            ax3 = plt.subplot(313, sharex=ax1)
            plt.plot(t,test_output[:number,2],color='blue')
            plt.plot(t,pred_test[:number,2],color='red',linestyle="dashed") 
            plt.ylabel("z(t)")
            plt.xlim(0,round(t[-1]))
            plt.xlabel("$\lambda_{max}t$")
            plt.title(name)
            #plt.savefig(name,format='pdf',bbox_inches='tight')
            if name is not None:
                plt.savefig(name,format='pdf',bbox_inches='tight')
            plt.show()
            
#    print(t[Etindex])
    return t[Etindex] 


    
    
def generate_graph(Index_Network=None,Network_weight=None,network_size=300,Seed=None,density=0.2,Depth=5,P_SM=0.3,R_index=None):
    #R_index 判断是否使用相同的网络结构
    if Index_Network==0:
        #UN-Regular randomness 规则网络
        ##无向网络需要考虑对称的
        if R_index is None:
            R_network_0=(Network_initial(network_index=Index_Network,network_size=network_size,Seed=Seed,density=density))
        else:
            R_network_0=R_index
        R_network=np.triu(np.multiply(R_network_0,Network_weight))+np.triu(np.multiply(R_network_0,Network_weight)).T
    elif Index_Network==1:
        #SW randomness
        if R_index is None:
            R_network_0=(Network_initial(network_index=Index_Network,network_size=network_size,Seed=Seed,density=density,P_SM=P_SM))
        else:
            R_network_0=R_index
        R_network=np.triu(np.multiply(R_network_0,Network_weight))+np.triu(np.multiply(R_network_0,Network_weight)).T
    elif Index_Network==2:
        #ER randomness
        if R_index is None:
            R_network_0=(Network_initial(network_index=Index_Network,network_size=network_size,Seed=Seed,density=density))
        else:
            R_network_0=R_index
        R_network=np.triu(np.multiply(R_network_0,Network_weight))+np.triu(np.multiply(R_network_0,Network_weight)).T  
        
    elif Index_Network==3:
        #DCG randomness
        if R_index is None:
            R_network_0=(Network_initial(network_index=Index_Network,network_size=network_size,Seed=Seed,density=density))
        else:
            R_network_0=R_index
        R_network=np.multiply(R_network_0,Network_weight)
        W_index=R_network_0!=0
        W_index_1=W_index*1+W_index.T*1
        W_index_2=W_index_1==2 #找出对称的元素位置
        R_network[np.tril(W_index_2)]=R_network.T[np.tril(W_index_2)]
    elif Index_Network==4:
        #DAG randomness
        if R_index is None:
            R_network_0=(Network_initial(network_index=Index_Network,network_size=network_size,density=density,
                              Depth=Depth))
        else:
            R_network_0=R_index
       
    #            G = nx.from_numpy_matrix(R_network_0,create_using=nx.DiGraph())
    #            nx.dag_longest_path_length(G)+1
        R_network=np.multiply(R_network_0,Network_weight)
    elif Index_Network==5:
        #DL randomness 有向线性网络 网络结构本质上就不变
        Network_weight=Network_weight
        R_network_0=np.eye(network_size,k=1)
        R_network=(R_network_0)*(Network_weight)
    elif Index_Network==6:
        #Ring 网络 有向线性网络 网络结构本质上就不变
        ring_network = nx.cycle_graph(network_size)
        R_network_0=nx.adjacency_matrix(ring_network).toarray() 
        R_network=np.triu(np.multiply(R_network_0,Network_weight))+np.triu(np.multiply(R_network_0,Network_weight)).T
    #R_network_0 是网络结构 R_network 加权的复杂网络
    return R_network_0,np.round(R_network,Data_precision)

#实验数据，模型数据和真实数据预测
def generate(data_length, odes, state, parameters,index=1):
    data = np.zeros([data_length,state.shape[0]])
    
    # the first 5000 data should be abandoned
    if index==1:
        for i in range(1000):
            state = rk4(odes, state, parameters)
    
    for i in range(data_length):
        state = rk4(odes, state, parameters)
        data[i,:] = state

    return np.round(data,Data_precision)

def rk4(odes, state, parameters, dt=0.01):
    k1 = dt * odes(*state, *parameters)
    k2 = dt * odes(*(state + 0.5 * k1), *parameters)
    k3 = dt * odes(*(state + 0.5 * k2), *parameters)
    k4 = dt * odes(*(state + k3), *parameters)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def Rabinovich_Fabrikant_odes(x, y, z,gamma,alpha):
    return np.array([y*(z-1 +x**2)+gamma*x, x * (3*z+1-x**2) +gamma*y, -2*z*(alpha+x*y)])

def Rabinovich_Fabrikant_generate(data_length):
    return generate(data_length, Rabinovich_Fabrikant_odes, \
        np.array([-0.4, 0.1,0.7]), np.array([0.1,0.14]))
        
        
        
def lorenz_odes(x, y, z, sigma, beta, rho):
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

def lorenz_generate(data_length):
    return generate(data_length, lorenz_odes, \
        np.array([-8.0, 8.0, 27.0]), np.array([10.0,8.0/3.0,28.0]))
        
        
        
def hyperchaotic_ode(x, y, z, w, a, b, c, d):
    return np.array([a*x-y*z+w,x*z-b*y, x * y - c*z,-y+d])

def hyperchaotic_generate(data_length):
    return generate(data_length, hyperchaotic_ode, \
        np.array([10.0,1.0,10.0,1.0]), np.array([8, 40, 15,-0.1]))
        
def cos_simi(X1,X2):
    vec1=X1.reshape(-1)
    vec2=X2.reshape(-1)
    simi_V=np.ndarray.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return simi_V




def New_MP(Index_Network=0,N=3,Dr=300,rho=1,Path=0,P_SM=0,density=0,delta=0.1,b=0,
           index_activation=0,eplison=0,p=0.95,MP_1_U=None,x_M=1,x_L=0):


    MP_0=eplison
    if index_activation==0: #线性激活函数   
        # print(0)
        #模型输入层的最大上界和下界
        mean=delta*N*1+b   # 正态分布均值为1
        MP_beta_U=delta*N # 这里没有进行缩放 输入在0-1之间   Win的最大值
        
        if np.log(MP_0/MP_beta_U)>0:
            MP_U=1
        else:
            #输入不在(D,U) 之间，
           #都是预估输入的上界，所以有下面一行的结论
            # MP_U=np.log(MP_0/MP_beta_U)/np.log(MP_1_U)      #如果MP_1_U 小于1 是上界， 大于1则是下界  这个证明是从 随机游走出发的  
            if log_uniform_stats(eplison, rho)[0]>0:
                 MP_U=100000  # 这表明无穷   
            else:
                MP_U=find_n_for_p(rho,p,eplison,MP_0/MP_beta_U)+1
    
    else:
        
        if rho>=1:
            MP_0=np.sqrt(1-1.0/(rho))-b*1.0/(rho)
            MP_beta_U=delta*N*(1-np.tanh(b)**2) # 初值状态的更改速率
        
           
            if np.log(MP_0/MP_beta_U)<0: #MP_0>MP_beta_U  当前记忆就卡住了，不会传递到下面
                MP_U=1
            else:
                #如果MP_1_U 小于1 是上界， 大于1则是下界  这个证明是从 随机游走出发的            
                MP_U=np.log(MP_0/MP_beta_U)/np.log(rho+eplison)+1
                # print(MP_U)
        else:
            MP_0=eplison
            
            MP_beta_U=delta*N*(1-np.tanh(b)**2) # 初值状态的更改速率
        
           
            if np.log(MP_0/MP_beta_U)>0: #MP_0>MP_beta_U  当前记忆就卡住了，不会传递到下面
                MP_U=1
            else:
                #如果MP_1_U 小于1 是上界， 大于1则是下界  这个证明是从 随机游走出发的            
                MP_U=np.log(MP_0/MP_beta_U)/np.log(rho)+1
                # print(MP_U)

    return MP_U

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

def richness(R_state=None):
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

def renyi_entropy(data=None, alpha=2,eplison=0.001):
    """计算 Rényi 熵，默认 alpha=2"""
    
    custom_bins = np.tanh(np.arange(-1, 1 + eplison, eplison))   # 创建区间边界（左闭右开）
    counts, bin_edges = np.histogram(data, bins=custom_bins)
    probabilities = counts / np.sum(counts)

    if alpha == 1:
        return -np.sum(probabilities * np.log2(probabilities))  # 退化为香农熵
    return (1 / (1 - alpha)) * np.log2(np.sum(probabilities ** alpha))


def count_different_range(data=None, eplison=0.001):
    bins = np.tanh(np.arange(-1, 1 + eplison, eplison)) # 创建区间边界（左闭右开）
        
    # 使用 np.digitize 分配区间索引
    bin_indices = np.digitize(data, bins, right=False)  # right=False 表示左闭右开
    return len(np.unique(bin_indices))*1.0/data.shape[0]


def mackey_glass(beta=0.2, gamma=0.1, n=10, tau=5, T=20000, dt=1):
    # 初始化参数
    steps = int(T / dt)
    delay_steps = int(tau / dt)
    x = np.zeros(steps)
    
    # 初始条件，假设 x(0) = 1
    x[:delay_steps] = 1.0

    # 通过 Euler 方法求解
    for t in range(delay_steps, steps - 1):
        x_tau = x[t - delay_steps]  # 延迟的值
        dx = beta * x_tau / (1 + x_tau**n) - gamma * x[t]
        x[t + 1] = x[t] + dx * dt

    return x[1000:]





def count_unique_column(matrix):
        columns = matrix 
        
        # 去除全为 0 的列
        filtered_columns = columns[~np.all(columns == 0, axis=1)]
        
        if filtered_columns.size != 0:
               
            # 统计不同列的数量
            unique_columns = np.unique(filtered_columns, axis=0)
            count_unique_columns = unique_columns.shape[0]
        else:
            count_unique_columns=0
        return count_unique_columns

def calculate_autocorrelation(sequence, max_lag):
    n = len(sequence)
    mean = np.mean(sequence)
    var = np.var(sequence)
    autocorrelations = []
    for lag in range(1, max_lag + 1):
        cov = np.sum((sequence[:n-lag] - mean) * (sequence[lag:] - mean)) / n
        autocorrelations.append(cov / var)
    return np.arange(1, max_lag + 1), np.array(autocorrelations)

def adf_test(sequence):
    result = adfuller(sequence)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print('Critical Values:')
        print(f'   {key}, {value}')
    return result[1]  # Return p-value for easy use

def min_max_scale(sequence, feature_range=(-1, 1)):
    min_val = np.min(sequence)
    max_val = np.max(sequence)
    scale = (feature_range[1] - feature_range[0]) / (max_val - min_val)
    min_adj = feature_range[0] - min_val * scale
    return scale * sequence + min_adj

#从位置0开始最大片段
def max_consecutive_trues(sequence):

    max_count = 0
    
    if sequence[0]:
        for value in sequence:
            if value:
                max_count  += 1
            else:
                break
    else:
        max_count =0
        
    
    return max_count

#生成W_in

def generate_win(N_row=0,Num_V=0,V_distri=None,seed=None):
    
    num_variables =Num_V
    variable_indices = list(range(num_variables)) # 结果是 [0, 1, 2]
    patterns = []
    rng1=random.Random(seed)
    
    if V_distri is None:
        V_distri=np.ones(Num_V)
        R=np.ones(N_row)
        R0=[]
        for i in range(Num_V):
            V_distri[i]=comb(Num_V,i+1)/(2**Num_V-1)
            R0.append([i+1] * int(V_distri[i]*N_row))
        flat_list = list(itertools.chain.from_iterable(R0))
        if len(flat_list)<N_row:
            R[:len(flat_list)]=np.array(flat_list)
            R[len(flat_list):]=Num_V
            
        else:
            R=np.array(flat_list)
            
      
        final_rows = []
        
        
        for i in range(N_row):
        
            r=int(R[i])
            CS=rng1.sample(variable_indices, r)#第二步无重复生成随机序列
        
            # 创建一个全零的列表
            pattern =  [0] * num_variables
            # 将组合中索引的位置设置为 1
            for index in CS:
                pattern[index] = 1
            final_rows.append(pattern)
       
        final_matrix = np.array(final_rows)

    else:
        rng1=random.Random(seed)
        np.random.seed(seed)
        final_rows = []
        R = V_distri
        for i in range(N_row):
            
            r = int(R[i]) #第一步随机生成抽取数量
            CS=rng1.sample(variable_indices, r)#第二步无重复生成随机序列
        
            # 创建一个全零的列表
            pattern = [0] * num_variables
            # 将组合中索引的位置设置为 1
            for index in CS:
                pattern[index] = 1
            final_rows.append(pattern)
       
        final_matrix = np.array(final_rows)
  

    final_matrix=final_matrix*(Num_V/np.sum(final_matrix,1).reshape(-1,1))
    return(final_matrix)


def analyze_granger_pairwise(X, R, max_lag=3, verbose=False):
    """
    接收X和R两个NumPy数组，对它们进行逐一的成对格兰杰因果检验。
    会同时计算 R -> X 和 X -> R 两个方向。

    Args:
        X (np.ndarray): 影响变量的时间序列 (N_samples, M_features_X)
        R (np.ndarray): 被影响变量的时间序列 (N_samples, K_features_R)
        max_lag (int): 测试的最大滞后阶数。
        verbose (bool): 是否打印 statsmodels 函数的详细输出。

    Returns:
        tuple: (p_values_X_to_R, p_values_R_to_X)
               两个p值矩阵，如果出错则返回 (None, None)。
    """
    # print("--- 开始成对格兰杰因果分析 ---")

    
    # --- 1. 内部数据准备 ---
    # 根据输入的数组形状，自动创建列名
    x_cols = [f'X_{i}' for i in range(X.shape[1])]
    r_cols = [f'R_{i}' for i in range(R.shape[1])]
    
    # 将两个NumPy数组合并成一个Pandas DataFrame
    df = pd.DataFrame(np.hstack([X, R]), columns=x_cols + r_cols)
    
    # --- 2. 分析 X -> R ---
    # print("\n正在检验 X -> R ...")
    p_values_X_to_R = pd.DataFrame(np.nan, index=x_cols, columns=r_cols)
    for caused_var in r_cols:
        for causing_var in x_cols:
            if np.var(df[causing_var])>0.00001:
                test_data = df[[caused_var, causing_var]]
                gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=verbose)
                min_p_value = min([result[0]['ssr_ftest'][1] for lag, result in gc_result.items()])
                p_values_X_to_R.loc[causing_var, caused_var] = min_p_value
            else:
                p_values_X_to_R.loc[causing_var, caused_var] = 1
            
            

    
    return p_values_X_to_R

def analyze_transfer_entropy(X, R, history_k=1, n_bins=4, verbose=False):
    """
    计算从多元序列X到多元序列R的成对传递熵。

    Args:
        X (np.ndarray): 影响变量的时间序列 (N_samples, M_features_X)
        R (np.ndarray): 被影响变量的时间序列 (N_samples, K_features_R)
        history_k (int): 使用的历史数据长度
        n_bins (int): 数据离散化的箱子数量
        verbose (bool): 是否打印详细输出

    Returns:
        np.ndarray: 一个 M x K 的矩阵，包含了从每个X变量到每个R变量的传递熵。
    """
    if verbose:
        print("\n\n--- 2. 传递熵分析 (Transfer Entropy) ---")
    
    n_samples, n_features_X = X.shape
    _, n_features_R = R.shape

    # 数据离散化：将所有数据一起计算分箱边界，以保证尺度一致
    all_data = np.hstack([X, R])
    # 为避免数据边界问题，增加一个极小量
    bins = np.linspace(np.min(all_data) - 1e-6, np.max(all_data) + 1e-6, n_bins + 1)
    X_discrete = np.digitize(X, bins) - 1
    R_discrete = np.digitize(R, bins) - 1

    # 初始化结果矩阵
    te_matrix = np.zeros((n_features_X, n_features_R))
    
    if verbose:
        print(f"正在计算 {n_features_X}x{n_features_R} 个成对传递熵...")

    # 成对计算传递熵
    for i in range(n_features_X):
        for j in range(n_features_R):
            source_series = X_discrete[:, i]
            target_series = R_discrete[:, j]
            te = pyinform.transfer_entropy(source_series, target_series, k=history_k)
            te_matrix[i, j] = te

    if verbose:
        print("\n传递熵矩阵 (行: X的变量, 列: R的变量):")
        print(np.round(te_matrix, 4))
        avg_te = np.mean(te_matrix)
        print(f"\n从 X 到 R 的平均传递熵: {avg_te:.4f}")
        max_te_val = np.max(te_matrix)
        max_idx = np.unravel_index(np.argmax(te_matrix, axis=None), te_matrix.shape)
        print(f"最强的信息流路径: X_{max_idx[0]} -> R_{max_idx[1]}，传递熵为: {max_te_val:.4f}")
        
    return te_matrix



# def decompose_R_on_X_basis(X, R, max_interaction_order=None, verbose=True):
#     """
#     将R的每个维度分解到由X的所有组合变量构成的基上，并将所有结果整合到一个单一的DataFrame矩阵中。

#     Args:
#         X (np.ndarray): 输入时间序列 (N_samples, M_features)
#         R (np.ndarray): 目标时间序列 (N_samples, K_features)
#         max_interaction_order (int, optional): 考虑的最高交互阶数。默认为None，即计算所有阶。

#     Returns:
#         pd.DataFrame: 一个单一的、长格式的DataFrame，包含所有分解结果。
#     """
#     if verbose:
#         print("--- 开始分解R，并将结果整合为单一矩阵 ---")

#     n_samples, n_features_X = X.shape
#     _, n_features_R = R.shape
    
#     if max_interaction_order is None:
#         max_interaction_order = n_features_X
        
#     # 步骤 1: 创建所有组合的交互特征
#     basis_functions = []
#     basis_names = []
    
#     for r in range(1, max_interaction_order + 1):
#         for indices in combinations(range(n_features_X), r):
#             if r == 1:
#                 basis_series = X[:, indices[0]]
#                 name = f'X_{indices[0]}'
#             else:
#                 basis_series = np.prod(X[:, indices], axis=1)
#                 name = ' * '.join([f'X_{i}' for i in indices])
#             basis_functions.append(basis_series)
#             basis_names.append(name)
            
#     X_basis = np.array(basis_functions).T
    
#     if verbose:
#         print(f"已生成 {X_basis.shape[1]} 个基函数。")

#     # =============================================================
#     # 核心改动：初始化一个列表来收集每个R维度的结果DataFrame
#     # =============================================================
#     results_list = []
    
#     # 步骤 2: 遍历R的每一个维度进行分解
#     print("正在对R的每个维度进行分解和能量计算...")
#     for j in range(n_features_R):
#         target_r = R[:, j]
#         r_name = f'R_{j}'
        
#         # 线性回归分解
#         model = LinearRegression(fit_intercept=False)
#         model.fit(X_basis, target_r)
#         weights = model.coef_
        
#         # 计算能量
#         decomposed_components = X_basis * weights
#         component_energies = np.sum(decomposed_components**2, axis=0)
#         total_energy = np.sum(component_energies)
#         energy_percentages = (component_energies / total_energy) * 100 if total_energy > 0 else np.zeros_like(component_energies)
        
#         # 创建一个临时的DataFrame用于存储当前R维度的结果
#         temp_df = pd.DataFrame({
#             'Basis_Function': basis_names,
#             'Weight': weights,
#             'Energy': component_energies,
#             'Energy_Percentage': energy_percentages
#         })
        
#         # 增加一列来标识这个结果属于哪个R的维度
#         temp_df['R_Dimension'] = r_name
        
#         # 将这个临时的DataFrame添加到列表中
#         results_list.append(temp_df)

#     # =============================================================
#     # 步骤 3: 将列表中的所有DataFrame合并成一个大的DataFrame
#     # =============================================================
#     final_matrix = pd.concat(results_list, ignore_index=True)
    
#     # 调整列的顺序以符合您的要求
#     final_matrix = final_matrix[[
#         'R_Dimension', 
#         'Basis_Function', 
#         'Weight', 
#         'Energy', 
#         'Energy_Percentage'
#     ]]
    
#     print("分析完成！")
#     return final_matrix
# def analyze_all_interactions_te(X, R, max_interaction_order=None, history_k=1, n_bins=4, verbose=True):
#     """
#     计算 X 的所有特征组合（单个、两两乘积、三三乘积...）对 R 的成对传递熵。

#     Args:
#         X (np.ndarray): 输入时间序列 (N_samples, M_features)
#         R (np.ndarray): 目标时间序列 (N_samples, K_features)
#         max_interaction_order (int, optional): 考虑的最高交互阶数。默认为None，即计算所有阶。
#         history_k (int): 使用的历史数据长度。
#         n_bins (int): 数据离散化的箱子数量。
#         verbose (bool): 是否打印详细输出。

#     Returns:
#         pd.DataFrame: 互信息得分矩阵 (行: X的组合项, 列: R的特征)。
#     """
#     if verbose:
#         print("\n\n--- 开始分析 X 的所有特征组合 -> R 的传递熵 ---")

#     n_samples, n_features_X = X.shape
#     _, n_features_R = R.shape
    
#     if max_interaction_order is None:
#         max_interaction_order = n_features_X
        
#     # =============================================================
#     # 步骤 1: 创建所有组合的交互特征
#     # =============================================================
#     interaction_features = []
#     interaction_names = []
    
#     print(f"正在生成从1阶到{max_interaction_order}阶的所有特征组合...")
    
#     # 外层循环：遍历交互的阶数（1=单个，2=两两，3=三三...）
#     for r in range(1, max_interaction_order + 1):
#         # 内层循环：获取当前阶数的所有特征组合
#         for indices in combinations(range(n_features_X), r):
            
#             # 如果阶数r=1，就是原始特征本身
#             if r == 1:
#                 interaction_series = X[:, indices[0]]
#             # 如果阶数r>1，就是这些特征列的元素级乘积
#             else:
#                 interaction_series = np.prod(X[:, indices], axis=1)
                
#             interaction_features.append(interaction_series)
            
#             # 创建交互项的名称，例如 'X_0', 'X_0 * X_1'
#             name = ' * '.join([f'X_{i}' for i in indices])
#             interaction_names.append(name)
            
#     # 将交互特征列表转换为一个NumPy矩阵
#     X_all_interactions = np.array(interaction_features).T
    
#     if verbose:
#         print(f"总共生成了 {X_all_interactions.shape[1]} 个源特征。")

#     # =============================================================
#     # 步骤 2: 执行传递熵分析
#     # =============================================================
    
#     # 数据离散化
#     all_data = np.hstack([X_all_interactions, R])
#     bins = np.linspace(np.min(all_data) - 1e-6, np.max(all_data) + 1e-6, n_bins + 1)
#     X_interactions_discrete = np.digitize(X_all_interactions, bins) - 1
#     R_discrete = np.digitize(R, bins) - 1

#     # 初始化结果矩阵
#     r_cols = [f'R_{j}' for j in range(n_features_R)]
#     te_matrix = pd.DataFrame(index=interaction_names, columns=r_cols, dtype=float)

#     if verbose:
#         print(f"正在计算 {X_all_interactions.shape[1]}x{n_features_R} 个成对传递熵...")

#     # 成对计算传递熵
#     for i, source_name in enumerate(interaction_names):
#         for j, target_name in enumerate(r_cols):
#             source_series = X_interactions_discrete[:, i]
#             target_series = R_discrete[:, j]
#             te = pyinform.transfer_entropy(source_series, target_series, k=history_k)
#             te_matrix.loc[source_name, target_name] = te

#     if verbose:
#         print("\n交互项传递熵矩阵 (行: X的组合项, 列: R的变量):")
#         print(te_matrix.round(4))
    
#     return te_matrix



def generate_orthonormal_matrix(n_rows, n_cols, seed=None):
    """
    生成一个具有标准正交列的矩阵。

    Args:
        n_rows (int): 矩阵的行数 (例如 100)。
        n_cols (int): 矩阵的列数 (例如 3)。
        seed (int, optional): 随机种子，用于保证结果可复现。

    Returns:
        np.ndarray: 一个形状为 (n_rows, n_cols) 的矩阵，其列是标准正交的。
    """
    if n_rows < n_cols:
        raise ValueError("行数必须大于或等于列数才能生成标准正交列。")
        
    # 1. 创建一个随机的 n_rows x n_cols 矩阵
    # 使用随机种子来确保每次运行结果一致
    if seed is not None:
        np.random.seed(seed)
    random_matrix = np.random.randn(n_rows, n_cols)
    
    # 2. 对这个随机矩阵进行QR分解
    # np.linalg.qr 会返回一个元组 (Q, R)
    q, r = np.linalg.qr(random_matrix)
    
    

    return q/np.max(q)
