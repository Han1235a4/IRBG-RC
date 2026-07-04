# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 15:30:00 2025

@author: 1235a4
"""
#计算结果 5 实际对应 R(t) 受到 x(t-1),....x(t-5)的影响 对应周期为 5 记忆容量为 5 
# 
#且 长度为100的随机扰动 最长识别也是 99 因为 数据只使用了99
import numpy as np
from tools import *
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
def comput_probablity(X_perturbation,number=100):
    # X_perturbation=rng.uniform(low=-1, high=1,size=10000)
    custom_bins =np.arange(-1, 1,2.0/(number+1))
    counts, bin_edges = np.histogram(X_perturbation, bins=custom_bins  )
    
    
    #整体评估指标
    probabilities = counts / np.sum(counts)
    return probabilities
def compute_memory_profile(Node_memory_0,eplison=0.00015):
    
    number_trail=int(Node_memory_0.shape[2])
    hidden_size=int(Node_memory_0.shape[0])
    length=Node_memory_0.shape[1]
    
    
    
    Node_memory_0=np.round((Node_memory_0),7)
    
    #计算变化值
    xx=np.abs(Node_memory_0-Node_memory_0[:, :, 0][:, :, np.newaxis])


    Node_memory=(np.abs(xx)>=eplison)
 
    # 判断每个节点 整个时间段内 的不同值的个数 平稳点  识别
    Node_MC_0=np.zeros((number_trail,hidden_size))
    unique_counts=np.zeros((number_trail,Node_memory_0.shape[0]))#每次取样的节点的 最大记忆能力 
    # indices=np.minimum(unique_counts_0,unique_counts_1)
    for i_unique in range(number_trail):
        unique_counts[i_unique,:]=np.apply_along_axis(
                 lambda x: len(np.unique(x)), axis=1, arr=Node_memory_0[:,:,i_unique])
        for row_idx,value in enumerate(unique_counts[i_unique,:]):
            # print(row_idx)
            if value<length:
                Node_memory[row_idx,int(value),i_unique] = False 
        Node_MC_0[i_unique,:] = np.array([max_consecutive_trues(Node_memory[j,:,i_unique]) for j in range(Node_memory.shape[0])])
             
    # 判断每个时间点是否有对应个数的不同值
    unique_counts_1=np.zeros((Node_memory_0.shape[0],Node_memory_0.shape[1]))#判断每一次 是否都会随机残生10个数
    for i_uniques in range(Node_memory_0.shape[1]):
        unique_counts_1[:,i_uniques]=np.apply_along_axis(
            lambda x: len(np.unique(x)), axis=1,  arr=Node_memory_0[:,i_uniques,:])==number_trail
    Node_MC_0R=np.array([max_consecutive_trues(unique_counts_1[j,:]) for j in range(Node_memory.shape[0])])    
    
    Node_MC=np.max(Node_MC_0,0)
    Node_MC=np.minimum(Node_MC_0R,Node_MC)

#    Node_MC=np.max(Node_MC_0,0)

    #富集程度可以用 频数代替
    # 相同记忆容量 可以用频数分布对比
    # 统计固定间隔内，每个记忆成分的数量
    
    #等于100 表明 没有记忆
    R_wass=100
    R_js=100
    R_haige=100
    Node_rich_range=np.zeros((Node_memory_0.shape[0],Node_memory_0.shape[1]))

        



    values, counts = np.unique(Node_MC, return_counts=True)

    unique_memory=int(values.shape[0])
    
    #自定义的richness
    my_richness=values.shape[0]/np.sum(counts)
    
    #整体评估指标
    probabilities=np.zeros(unique_memory)
    probabilities = counts / np.sum(counts)
    
    
    
    # 辛普森多样性指数 值越大 越分散
    nonzero_probs = probabilities[probabilities > 0]
    simpson_diversity=1-np.sum(probabilities**2)   
    shannon =-np.sum(nonzero_probs * np.log(nonzero_probs))
    
    
    
    # JS 散度 值越大 越不富集 最大为1 等于0 说明没差别
    #最丰富的是均匀分布
    P_stand=np.ones(unique_memory)*1.0/unique_memory
    R_js=jensenshannon(probabilities, P_stand)
    
    #Wasserstein 距离 最小传输代价
    X=np.arange(unique_memory)
    R_wass=wasserstein_distance(X, X, probabilities, P_stand)
        
    #海林格距离  0 表示一样 1 表示完全不一样
    # Xresult[zzz,11]=np.sqrt( 0.5 * np.sum(np.abs(probabilities - P_stand)**2))
    # 总偏差距离 Total Variation Distance, TVD
    R_haige=0.5 * np.sum(np.abs(probabilities - P_stand))
     
    
     
    # 计算 单个神经元的 富集程度
    # number=100 # 区间里面考虑多少均匀区间
    # Node_rich_js=np.zeros((Node_memory_0.shape[0],Node_memory_0.shape[1]))
    # Node_rich_wass=np.zeros((Node_memory_0.shape[0],Node_memory_0.shape[1]))
    # Node_rich_haiges=np.zeros((Node_memory_0.shape[0],Node_memory_0.shape[1]))
    # node_distribution=np.zeros((Node_memory_0.shape[0],Node_memory_0.shape[1],number))
    
    # 计算单个神经元的极值 以此判断 记忆量的多少


    for i in range(hidden_size):
        # node_distribution[i,:]=np.apply_along_axis(
        #          lambda x: comput_probablity(x,number=number), axis=1, arr=Node_memory_h[i,:,:])
        Node_rich_range[i,:]=np.apply_along_axis(
                  lambda x: (np.max(x)-np.min(x)), axis=1, arr=Node_memory_0[i,:,:])
        Node_rich_range[i,int(Node_MC[i]):]=0
        
       
 
        
     
    return {
        'memory_for_node': Node_MC,
        'memory_for_model':np.max(Node_MC),
        'Richness_jS': R_js, 
        'Richness_Wass': R_wass, 
        'Richness_haige': R_haige,  
        'Memory_rich_node':Node_rich_range,
        'My_richness':my_richness
    }



