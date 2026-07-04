# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:08:02 2026

@author: Administrator
"""

import numpy as np
import time
import copy
import itertools
import networkx as nx
from model_RCs import *
# from models import *
from tools import *
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
Data_precision=7
import pickle
from compute_memory_profile import *
from sklearn.preprocessing import StandardScaler
from node_feature_importance import *


data_names=['Lorenz','hyper_chaotic','hadcet','Pems_bay']



data_names=['Lorenz','hyper_chaotic','hadcet','Pems_bay']; 
Lyapunov_E={'Lorenz':0.895,'hyper_chaotic':3.03,'hadcet':1,'Pems_bay':1}; 
Data_Length={'Lorenz':20000,'hyper_chaotic':20000,'hadcet':4343,'Pems_bay':52116}; 
L_train_0={'Lorenz':16000,'hyper_chaotic':16000,'hadcet':4260,'Pems_bay':10 * 24 * 12}; 
L_test_0={'Lorenz':3000,'hyper_chaotic':1000,'hadcet':83,'Pems_bay':600}; 
transient_0={'Lorenz':1000,'hyper_chaotic':1000,'hadcet':600,'Pems_bay':5 * 24 * 12}; 
DT={'Lorenz':0.01,'hyper_chaotic':0.1,'hadcet':1,'Pems_bay':1}; 
N_dim={'Lorenz':3,'hyper_chaotic':4,'hadcet':1,'Pems_bay':325}


#网络结构类型
Graph_name=["UG-regular","UG-SW","UG-ER","DCG","DAG","DLG","Ring"]
Index_Network=2
Path=5
Index_method=2 # 0 传统 1 ridge 2 selectbest 3 RFECV
if Index_Network<3:
    U_index=1
else:
    U_index=0
P_SM=0.5 # Index_Network=1 时候有用



L2_index=1 # 1选择充分条件 0还是必要条件 其他不进行缩放

#数据index
data_index=0
               
if data_names[data_index]=='Pems_bay':
    data_initial_0 = pd.read_hdf('./data/datasets/pems-bay.h5')
    data_initial=(data_initial_0.values)
    data_initial[data_initial<0]=0
else:
    
    data_initial=(((pd.read_csv(os.path.join('./data/datasets/',data_names[data_index]+'.csv')).values))[:,1:(N_dim[data_names[data_index]]+1)]).astype(float)


    
    

# data_initial=(data_initial-np.min(data_initial))/(np.max(data_initial)-np.min(data_initial))+0.001
L_train=L_train_0[data_names[data_index]]
L_test=L_test_0[data_names[data_index]]

train_data=copy.deepcopy(data_initial[:L_train,:])
test_data=copy.deepcopy(data_initial[L_train:(L_train+L_test),:])

Train_input=train_data[:-1,:]
Train_expect=train_data[1:,:]
test_data=copy.deepcopy(data_initial[L_train:(L_train+L_test),:])

transient=transient_0[data_names[data_index]]#需要删除的池子初始状态的数量。
dt=DT[data_names[data_index]]
Lyapunov=Lyapunov_E[data_names[data_index]]



        
#超参数设置，j选择不同的模型 

j=0  #0---传统 1--leaky lr不等于1 2--ES2N 3---DeepRC   4--MCI-ESN


start_time = time.time()
#随机种子
Seed=42#
rng=np.random.RandomState(Seed)
density=0.7#池子网络的连接密度
R_depth=5
Dr=400
lr=0.05
N=Train_input.shape[1]

if j==3:
    Dr=int(Dr/R_depth)       

# z=int(Hyper_matrix_sampled[zzz,4])

W_in_0=(rng.rand(Dr,N)*2-1) 
 

    
Network_weight=rng.rand(Dr,Dr)
 
R_network_00,R_network=generate_graph(Index_Network=Index_Network,
                                      Network_weight=Network_weight,network_size=Dr,Depth=Path,Seed= Seed,density=density,P_SM=P_SM)
while np.sum(R_network_00)==0: #避免出现没有节点的情况
    R_network_00,R_network=generate_graph(Index_Network=Index_Network,
                                          Network_weight=Network_weight,network_size=Dr,Depth=Path,Seed= Seed,density=density,P_SM=P_SM)
# print(R_network)
# print(W_in)
# print(np.max(np.sum(np.linalg.matrix_power(R_network,2),1)))
# print(np.max(np.sum(np.linalg.matrix_power(R_network_00,1),1)))


#                              
if Index_Network<=4 or Index_Network==6:
        if L2_index==1:#最大奇异值=二范数 是充分条件 比谱半径值大 导致权重比较小 直接推导出来的
            s,u,v=np.linalg.svd(R_network)
              # print(np.max(np.abs(u)))
            R_network=R_network/np.max(np.abs(u))
        if L2_index==0:
            print("RCS")
            # #最大特征值（谱半径）是必要条件  一般选择这个条件 更宽松 从反条件推导出来的
            E,V=np.linalg.eig(R_network) 
            R_network=R_network/np.max(np.abs(E))

if j==3:
    R_network=np.zeros((R_depth*2-1,Dr,Dr))
    
    for zz in range(R_depth*2-1):
                           
        Network_weight=rng.rand(Dr,Dr)
     
        R_network_00,R_network_ini=generate_graph(Index_Network=Index_Network,
                                              Network_weight=Network_weight,network_size=Dr,Depth=Path,Seed= Seed,density=density,P_SM=P_SM)
        while np.sum(R_network_00)==0: #避免出现没有节点的情况
            R_network_00,R_network_ini=generate_graph(Index_Network=Index_Network,
                                                  Network_weight=Network_weight,network_size=Dr,Depth=Path,Seed= Seed,density=density,P_SM=P_SM)
        # print(np.max(np.sum(np.linalg.matrix_power(R_network,2),1)))
        # print(np.max(np.sum(np.linalg.matrix_power(R_network_00,1),1)))
        
    
    #                              
        if Index_Network<=4 or Index_Network==6:
                if L2_index==1:#最大奇异值=二范数 是充分条件 比谱半径值大 导致权重比较小 直接推导出来的
                    s,u,v=np.linalg.svd(R_network_ini)
                      # print(np.max(np.abs(u)))
                    R_network_ini=R_network_ini/np.max(np.abs(u))
                if L2_index==0:
                    # print("++++++++++++++++++")
                    # #最大特征值（谱半径）是必要条件  一般选择这个条件 更宽松 从反条件推导出来的
                    E,V=np.linalg.eig(R_network_ini) 
                    R_network_ini=R_network_ini/np.max(np.abs(E))
        R_network[zz,:]=(R_network_ini)
if j==4:#MCI-ESN 两个极简 Reservoir
   v1=(rng.rand(1)*delta).item()
   v2=(rng.rand(1)*delta).item()
   signs = [-1, 1]
   sign_matrix = np.random.choice(signs, size=(int(Dr/2), N))
   #随机初始化--z=0 传统，z=1 高阶
   z=1
   if z==0:
       W_in_00=np.ones((Dr,N))
       W_in_0=np.ones((Dr,N))
       W_in_0[:int(Dr/2),:]=sign_matrix*(v1-v2)
       W_in_0[int(Dr/2):,:]=sign_matrix*(v1+v2)
       
       
   mu=(rng.rand(1)*RHO_R).item()
   R_network=np.zeros((Dr,Dr))
   
   R_network[:int(Dr/2),:int(Dr/2)]=np.eye(int(Dr/2),k=-1)*mu
   R_network[0,int(Dr/2)-1]=mu
   R_network[int(Dr/2),Dr-1]=RHO_R-mu
   R_network[Dr-1,int(Dr/2)]=RHO_R-mu
   Dr=int(Dr/2)
#    b=0
 
# # 无改进模型
W_in=W_in_0
RC=RCs(
        N = N,
        Dr =400,
        rho=1.3,
        delta=0.01,
        b=0.001,
        transient= transient,
        R_network=np.round(R_network,Data_precision),
        W_in=np.round(W_in,Data_precision),
        RC_index=j,
        leaky_rate=lr
        )   
   
   
Train_real_output,W_out0,R_state=RC.Training_phase(Train_input,
                                                Train_expect,
                                                index_method=Index_method,
                                                alpha=0.9)
  # 正常量化指标 
Train_Rmse=Loss_Prediciton(Prediction=Train_real_output[transient:,:],
      Real_data=Train_expect[(transient):,:],Method=1)  

  #lyapunov 时间
Train_Lya=plot_figure(Train_real_output[transient:,:],
          Train_expect[(transient):,:],L_train-transient,Lt=Lyapunov,dt=dt,index=0)

   
  # #W_out变化
Pred_test_0=RC.Predicting_phase(Pre_L=L_test)  

   
  # 计算损失函数 0--MAE 1--RMSE 2--MAPE
Test_Rmse=Loss_Prediciton(Prediction=Pred_test_0[:L_test,:],
              Real_data=test_data[:L_test,:],Method=1)

Test_Lya=plot_figure(Pred_test_0,test_data,L_test,Lt=Lyapunov,dt=dt,index=1)
print(Test_Lya)
mid_time = time.time()

    
#改进版本
Seed=42#
rng=np.random.RandomState(Seed)
density=0.4#池子网络的连接密度
R_depth=5
Dr=300

N=Train_input.shape[1]

if j==3:
    Dr=int(Dr/R_depth)       

# z=int(Hyper_matrix_sampled[zzz,4])

W_in_0=(rng.rand(Dr,N)*2-1) 
 

    
Network_weight=rng.rand(Dr,Dr)
 
R_network_00,R_network=generate_graph(Index_Network=Index_Network,
                                      Network_weight=Network_weight,network_size=Dr,Depth=Path,Seed= Seed,density=density,P_SM=P_SM)
while np.sum(R_network_00)==0: #避免出现没有节点的情况
    R_network_00,R_network=generate_graph(Index_Network=Index_Network,
                                          Network_weight=Network_weight,network_size=Dr,Depth=Path,Seed= Seed,density=density,P_SM=P_SM)
# print(R_network)
# print(W_in)
# print(np.max(np.sum(np.linalg.matrix_power(R_network,2),1)))
# print(np.max(np.sum(np.linalg.matrix_power(R_network_00,1),1)))


#                              
if Index_Network<=4 or Index_Network==6:
        if L2_index==1:#最大奇异值=二范数 是充分条件 比谱半径值大 导致权重比较小 直接推导出来的
            s,u,v=np.linalg.svd(R_network)
              # print(np.max(np.abs(u)))
            R_network=R_network/np.max(np.abs(u))
        if L2_index==0:
            print("RCS")
            # #最大特征值（谱半径）是必要条件  一般选择这个条件 更宽松 从反条件推导出来的
            E,V=np.linalg.eig(R_network) 
            R_network=R_network/np.max(np.abs(E))

if j==3:
    R_network=np.zeros((R_depth*2-1,Dr,Dr))
    
    for zz in range(R_depth*2-1):
                           
        Network_weight=rng.rand(Dr,Dr)
     
        R_network_00,R_network_ini=generate_graph(Index_Network=Index_Network,
                                              Network_weight=Network_weight,network_size=Dr,Depth=Path,Seed= Seed,density=density,P_SM=P_SM)
        while np.sum(R_network_00)==0: #避免出现没有节点的情况
            R_network_00,R_network_ini=generate_graph(Index_Network=Index_Network,
                                                  Network_weight=Network_weight,network_size=Dr,Depth=Path,Seed= Seed,density=density,P_SM=P_SM)
        # print(np.max(np.sum(np.linalg.matrix_power(R_network,2),1)))
        # print(np.max(np.sum(np.linalg.matrix_power(R_network_00,1),1)))
        
    
    #                              
        if Index_Network<=4 or Index_Network==6:
                if L2_index==1:#最大奇异值=二范数 是充分条件 比谱半径值大 导致权重比较小 直接推导出来的
                    s,u,v=np.linalg.svd(R_network_ini)
                      # print(np.max(np.abs(u)))
                    R_network_ini=R_network_ini/np.max(np.abs(u))
                if L2_index==0:
                    # print("++++++++++++++++++")
                    # #最大特征值（谱半径）是必要条件  一般选择这个条件 更宽松 从反条件推导出来的
                    E,V=np.linalg.eig(R_network_ini) 
                    R_network_ini=R_network_ini/np.max(np.abs(E))
        R_network[zz,:]=(R_network_ini)
if j==4:#MCI-ESN 两个极简 Reservoir
   v1=(rng.rand(1)*delta).item()
   v2=(rng.rand(1)*delta).item()
   signs = [-1, 1]
   sign_matrix = np.random.choice(signs, size=(int(Dr/2), N))
   #随机初始化--z=0 传统，z=1 高阶
   z=1
   if z==0:
       W_in_00=np.ones((Dr,N))
       W_in_0=np.ones((Dr,N))
       W_in_0[:int(Dr/2),:]=sign_matrix*(v1-v2)
       W_in_0[int(Dr/2):,:]=sign_matrix*(v1+v2)
       
       
   mu=(rng.rand(1)*RHO_R).item()
   R_network=np.zeros((Dr,Dr))
   
   R_network[:int(Dr/2),:int(Dr/2)]=np.eye(int(Dr/2),k=-1)*mu
   R_network[0,int(Dr/2)-1]=mu
   R_network[int(Dr/2),Dr-1]=RHO_R-mu
   R_network[Dr-1,int(Dr/2)]=RHO_R-mu
   Dr=int(Dr/2)
   b=0
 
Index_node_importance=2

if j==3:
    W_in,Corr_value,Corr_rank,permut=generate_transfor(W_in=W_in_0,R_network=R_network[0,:],metr_index=Index_node_importance,corr=0.7)
else:
    W_in,Corr_value,Corr_rank,permut=generate_transfor(W_in=W_in_0,R_network=R_network,metr_index=Index_node_importance,corr=-0.8)
    
mid_time_1 = time.time()

RC=RCs(
        N = N,
        Dr =300,
        rho=0.7,
        delta=0.1,
        b=0.1,
        transient= transient,
        R_network=np.round(R_network,Data_precision),
        W_in=np.round(W_in,Data_precision),
        RC_index=j,
        leaky_rate=0.1
        )   
   
   
Train_real_output,W_out0,R_state=RC.Training_phase(Train_input,
                                                Train_expect,
                                                index_method=Index_method,
                                                alpha=0.7)
  # 正常量化指标 
Train_Rmse=Loss_Prediciton(Prediction=Train_real_output[transient:,:],
      Real_data=Train_expect[(transient):,:],Method=1)  

  #lyapunov 时间
Train_Lya=plot_figure(Train_real_output[transient:,:],
          Train_expect[(transient):,:],L_train-transient,Lt=Lyapunov,dt=dt,index=0)

   
  # #W_out变化
Pred_test_1=RC.Predicting_phase(Pre_L=L_test)  

   
  # 计算损失函数 0--MAE 1--RMSE 2--MAPE
Test_Rmse=Loss_Prediciton(Prediction=Pred_test_1[:L_test,:],
              Real_data=test_data[:L_test,:],Method=1)

Test_Lya=plot_figure(Pred_test_1,test_data,L_test,Lt=Lyapunov,dt=dt,index=1)
print(Test_Lya)


