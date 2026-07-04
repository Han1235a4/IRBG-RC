# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:34:39 2025

@author: 1235a4


leaky echo state 
ES2N model
Deep RC model
"""


import numpy as np
from scipy import special, stats
from tools import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 
Data_precision=7
global indexx

Seed=42
rng=np.random.RandomState(Seed)

indexx=1 # 1是有激活函数 0 是没有激活函数
def activation_function(A=None):
    if indexx==1:
        return np.tanh(A)
    else:
        return A
    
import numpy as np

def random_orthogonal_matrix(n):
    # 随机生成一个 n x n 的矩阵
    A = rng.rand(n, n)*2-1
    # 做 QR 分解
    Q, R = np.linalg.qr(A)
    return Q




#0---传统 1--leaky lr不等于1 2--ES2N 3---DeepRC  4-MCI-ESN  当选择不是tanh时候 还可以生成linear RC
class RCs():
    def __init__(self, N=3,Dr=300, rho=1,delta=0.1,b=0,transient=1000,R_network=None,W_in=None,RC_index=0,leaky_rate=0.5):
        """
        L: 
        N: the input dimension
        Dr: the reservoir network size
        rho: spectral radius of the reservoir network weight matrix
        delta： the scaling parameter of the input-to-reservoir matrix
        b： the bias term
        transient: the number of reservoir state data  to be deleted  
        
        R_network: the reservoir network weight matrix
        W_in : the input-to-reservoir matrix
        """
        self.N=N
        self.Dr=Dr
        self.rho=rho
        self.delta=delta
        self.b=b
        self.transient=transient
        
        self.R_network=R_network.T
        self.W_in=W_in
        self.lr=leaky_rate
        self.RI=RC_index
        if self.RI==3:
            self.R_network= R_network.transpose(0, 2, 1)
        else:
            
            self.R_network=R_network.T
            
        
    
    def Training_phase(self, train_data=None,Train_expect=None,index_method=0,alpha=0,F_wout=2,Node_MC=None):
        L=train_data.shape[0]
        R_state = np.zeros((L, self.Dr))
        W_out= np.zeros((self.Dr, self.N))
        Pre_train_output= np.zeros((L, self.N))
         
        if self.RI==0:
            R_state[0,:]=activation_function(np.dot(self.rho*self.R_network, R_state[0,:])
                                 +self.delta*np.dot(self.W_in,train_data[0,:])+self.b)
        elif self.RI==1:
            R_state[0,:]=self.lr*activation_function(np.dot(self.rho*self.R_network, R_state[0,:])
                                 +self.delta*np.dot(self.W_in,train_data[0,:])+self.b)+ (1-self.lr)*R_state[0,:]
        elif self.RI==2:
            self.O=random_orthogonal_matrix(self.Dr)
            R_state[0,:]=self.lr*activation_function(np.dot(self.rho*self.R_network, R_state[0,:])
                                 +self.delta*np.dot(self.W_in,train_data[0,:])+self.b)
            +(1-self.lr)*np.dot(self.O, R_state[0,:])
        elif self.RI==3:
            self.R_depth=int((self.R_network.shape[0]+1)/2.0)
            R_state = np.zeros((self.R_depth,L,self.Dr))
            W_out=np.zeros((self.Dr*self.R_depth, self.N))
            R_state[0,0,:]=self.lr*activation_function(np.dot(self.rho*self.R_network[0,:], R_state[0,0,:])
                             +self.delta*np.dot(self.W_in,train_data[0,:])+self.b)+(1-self.lr)*R_state[0,0,:]
            for ii in range(1,self.R_depth):
                R_state[ii,0,:]=self.lr*activation_function(np.dot(self.rho*self.R_network[ii,:], R_state[ii,0,:])
                                 +np.dot(self.R_network[ii+self.R_depth-1,:], R_state[ii-1,0,:]))+(1-self.lr)*R_state[ii,0,:]
        elif self.RI==4:
            self.Wres=self.R_network[:self.Dr,:self.Dr]
            self.Wcn=self.R_network[self.Dr:,self.Dr:]
            self.Win1=self.W_in[:self.Dr,:]
            self.Win2=self.W_in[self.Dr:,:]
            self.R0_0=self.lr*activation_function(np.dot(self.Wres, R_state[0,:])
                                 +self.delta*np.dot(self.Win1,train_data[0,:])+self.b)
            self.R1_0=(1-self.lr)*activation_function(np.dot(self.Wres, R_state[0,:])
                                 +self.delta*np.dot(self.Win2,train_data[0,:])+self.b)
            R_state[0,:]=self.R0_0+self.R1_0
        

        for i in range(1,L):
            if self.RI==0:
                R_state[i,:]=activation_function(np.dot(self.rho*self.R_network, R_state[i-1,:])
                                     +self.delta*np.dot(self.W_in,train_data[i,:])+self.b)
            elif self.RI==1:
                R_state[i,:]=self.lr*activation_function(np.dot(self.rho*self.R_network, R_state[i-1,:])
                             +self.delta*np.dot(self.W_in,train_data[i,:])+self.b)+ (1-self.lr)*R_state[i-1,:]
            elif self.RI==2:
                R_state[i,:]=self.lr*activation_function(np.dot(self.rho*self.R_network, R_state[i-1,:])
                                     +self.delta*np.dot(self.W_in,train_data[i,:])+self.b)
                +(1-self.lr)*np.dot(self.O, R_state[i-1,:])
            elif self.RI==3:
                R_state[0,i,:]=self.lr*activation_function(np.dot(self.rho*self.R_network[0,:], R_state[0,i-1,:])
                                 +self.delta*np.dot(self.W_in,train_data[i,:])+self.b)+(1-self.lr)*R_state[0,i-1,:]
                for ii in range(1,self.R_depth):
                    R_state[ii,i,:]=self.lr*activation_function(np.dot(self.rho*self.R_network[ii,:], R_state[ii,i-1,:])
                                     +np.dot(self.R_network[ii+self.R_depth-1,:], R_state[ii-1,i,:]))+(1-self.lr)*R_state[ii,i-1,:]
            elif self.RI==4:
                self.R0_0=self.lr*activation_function(np.dot(self.Wres, self.R0_0)+np.dot(self.Wcn, self.R1_0)
                                     +self.delta*np.dot(self.Win1,train_data[i,:])+self.b)
                self.R1_0=(1-self.lr)*activation_function(np.dot(self.Wres, self.R1_0)+np.dot(self.Wcn,self.R0_0)
                                     +self.delta*np.dot(self.Win2,train_data[i,:])+self.b)
                R_state[i,:]=self.R0_0+self.R1_0
                
                    

            # print(i)
            # print(R_state.shape)
            
          
        if self.RI==3:
            R_state=R_state.transpose(1, 0, 2).reshape(L,-1)                
        R_state=np.round(R_state,Data_precision)
        # print(R_state) #用不用训练
        if F_wout==2:
               if Node_MC is None:
                     W_out=traing_Wout(Train_expect[self.transient:,:],R_state[self.transient:,:],index=index_method,
                              alpha=alpha)
               else:
                     # unique_elements, counts = np.unique(Node_MC, return_counts=True)
                     # unique_index=np.where(counts==1)
                     # #对于等于1的 统一处理
                     # if len(unique_index[0])==1:
                     #     index_Wout=np.where(np.isin(Node_MC,unique_elements[unique_index]))[0]
                     #     W_out[index_Wout,:]=0
                     # elif len(unique_index[0])>1:
                                                 
                     #     index_Wout=np.where(np.isin(Node_MC,unique_elements[unique_index]))[0]
                     #     W_out[index_Wout,:]=traing_Wout(Train_expect[self.transient:,:],R_state[self.transient:,index_Wout],index=index_method,
                     #  alpha=alpha)
                         
                     # unique_index=np.where(counts>1)
                     
                     # for iii in range(unique_elements[unique_index].shape[0]):
                     #        # print(iii)
                     #        index_Wout=np.where(np.isin(Node_MC,unique_elements[unique_index][iii]))[0]
                     #        W_out[index_Wout,:]=traing_Wout(Train_expect[self.transient:,:],R_state[self.transient:,index_Wout],index=index_method,
                     #         alpha=alpha)


                     # # 每层提取的特征不一样
                                    
                     # # final_index=np.where(W_out!=0)
                     # # # print(final_index)
                     # # W_out= np.zeros((self.Dr, self.N))  
                     # # if self.RI==3:
                     # #     W_out=np.zeros((self.Dr*self.R_depth, self.N))
                     
                     # for i_x in range(Train_expect.shape[1]):
                     #     final_index=np.where(W_out[:,i_x]!=0)[0]
                     #     # final_index=np.where(W_out[:,i_x]!=0)[0]
                     #     W_out[final_index,i_x]=traing_Wout(Train_expect[self.transient:,i_x],R_state[self.transient:,final_index],index=0,
                     #          alpha=alpha)[:,0]
                         
                     #     # W_out[final_index,i_x]=np.linalg.lstsq(R_state[self.transient:,final_index], Train_expect[self.transient:,i_x], rcond=None)[0]
                     
                     
                     W_out=traing_Wout(Train_expect[self.transient:,:],R_state[self.transient:,:],index=4,
                              alpha=alpha,Node_MC=Node_MC)

                                    
                                   
                    
            
            
#        print(W_out.shape)
        
        Pre_train_output=np.dot(R_state,W_out)
         
        self.W_out=W_out
        self.laststate=R_state[-1,:]
        self.lastinput=Train_expect[-1,:]
         
        return Pre_train_output,W_out,R_state
    
    def Predicting_phase(self,Pre_L=1000,W_out=None):

        outputs=np.zeros((Pre_L,self.N))
        R_state=np.zeros((Pre_L,self.Dr))
        
        if self.RI==0:
            R_state[0,:]=activation_function((np.dot(self.rho*self.R_network,self.laststate)
                                 +self.delta*np.dot(self.W_in,self.lastinput)+self.b))
        elif self.RI==1:
            R_state[0,:]=self.lr*activation_function((np.dot(self.rho*self.R_network,self.laststate)
                                 +self.delta*np.dot(self.W_in,self.lastinput)+self.b))+ (1-self.lr)*self.laststate
        elif self.RI==2:
            R_state[0,:]=self.lr*activation_function((np.dot(self.rho*self.R_network,self.laststate)
                                 +self.delta*np.dot(self.W_in,self.lastinput)+self.b))
            + (1-self.lr)*np.dot(self.O, self.laststate)

        elif self.RI==3:
            R_state = np.zeros((self.R_depth,Pre_L,self.Dr))
            self.laststate=self.laststate.reshape(self.R_depth,-1)
            R_state[0,0,:]=self.lr*activation_function(np.dot(self.rho*self.R_network[0,:], self.laststate[0,:])
                             +self.delta*np.dot(self.W_in,self.lastinput)+self.b)+(1-self.lr)*self.laststate[0,:]
            for ii in range(1,self.R_depth):
                R_state[ii,0,:]=self.lr*activation_function(np.dot(self.rho*self.R_network[ii,:], self.laststate[ii,:])
                                 +np.dot(self.R_network[ii+self.R_depth-1,:], R_state[ii-1,0,:]))+(1-self.lr)*self.laststate[ii,:]
        elif self.RI==4:
            self.R0_0=self.lr*activation_function(np.dot(self.Wres, self.R0_0)+np.dot(self.Wcn, self.R1_0)
                                 +self.delta*np.dot(self.Win1,self.lastinput)+self.b)
            self.R1_0=(1-self.lr)*activation_function(np.dot(self.Wres, self.R1_0)+np.dot(self.Wcn, self.R0_0)
                                 +self.delta*np.dot(self.Win2,self.lastinput)+self.b)
            R_state[0,:]=self.R0_0+self.R1_0
        

        if W_out is not None:
                   
 
            
            if self.RI==3:
                R_state_d=R_state.transpose(1, 0, 2).reshape(Pre_L,-1)  
                outputs[0,:]=np.dot(R_state_d[0, :],W_out)
            else:
                outputs[0,:]=np.dot(R_state[0, :],W_out)
            for i in range(1,Pre_L):
                
                if self.RI==0:
                    R_state[i,:]=activation_function(np.dot(self.rho*self.R_network, R_state[i-1,:])
                                         +self.delta*np.dot(self.W_in,outputs[i-1,:])+self.b)
                elif self.RI==1:
                    R_state[i,:]=self.lr*activation_function(np.dot(self.rho*self.R_network, R_state[i-1,:])
                                 +self.delta*np.dot(self.W_in,outputs[i-1,:])+self.b)+ (1-self.lr)*R_state[i-1,:]
                elif self.RI==2:
                    R_state[i,:]=self.lr*activation_function(np.dot(self.rho*self.R_network, R_state[i-1,:])
                                         +self.delta*np.dot(self.W_in,outputs[i-1,:])+self.b)
                    +(1-self.lr)*np.dot(self.O, R_state[i-1,:])
                elif self.RI==3:
                    R_state[0,i,:]=self.lr*activation_function(np.dot(self.rho*self.R_network[0,:], R_state[0,i-1,:])
                                     +self.delta*np.dot(self.W_in,outputs[i-1,:])+self.b)+(1-self.lr)*R_state[0,i-1,:]
                    for ii in range(1,self.R_depth):
                        R_state[ii,i,:]=self.lr*activation_function(np.dot(self.rho*self.R_network[ii,:], R_state[ii,i-1,:])
                                         +np.dot(self.R_network[ii+self.R_depth-1,:], R_state[ii-1,i,:]))+(1-self.lr)*R_state[ii,i-1,:]
                elif self.RI==4:
                    self.R0_0=self.lr*activation_function(np.dot(self.Wres, self.R0_0)+np.dot(self.Wcn, self.R1_0)
                                         +self.delta*np.dot(self.Win1,outputs[i-1,:])+self.b)
                    self.R1_0=(1-self.lr)*activation_function(np.dot(self.Wres, self.R1_0)+np.dot(self.Wcn, self.R0_0)
                                         +self.delta*np.dot(self.Win2,outputs[i-1,:])+self.b)
                    R_state[i,:]=self.R0_0+self.R1_0
                        

                

                if self.RI==3:
                    R_state_d=R_state.transpose(1, 0, 2).reshape(Pre_L,-1)  
                    outputs[i, :] = np.dot(R_state_d[i, :],W_out)
                else:
                    outputs[i, :] = np.dot(R_state[i, :],W_out)
        else:

            if self.RI==3:
                R_state_d=R_state.transpose(1, 0, 2).reshape(Pre_L,-1)  
                outputs[0,:]=np.dot(R_state_d[0, :],self.W_out)
            else:
                outputs[0,:]=np.dot(R_state[0, :],self.W_out)
                
            for i in range(1,Pre_L):
                if self.RI==0:
                    R_state[i,:]=activation_function(np.dot(self.rho*self.R_network, R_state[i-1,:])
                                         +self.delta*np.dot(self.W_in,outputs[i-1,:])+self.b)
                elif self.RI==1:
                    R_state[i,:]=self.lr*activation_function(np.dot(self.rho*self.R_network, R_state[i-1,:])
                                 +self.delta*np.dot(self.W_in,outputs[i-1,:])+self.b)+ (1-self.lr)*R_state[i-1,:]
                elif self.RI==2:
                    R_state[i,:]=self.lr*activation_function(np.dot(self.rho*self.R_network, R_state[i-1,:])
                                         +self.delta*np.dot(self.W_in,outputs[i-1,:])+self.b)
                    +(1-self.lr)*np.dot(self.O, R_state[i-1,:])
                elif self.RI==3:
                    R_state[0,i,:]=self.lr*activation_function(np.dot(self.rho*self.R_network[0,:], R_state[0,i-1,:])
                                     +self.delta*np.dot(self.W_in,outputs[i-1,:])+self.b)+(1-self.lr)*R_state[0,i-1,:]
                    for ii in range(1,self.R_depth):
                        R_state[ii,i,:]=self.lr*activation_function(np.dot(self.rho*self.R_network[ii,:], R_state[ii,i-1,:])
                                         +np.dot(self.R_network[ii+self.R_depth-1,:], R_state[ii-1,i,:]))+(1-self.lr)*R_state[ii,i-1,:]
                        
                        
                elif self.RI==4:
                    self.R0_0=self.lr*activation_function(np.dot(self.Wres, self.R0_0)+np.dot(self.Wcn, self.R1_0)
                                         +self.delta*np.dot(self.Win1,outputs[i-1,:])+self.b)
                    self.R1_0=(1-self.lr)*activation_function(np.dot(self.Wres, self.R1_0)+np.dot(self.Wcn, self.R0_0)
                                         +self.delta*np.dot(self.Win2,outputs[i-1,:])+self.b)
                    R_state[i,:]=self.R0_0+self.R1_0
                
                
                if self.RI==3:
                    R_state_d=R_state.transpose(1, 0, 2).reshape(Pre_L,-1)  
                    outputs[i, :] = np.dot(R_state_d[i, :],self.W_out)
                else:
                    outputs[i, :] = np.dot(R_state[i, :],self.W_out)
            
        return outputs