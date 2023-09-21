#%%
import numpy as np
import pandas as pd
import math
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
#%matplotlib notebook
sns.set()
import os
import time
import random
import pygraphviz as pgv
import cvxpy as cp
import scipy as scipy
import statistics as stat
from scipy.spatial import KDTree
import random
import json
import math
# Generate points in circle
pi = math.pi
def PointsInCircum(r,n):
    l=[((math.cos(2*pi/n*x)*r)+600,(math.sin(2*pi/n*x)*r)+600) for x in range(0,n)]
    return l
#%%
##time series on edges
N=20
##time series on node coordinates

# M_nd=2*N #multiplied by 2 because of x and y coordinated for each node
# X_nd=np.empty((M_nd,0)) #this will the system series for nodes
# MUs_nd=np.empty((M_nd,0))
# Y_nd=np.empty((M_nd,0)) #this will the observed series for nodes
# Y_nd=np.append(Y_nd,np.zeros((M_nd,1)),axis=1)

# l = np.array(PointsInCircum(400, M_nd))
# m=0
# L=np.empty((M_nd,1))
# for i in range(0,M_nd,2):
#     L[i]=l[m,0]
#     L[i+1]=l[m,1]
#     m=m+1
    
# X_nd=np.append(X_nd,L,axis=1) #initializing node positions at 0th time point


# t=100 #time points

# # assign parameter values to your liking
# # C=0.95*np.eye(M)+0.01*np.random.normal(0, 1,(M,M))
# C_nd=np.eye(M_nd)
# # D_nd=1.1*np.eye(M_nd)+0.2*np.random.normal(0, 1,(M_nd,M_nd))
# random.seed(89)
# D_nd=np.eye(M_nd)+3*np.random.normal(0, 0.01,(M_nd,M_nd))
# # D_nd=np.eye(M_nd)
# Gam_nd=1*np.eye(M_nd)
# sigsq1_nd=100
# Q_nd=sigsq1_nd*np.eye(M_nd)
# # Q=0.2*np.random.normal(0, 1,(M,M))+np.eye(M)
# sigsq2_nd=10 #10 #0.1
# Lam_nd=sigsq2_nd*np.eye(M_nd)

# random.seed(65)

# #generate series
# MUs_nd=np.append(MUs_nd,np.array([np.random.multivariate_normal(np.zeros(M_nd), Q_nd)]).T,axis=1)
# Xprev_nd=X_nd.copy()
# mu_nd=MUs_nd.copy()
# for i in range(t):
#     Xnew_nd=np.dot(C_nd,Xprev_nd)+mu_nd #+np.dot(Gam,mu)
#     X_nd=np.append(X_nd,Xnew_nd,axis=1)
#     random.seed(i+86)
#     w_nd=np.array([np.random.multivariate_normal(np.zeros(M_nd), Lam_nd)]).T
#     Ynew_nd=np.dot(D_nd,Xnew_nd)+w_nd
#     Y_nd=np.append(Y_nd,Ynew_nd,axis=1)
#     Xprev_nd=Xnew_nd.copy()
#     random.seed(i+59)
#     mu_nd=np.array([np.random.multivariate_normal(np.zeros(M_nd), Q_nd)]).T
#     MUs_nd=np.append(MUs_nd,mu_nd,axis=1)
    
# print(X_nd.shape)
# print(Y_nd.shape)

M=int(N*(N-1)/2) #number of unique edges, since the graph is symmetric
S = 0.3
X=np.empty((M,0)) #this will the system series for edges
MUs=np.empty((M,0))
Y=np.empty((M,0)) #this will the observed series for edges
Y=np.append(Y,np.zeros((M,1)),axis=1)
random.seed(42)
X=np.append(X,np.random.rand(M,1),axis=1) #initializing edge weights at 0th time point
# ind = np.random.choice(M,int(0.3*M),replace=False)
# X[ind]=0
t=1000 #time points
# assign parameter values to your liking
C=0.95*np.eye(M)
# upper_ind = np.triu_indices(M)
# C_r_ind = np.random.choice(upper_ind[0],int(S*len(upper_ind[0])),replace=False)
# C_c_ind = np.random.choice(upper_ind[1],int(S*len(upper_ind[1])),replace=False)
# ind = (C_r_ind,C_c_ind)
# C[ind]=np.random.normal(0,0.005,C[ind].shape[0])
# C[C==0].shape[0]/(M*M)
#C=0.95*np.eye(M)
#D=2*np.eye(M)+0.1*np.random.normal(0, 1,(M,M))

random.seed(32)
D=np.eye(M)+0.001*np.random.normal(0, 1,(M,M))
# D_ind = np.random.choice(M*M,int(0.6*M*M),replace=False)
# D = D.flatten()
# D[D_ind]=0
# D= D.reshape((M,M))
# D[range(M),range(M)]=1

#D=np.eye(M)
Gam=1*np.eye(M)
sigsq1=0.01 #1
Q=sigsq1*np.eye(M)
#Q=0.2*np.random.normal(0, 1,(M,M))+np.eye(M)
sigsq2=0.1 #0.01
Lam=sigsq2*np.eye(M)
random.seed(54)

#%%
#generate series
MUs=np.append(MUs,np.array([np.random.multivariate_normal(np.zeros(M), Q)]).T,axis=1)
Xprev=X.copy()
mu=MUs.copy()
for i in range(t): 
    Xnew=np.dot(C,Xprev)+mu #+np.dot(Gam,mu)
    # Xnew[Xnew<1e-5]=0
    X=np.append(X,Xnew,axis=1)
    random.seed(i+23)
    w=np.array([np.random.multivariate_normal(np.zeros(M), Lam)]).T
    Ynew=np.dot(D,Xnew)+w
    Y=np.append(Y,Ynew,axis=1)
    Xprev=Xnew.copy()
    random.seed(i+76)
    mu=np.array([np.random.multivariate_normal(np.zeros(M), Q)]).T
    MUs=np.append(MUs,mu,axis=1)
Lam_E = np.eye(M)
Q_E = np.eye(M)    
C_E = np.eye(M)
D_E = np.eye(M)    
P = np.empty((M,M,t+1))
Xk =np.empty((M,2))
Xp = np.empty((M,t+1))
Xk[:,1]=Y[:,1]
Xk[:,0]=X[:,0]
P[:,:,0]=np.eye(M)*10
PE = np.empty((M,M,t))
C_s = scipy.sparse.csc_matrix(C_E)
D_s = scipy.sparse.csc_matrix(D_E)
for i in range(1,t+1):

    Xp[:,i] = C_s@Xk[:,i]
    PE[:,:,i-1]= C_s@P[:,:,i-1]@C_s.T+Q_E
    SK =D_s@PE[:,:,i-1]@D_s.T+Lam_E

    if i%1 == 0 or i ==1:
        sk_inv = np.linalg.inv(SK)

    Kk = PE[:,:,i-1]@D_s.T@sk_inv
    P[:,:,i]=np.matmul((np.eye(M)-Kk@D_s),PE[:,:,i-1])
    Xknew = Xp[:,i].reshape(M,1)+np.matmul(Kk,(Y[:,i].reshape(M,1)-D_s@Xp[:,i].reshape(M,1)))
    Xk = np.append(Xk,Xknew,axis=1)

print(Xk[1,-10:])
print(Xp[1,-10:])

X_s = np.empty((M,t+1))
X_s[:,-1]=Xk[:,-1]
P_s = np.empty((M,M,t))
P_s[:,:,-1]=P[:,:,-1]
Ht = P[:,:,-2]@C_E@np.linalg.pinv(PE[:,:,-1])
X_s[:,-2] = Xk[:,-2]+Ht@(X_s[:,-1]-Xp[:,-1])
print('X_s 倒数第二个',X_s[1,-2])
from pykalman import KalmanFilter
kf = KalmanFilter(transition_matrices = C_E, observation_matrices =D_E,transition_covariance=Q_E,observation_covariance=Lam_E,initial_state_mean = Y[:,1].T, initial_state_covariance=P[:,:,0], em_vars=[
      'transition_matrices', 'observation_matrices',
      'transition_covariance', 'observation_covariance',
    ])
smoothed_state_means=kf.smooth(Y[:,1:].T)[0]
_s_state = smoothed_state_means.T