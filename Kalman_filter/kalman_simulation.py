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

def _last_dims(X, t, ndims=2):
    """Extract the final dimensions of `X`

    Extract the final `ndim` dimensions at index `t` if `X` has >= `ndim` + 1
    dimensions, otherwise return `X`.

    Parameters
    ----------
    X : array with at least dimension `ndims`
    t : int
        index to use for the `ndims` + 1th dimension
    ndims : int, optional
        number of dimensions in the array desired

    Returns
    -------
    Y : array with dimension `ndims`
        the final `ndims` dimensions indexed by `t`
    """
    X = np.asarray(X)
    if len(X.shape) == ndims + 1:
        return X[t]
    elif len(X.shape) == ndims:
        return X
    else:
        raise ValueError(("X only has %d dimensions when %d" +
                " or more are required") % (len(X.shape), ndims))

def _smooth_update(transition_matrix, filtered_state_mean,
                   filtered_state_covariance, predicted_state_mean,
                   predicted_state_covariance, next_smoothed_state_mean,
                   next_smoothed_state_covariance):
    r"""Correct a predicted state with a Kalman Smoother update

    Calculates posterior distribution of the hidden state at time `t` given the
    observations all observations via Kalman Smoothing.

    Parameters
    ----------
    transition_matrix : [n_dim_state, n_dim_state] array
        state transition matrix from time t to t+1
    filtered_state_mean : [n_dim_state] array
        mean of filtered state at time t given observations from
        times [0...t]
    filtered_state_covariance : [n_dim_state, n_dim_state] array
        covariance of filtered state at time t given observations from
        times [0...t]
    predicted_state_mean : [n_dim_state] array
        mean of filtered state at time t+1 given observations from
        times [0...t]
    predicted_state_covariance : [n_dim_state, n_dim_state] array
        covariance of filtered state at time t+1 given observations from
        times [0...t]
    next_smoothed_state_mean : [n_dim_state] array
        mean of smoothed state at time t+1 given observations from
        times [0...n_timesteps-1]
    next_smoothed_state_covariance : [n_dim_state, n_dim_state] array
        covariance of smoothed state at time t+1 given observations from
        times [0...n_timesteps-1]

    Returns
    -------
    smoothed_state_mean : [n_dim_state] array
        mean of smoothed state at time t given observations from times
        [0...n_timesteps-1]
    smoothed_state_covariance : [n_dim_state, n_dim_state] array
        covariance of smoothed state at time t given observations from
        times [0...n_timesteps-1]
    kalman_smoothing_gain : [n_dim_state, n_dim_state] array
        correction matrix for Kalman Smoothing at time t
    """
    kalman_smoothing_gain = (
        np.dot(filtered_state_covariance,
               np.dot(transition_matrix.T,
                      linalg.pinv(predicted_state_covariance)))
    )

    smoothed_state_mean = (
        filtered_state_mean
        + np.dot(kalman_smoothing_gain,
                 next_smoothed_state_mean - predicted_state_mean)
    )
    smoothed_state_covariance = (
        filtered_state_covariance
        + np.dot(kalman_smoothing_gain,
                 np.dot(
                    (next_smoothed_state_covariance
                        - predicted_state_covariance),
                    kalman_smoothing_gain.T
                 ))
    )

    return (smoothed_state_mean, smoothed_state_covariance,
            kalman_smoothing_gain)

def _smooth(transition_matrices, filtered_state_means,
            filtered_state_covariances, predicted_state_means,
            predicted_state_covariances):
    """Apply the Kalman Smoother

    Estimate the hidden state at time for each time step given all
    observations.

    Parameters
    ----------
    transition_matrices : [n_timesteps-1, n_dim_state, n_dim_state] or \
    [n_dim_state, n_dim_state] array
        `transition_matrices[t]` = transition matrix from time t to t+1
    filtered_state_means : [n_timesteps, n_dim_state] array
        `filtered_state_means[t]` = mean state estimate for time t given
        observations from times [0...t]
    filtered_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
        `filtered_state_covariances[t]` = covariance of state estimate for time
        t given observations from times [0...t]
    predicted_state_means : [n_timesteps, n_dim_state] array
        `predicted_state_means[t]` = mean state estimate for time t given
        observations from times [0...t-1]
    predicted_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
        `predicted_state_covariances[t]` = covariance of state estimate for
        time t given observations from times [0...t-1]

    Returns
    -------
    smoothed_state_means : [n_timesteps, n_dim_state]
        mean of hidden state distributions for times [0...n_timesteps-1] given
        all observations
    smoothed_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
        covariance matrix of hidden state distributions for times
        [0...n_timesteps-1] given all observations
    kalman_smoothing_gains : [n_timesteps-1, n_dim_state, n_dim_state] array
        Kalman Smoothing correction matrices for times [0...n_timesteps-2]
    """
    n_timesteps, n_dim_state = filtered_state_means.shape

    smoothed_state_means = np.zeros((n_timesteps, n_dim_state))
    smoothed_state_covariances = np.zeros((n_timesteps, n_dim_state,
                                           n_dim_state))
    kalman_smoothing_gains = np.zeros((n_timesteps - 1, n_dim_state,
                                       n_dim_state))

    smoothed_state_means[-1] = filtered_state_means[-1]
    smoothed_state_covariances[-1] = filtered_state_covariances[-1]

    for t in reversed(range(n_timesteps - 1)):
        transition_matrix = _last_dims(transition_matrices, t)
        (smoothed_state_means[t], smoothed_state_covariances[t],
         kalman_smoothing_gains[t]) = (
            _smooth_update(
                transition_matrix,
                filtered_state_means[t],
                filtered_state_covariances[t],
                predicted_state_means[t + 1],
                predicted_state_covariances[t + 1],
                smoothed_state_means[t + 1],
                smoothed_state_covariances[t + 1]
            )
        )
    return (smoothed_state_means, smoothed_state_covariances,
            kalman_smoothing_gains)
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
D=2*np.eye(M)+0.1*np.random.normal(0, 1,(M,M))

random.seed(32)
# D=np.eye(M)+0.001*np.random.normal(0, 1,(M,M))
# D_ind = np.random.choice(M*M,int(0.6*M*M),replace=False)
# D = D.flatten()
# D[D_ind]=0
# D= D.reshape((M,M))
# D[range(M),range(M)]=1

# D=np.eye(M)
Gam=1*np.eye(M)
sigsq1=0.02 #1
Q=sigsq1*np.eye(M)
#Q=0.2*np.random.normal(0, 1,(M,M))+np.eye(M)
sigsq2=0.05 #0.01
Lam=sigsq2*np.eye(M)
random.seed(54)

#generate series
MUs=np.append(MUs,np.array([np.random.multivariate_normal(np.zeros(M), Q)]).T,axis=1)
Xprev=X.copy()
mu=MUs.copy()
for i in range(t): 
    Xnew=np.dot(C,Xprev)+mu #+np.dot(Gam,mu)
    X=np.append(X,Xnew,axis=1)
    random.seed(i+23)
    w=np.array([np.random.multivariate_normal(np.zeros(M), Lam)]).T
    Ynew=np.dot(D,Xnew)+w
    Y=np.append(Y,Ynew,axis=1)
    Xprev=Xnew.copy()
    random.seed(i+76)
    mu=np.array([np.random.multivariate_normal(np.zeros(M), Q)]).T
    MUs=np.append(MUs,mu,axis=1)
    


# GXs=[] #system
# t = 100
# for tm in range(t):
#     Xmat = np.zeros((N, N))
#     Xmat[np.triu_indices(N, 1)] = X[:,tm].copy()
#     Xmat= Xmat+ Xmat.T - np.diag(Xmat.diagonal())

    
#     GX = nx.from_numpy_matrix(Xmat)
#     m=0
#     for i in range(N):
#         GX.nodes[i]['pos']=[X_nd[m,tm],X_nd[m+1,tm]]
#         m=m+2
#     null_nodes=[]
#     for nd in GX.nodes:
#         if GX.nodes[nd]['pos']==[0,0]:
#             null_nodes.append(nd)
#     GX.remove_nodes_from(null_nodes)
#     GX=nx.relabel.convert_node_labels_to_integers(GX,first_label=0)
#     labels={nn: str(nn) for nn in range(len(GX))}
#     nx.set_node_attributes(GX, labels, 'label')  
#     GXs.append(GX)


# for ts in range(5):
#     plt.figure(figsize=(5,5))
#     GX=GXs[ts].copy()
#     pos=nx.get_node_attributes(GX, 'pos')
#     labels=nx.get_node_attributes(GX, 'label')
#     weight = nx.get_edge_attributes(GX, 'weight')
#     wts=list(weight.values())
#     min_wts=min(wts)
#     wts=[abs(min_wts)+0.001+ele for ele in wts]
#     wts = [i * 0.5 for i in wts]
#     nx.draw(GX,pos=pos,width=wts,labels=labels, font_size=13, node_color='lightcoral',edge_color='lightcoral',font_color='white')
def Para_est(Xk,Y,M,t):

    C_E_U = np.sum(np.array([Xk[:,i].reshape(M,1)@Xk[:,i-1].reshape(M,1).T for i in range(1,t)]),axis=0)
    C_E_D = np.sum(np.array([Xk[:,i-1].reshape(M,1)@Xk[:,i-1].reshape(M,1).T for i in range(1,t)]),axis=0)
    C_E = C_E_U@np.linalg.inv(C_E_D)
    
    D_E_U = np.sum(np.array([Y[:,i].reshape(M,1)@Xk[:,i].reshape(M,1).T for i in range(1,t)]),axis=0)
    D_E_D = np.sum(np.array([Xk[:,i].reshape(M,1)@Xk[:,i].reshape(M,1).T for i in range(1,t)]),axis=0)
    D_E = D_E_U@np.linalg.inv(D_E_D)
    Lam_E = 1/(t-1)*np.sum(np.array([(Y[:,i].reshape(M,1)-D_E@Xk[:,i].reshape(M,1))@(Y[:,i].reshape(M,1)-D_E@Xk[:,i].reshape(M,1)).T for i in range(1,t)]),axis=0)
    Q_E =  1/(t-1)*np.sum(np.array([(Xk[:,i].reshape(M,1)-C_E@Xk[:,i-1].reshape(M,1))@(Xk[:,i].reshape(M,1)-C_E@Xk[:,i-1].reshape(M,1)).T for i in range(1,t)]),axis=0)
    return C_E, D_E, Lam_E, Q_E

Lam_E = Lam
Q_E = Q
C_E = 5*np.eye(M)
D_E = D
for j in range(5):
    P = np.empty((M,M,t+1))
    Xk =np.empty((M,1))
    Xp = np.empty((M,t))
    Xk[:,0]=Y[:,0]
    P[:,:,0]=np.eye(M)
    PE = np.empty((M,M,t))
    C_s = scipy.sparse.csc_matrix(C_E)
    D_s = scipy.sparse.csc_matrix(D_E)
    dift1 = []
    dift2 = []
    dift3 = []
    dift4 = []
    SK = np.empty((M,M))    
    # flag = True
    temp = 0
    count = 0
    for i in range(1,t+1):

        st1 = time.time()
        
        Xp[:,i-1] = C_s@Xk[:,i-1]
        PE[:,:,i-1]= C_s@P[:,:,i-1]@C_s.T+Q_E
        et1 = time.time()
        if count > 100:
            P[:,:,i] = np.matmul((np.eye(M)-Kk@D_s),PE[:,:,i-1])
        else:
            SK =D_s@PE[:,:,i-1]@D_s.T+Lam_E
            # if np.linalg.norm(temp-SK)<1e-9 and type(temp) != type(0):
            #     flag = False
            # else:
            #     flag = True
            # if flag:
            #     sk_inv = np.linalg.inv(SK)
            #     temp = SK
            sk_inv = np.linalg.inv(SK)
            
            # else:
            #     L = np.matmul(np.matmul(U.T,SK),U)
            #     Linv = np.diag(1/np.diag(L))
            
            
            # sk_inv = np.matmul(np.matmul(U,Linv),U.T)
            
            Kk = PE[:,:,i-1]@D_s.T@sk_inv
            if np.linalg.norm(temp - Kk)<1e-12:
                count +=1
            else:
                count = 0
            temp = Kk
            P[:,:,i]=np.matmul((np.eye(M)-Kk@D_s),PE[:,:,i-1])
        zt = time.time()
        Xknew = Xp[:,i-1].reshape(M,1)+np.matmul(Kk,(Y[:,i].reshape(M,1)-D_s@Xp[:,i-1].reshape(M,1)))
        ft = time.time()
        Xk = np.append(Xk,Xknew,axis=1)
        dift1 = np.append(dift1,et1-st1)
        dift3 = np.append(dift3,zt-et1)
        dift4 = np.append(dift4,ft-zt)
    # print(np.mean(dift1),"s")
    # print(np.mean(dift3),"s")
    # print(np.mean(dift4),"s")
    # print(np.sum(dift1+dift3+dift4),"s, total filter")
    Xp = np.append(Xp,C_s@Xk[:,t].reshape(M,1),axis=1)
    PE = np.append(PE,(C_s@P[:,:,t]@C_s.T+Q_E).reshape(M,M,1),axis=2)
    _at = time.time()
    Xs = _smooth(C_E,Xk.T,P.T,Xp.T,PE.T)[0].T
    _bt = time.time()
    # print(_bt-_at,"s smooth time")
    print(np.linalg.norm(Xs-X))
    # X_s = np.empty((M,t+1))
    # X_s[:,t]=Xk[:,t]
    # P_s = np.empty((M,M,t+1))
    # P_s[:,:,t]=P[:,:,t]
    # for i in range(t-1,-1,-1):
    #     Ht = P[:,:,i]@C_E@np.linalg.inv(PE[:,:,i+1])
    #     X_s[:,i] = Xk[:,i]+Ht@(X_s[:,i+1]-Xp[:,i+1])
    #     P_s[:,:,i] = P[:,:,i]+Ht@(P_s[:,:,i+1]-PE[:,:,i+1])@Ht.T
    _ct = time.time()
    C_E, _,_,_ = Para_est(Xs,Y,M,t+1)
    print(np.linalg.norm(C_E-C))
    _dt = time.time()
    # print(_dt-_ct,"s para est time")
    # C_E[C_E<1e-7]=0
    # D_E[D_E<1e-7]=0
    # Lam_E[Lam_E<1e-7]=0
    # Q_E[Q_E<1e-7]=0


for j in range(5):

    P = np.empty((M,M,t+1))
    Xk =np.empty((M,1))
    Xp = np.empty((M,t))
    Xk[:,0]=Y[:,0]
    P[:,:,0]=np.eye(M)
    PE = np.empty((M,M,t))
    C_s = scipy.sparse.csc_matrix(C_E)
    D_s = scipy.sparse.csc_matrix(D_E)
    dift1 = []
    dift2 = []
    dift3 = []
    dift4 = []
    SK = np.empty((M,M))
    flag = True
    temp = 0
    for i in range(1,t+1):
        st1 = time.time()

        Xp[:,i-1] = C_s@Xk[:,i-1]
        PE[:,:,i-1]= C_s@P[:,:,i-1]@C_s.T+Q_E
        SK =D_s@PE[:,:,i-1]@D_s.T+Lam_E
        st2 = time.time()
        # if np.linalg.norm(temp-SK)<1e-9 and type(temp) != type(0):
        #     flag = False
        # else:
        #     flag = True
        # if flag:
        #     sk_inv = np.linalg.inv(SK)
        #     temp = SK
        if i%5 == 0 or i == 1:
            sk_inv = np.linalg.inv(SK)

        # else:
        #     L = np.matmul(np.matmul(U.T,SK),U)
        #     Linv = np.diag(1/np.diag(L))

        et1 = time.time()

        # sk_inv = np.matmul(np.matmul(U,Linv),U.T)

        Kk = PE[:,:,i-1]@D_s.T@sk_inv

        P[:,:,i]=np.matmul((np.eye(M)-Kk@D_s),PE[:,:,i-1])

        Xknew = Xp[:,i-1].reshape(M,1)+np.matmul(Kk,(Y[:,i].reshape(M,1)-D_s@Xp[:,i-1].reshape(M,1)))
        zt = time.time()
        Xk = np.append(Xk,Xknew,axis=1)
        dift1 = np.append(dift1,st2-st1)
        dift2 = np.append(dift2,et1-st2)
        dift3 = np.append(dift3,zt-et1)
    print("difference between Xk and X", np.linalg.norm(Xk - X) )
    X_s = np.empty((M,t+1))
    X_s[:,t]=Xk[:,t]
    P_s = np.empty((M,M,t+1))
    P_s[:,:,t]=P[:,:,-1]
    for i in range(t-1,-1,-1):
        Ht = P[:,:,i]@C_E@np.linalg.pinv(PE[:,:,i])
        X_s[:,i] = Xk[:,i]+Ht@(X_s[:,i+1]-Xp[:,i])
        P_s[:,:,i] = P[:,:,i]+Ht@(P_s[:,:,i+1]-PE[:,:,i])@Ht.T

    C_E, D_E, Lam_E, Q_E = Para_est(X_s,Y,M,t+1)
    C_E[C_E<1e-4]=0
    D_E[D_E<1e-4]=0
    Lam_E[Lam_E<1e-4]=0
    Q_E[Q_E<1e-4]=0
    print(np.linalg.norm(C_E - C) )
    print(np.linalg.norm(Q_E - Q) )
    print(np.linalg.norm(D_E - D) )
    print(np.linalg.norm(Lam_E - Lam) )

print(np.mean(dift1)+np.mean(dift2)+np.mean(dift3))

Lam_E = np.eye(M)
Q_E = np.eye(M)
C_E = np.eye(M)
D_E = np.eye(M)
for i in range(5):
    kf = KalmanFilter(C_E,D_E,Q_E,Lam_E)
    Xs_a = kf.smooth(Y.T)[0]
    print(np.linalg.norm(Xs_a.T-X))
    C_E, D_E, Lam_E, Q_E = Para_est(Xs_a.T,Y,M,t+1)
    C_E[C_E<1e-6]=0
    D_E[D_E<1e-6]=0
    Lam_E[Lam_E<1e-6]=0
    Q_E[Q_E<1e-6]=0
    