""" Main routines for QFAMES

Goal: Given a signal data matrix, output estimatation of dominant frequencies, numbers, and orthogonal space.

-Input:

Z_est: L \times R \times N   data matrix

L: number of left initial states

R: number of right initial states

N: samples

d_x: space step

t_list: np.array of time points

K: number of dominant frequencies

alpha: interval constant

T: maximal time

tau: singular value threshold)

-Output:

Dominant_freq: np.array of estimation of locations of clusters (up to adjustment when there is no gap)

Dominant_number: np.array of estimation of number of eigenvalues in each cluster

Dominant_vector: np.array of orthogonal vectors corresponding to the eigenvector space of each cluster (only for verbose=True)

Last revision: 08/03/2025
"""

import numpy as np

def QFAMES(Z_est, d_x, t_list, K, alpha, T, tau, verbose='False'):
    """
    nonorthogonal QMEGS algorithm

    To avoid long classical running time, we first do a rough search
    then do a detailed search around the rough maximal point.
    """
    L = len(Z_est[:,0,0])
    R = len(Z_est[0,:,0])
    N = len(Z_est[0,0,:])
    W = np.zeros((L, R), dtype='complex') #Store data matrix
    num_x=int(2*np.pi/(d_x*10))
    num_x_detail=int(2/d_x/T)
    x_rough=np.arange(0,num_x)*d_x*10-np.pi
    G = np.zeros(len(x_rough), dtype='complex') #Store Frobenius norm of the data matrix (for rough search)
    for l in range(L):
        for r in range(R):
            G += np.abs(Z_est[l, r, :].dot(np.exp(1j*np.outer(t_list,x_rough)))/N)**2
    Dominant_freq=np.zeros(K,dtype='complex')
    Dominant_num=np.zeros(K,dtype='int')
    Dominant_vector=[]
    for k in range(K):
        max_idx_rough = np.argmax(G) # Rough search
        # print('maxG',max(G))
        if G[max_idx_rough] < 1e-6:
            print('No more clusters found, exiting early.')
            break
        Dominant_potential=x_rough[max_idx_rough]
        x=np.arange(0,num_x_detail)*d_x+Dominant_potential-1/T
        G_detail = np.zeros(len(x), dtype='complex') #Store Frobenius norm of the data matrix (for detail search)
        for l in range(L):
            for r in range(R):
                G_detail += np.abs(Z_est[l, r, :].dot(np.exp(1j*np.outer(t_list,x)))/N)**2
        max_idx_detail = np.argmax(G_detail) # Detail search
        Dominant_freq[k]=x[max_idx_detail] #one cluster found

        # Calculate the number of eigenvalues in the cluster
        for l in range(L):
            for r in range(R):
                W[l,r] = Z_est[l, r, :].dot(np.exp(1j*t_list*Dominant_freq[k]))/N
        U, S, V = np.linalg.svd(W)
        V = V.conj().T
        # print('k=',k,'Dominant_freq=',Dominant_freq[k])
        # print(G_detail[max_idx_detail])
        # print(k,S)
        Dominant_num[k] = int(np.sum(S > tau))
        if Dominant_num[k]>0: #if there is a cluster, calculate the orthogonal vector space
        #    print('k=',k,'S=',S)
            # Calculate the orthogonal vector space of the cluster
           if verbose=='True':
                if L>R:
                    Dominant_vector.append(U[:,Dominant_num[k]:])
                else:
                    Dominant_vector.append(V[:,Dominant_num[k]:])
        interval_max=x[max_idx_detail]+alpha/T
        interval_min=x[max_idx_detail]-alpha/T
        # print('interval_min=',interval_min,'interval_max=',interval_max)
        G=np.multiply(G,x_rough>interval_max)+np.multiply(G,x_rough<interval_min) #eliminate interval
    if verbose=='True':
        return Dominant_freq, Dominant_num, Dominant_vector
    else:
        return Dominant_freq, Dominant_num
