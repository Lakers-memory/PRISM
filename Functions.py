import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform, euclidean
import matplotlib.pyplot as plt



def Kernel_matrix(df, epsilon):
    """Compute a kernel matrix with adaptive bandwidth
    df = data frame on which to compute the kernel matrix
    epsilon = number of neighbors to consider in the bandwidth calculation"""
    df = df.T    # Transpose the data so that each column represents a sample. This modification allows the calculation of the kernel matrix between features.

    D = squareform(pdist(df, 'euclidean')) # euclidian distance matrix
    dist_sort = np.sort(D, axis=1)     
    sigmas = dist_sort[:, round(epsilon)]   # Take the distance to the epsilon-th nearest neighbor of each sample
    Sig = np.outer(sigmas, sigmas)  # the adaptive bandwidth
    kernel_matrix = np.exp(-(D ** 2) / Sig)
    return kernel_matrix


def LG_RW(W, k=None, initial=None):
    """ compute the Laplacian graph (type=random walk): L = D^(-1)K"""
    D = np.diag(np.power(np.sum(W, axis=1), -1)) # inverse of the degree matrix
    Lrw = D @ W # the random walk laplacian

    v, d, _ = np.linalg.svd(Lrw) # calculate its singular value decomposition
    idx_ = np.argsort(d)[::-1] # sort by the singular values
    v = v[:, idx_]
    if k:
        v = v[:, :k]
        d = d[:k]   
    return Lrw, d, v


def LG_K(W, k=None, initial=None):
    """ compute Laplacian graph (type=regular): L = D - K"""
    D = np.diag(np.sum(W, axis=1)) # the degree matrix
    Lrw = D - W # the laplacian matrix

    v, d, _ = np.linalg.svd(Lrw) # calculate its singular value decomposition
    idx_ = np.argsort(d)[::-1] # sort by the singular values
    v = v[:, idx_]
    if k:
        v = v[:, :k]
        d = d[:k]
    return Lrw, d, v


def LG_sym(W, k=None, initial=None):
    """ compute Laplacian graph (type=Symmetric): L = D^(-0.5)K D^(-0.5)"""
    D = np.diag(np.sum(W, axis=1) ** (-0.5)) # D ** (-0.5), D = degree matrix
    Lrw = D @ W @ D # the operator of the symmetric matrix

    d, v = np.linalg.eigh(Lrw) # calculate its eigen decomposition
    idx_ = np.argsort(d)[::-1] # sort by the eigen values
    # d = d[idx_]
    v = v[:, idx_]
    if k != None:
        v = v[:, :k]
        d = d[:k]
    return Lrw, d, v



def calc_differential_vec(L_A, v_B, k, Q=None):
    """ Calculate differential vectors, as describes in Algorithm 1.
    L_A = the Laplacian matrix we filter.
    v_B = the eigenvectors of the second class we filter L_A with.
    k = the number of leading eigenvectors we use in the filter."""
    U1 = v_B[:, :k]    
    Q1 = U1 @ U1.T     # Q1 is a projection matrix that can project any n-dimensional vector onto the subspace spanned by the columns of U1.
    Q1 = np.eye(Q1.shape[0]) - Q1   # complementary projection matrix
    Q1 = Q1 @ L_A @ Q1   # Filtering Operation

    s, u1 = np.linalg.eigh(Q1)
    idx_order = np.argsort(s)[::-1]
    u1 = u1[:, idx_order]
    if Q:
        return Q1, s, u1
    else:
        return s, u1


