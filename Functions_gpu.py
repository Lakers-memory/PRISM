''' GPU accelerated version'''
import cupy as cp
from cupyx.scipy.spatial.distance import pdist
from cuml.svm import SVC
import matplotlib.pyplot as plt


def squareform(pdist_matrix):
    """Convert the compressed distance matrix output by pdist to a square matrix"""
    n = int(cp.sqrt(2 * pdist_matrix.shape[0])) + 1
    square = cp.zeros((n, n), dtype=pdist_matrix.dtype)
    square[cp.triu_indices(n, k=1)] = pdist_matrix
    square += square.T
    return square

def Kernel_matrix(df, epsilon):
    """GPU accelerated kernel matrix calculations"""
    df = cp.asarray(df.T, dtype=cp.float32)  # use float32 to save memory
    
    # calculate distance matrix using pdist of cupy
    D = squareform(pdist(df, 'euclidean'))  
    dist_sort = cp.sort(D, axis=1)
    sigmas = dist_sort[:, round(epsilon)]
    Sig = cp.outer(sigmas, sigmas)
    kernel_matrix = cp.exp(-(D ** 2) / Sig)
    return kernel_matrix

def LG_sym(W, k=None, initial=None):
    """GPU accelerated symmetric Laplacian matrix calculation"""
    W = cp.asarray(W, dtype=cp.float32)
    D = cp.diag(cp.sum(W, axis=1) ** (-0.5))
    Lrw = D @ W @ D
    
    # eigen decomposition using cupy
    d, v = cp.linalg.eigh(Lrw)
    idx_ = cp.argsort(d)[::-1]
    v = v[:, idx_]
    if k is not None:
        v = v[:, :k]
        d = d[:k]
    return Lrw, d, v  

def calc_differential_vec(L_A, v_B, k, Q=None):
    """GPU accelerated differential vector calculation"""
    L_A = cp.asarray(L_A, dtype=cp.float32)
    v_B = cp.asarray(v_B, dtype=cp.float32)
    
    U1 = v_B[:, :k]
    Q1 = U1 @ U1.T
    Q1 = cp.eye(Q1.shape[0]) - Q1
    Q1 = Q1 @ L_A @ Q1
    
    s, u1 = cp.linalg.eigh(Q1)
    idx_order = cp.argsort(s)[::-1]
    u1 = u1[:, idx_order]
    return s, u1

