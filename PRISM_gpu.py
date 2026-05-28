from Functions_gpu import Kernel_matrix, LG_sym, calc_differential_vec
import cupy as cp
import numpy as np
from cuml.model_selection import GridSearchCV

def Differential_method(L1, L2, v1, v2, k=30):
    """GPU accelerated differential method"""
    # X1_gpu = cp.asarray(X1, dtype=cp.float32)
    # X2_gpu = cp.asarray(X2, dtype=cp.float32)
    
    # K1 = Kernel_matrix(X1, K)
    # K2 = Kernel_matrix(X2, K)
    
    # L1, d1, v1 = LG_sym(K1)
    # L2, d2, v2 = LG_sym(K2)
    
    s2, u2 = calc_differential_vec(L2, v1[:, 1:], k)
    s1, u1 = calc_differential_vec(L1, v2[:, 1:], k)
    
    return s1, u1, s2, u2

def Shared_space(P1, P2, k0=None):
    """GPU-accelerated shared space computing"""
    # X1_gpu = cp.asarray(X1, dtype=cp.float32)
    # X2_gpu = cp.asarray(X2, dtype=cp.float32)
    
    # K1 = Kernel_matrix(X1, K)
    # K2 = Kernel_matrix(X2, K)
    
    # D1 = cp.diag(cp.sum(K1, axis=1) ** (-0.5))
    # D2 = cp.diag(cp.sum(K2, axis=1) ** (-0.5))
    # P1 = D1 @ K1 @ D1
    # P2 = D2 @ K2 @ D2
    
    P_theta = P1 @ P2 + P2 @ P1
    
    d, v = cp.linalg.eigh(P_theta)
    idx_ = cp.argsort(d)[::-1]
    d = d[idx_]
    v = v[:, idx_]
    if k0 is not None:
        v = v[:, :k0]
        d = d[:k0]
    return P_theta, d, v

def Multiple_latent_variables(X1, X2, N=5, K=200, k=100, k0=100):
    """GPU accelerated multi-latent variable computation"""
    # X1_gpu = cp.asarray(X1, dtype=cp.float32)
    # X2_gpu = cp.asarray(X2, dtype=cp.float32)

    K1 = Kernel_matrix(X1, K)
    K2 = Kernel_matrix(X2, K)
    
    L1, _, v1 = LG_sym(K1)
    L2, _, v2 = LG_sym(K2)
    
    _, u1, _, u2 = Differential_method(L1, L2, v1, v2, k=k)
    deltas1 = [u1[:, 0]]
    deltas2 = [u2[:, 0]]
    
    _, _, v_shared = Shared_space(L1, L2, k0=k0)
    V1 = v_shared
    V2 = v_shared
    
    for i in range(1, N):
        V1 = cp.hstack([V1, deltas1[i-1].reshape(-1, 1)])
        Kv1 = Kernel_matrix(V1.T, K)
        Lv1, _, vv1 = LG_sym(Kv1)
        _, u1_new, _, _ = Differential_method(L1, Lv1, v1, vv1, k=k)
        deltas1.append(u1_new[:, 0])

        V2 = cp.hstack([V2, deltas2[i-1].reshape(-1, 1)])
        Kv2 = Kernel_matrix(V2.T, K)
        Lv2, _, vv2 = LG_sym(Kv2)
        _, u2_new, _, _ = Differential_method(L2, Lv2, v2, vv2, k=k)
        deltas2.append(u2_new[:, 0])
    
    return cp.array(deltas1).T, cp.array(deltas2).T


def PRISM(X1, X2, N=2, K=200, k=500, k0=400):
    """ iteratively calculate the differential vector: take the larger value at the corresponding position as the final score"""
    # delta1 = Multiple_latent_variables(X1, X2, N=N, K=K, k=k, k0=k0)
    # delta2 = Multiple_latent_variables(X2, X1, N=N, K=K, k=k, k0=k0)

    delta1, delta2 = Multiple_latent_variables(X1, X2, N=N, K=K, k=k, k0=k0)

    # ===== intra-category aggregation =====
    # calculate score1 and score2
    score1 = delta1[:, -1] ** 2
    score2 = delta2[:, -1] ** 2

    # ===== aggregation across categories =====
    # for score1 and score2, take the larger value in the corresponding position
    score = np.maximum(score1, score2)

    return score            # delta1, delta2, score1, score2, score

