exec(open("Functions.py", encoding="utf-8").read())
from Functions import Kernel_matrix, LG_sym, calc_differential_vec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.linalg import eig, svd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

csfont = {'fontname': 'Times New Roman'}
import matplotlib.font_manager as font_manager

font = font_manager.FontProperties(family='Times New Roman', size=24)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid
from sklearn.svm import SVC
import time
import json
from joblib import Parallel, delayed
import re


def Differential_method(X1, X2, K=200, k=30, L1=None, L2=None, v1=None, v2=None):
    """ Function to compute Differential Method.
    X1, X2 are numpy arrays containing the two datsets.
    K is the number of neighbors to consider in the kernel's adaptive bandwidth.
    k is the number of neighbors to vectors we use in the filtering"""

    # Differential vectors method
    if L1 is None or L2 is None or v1 is None or v2 is None:
        K1 = Kernel_matrix(X1, K)
        K2 = Kernel_matrix(X2, K)

        L1, d1, v1 = LG_sym(K1)
        L2, d2, v2 = LG_sym(K2)

    s2, u2 = calc_differential_vec(L2, v1[:, 1:], k)  
    s1, u1 = calc_differential_vec(L1, v2[:, 1:], k)

    return s1, u1, s2, u2


def Shared_space(X1, X2, K=200, k0=None, K1=None, K2=None, P1=None, P2=None):
    """ Compute manifold products to represent shared variables between categories: L = D^(-0.5)K D^(-0.5)"""
    if P1 is None or P2 is None:
        if K1 is None or K2 is None:
            K1 = Kernel_matrix(X1, K)
            K2 = Kernel_matrix(X2, K)

        D1 = np.diag(np.sum(K1, axis=1) ** (-0.5))  # D ** (-0.5), D = degree matrix
        D2 = np.diag(np.sum(K2, axis=1) ** (-0.5))
        P1 = D1 @ K1 @ D1  # the operator of the symmetric matrix
        P2 = D2 @ K2 @ D2

    P_theta = P1 @ P2 + P2 @ P1

    d, v = np.linalg.eigh(P_theta)  # calculate its eigen decomposition
    idx_ = np.argsort(d)[::-1]  # sort by the eigen values
    d = d[idx_]
    v = v[:, idx_]
    if k0 != None:
        v = v[:, :k0]
        d = d[:k0]
    return P_theta, d, v


def Multiple_latent_variables(X1, X2, N=5, K=200, k=100, k0=100):
    """ Iterative calculation of differential vectors
    X1 and X2 are samples of different categories
    N is the number of iterations
    K is the number of neighbors considered in calculating the kernel matrix
    k is the number of eigenvectors considered in the filtering operation
    k0 is the number of eigenvectors considered in the shared space
    """
    # Step 1: Calculate the initial differential vector Δ0^A
    K1 = Kernel_matrix(X1, K)
    K2 = Kernel_matrix(X2, K)

    L1, d1, v1 = LG_sym(K1)
    L2, d2, v2 = LG_sym(K2)

    _, u1, _, u2 = Differential_method(X1, X2, K=K, k=k, L1=L1, L2=L2, v1=v1, v2=v2)
    deltas1 = [u1[:, 0]]
    deltas2 = [u2[:, 0]]

    # Step 2: Compute the shared space V^(0)
    _, _, v_shared = Shared_space(X1, X2, K=K, k0=k0, K1=K1, K2=K2)
    V = v_shared  # Initial shared space vector
    V1 = V
    V2 = V

    # Step 3: Iteratively calculate subsequent differential vectors
    for i in range(1, N):
        # Concatenate new representation V^(i) = [V^(i-1) | Δ_{i-1}^A]
        V1 = np.hstack([V1, deltas1[i - 1].reshape(-1, 1)])
        K_v1 = Kernel_matrix(V1.T, K)
        L_v1, d_v1, v_v1 = LG_sym(K_v1)
        _, u1_new, _, _ = Differential_method(X1, V1.T, K=K, k=k, L1=L1, L2=L_v1, v1=v1, v2=v_v1)
        deltas1.append(u1_new[:, 0])

        V2 = np.hstack([V2, deltas2[i - 1].reshape(-1, 1)])
        K_v2 = Kernel_matrix(V2.T, K)
        L_v2, d_v2, v_v2 = LG_sym(K_v2)
        _, u2_new, _, _ = Differential_method(X2, V2.T, K=K, k=k, L1=L2, L2=L_v2, v1=v2, v2=v_v2)
        deltas2.append(u2_new[:, 0])

    return np.array(deltas1).T, np.array(deltas2).T     # Returns an n×N matrix of differential vectors

def PRISM(X1, X2, N=2, K=200, k=500, k0=400):
    """ iteratively calculate the differential vector: take the larger value at the corresponding position as the final score"""
    delta1, delta2 = Multiple_latent_variables(X1, X2, N=N, K=K, k=k, k0=k0)

    # ===== intra-category aggregation =====
    # calculate score1 and score2
    score1 = delta1[:, -1] ** 2
    score2 = delta2[:, -1] ** 2

    # normalize score1 and score2 to 0-1
    scaler = MinMaxScaler()
    score1 = scaler.fit_transform(score1.reshape(-1, 1)).flatten()
    score2 = scaler.fit_transform(score2.reshape(-1, 1)).flatten()

    # ===== aggregation across categories =====
    # for score1 and score2, take the larger value in the corresponding position
    score = np.maximum(score1, score2)

    return delta1, delta2, score1, score2, score






