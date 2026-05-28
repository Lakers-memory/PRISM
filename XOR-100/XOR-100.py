# exec(open("Functions.py", encoding="utf-8").read())
# exec(open("PRISM.py", encoding="utf-8").read())
from Functions import Kernel_matrix, LG_sym, calc_differential_vec
from PRISM import Differential_method, Shared_space, Multiple_latent_variables
from ManiFeSt import ManiFeSt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time



def generate_xor_dataset(n_samples=50, n_features=100, relevant_features=(0, 4), seed=42):
    """
    Generate XOR dataset
    n_samples: number of samples (default 50)
    n_features: number of features (default 100)
    relevant_features: the index of the relevant feature (default f1=0 and f5=4)
    return: X (feature matrix), y (label)
    """
    np.random.seed(seed)
    # generate binary features (Bernoulli distribution, p=0.5)
    X = np.random.binomial(n=1, p=0.5, size=(n_samples, n_features))

    # calculate the label（f1 XOR f5）
    f1 = X[:, relevant_features[0]]
    f5 = X[:, relevant_features[1]]
    y = np.logical_xor(f1, f5).astype(int)

    return X, y



# PRISM：Only the differential vector of the last iteration is considered
def PRISM(X1, X2, N=2, K=200, k=500, k0=400, w1=0.5):
    """ iteratively calculate the differential vector: take the larger value at the corresponding position as the final score"""
    delta1 = Multiple_latent_variables(X1, X2, N=N, K=K, k=k, k0=k0)
    delta2 = Multiple_latent_variables(X2, X1, N=N, K=K, k=k, k0=k0)

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


def monte_carlo_simulation(n_iter=200, prism_params={'K':7, 'k':100, 'k0':90, 'N':2, 'w1':0.5, 'seed':1337}):
    
    all_scores_prism = np.zeros((n_iter, 100))
    all_scores_manifest = np.zeros((n_iter, 100))
    correct_selections_prism  = 0
    correct_selections_manifest = 0

    for i in range(n_iter):
        X, y = generate_xor_dataset(seed=prism_params['seed']+i)    # seed=42+i
        print(X)
        # PRISM
        X1 = X[y == 0]  # classify by the label
        X2 = X[y == 1]
        delta1, delta2, score1, score2, scores_prism = PRISM(X1, X2,
                                                    N=prism_params['N'],
                                                    K=prism_params['K'],
                                                    k=prism_params['k'],
                                                    k0=prism_params['k0'],
                                                    w1=prism_params['w1'])
        # print(delta1)

        # ManiFeSt
        scores_manifest, idx, _ = ManiFeSt(X, y, kernel_scale_factor=0.1, use_spsd=True, percentile=50)

        # normalize the scores
        scaler = MinMaxScaler()
        scores_prism = scaler.fit_transform(scores_prism.reshape(-1, 1)).flatten()
        all_scores_prism[i] = scores_prism

        scores_manifest = scaler.fit_transform(scores_manifest.reshape(-1, 1)).flatten()
        all_scores_manifest[i] = scores_manifest

        # get the two features with the highest scores
        top2_prism = set(np.argsort(scores_prism)[-2:])
        top2_manifest = set(np.argsort(scores_manifest)[-2:])
        relevant = {0, 4}

        # counting hits
        hit_count_prism = len(top2_prism & relevant)
        correct_selections_prism += hit_count_prism

        hit_count_manifest = len(top2_manifest & relevant)
        correct_selections_manifest += hit_count_manifest  

    # calculate mean score and standard deviation
    mean_scores_prism = np.mean(all_scores_prism, axis=0)
    std_scores_prism = np.std(all_scores_prism, axis=0)

    mean_scores_manifest = np.mean(all_scores_manifest, axis=0)
    std_scores_manifest = np.std(all_scores_manifest, axis=0)

    # calculate the average number of correct selections (total number of correct selections / number of iterations)
    avg_num_prism = correct_selections_prism / n_iter
    avg_num_manifest = correct_selections_manifest / n_iter

    return mean_scores_prism, std_scores_prism, avg_num_prism, all_scores_prism, mean_scores_manifest, std_scores_manifest, avg_num_manifest, all_scores_manifest


# run the simulation
start_time = time.time()  
mean_scores_prism, std_scores_prism, avg_num_prism, all_scores_prism, mean_scores_manifest, std_scores_manifest, avg_num_manifest, all_scores_manifest = monte_carlo_simulation(n_iter=200)
results_prism = pd.DataFrame({
    "mean_scores": mean_scores_prism,
    "std_scores": std_scores_prism
})

results_manifest = pd.DataFrame({
    "mean_scores": mean_scores_manifest,
    "std_scores": std_scores_manifest
})

print(f"PRISM: {avg_num_prism}")
print(f"ManiFeSt: {avg_num_manifest}")
results_prism.to_csv("/XOR-100/results_prism.csv", index=False)
results_manifest.to_csv("/XOR-100/results_manifest.csv", index=False)
pd.DataFrame(all_scores_prism).to_csv("/XOR-100/all_scores_prism.csv", index=False)
pd.DataFrame(all_scores_manifest).to_csv("/XOR-100/all_scores_manifest.csv", index=False)

end_time = time.time()  
print(f"Total time taken: {end_time - start_time:.2f} seconds") 
