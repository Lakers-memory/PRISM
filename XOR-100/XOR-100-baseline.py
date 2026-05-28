# exec(open("Functions.py", encoding="utf-8").read())
# exec(open("PRISM.py", encoding="utf-8").read())
from Functions import Kernel_matrix, LG_sym, calc_differential_vec
from PRISM import Differential_method, Shared_space, Multiple_latent_variables
from ManiFeSt import ManiFeSt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from sklearn.feature_selection import mutual_info_classif, f_classif
from skfeature.function.similarity_based import fisher_score, lap_score, reliefF, SPEC
from skfeature.function.statistical_based import gini_index, t_score
from skfeature.utility.construct_W import construct_W


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



def monte_carlo_simulation(n_iter=200, prism_params={'K':7, 'k':100, 'k0':90, 'N':2, 'w1':0.5, 'seed':1337}):
    
    all_scores = np.zeros((n_iter, 100))
    correct_selections = 0

    for i in range(n_iter):
        X, y = generate_xor_dataset(seed=prism_params['seed']+i)

        # feature selection: ManiFeSt
        # scores, idx, _ = ManiFeSt(X, y, kernel_scale_factor=0.1, use_spsd=True, percentile=50)

        # feature selection: IG
        scores = mutual_info_classif(X, y, n_neighbors=3)   # bigger is better

        # feature selection: ANOVA
        # scores, _ = f_classif(X, y)   # bigger is better

        # feature selection: Pearson
        # df_X = pd.DataFrame(X)   
        # df_y = pd.Series(y)
        # scores = df_X.apply(lambda x: x.corr(df_y), axis=0)   
        # scores = scores.abs().to_numpy()

        # feature selection: Gini-index
        # scores = gini_index.gini_index(X, y)   # the smaller the better
        # scores = -scores

        # feature selection: t-test
        # scores = t_score.t_score(X, y)  # bigger is better

        # feature selection: Fisher
        # scores = fisher_score.fisher_score(X, y)  # bigger is better

        # feature selection: Laplacian
        # W = construct_W(X, metric='euclidean', weight_mode='heat_kernel')   
        # scores = lap_score.lap_score(X, W=W)  # the smaller the better
        # scores = -scores

        # feature selection: ReliefF
        # scores = reliefF.reliefF(X, y, k=5)  # bigger is better (default k=5)


        # normalize the scores
        scaler = MinMaxScaler()
        scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
        all_scores[i] = scores

        # get the two features with the highest scores
        top2 = set(np.argsort(scores)[-2:])
        relevant = {0, 4}

        # counting hits
        hit_count = len(top2 & relevant)
        correct_selections += hit_count  

    # calculate mean score and standard deviation
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)

    # calculate the average number of correct selections (total number of correct selections / number of iterations)
    avg_num = correct_selections / n_iter

    return mean_scores, std_scores, avg_num, all_scores


# run the simulation
start_time = time.time()  
mean_scores, std_scores, avg_num, all_scores = monte_carlo_simulation(n_iter=200)
results = pd.DataFrame({
    "mean_scores": mean_scores,
    "std_scores": std_scores
})

print(avg_num)
results.to_csv("/XOR-100/baseline/results.csv", index=False)
pd.DataFrame(all_scores).to_csv("/XOR-100/baseline/all_scores.csv", index=False)

end_time = time.time()  
print(f"Total time taken: {end_time - start_time:.2f} seconds")  

