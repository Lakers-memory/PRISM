import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
# from sklearn.feature_selection import mutual_info_classif, f_classif
# from skfeature.function.similarity_based import fisher_score, lap_score, reliefF, SPEC
# from skfeature.function.statistical_based import gini_index, t_score
# from skfeature.utility.construct_W import construct_W
import os
import scipy.io
from joblib import Parallel, delayed
from itertools import product
from PRISM_gpu import PRISM
import cupy as cp
from cuml.svm import SVC as cuSVC
from cuml.model_selection import GridSearchCV as cuGridSearchCV
from cupyx.scipy.spatial.distance import pdist
from cuml.preprocessing import MinMaxScaler
import json
from ast import literal_eval
import gc   

# ================== Cache management module ==================
import hashlib
import pickle
from filelock import FileLock

CACHE_DIR = "/Prostate/Cache"  
os.makedirs(CACHE_DIR, exist_ok=True)

def get_train_cache_key(params, seeds, outer_iter, fold_idx, n_inner_folds):
    """generate training phase cache key (including fold information)"""
    key_str = f"train_{params}_{seeds}_iter{outer_iter}_fold{fold_idx}_inner{n_inner_folds}"
    return hashlib.md5(key_str.encode()).hexdigest()

def get_test_cache_key(params, seeds, outer_iter):
    """generate test cache key"""
    key_str = f"test_{params}_{seeds}_iter{outer_iter}"
    return hashlib.md5(key_str.encode()).hexdigest()

def save_cache(scores, cache_key):
    """save cache to file"""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    with FileLock(cache_path + ".lock"):
        with open(cache_path, "wb") as f:
            scores_cpu = cp.asnumpy(scores) if isinstance(scores, cp.ndarray) else np.asarray(scores)
            pickle.dump(scores_cpu, f)  

def load_cache(cache_key):
    """loading cache from file"""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    if not os.path.exists(cache_path):
        return None
    with FileLock(cache_path + ".lock"):
        with open(cache_path, "rb") as f:
            return cp.array(pickle.load(f))  


def task_to_hashable(task):
    """convert task arguments to a hashable tuple"""
    prism_combinations, seeds, n_feat, outer_iter = task
    prism_combinations = tuple(
        tuple(float(p) if isinstance(p, cp.ndarray) else p for p in params)
        for params in prism_combinations
    )
    seeds = (seeds['seed1'], seeds['seed3'])
    return (prism_combinations, seeds, n_feat, outer_iter)

def save_checkpoint(results, processed_tasks, result_path, checkpoint_path):
    """save the results and remove duplicates"""
    columns = [
        'n_features',
        'best_prism_params',
        'seeds',
        'outer_iter',
        'inner_cv_accuracy_mean',
        'inner_cv_accuracy_std',
        'test_accuracy',
        'test_error',
        'best_svm_params'
    ]
    
    for res in results:
        for col in columns:
            res[col] = res.get(col, np.nan)
    
    new_df = pd.DataFrame(results, columns=columns)
    
    if os.path.exists(result_path):
        existing_df = pd.read_csv(result_path)
        for col in columns:
            if col not in existing_df.columns:
                existing_df[col] = np.nan
        
        # generate unique composite key
        key_cols = ['n_features', 'seeds', 'outer_iter']
        existing_df['_key'] = existing_df[key_cols].astype(str).agg('|'.join, axis=1)
        new_df['_key'] = new_df[key_cols].astype(str).agg('|'.join, axis=1)
        
        new_df = new_df[~new_df['_key'].isin(existing_df['_key'])]
        new_df = new_df.drop(columns=['_key'])
        existing_df = existing_df.drop(columns=['_key'])
        output_df = pd.concat(
            [existing_df[columns], new_df[columns]],
            ignore_index=True
        )
    else:
        output_df = new_df

    output_df.to_csv(result_path, index=False)
    
    # save Checkpoint
    with open(checkpoint_path, 'w') as f:
        serializable_tasks = [
            [[list(params) for params in task[0]], list(task[1]), task[2], task[3]]
            for task in processed_tasks
        ]
        json.dump(serializable_tasks, f)


# ==================Result processing function ==================
def analyze_results(df):
    """
    Average performance of multiple outer layer iterations by number of features
    """
    grouped = df.groupby(['n_features', 'seeds']).agg({
        'inner_cv_accuracy_mean': ['mean', 'std'],
        'test_accuracy': ['mean', 'std'],
        'test_error': ['mean', 'std']
    }).reset_index()

    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

    qualified = grouped[
        (grouped['test_accuracy_mean'] > 0.95)
        ]

    return grouped, qualified


# ================== Nested Cross Validation ==================
def to_python_float(value):
    """Convert CuPy/cuML scalar outputs to plain Python float."""
    if isinstance(value, cp.ndarray):
        return float(cp.asnumpy(value).item())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def get_prism_scores(X_data_gpu, y_data_gpu, params, cache_key):
    """Run PRISM on the supplied training split only, with score caching."""
    cached_scores = load_cache(cache_key)
    if cached_scores is not None:
        return cached_scores

    X1_gpu = X_data_gpu[y_data_gpu == 0]
    X2_gpu = X_data_gpu[y_data_gpu == 1]
    scores = cp.asarray(PRISM(X1_gpu, X2_gpu, *params))
    save_cache(scores, cache_key)

    del X1_gpu, X2_gpu
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return scores


def evaluate_prism_params_inner_cv(params, X_train_gpu, y_train_gpu, n_features,
                                  inner_splits, seeds, outer_iter, n_inner_folds):
    """Evaluate one PRISM parameter set with inner CV and default SVM."""
    fold_accuracies = []

    for fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
        X_inner_train_gpu = X_train_gpu[inner_train_idx]
        y_inner_train_gpu = y_train_gpu[inner_train_idx]
        X_inner_val_gpu = X_train_gpu[inner_val_idx]
        y_inner_val_gpu = y_train_gpu[inner_val_idx]

        train_cache_key = get_train_cache_key(
            params,
            seeds,
            outer_iter,
            fold_idx,
            n_inner_folds
        )
        scores_inner = get_prism_scores(
            X_inner_train_gpu,
            y_inner_train_gpu,
            params,
            train_cache_key
        )
        selected_inner = cp.argsort(scores_inner)[-n_features:]

        svm_model = cuSVC()
        svm_model.fit(X_inner_train_gpu[:, selected_inner], y_inner_train_gpu)
        fold_acc = to_python_float(
            svm_model.score(X_inner_val_gpu[:, selected_inner], y_inner_val_gpu)
        )
        fold_accuracies.append(fold_acc)

        del (
            X_inner_train_gpu,
            y_inner_train_gpu,
            X_inner_val_gpu,
            y_inner_val_gpu,
            scores_inner,
            selected_inner,
            svm_model
        )
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()

    return float(np.mean(fold_accuracies)), float(np.std(fold_accuracies))


def process_iteration(seeds, n_features, outer_iter, prism_combinations,
                      n_inner_folds, svm_param_grid):
    """GPU full-process computing, data does not leave the video memory"""
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device(0).use()
    
    # Global data preloaded to GPU
    global X_gpu, y_gpu  
    
    train_idx, test_idx = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seeds['seed1'] + outer_iter)   
        .split(cp.zeros(len(y_gpu)), y_gpu.get())  
    )
    
    X_train_gpu = X_gpu[train_idx]
    y_train_gpu = y_gpu[train_idx]
    X_test_gpu = X_gpu[test_idx]
    y_test_gpu = y_gpu[test_idx]

    # ===== Inner CV parameter selection and outer test evaluation =====
    test_acc = np.nan
    test_error = np.nan
    best_prism_params = None
    best_inner_mean = np.nan
    best_inner_std = np.nan
    best_svm_params = None
    try:
        y_train_cpu = cp.asnumpy(y_train_gpu).astype(np.int32)
        inner_cv = StratifiedKFold(
            n_splits=n_inner_folds,
            shuffle=True,
            random_state=seeds['seed3']
        )
        inner_splits = list(
            inner_cv.split(np.zeros(len(y_train_cpu)), y_train_cpu)
        )

        for params in prism_combinations:
            params = tuple(params)
            inner_mean, inner_std = evaluate_prism_params_inner_cv(
                params,
                X_train_gpu,
                y_train_gpu,
                n_features,
                inner_splits,
                seeds,
                outer_iter,
                n_inner_folds
            )
            if np.isnan(inner_mean):
                continue
            if best_prism_params is None or inner_mean > best_inner_mean:
                best_prism_params = params
                best_inner_mean = inner_mean
                best_inner_std = inner_std

        if best_prism_params is None:
            raise RuntimeError("No valid PRISM parameter set was selected by inner CV.")

        test_cache_key = get_test_cache_key(best_prism_params, seeds['seed1'], outer_iter)
        scores_full = get_prism_scores(
            X_train_gpu,
            y_train_gpu,
            best_prism_params,
            test_cache_key
        )
        selected_full = cp.argsort(scores_full)[-n_features:]

        svm_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seeds['seed3'])
        svm_cv_splits = list(
            svm_cv.split(X_train_gpu[:, selected_full].get(), y_train_cpu)
        )
        grid_search = cuGridSearchCV(
            cuSVC(kernel='rbf'),
            svm_param_grid,
            cv=svm_cv_splits
        )
        grid_search.fit(X_train_gpu[:, selected_full], y_train_gpu)
        test_acc = to_python_float(
            grid_search.score(X_test_gpu[:, selected_full], y_test_gpu)
        )
        test_error = 1.0 - test_acc
        best_svm_params = {
            k: v.item() if hasattr(v, "item") else v
            for k, v in grid_search.best_params_.items()
        }

        del y_train_cpu, inner_splits, scores_full, selected_full, svm_cv_splits, grid_search
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    
    except Exception as e:
        print(f"Nested CV phase error: {str(e)}")
        test_error = np.nan
        import traceback
        traceback.print_exc()  

    try:
        del X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu
    except NameError:
        pass
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp._default_memory_pool.free_all_blocks()
    gc.collect()
    
    return {
        'n_features': n_features,
        'best_prism_params': str(best_prism_params),
        'seeds': str(seeds),
        'outer_iter': outer_iter,
        'inner_cv_accuracy_mean': best_inner_mean,
        'inner_cv_accuracy_std': best_inner_std,
        'test_accuracy': test_acc,
        'test_error': test_error,
        'best_svm_params': str(best_svm_params)
    }


# ================== Performing parallel computations ==================
if __name__ == "__main__":
    mat_data = scipy.io.loadmat('Prostate_GE.mat')
    X_gpu = cp.asarray(mat_data['X'], dtype=cp.float32)  
    y_gpu = cp.asarray(mat_data['Y'].ravel(), dtype=cp.int32)
    y_gpu = (y_gpu < 2).astype(cp.int32)

    # experimental parameters
    param_configs = [
        {
            'n_features': [10, 50, 100, 150, 200],    
            'prism_params': {
                'N': [5], 'K': [500], 'k': [5800], 'k0': [5700]
            },
            'seeds': {
                'seed1': [56],   
                'seed3': [42]
            }
        },
    ]
    svm_param_grid = {
        'C': [2 ** i for i in [-5, -2, 1, 4, 7, 10, 13]],
        'gamma': [2 ** i for i in [-15, -12, -9, -6, -3, 0, 3]]
    }
    n_outer_iter = 30  # outer loop times (30 independent divisions)  
    n_inner_folds = 10  # inner cross validation folds  

    # generate all task combinations
    tasks = []
    for config in param_configs:
        n_feature_list = config['n_features']
        prism_param_grid = config['prism_params']
        seeds_grid = config['seeds']
        prism_combinations = list(product(*[
            [float(p) if isinstance(p, cp.ndarray) else p for p in values]  
            for values in prism_param_grid.values()
        ]))
        seeds_combinations = [
            {'seed1': s1, 'seed3': s3} 
            for s1, s3 in product(
                seeds_grid['seed1'],
                seeds_grid['seed3']
            )
        ]
        
        for n_feat in n_feature_list:
            for seeds in seeds_combinations:
                for iter in range(n_outer_iter):
                    tasks.append((tuple(prism_combinations), seeds, n_feat, iter))

    cp.cuda.Device(0).use()

    # ================== Breakpoint resume logic ==================
    result_path = '/Prostate/partial_results.csv'
    checkpoint_path = '/Prostate/processed_tasks.json'
    # Loading processed tasks
    processed_tasks = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            processed = json.load(f)
            processed_tasks = {
                (
                    tuple(tuple(params) for params in task[0]),
                    tuple(task[1]),
                    task[2],
                    task[3]
                )
                for task in processed
                if len(task) == 4
            }
    # generate a list of remaining tasks
    remaining_tasks = [
        task for task in tasks 
        if task_to_hashable(task) not in processed_tasks
    ]
    print(f"Number of remaining tasks: {len(remaining_tasks)}/{len(tasks)}")
    
    # ================== Batch Processing Logic ==================
    batch_size = 30
    for i in range(0, len(remaining_tasks), batch_size):
        batch_tasks = remaining_tasks[i:i+batch_size]
        current_batch_results = []
        try:
            with Parallel(n_jobs=1, prefer="processes", verbose=10) as parallel:
                batch_results = parallel(
                    delayed(process_iteration)(
                        seeds,
                        n_feat,
                        iter,
                        prism_combinations,
                        n_inner_folds,
                        svm_param_grid
                    )
                    for prism_combinations, seeds, n_feat, iter in tqdm(batch_tasks)
                )
                current_batch_results.extend(batch_results)
            batch_hashes = {task_to_hashable(task) for task in batch_tasks}
            processed_tasks.update(batch_hashes)  

            save_checkpoint(
                current_batch_results,
                processed_tasks,  
                result_path,
                checkpoint_path
            )
        except KeyboardInterrupt:
            print("Server or user interruption, saving processed tasks...")
            save_checkpoint(
                current_batch_results,
                processed_tasks,  
                result_path,
                checkpoint_path
            )
            exit()
        finally:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            gc.collect()


    # ================== Final result ==================
    final_df = pd.read_csv(result_path)
    grouped_df, qualified_df = analyze_results(final_df)
    final_df.to_csv('/Prostate/full_results.csv', index=False)
    grouped_df.to_csv('/Prostate/grouped_results.csv', index=False)

   
