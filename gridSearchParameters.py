# -*- coding: utf-8 -*-
"""Parameters for SVC, kNN and LDA, suitable for GridSearchCV from Sklearn."""

def para_svc():
    """GridSearchCV parameteres for SVC."""

    para_svc = [{
        'C': [10000, 1000, 100, 10, 1, 0.1, 0.01, 1e-3, 1e-4, 1e-5],
        'cache_size': [1000],
        # 'class_weight': [],
        # 'coef0': [],
        # 'decision_function_shape': ['ovo'],
        # 'degree': [],
        'gamma': ['auto'],
        'kernel': ['linear', 'rbf'],
        'max_iter': [100, 1000, 5000, 10000, 100000, -1],
        # 'probability': [True],
        # 'random_state': [1000],
        'shrinking': [True, False],
        'tol': [1e-2, 1e-3, 1e-4]},
        {   
            # svc variant with sigmoidal kernel
        'C': [10000, 1000, 100, 10, 1, 0.1, 0.01, 1e-3, 1e-4, 1e-5],
        'cache_size': [1000],
        'coef0': [0, 1],
        'gamma': ['auto'],
        'kernel': ['sigmoid'],
        'max_iter': [100, 1000, 5000, 10000, 100000, -1],
        'shrinking': [True, False],
        'tol': [1e-2, 1e-3, 1e-4]},
        {
            # svc variant with polynomial kernel
        'C': [10000, 1000, 100, 10, 1, 0.1, 0.01, 1e-3, 1e-4, 1e-5],
        'cache_size': [1000],
        'coef0': [0, 1],
        'degree': [1,2,3,4,5],
        'gamma': ['auto'],
        'kernel': ['poly'],
        'max_iter': [100, 1000, 5000, 10000, 100000, -1],
        'shrinking': [True, False],
        'tol': [1e-2, 1e-3, 1e-4]}]
    return para_svc

def para_knn():
    """GridSearchCV parameteres for kNN."""

    para_knn = [{
        'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20, 21, 25, 30, 31],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree'],
        'leaf_size': [5, 10, 15, 30, 50],
        'p': [1, 2],
        # 'metric': ('minkowski'),
        # 'metric_params': (),
        # 'n_jobs': (None),
        },
        {
        'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20, 21, 25, 30, 31],
        'weights': ['uniform', 'distance'],
        'algorithm': ['brute'],
        'p': [1, 2]}]
    return para_knn

def para_lda():
    """GridSearchCV parameteres for LDA."""
    
    para_lda = [{
        'solver': ['svd'],
        # 'shrinkage': ['auto'],
        'n_components': [5, 10, 15, 20, 25, 30],
        # 'priors': [],
        # 'store_covariance': [],
        'tol': [1e-2, 1e-3, 1e-4]},
        {
        'solver': ['lsqr', 'eigen'],
        'shrinkage': ['auto'],
        'n_components': [5, 10, 15, 20, 25, 30],
        'tol': [1e-2, 1e-3, 1e-4]}]
    return para_lda