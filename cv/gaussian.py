from skopt import gp_minimize, dump, load
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective, plot_histogram
from skopt.utils import use_named_args
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
#from ml_kitchen_sink.cv import gauss_grid
#from ml_kitchen_sink.cv import models

import pandas as pd
import numpy as np
from pickle import dump

'''
def on_step(optim_result):
    """
    View scores after each iteration
    while performing Bayesian
    Optimization in Skopt
    """
    score = reg_gp.best_score_
    print("best score: %s" % score)
    if score >= -0.90:
        print('Interrupting!')
        return True
'''

def gaussian(test_atoms, alg, file):

    atoms_df = pd.read_pickle(test_atoms)
    #pairs_df = pd.read_pickle(pairs_file)

    X = np.array(atoms_df['atomic_rep']).tolist()
    y = atoms_df['shift']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    if alg == 'DTR':
        space = [
            Categorical(['mse', 'friedman_mse', 'mae'], name='criterion'),
            Categorical(['best', 'random'], name='splitter'),
            Integer(2, 40, name='min_samples_split'),
            Integer(1, 40, name='min_samples_leaf'),
            Real(0, 0.5, prior='uniform', name='min_weight_fraction_leaf'),
            Categorical(['auto', 'sqrt', 'log2'], name='max_features'),
            Integer(0,10, name='min_impurity_decrease'),
            #'ccp_alpha': Real(0, 10, prior='log-uniform'),
            ]
        regressor = DecisionTreeRegressor()

    elif alg == 'KRR':
        space = [
            Real(0.0, 5.0, prior='uniform', name='alpha'),
            Categorical(['linear', 'laplacian', 'polynomial', 'rbf', 'sigmoid', 'chi2', 'additive_chi2'], name='kernel'),
            Real(0.0, 5.0, prior='uniform', name='gamma'),
            Real(0.0, 10.0, prior='uniform', name='degree'),
            Real(0.0, 5.0, prior='uniform', name='coef0'),
            ]
        regressor = KernelRidge()

    elif alg == 'KNN':
        space = [
            Integer(1, 50, name='n_neighbors'),
            Categorical(['uniform', 'distance'], name='weights'),
            Categorical(['auto'], name='algorithm'),
            Integer(10, 100, name='leaf_size'),
            Integer(1, 10, name='p'),
            Categorical(['euclidean', 'manhattan', 'chebyshev', 'minkowski'], name='metric'),
            ]
        regressor = KNeighborsRegressor()

    elif alg == 'RFR':
        space = [
            Integer(100, 1000, name='n_estimators'),
            Categorical(['mse', 'mae'], name='criterion'),
            Integer(2, 100, name='min_samples_split'),
            Integer(1, 100, name='min_samples_leaf'),
            Real(0, 0.5, prior='uniform', name='min_weight_fraction_leaf'),
            Categorical(['auto', 'sqrt', 'log2'], name='max_features'),
            Integer(0,10, name='min_impurity_decrease'),
            Categorical(['True', 'False'], name='bootstrap'),
            Categorical(['True', 'False'], name='oob_score'),
            Real(0.0, 1.0, prior='log-uniform', name='ccp_alpha'),
            ]
        regressor = RandomForestRegressor()

    @use_named_args(space)
    def objective(**params):
        regressor.set_params(**params)
        return -np.mean(cross_val_score(regressor, X_train, y_train, cv=5, scoring='neg_mean_absolute_error'))

    reg_gp = gp_minimize(objective, space, verbose=True)

    print('best score: {}'.format(reg_gp.fun))

    print('best params:')

    if alg == 'DTR':
        print('               criterion: {}'.format(reg_gp.x[0]))
        print('                splitter: {}'.format(reg_gp.x[1]))
        print('       min_samples_split: {}'.format(reg_gp.x[2]))
        print('        min_samples_leaf: {}'.format(reg_gp.x[3]))
        print('min_weight_fraction_leaf: {}'.format(reg_gp.x[4]))
        print('            max_features: {}'.format(reg_gp.x[5]))
        print('   min_impurity_decrease: {}'.format(reg_gp.x[6]))

    elif alg == 'KRR':
        print('                   alpha: {}'.format(reg_gp.x[0]))
        print('                  kernel: {}'.format(reg_gp.x[1]))
        print('                   gamma: {}'.format(reg_gp.x[2]))
        print('                  degree: {}'.format(reg_gp.x[3]))
        print('                   coef0: {}'.format(reg_gp.x[4]))

    elif alg == 'KNN':
        print('             n_neighbors: {}'.format(reg_gp.x[0]))
        print('                 weights: {}'.format(reg_gp.x[1]))
        print('               algorithm: {}'.format(reg_gp.x[2]))
        print('               leaf_size: {}'.format(reg_gp.x[3]))
        print('                       p: {}'.format(reg_gp.x[4]))
        print('                  metric: {}'.format(reg_gp.x[5]))

    elif alg == 'RFR':
        print('            n_estimators: {}'.format(reg_gp.x[0]))
        print('               criterion: {}'.format(reg_gp.x[1]))
        print('       min_samples_split: {}'.format(reg_gp.x[2]))
        print('        min_samples_leaf: {}'.format(reg_gp.x[3]))
        print('min_weight_fraction_leaf: {}'.format(reg_gp.x[4]))
        print('            max_features: {}'.format(reg_gp.x[5]))
        print('   min_impurity_decrease: {}'.format(reg_gp.x[6]))
        print('               bootstrap: {}'.format(reg_gp.x[7]))
        print('               oob_score: {}'.format(reg_gp.x[8]))
        print('               ccp_alpha: {}'.format(reg_gp.x[9]))

    dump(reg_gp, file)

    print("val. score: %s" % reg_gp.best_score_)
    print("test score: %s" % reg_gp.score(X_test, y_test))
    print("best params: %s" % str(reg_gp.best_params_))
