from skopt import gp_minimize, dump, load
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective, plot_histogram
from skopt.utils import use_named_args
from sklearn.tree import DecisionTreeRegressor
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score

import pandas as pd
import numpy as np
'''
To Do:
    Utilize pipeline fully
    Tweak surrogate method
    Implement more models
    Add in loop for multiple models and params
'''
#point to df with x, y data
test_atoms = "/mnt/storage/home/bd20841/scratch/test_files/ml_cv/df_gen/atoms_df_data4.pkl"

atoms_df = pd.read_pickle(test_atoms)
#pairs_df = pd.read_pickle(pairs_file)

X = np.array(atoms_df['atomic_rep']).tolist()
y = atoms_df['shift']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

dtr_space = [
    Categorical(['mse', 'friedman_mse', 'mae'], name='criterion'),
    Categorical(['best', 'random'], name='splitter'),
    Integer(2, 20, name='min_samples_split'),
    Integer(1, 20, name='min_samples_leaf'),
    Real(0, 0.5, prior='uniform', name='min_weight_fraction_leaf'),
    Categorical(['auto', 'sqrt', 'log2'], name='max_features'),
    Integer(0,10, name='min_impurity_decrease'),
    #'ccp_alpha': Real(0, 10, prior='log-uniform'),
]

regressor = DecisionTreeRegressor()

@use_named_args(dtr_space)
def objective(**params):
        regressor.set_params(**params)
        return -np.mean(cross_val_score(regressor, X_train, y_train, cv=5, scoring='neg_mean_absolute_error'))

reg_gp = gp_minimize(objective, dtr_space, verbose=True)

print('best score: {}'.format(reg_gp.fun))

print('best params:')
print('               criterion: {}'.format(reg_gp.x[0]))
print('                splitter: {}'.format(reg_gp.x[1]))
print('       min_samples_split: {}'.format(reg_gp.x[2]))
print('        min_samples_leaf: {}'.format(reg_gp.x[3]))
print('min_weight_fraction_leaf: {}'.format(reg_gp.x[4]))
print('            max_features: {}'.format(reg_gp.x[5]))
print('   min_impurity_decrease: {}'.format(reg_gp.x[6]))

dump(reg_gp, 'dtr_result.pkl')
'''
def on_step(optim_result):
    """
    View scores after each iteration
    while performing Bayesian
    Optimization in Skopt"""
    score = dtr_opt.best_score_
    print("best score: %s" % score)
    if score >= -0.90:
        print('Interrupting!')
        return True

    dtr_opt.fit(X_train, y_train)

    print("val. score: %s" % dtr_opt.best_score_)
    print("test score: %s" % dtr_opt.score(X_test, y_test))
    print("best params: %s" % str(dtr_opt.best_params_))
'''
