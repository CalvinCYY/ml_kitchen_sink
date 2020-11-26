# Copyright 2020 Will Gerrard, Calvin Yiu
#This file is part of autoenrich.

#autoenrich is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#autoenrich is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with autoenrich.  If not, see <https://www.gnu.org/licenses/>.

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective, plot_histogram

from sklearn.tree import DecisionTreeRegressor
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

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
def model_selection_cv(atoms_file,
                       pairs_file = None,
                       splits = 5,
                       rs = 42,
                       ts = 0.2,
                       scoring = 'neg_mean_absolute_error',
                       ):

    atoms_df = pd.read_pickle(atoms_file)

    if pairs_file != None:
        pairs_df = pd.read_pickle(pairs_file)

    X = np.array(atoms_df['atomic_rep']).tolist()
    y = atoms_df['shift']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs)

    dtr_search = {
        'criterion': Categorical(['mse', 'friedman_mse', 'mae']),
        'splitter': Categorical(['best', 'random']),
        #'max_depth': Categorical(['None']),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 20),
        'min_weight_fraction_leaf':Real(0, 0.5, prior='uniform'),
        'max_features': Categorical(['auto', 'sqrt', 'log2']),
        #'max_leaf_nodes': Categorical(['None']),
        'min_impurity_decrease':Integer(0,10),
        #'ccp_alpha': Real(0, 10, prior='log-uniform'),
    }

    dtr_opt = BayesSearchCV(
        DecisionTreeRegressor(),
        # (parameter space, # of evaluations)
        dtr_search,
        n_iter=50,
        scoring=scoring,
        cv=splits,
        random_state=rs
    )

    return dtr_opt, X_train, y_train, X_test, y_test

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
