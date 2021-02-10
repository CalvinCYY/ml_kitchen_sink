from sklearn.model_selection import train_test_split, cross_val_score
from ml_kitchen_sink.kitchen_sink.modules.search_space import search_space
from ml_kitchen_sink.kitchen_sink.modules.print_scores import print_scores
from skopt import gp_minimize, dump, load, callbacks
from skopt.callbacks import CheckpointSaver
from skopt.utils import use_named_args
from functools import wraps

#from sklearn.pipeline import Pipeline
#from ml_kitchen_sink.cv import gauss_grid
#from ml_kitchen_sink.cv import models

import pandas as pd
import numpy as np

'''
def on_step(optim_result):
    """
    View scores after each iteration
    while performing Bayesian
    Optimization in Skopt
    """
    score = reg_gp.best_score_
    print("best score: %s" % score)
    if score >= 0.90:
        print('Interrupting!')
        return True
'''
class cross_validation(object):
    def __init__(self, id, algorithm=None, scoring='neg_mean_absolute_error', checkpoint=False):
        self.id = id
        self.algorithm = algorithm
        self.search_space = None
        self.sklearn_model = None
        self.X = []
        self.y = []
        self.test_X = []
        self.test_y = []
        self.cv_model = None
        self.scoring = scoring
        self.checkpoint = checkpoint

    def check_args(self):

        implemented_models = ['DTR', 'KRR', 'KNN', 'RFR']

        if self.algorithm not in implemented_models:
            print(f'The algorithm {self.algorithm} has not been implemented (yet)')

    def get_input(self, train_data, test_data, split=False):

        X_train = train_data['atomic_rep'].to_list()
        X_test = train_data['shift']

        if split:
            X_train, X_test, y_train, y_test = train_test_split(X_train, X_test)
            self.X = X_train
            self.y = X_test
            self.test_X = y_train
            self.test_y = y_test

        else:
            self.X = X_train
            self.y = X_test
            self.test_X = test_data['atomic_rep'].to_list()
            self.test_y = test_data['shift']

    def get_space(self):

        self.search_space, self.sklearn_model = search_space(self.algorithm)

    def save_model(self, reg_gp, file, compress=0):
        dump(reg_gp, file, compress=compress)

    def from_checkpoint(self, checkpoint_file):
        reg_gp = load(checkpoint_file)
        print(reg_gp.fun)
        self.checkpoint = True
        return reg_gp

    def gaussian(self, save=True):

        regressor = self.sklearn_model
        space = self.search_space

        @use_named_args(space)
        def objective(**params):
            regressor.set_params(**params)
            return -np.mean(cross_val_score(regressor, self.X, self.y, cv=5, scoring=self.scoring))

        checkpoint_saver = CheckpointSaver(f"{self.algorithm}_{self.id}_checkpoint.pkl", compress=9)

        if self.checkpoint:
            x0 = reg_gp.x_iters
            y0 = reg_gp.func_vals

            reg_gp = gp_minimize(objective, space, x0=x0, y0=y0, verbose=True, callback=[checkpoint_saver], n_jobs=-1)

        else:
            reg_gp = gp_minimize(objective, space, verbose=True, callback=[checkpoint_saver], n_jobs=-1)

        self.cv_model = reg_gp

        if save:
            filename = f"{self.algorithm}_{self.id}.pkl"
            self.save_model(reg_gp, filename)

        print(f'{self.algorithm} best score: ', reg_gp.fun)
        print(f'{self.algorithm} best params:')
        print_scores(self.algorithm, reg_gp)
        print(f'{self.algorithm} val. score: %s' % reg_gp.best_score_)
        print(f'{self.algorithm} test score: %s' % reg_gp.score(self.test_X, self.test_y))
        #print(f"best params: %s" % str(reg_gp.best_params_))
