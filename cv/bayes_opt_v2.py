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

from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#point to df with x, y data
def model_selection_cv(atoms_file,
                       pairs_file = None,
                       splits = 5,
                       type_of_pred = 'regression',
                       type_of_opt = 'random',
                       rs = 42,
                       ts = 0.2,
                       scoring = 'neg_mean_absolute_error',
                       ):

    atoms_df = pd.read_pickle(atoms_file)

    if pairs_file != None:
        pairs_df = pd.read_pickle(pairs_file)

    X = np.array(atoms_df['atomic_rep']).tolist()
    Y = atoms_df['shift']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    pipe = Pipeline([
        ('model', DecisionTreeRegressor())
        ])

        dtr_search = {
        'model': Categorical([DecisionTreeRegressor()]),
        'model__criterieon': Categorical(['mse', 'friedman_mse', 'mae']),
        'model__splitter': Categorical(['best', 'random']),
        'model__max_depth': Categorical(['None']),
        'model__min_samples_split': Integer(2, 20),
        'model__min_samples_leaf': Integer(1, 20),
        'model__min_weight_fraction_leaf':,
        'model__max_features': Categorical(['auto', 'sqrt', 'log2']),
        'model__max_leaf_nodes': Categorical(['None']),
        'model__min_impurity_decrease':Integer(0,10),
        'model__ccp_alpha': Real(1e-7, 1e+1, prior='log-uniform'),
    }

    opt = BayesSearchCV(
        pipe,
        # (parameter space, # of evaluations)
        (dtr_search, 40),
        cv=5
    )

    opt.fit(X_train, y_train)

    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(X_test, y_test))
    print("best params: %s" % str(opt.best_params_))
