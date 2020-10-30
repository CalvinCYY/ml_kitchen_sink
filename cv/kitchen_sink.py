import pandas as pd
import numpy as np
import tqdm

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

grid_search = dict()
grid_search['criterion'] = ['mse', 'friedman_mse', 'mae']
grid_search['splitter'] = ['best', 'random']
grid_search['max_depth'] = [2, 4, 6, 8, 10, None]
grid_search['min_samples_split'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
grid_search['min_samples_leaf'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#grid_search['min_weight_fraction_leaf'] = []
grid_search['max_features'] = ['auto', 'sqrt', 'log2', None]
grid_search['random_state'] = [rs]
#grid_search['max_leaf_nodes'] = [None]
#grid_search['min_impurity_decrease'] = []
#grid_search['ccp_alpha'] = []

search = GridSearchCV(model, grid_search, scoring='neg_mean_absolute_error', cv=kfold, verbose=5)
result = search.fit(X_train, Y_train)

print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

def model_selection_cv(atoms_file, pairs_file, rs=100, ts=0.2, cv=kfold):

    atoms_df = pd.read_pickle(atoms_file)
    pairs_df = pd.read_pickle(pairs_file)
    X = np.array(atoms_df['atomic_rep']).tolist()
    Y = atoms_df['shift']

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=ts, random_state=rs)

    kfold = model_selection.KFold(n_splits = 10, shuffle = True, random_state = rs)
