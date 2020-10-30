import pandas as pd
import numpy as np
import tqdm

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from ml_kitchen_sink.cv import params, models

def model_selection_cv(atoms_file, pairs_file, type_of_pred='regression' rs=100, ts=0.2, n_splits=5):

    atoms_df = pd.read_pickle(atoms_file)
    pairs_df = pd.read_pickle(pairs_file)
    X = np.array(atoms_df['atomic_rep']).tolist()
    Y = atoms_df['shift']

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=ts, random_state=rs)

    kfold = model_selection.KFold(n_splits = n_splits, shuffle = True, random_state = rs)

    if type_of_pred='regression':
        models = Get_reg_models()
        grids = Get_reg_params_grid()
    elif type_of_pred='classification':
        models = Get_class_models()
        grids = Get_class_params_grid()

    for model in models:
        for grid in grids:
            if model.keys() == grid.keys():
                search = GridSearchCV(model, grid_search, scoring='neg_mean_absolute_error', cv=kfold, verbose=5)
                result = search.fit(X_train, Y_train)
                print('Best Score: %s' % result.best_score_)
                print('Best Hyperparameters: %s' % result.best_params_)
            else:
                continue
