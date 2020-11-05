import pandas as pd
import numpy as np
import tqdm

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from ml_kitchen_sink.cv import params, models

def random_opt(estimator=model, param_distrubution=grid, n_iter =50, cv=kfold, verbose=1, random_state=rs, n_jobs=-1):
    search = RandomizedSearchCV(estimator=estimator, param_distrubution=param_distrubution, n_iter =n_iter, cv=cv, verbose=verbose, random_state=random_state, n_jobs=n_jobs)
    result = search.fit(X_train, Y_train)
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    return result

def grid_opt(estimator=model, param_grid=grid, scoring=scoring, cv=kfold, verbose=1):
    search = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=cv, verbose=verbose)
    result = search.fit(X_train, Y_train)
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    return result

def model_selection_cv(atoms_file, pairs_file, type_of_pred='regression', type_of_opt='random' rs=100, ts=0.2, n_splits=5, scoring='neg_mean_absolute_error'):

    atoms_df = pd.read_pickle(atoms_file)
    pairs_df = pd.read_pickle(pairs_file)
    X = np.array(atoms_df['atomic_rep']).tolist()
    Y = atoms_df['shift']

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=ts, random_state=rs)

    kfold = model_selection.KFold(n_splits = n_splits, shuffle = True, random_state = rs)

    if type_of_pred='regression':
        models = models.Get_reg_models()
        grids = params.Get_reg_params_grid()

    elif type_of_pred='classification':
        models = models.Get_class_models()
        grids = params.Get_class_params_grid()

    else raise Exception("Only regression and classification predictions allowed")

    result_dict = {}

    for model in models:
        for grid in grids:
            if type_of_opt = 'random':
                if name.keys() == grid.keys():
                    result = random_opt(model, grid, scoring=scoring, cv=kfold, verbose=1, random_state=rs, n_jobs=-1)
                    result_dict

            else:
                continue
'''
    for name, model in models:
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s MAE: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
'''
