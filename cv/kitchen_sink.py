import pandas as pd
import numpy as np
import tqdm

from sklearn import model_selection
from ml_kitchen_sink.cv import params, models
from ml_kitchen_sink.cv.search_method import random_opt, grid_opt

def model_selection_cv(atoms_file, pairs_file=None, splits=5, type_of_pred='regression', type_of_opt='random', rs=42, ts=0.2, scoring='neg_mean_absolute_error'):

    atoms_df = pd.read_pickle(atoms_file)

    if pairs_file != None:
        pairs_df = pd.read_pickle(pairs_file)

    X = np.array(atoms_df['atomic_rep']).tolist()
    Y = atoms_df['shift']

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=ts, random_state=rs)

    kfold = model_selection.KFold(n_splits=splits, shuffle = True, random_state = rs)

    if type_of_pred == 'regression':
        model = models.Get_reg_models()
        grid = params.Get_reg_param_grid()

    elif type_of_pred == 'classification':
        model = models.Get_class_models()
        grid = params.Get_class_param_grid()

    else:
        raise Exception("Only regression and classification predictions allowed")

    result_dict = {}

    for model_key, model_value in model.items():
        for grid_key, grid_value in grid.items():
            if model_key == grid_key:
                print('grid match found')

                if type_of_opt == 'random':
                    search = random_opt(model_value, grid_value,  scoring=scoring, cv=kfold, verbose=1, random_state=rs, n_jobs=-1 )
                    result = search.fit(X_train, Y_train)
                    print('Best Score: %s' % result.best_score_)
                    print('Best Hyperparameters: %s' % result.best_params_)
                    result_dict[model_key] = result

                elif type_of_opt =='grid':
                    search = grid_opt(model_value, grid_value, scoring=scoring, n_jobs=-1, cv=kfold, verbose=1)
                    result = search.fit(X_train, Y_train)
                    print('Best Score: %s' % result.best_score_)
                    print('Best Hyperparameters: %s' % result.best_params_)
                    result_dict[model_key] = result

            else:
                continue
    return result_dict
    print(result_dict)

'''
    for name, model in models:
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s MAE: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
'''
