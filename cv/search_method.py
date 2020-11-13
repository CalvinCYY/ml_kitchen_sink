from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def random_opt(estimator, param_distrubution, n_iter=50, scoring='neg_mean_absolute_error', n_jobs=-1, cv=5, verbose=1, random_state='rs'):
    search = RandomizedSearchCV(estimator, param_distrubution, n_iter, scoring, n_jobs, cv=cv, verbose=verbose, random_state=random_state)

    return search

def grid_opt(estimator, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=5, verbose=1):
    search = GridSearchCV(estimator, param_grid, scoring, n_jobs, cv=cv, verbose=verbose)

    return search
