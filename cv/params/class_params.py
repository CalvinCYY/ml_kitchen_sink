DTC_param = {
    'criterion': ['mse', 'friedman_mse', 'mae'],
    'splitter': ['best', 'random'],
    'max_depth': [2, 4, 6, 8, 10, None],
    'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_weight_fraction_leaf': [],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'random_state': [rs],
    'max_leaf_nodes': [None],
    'min_impurity_decrease': [],
    'ccp_alpha': []
    }
