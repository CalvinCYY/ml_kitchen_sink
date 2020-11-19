def Get_reg_param_grid():
    params = {
    'DTR':{
        'criterion': ['mse', 'friedman_mse', 'mae'],
        #'splitter': [],
        'max_depth': [2, 4, 6, 8, 10, None],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [2, 4, 6, 8, 10],
        #'min_weight_fraction_leaf': [],
        'max_features': ['auto', 'sqrt', 'log2', None],
        #'random_state': [rs],
        #'max_leaf_nodes': [None],
        #'min_impurity_decrease': [],
        #'ccp_alpha': [],
        },
    }

    return params

def Get_class_param_grid():
    params = {
    'LR':{
        ''
        },
    '':{

        }
    }
    return params
