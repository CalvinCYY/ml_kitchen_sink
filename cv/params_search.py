def reg_param_search():
    params = {
    'DTR':{
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
    '   model__ccp_alpha': Real(1e-7, 1e+1, prior='log-uniform'),
},
}
