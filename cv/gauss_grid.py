from skopt.space import Categorical, Real, Integer

get_gauss_grid():
    {
    'DTR': [
        Categorical(['mse', 'friedman_mse', 'mae'], name='criterion'),
        Categorical(['best', 'random'], name='splitter'),
        Integer(2, 40, name='min_samples_split'),
        Integer(1, 40, name='min_samples_leaf'),
        Real(0, 0.5, prior='uniform', name='min_weight_fraction_leaf'),
        Categorical(['auto', 'sqrt', 'log2'], name='max_features'),
        Integer(0,10, name='min_impurity_decrease'),
        #'ccp_alpha': Real(0, 10, prior='log-uniform'),
        ],
    'KRR': [
        Real(0, 1, prior='log', name='alpha'),
        Categorical(['linear', 'chi2', 'laplacian', 'sigmoid', 'rbf', 'polynomial'], name='kernel'),
        Real(0, 1, prior='log,', name='gamma'),
        Integer(0, 10, name='degree'),
        Integer(0, 5, name='coef0'),
        #Categorical([], name='kernel_params'),
        ]
    }
