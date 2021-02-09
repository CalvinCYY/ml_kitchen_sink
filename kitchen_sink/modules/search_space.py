from skopt.space import Real, Categorical, Integer

from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

def search_space(alg):

    if alg == 'DTR':
        space = [
            Categorical(['mse', 'friedman_mse', 'mae'], name='criterion'),
            Categorical(['best', 'random'], name='splitter'),
            Integer(2, 40, name='min_samples_split'),
            Integer(1, 40, name='min_samples_leaf'),
            Real(0, 0.5, prior='uniform', name='min_weight_fraction_leaf'),
            Categorical(['auto', 'sqrt', 'log2'], name='max_features'),
            Integer(0,10, name='min_impurity_decrease'),
            #'ccp_alpha': Real(0, 10, prior='log-uniform'),
            ]
        regressor = DecisionTreeRegressor()

    elif alg == 'KRR':
        space = [
            Real(0.0, 5.0, prior='uniform', name='alpha'),
            Categorical(['linear', 'laplacian', 'polynomial', 'rbf', 'sigmoid', 'chi2', 'additive_chi2'], name='kernel'),
            Real(0.0, 5.0, prior='uniform', name='gamma'),
            Real(0.0, 10.0, prior='uniform', name='degree'),
            Real(0.0, 5.0, prior='uniform', name='coef0'),
            ]
        regressor = KernelRidge()

    elif alg == 'KNN':
        space = [
            Integer(1, 50, name='n_neighbors'),
            Categorical(['uniform', 'distance'], name='weights'),
            Categorical(['auto'], name='algorithm'),
            Integer(10, 100, name='leaf_size'),
            Integer(1, 10, name='p'),
            Categorical(['euclidean', 'manhattan', 'chebyshev', 'minkowski'], name='metric'),
            ]
        regressor = KNeighborsRegressor()

    elif alg == 'RFR':
        space = [
            Integer(100, 1000, name='n_estimators'),
            Categorical(['mse'], name='criterion'),
            Integer(2, 100, name='min_samples_split'),
            Integer(1, 100, name='min_samples_leaf'),
            Real(0, 0.5, prior='uniform', name='min_weight_fraction_leaf'),
            Categorical(['auto'], name='max_features'),
            Integer(0,10, name='min_impurity_decrease'),
            #Categorical(['True', 'False'], name='bootstrap'),
            #Categorical(['True', 'False'], name='oob_score'),
            Real(0.0, 1.0, prior='log-uniform', name='ccp_alpha'),
            ]
        regressor = RandomForestRegressor()

    return space, regressor
