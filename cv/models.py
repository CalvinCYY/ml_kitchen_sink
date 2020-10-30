from sklearn.linear_model import LogisticRegression, SGDRegressor, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

def Get_reg_models():
    models = {
    'DTR': DecisionTreeRegressor(),
    'KRR': KernelRidge(),
    'EN': ElasticNe(),
    'ADA': AdaBoostRegressor(),
    'SGD': SGDRegressor(),
    'MLP': MLPRegressor(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'GPR': GaussianProcessRegressor(),
    }

    return models
    
def Get_class_models():
    models = {

    }

    return models
