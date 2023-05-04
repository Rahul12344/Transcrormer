# Description: Model parameters for the different models
# Compare the performance of the different models over various parameters

# check over various scaling methods
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# import dimensionality reduction functions
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap

# random seed
RANDOM_SEED = 42

rfr = RandomForestRegressor()
mlp = MLPRegressor()
svr = SVR()


# random forest parameters
RFR_PARAMS = {
    'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler(), Normalizer(), QuantileTransformer(), PowerTransformer()],
    'regressor': [rfr],
    'regressor__warm_start': [False],
    'regressor__random_state': [RANDOM_SEED],
    'regressor__n_estimators': [10, 100, 1000],
    'regressor__criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
    'regressor__max_depth': [None, 4, 16, 256, 1024],
    'dimensionality_reduction': [PCA(), Isomap()],
    'dimensionality_reduction__n_components': [77, 384, 691]
}

# mlp classifier parameters
MLP_PARAMS = {
    'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler(), Normalizer(), QuantileTransformer(), PowerTransformer()],
    'regressor': [mlp],
    'regressor__warm_start': [False],
    'regressor__random_state': [RANDOM_SEED],
    'regressor__max_iter': [1000],
    'regressor__hidden_layer_sizes': [(64,32,4),(64,32,16,4),(64,32,16,8,4),(64,32,16,8,4,2)],
    'regressor__learning_rate': ['constant', 'invscaling', 'adaptive'],
    'regressor__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'regressor__solver': ['lbfgs', 'sgd', 'adam'],
    'dimensionality_reduction': [PCA(), Isomap()],
    'dimensionality_reduction__n_components': [77, 384, 691]
}

# svr parameters
SVR_CONST_PARAMS = {
    'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler(), Normalizer(), QuantileTransformer(), PowerTransformer()],
    'regressor': [svr],
    'regressor__max_iter': [1000],
    'regressor__gamma': ['auto', 'scale'],
    'regressor__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'regressor__degree': [2, 3, 4, 5],
    'regressor__shrinking': [True, False],
    'regressor__C': [0.1, 1, 10, 100, 1000],
    'regressor__coef0': [0.0, 0.1, 1.0, 10.0],
    'regressor__epsilon': [0.1, 0.5, 1.0, 10.0],
    'dimensionality_reduction': [PCA(), Isomap()],
    'dimensionality_reduction__n_components': [77, 384, 691]
}   

dimensionality_reduction_functions = {
    'PCA': PCA(),
    'TSNE': TSNE(),
    'MDS': MDS(),
    'Isomap': Isomap()
}

params = [RFR_PARAMS, MLP_PARAMS, SVR_CONST_PARAMS]
