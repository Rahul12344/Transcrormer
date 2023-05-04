# import sklearn.make_pipeline for easier pipeline creation
from sklearn.pipeline import Pipeline

# import sklearn.model_selection for hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

# create pipeline for training sklearn model
def train_sklearn_model_and_return_pipeline(scaler=None, dimensionality_reduction=None, model=None):
    classification_model_pipeline = Pipeline(steps=[('scaler', scaler), ('dimensionality_reduction', dimensionality_reduction), ('regressor', model)])
    return classification_model_pipeline

# predict from sklearn model
def predict_from_sklearn_pipeline(classification_model_pipeline, X_test=None, y_test=None):
    return classification_model_pipeline.predict(X_test), classification_model_pipeline.score(X_test, y_test)

# tune sklearn model
def tune_sklearn_pipeline(classification_model_pipeline, params, X_train=None, y_train=None):
    clf = RandomizedSearchCV(classification_model_pipeline, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1, n_iter=50)
    optimal_pipeline = clf.fit(X_train, y_train)
    return optimal_pipeline