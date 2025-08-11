from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def get_imputers(seed: int):
    """Return dict[name -> base_estimator] for IterativeImputer."""
    return {
        "RF": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "GradientBoosting": GradientBoostingRegressor(random_state=seed),
        "SVM": SVR(),
        "MLP": MLPRegressor(random_state=seed, max_iter=500),
        "DecisionTree": DecisionTreeRegressor(random_state=seed),
        "AdaBoost": AdaBoostRegressor(random_state=seed),
        "Bagging": BaggingRegressor(random_state=seed),
        "HGBR": HistGradientBoostingRegressor(random_state=seed),
        "RegressionTree": DecisionTreeRegressor(random_state=seed),  # legacy alias
        # "MICE" -> use IterativeImputer(estimator=None)
    }

def make_iterative_imputer(estimator, seed: int):
    return IterativeImputer(estimator=estimator, max_iter=10, random_state=seed, sample_posterior=False)
