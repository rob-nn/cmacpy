import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.metrics import euclidean_distances

class CMACR (BaseEstimator, RegressorMixin):
    def __init__(self, num_active_cells=5):
        self._num_active_cells = num_active_cells
        print "__init__"

    def fit(self, X, y):
        self._cmac = CMAC
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self,['X_', 'y_'])
        X = check_array(X)
        y = np.zeros((X.shape[0]))
        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


from sklearn.utils.estimator_checks import check_estimator

if (__name__ == "__main__"):
    check_estimator(CMACR)
