"""Module containing estimators for diverse learning tasks


.. moduleauthor:: Yassine LAGUEL
"""

from sklearn.base import RegressorMixin
from .risk_optimization import RiskOptimizer
from .losses import *
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


class DRLinearRegression(RiskOptimizer, RegressorMixin):

    def __init__(self, p=0.9, mu=1.0, lmbda=None, fit_intercept=True):

        # Linear Model Related Parameters
        self.lmbda = lmbda
        self.fit_intercept = fit_intercept

        params = {
            'algorithm': 'l-bfgs',
            'p': p,
            'mu': mu,
        }
        self.coef_ = 0.
        self.intercept_ = 0.
        super().__init__(self.l2, self.l2_prime, params=params)

    def fit(self, x, y, logs=False, verbose_mode=False, logs_freq=1):
        if self.lmbda is None:
            self.lmbda = 1.0/y.shape[0]
        if self.fit_intercept:
            dim_w = x.shape[1]+1
            formatted_x = np.ones((x.shape[0], dim_w))
            formatted_x[:, 1:] = x
        else:
            dim_w = x.shape[1]
            formatted_x = x

        if self.params['w_start'] is None:
            self.params['w_start'] = np.zeros(dim_w)
            self.algorithm.w = self.params['w_start']
            self._update_algoritm_params()

        super().fit(formatted_x, y, logs=logs, verbose_mode=verbose_mode, logs_freq=logs_freq)

        if self.fit_intercept:
            self.coef_ = self.solution[1:]
            self.intercept_ = self.solution[0]
        else:
            self.coef_ = self.solution

    def l2(self, w, x, y):
        return l2_loss(w, x, y, lmbda=self.lmbda)

    def l2_prime(self, w, x, y):
        return l2_prime(w, x, y, lmbda=self.lmbda)

    def predict(self, x):
        """ Gives a prediction of x
                :param ``numpy.array`` x: input whose label is to predict
                :return:  value of the prediction
        """

        return np.dot(x, self.algorithm.w)

    def score(self, X, y, sample_weights=None):
        return 0


class DRLogisticRegression(RiskOptimizer, ClassifierMixin):

    def __init__(self, p=0.9, mu=1.0, lmbda=None, fit_intercept=True, intercept_scaling=1.0):

        # Linear Model Related Parameters
        self.lmbda = lmbda

        self.n_features = None
        self.n_classes = None

        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling

        params = {
            'algorithm': 'l-bfgs',
            'p': p,
            'mu': mu,
        }

        self.coef_ = 0.
        self.intercept_ = 0.

        super().__init__(self.logistic_loss, self.logistic_grad, params=params)

    def fit(self, x, y, logs=False, verbose_mode=False, logs_freq=1):

        self.n_features = x.shape[1]
        le = LabelEncoder()
        le.fit(y)
        self.n_classes = len(le.classes_)
        formatted_y = le.transform(y)
        formatted_y = _recast_y(formatted_y, n_classes=self.n_classes)

        if self.lmbda is None:
            self.lmbda = 1.0/y.shape[0]

        # Todo improve code be removing the condition
        if self.fit_intercept:
            dim_w = (self.n_features+1) * self.n_classes
            formatted_x = np.ones((x.shape[0], self.n_features + 1))
            formatted_x[:, 1:] = x
        else:
            dim_w = self.n_features * self.n_classes
            formatted_x = x

        if self.params['w_start'] is None:
            self.params['w_start'] = np.zeros(dim_w)
            self.algorithm.w = self.params['w_start']
            self._update_algoritm_params()

        super().fit(formatted_x, formatted_y, logs=logs, verbose_mode=verbose_mode, logs_freq=logs_freq)

        if self.fit_intercept:
            self.coef_ = self.solution[1:]
            self.intercept_ = self.solution[0]
        else:
            self.coef_ = self.solution

    def logistic_loss(self, w, x, y):
        return logistic_loss(w, x, y, lmbda=self.lmbda, n_features=self.n_features + self.fit_intercept,
                             n_classes=self.n_classes)

    def logistic_grad(self, w, x, y):
        return logistic_grad(w, x, y, lmbda=self.lmbda, n_features=self.n_features + self.fit_intercept,
                             n_classes=self.n_classes)

    def predict(self, x):
        """ Gives a prediction of x
                :param ``numpy.array`` x: input whose label is to predict
                :return:  value of the prediction
        """
        formatted_x = np.ones((x.shape[0], self.n_features + self.fit_intercept))
        formatted_x[:, self.fit_intercept:] = x
        casted_sol = np.reshape(self.solution, (self.n_features + self.fit_intercept, self.n_classes))
        probas = np.dot(formatted_x, casted_sol)
        predictions = np.argmax(probas, axis=1)

        return predictions

    def score(self, X, y, sample_weights=None):
        return 0


@njit
def _recast_y(y, n_classes):
    res = np.zeros((len(y), n_classes), dtype=np.float64)
    for ii in range(len(y)):
        res[ii][y[ii]] = 1.0
    return res