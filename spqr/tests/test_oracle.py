import numpy as np
import pytest

from sklearn.datasets import make_blobs
from scipy.optimize import check_grad, approx_fprime
from functools import partial

from .. import oracle
from .. import losses
from .. import estimators

N = 100
D = 2
P = 0.8

# From scipy.optimize.check_grad, roughly 1.5e-8
EPS = np.sqrt(np.finfo(float).eps)
# EPS = 1e-15

PREC = 1e-4

gen = np.random.default_rng()

@pytest.fixture
def loss():
    return partial(losses.logistic_loss, n_classes=2, n_features=D)

@pytest.fixture
def loss_grad():
    return partial(losses.logistic_grad, n_classes=2, n_features=D)

@pytest.fixture
def w():
    return gen.normal(size=2*D)

@pytest.fixture
def dataset():
    x, y = make_blobs(n_samples=N, n_features=D, centers=2)
    assert x.shape == (N, D)
    assert y.shape == (N,)

    one_hot_y = np.identity(2)[y]
    assert one_hot_y.shape == (N, 2)
    assert (np.argmax(one_hot_y, axis=1) == y).all()

    return x, one_hot_y

@pytest.mark.parametrize("sq_smoothing_parameter", [1, 10, 100, 1000])
def test_estimator_gradients(dataset, sq_smoothing_parameter):
    est = estimators.DRLogisticRegression(p=P, mu=sq_smoothing_parameter, fit_intercept=False, lmbda=0.)
    x, one_hot_y = dataset
    y = np.argmax(one_hot_y, axis=1)
    est.fit(x, y)
    w = est.coef_
    assert w.shape == (2*D,)

    orc = est.oracle
    grad = orc.g(w, *dataset)
    approx_grad = approx_fprime(w, orc.f, EPS, *dataset)
    print(f"{grad=}, {approx_grad=}")
    assert check_grad(orc.f, orc.g, w, *dataset, epsilon=EPS) <= 1e-4

@pytest.mark.parametrize("sq_smoothing_parameter", [1, 10, 100, 1000])
def test_gradients(loss, loss_grad, w, dataset, sq_smoothing_parameter):
    orc = oracle.OracleSmoothGradient(loss, loss_grad, P, sq_smoothing_parameter)
    grad = orc.g(w, *dataset)
    approx_grad = approx_fprime(w, orc.f, EPS, *dataset)
    print(f"{grad=}, {approx_grad=}")
    assert check_grad(orc.f, orc.g, w, *dataset, epsilon=EPS) <= 1e-5


def test_logistic_single_gradients(loss, loss_grad, w, dataset):
	def f(w, x, y):
		def single_loss(w, x_i, y_i, l2_penalty=1, eps=0):
			y_hat = losses.softmax(np.dot(x_i, w))
			y_hat = losses.clip(y_hat, eps, 1.0 - eps)
			return - np.sum(y_i * np.log(y_hat)) + l2_penalty / 2 * np.linalg.norm(w) ** 2
		return single_loss(np.reshape(w, (D, 2)), x[0], y[0])
	
	def g(w, x, y):
		def single_gradient(w, x_i, y_i, l2_penalty=1):
			p_hat = losses.softmax(np.dot(x_i, w))  # vector of probabilities
			a = p_hat - y_i
			return np.outer(x_i, a) + l2_penalty * w
		return np.ravel(single_gradient(np.reshape(w, (D, 2)), x[0], y[0]))

	assert check_grad(f, g, w, *dataset) <= 1e-5

def test_logistic_full_gradients(loss, loss_grad, w, dataset):
	for i in range(N):
		def f(w, x, y):
			return loss(w, x, y)[i]
		
		def g(w, x, y):
			return loss_grad(w, x, y)[i]
		grad = g(w, *dataset)
		approx_grad = approx_fprime(w, f, EPS, *dataset)
		print(f"{grad=}, {approx_grad=}")
		assert check_grad(f, g, w, *dataset) <= 1e-5
