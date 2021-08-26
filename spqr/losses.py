"""
.. module:: losses
   :synopsis: Module with with fast implementation of selected learning losses.

.. moduleauthor:: Yassine LAGUEL
"""
from numba import njit
import numpy as np


def l2_loss(w, batched_x, batched_y, lmbda=0.0):
    return np.square(batched_y - np.dot(batched_x, w)) + 0.5 * lmbda * np.linalg.norm(w) ** 2


def l2_prime(w, batched_x, batched_y, lmbda=0.0):
    u = np.diag(batched_y - np.dot(batched_x, w))
    return -2.0 * np.dot(u, batched_x) + lmbda * w


@njit
def logistic_loss(w, batched_x, batched_y, lmbda=0.0, n_features=10, n_classes=10):

    def single_loss(w, x_i, y_i, l2_penalty=lmbda, eps=0):
        score = np.dot(x_i, w)
        log_y_hat = score - logsumexp(score)
        return - np.sum(y_i * log_y_hat) + l2_penalty / 2 * np.linalg.norm(w) ** 2

    w_mat = np.reshape(w, (n_features, n_classes))
    n = len(batched_y)
    res = np.zeros(n)
    for ii in range(n):
        res[ii] = single_loss(w_mat, batched_x[ii], batched_y[ii])
    return res


@njit
def logistic_grad(w, batched_x, batched_y, lmbda=0.0, n_features=10, n_classes=10):

    def single_gradient(w, x_i, y_i, l2_penalty=lmbda):
        p_hat = softmax(np.dot(x_i, w))  # vector of probabilities
        a = p_hat - y_i
        return np.outer(x_i, a) + l2_penalty * w

    w_mat = np.reshape(w, (n_features, n_classes))
    n = len(batched_y)
    res = np.zeros((n, *w.shape))
    for ii in range(n):
        g = single_gradient(w_mat, batched_x[ii], batched_y[ii])
        res[ii] = np.ravel(g)
    return res

# Auxiliary Functions for the logistic regression


@njit
def softmax(x):
    res = np.exp(x - np.max(x))
    s = np.sum(res)
    return res / s


@njit
def clip(a, a_min, a_max):
    out = np.empty_like(a)
    for ii in range(len(a)):
        if a[ii] < a_min:
            out[ii] = a_min
        elif a[ii] > a_max:
            out[ii] = a_max
        else:
            out[ii] = a[ii]
    return out

@njit
def logsumexp(a):
    i_max = np.argmax(a)
    return np.log1p(np.sum(np.delete(np.exp(a - a[i_max]), i_max))) + a[i_max] 




