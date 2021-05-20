    # Author: Yassine Laguel
    # License: BSD

import numpy as np
from numba import njit


def old_quantile(p, u):
    """ Computes the p-quantile of u

        :param ``float`` p: probability associated to the quantile
        :param ``numpy.array`` u: vector of realizations of the rqndom variable whose quantile is to be computed

        :return p-quantile of u
    """

    v = np.sort(u)

    if p == 0:
        return v[0]
    else:
        n = len(v)
        index = int(np.ceil(n*p)) - 1
        return v[index]


@njit
def quantile(p, u):
    if p == 0:
        k = 1
    else:
        k = np.ceil(p * len(u))
    return _quickselect(k, u)


@njit
def superquantile(p, u):
    """ Computes the p-superquantile of u

            :param ``float`` p: probability associated to the superquantile
            :param ``numpy.array`` u: vector of realizations of the random variable
            whose superquantile is to be computed

            :return p-superquantile of u
    """
    if p == 0:
        return np.mean(u)

    elif p >= 1.0 - 1.0/len(u):
        return np.max(u)
    else:
        n = len(u)
        q_p = quantile(p, u)
        higher_data = np.extract(u > q_p, u)
        if len(higher_data) == 0:
            return q_p
        else:
            next_jump = (u <= q_p).sum() / n
            cvar_plus = np.mean(higher_data)
            lmbda = (next_jump - p) / (1.0 - p)
            return lmbda * q_p + (1.0 - lmbda) * cvar_plus


@njit
def _quickselect(k, list_of_numbers):
    return _kthSmallest(list_of_numbers, k, 0, len(list_of_numbers) - 1)


@njit
def _kthSmallest(arr, k, start, end):

    pivot_index = _partition(arr, start, end)

    if pivot_index - start == k - 1:
        return arr[pivot_index]

    if pivot_index - start > k - 1:
        return _kthSmallest(arr, k, start, pivot_index - 1)

    return _kthSmallest(arr, k - pivot_index + start - 1, pivot_index + 1, end)


@njit
def _partition(arr, l, r):

    pivot = arr[r]
    i = l
    for j in range(l, r):

        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    arr[i], arr[r] = arr[r], arr[i]
    return i
