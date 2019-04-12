    # Author: Yassine Laguel
    # License: BSD

import numpy as np


def quantile(p, u):
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
        condition = u > q_p
        higher_data = np.extract(condition, u)
        cvar_plus = np.mean(higher_data)
        lmbda = (np.ceil(n * p)/n - p) / (1.0 - p)
        return lmbda * q_p + (1.0 - lmbda) * cvar_plus


def hyperquantile(p, u):
    """ Computes the p-hyperquantile of u

                :param ``float`` p: probability associated to the hyperquantile
                :param ``numpy.array`` u: vector of realizations of the random variable
                whose superquantile is to be computed

                :return p-hyperquantile of u
    """
    v = np.sort(u)
    n = len(v)

    if p == 1:
        return v[n - 1]

    def _integrate_superquantile(p1):

        index = int(np.ceil(n * p1)) - 1

        if p1 == 0.0:
            index = 0
        if p1 == 1.0:
            return 0.0

        if index == n - 1:
            return ((np.log((1 - p) / (1 - p1)) + 1) * (1 - p1)) * v[n - 1]

        p2 = float(index + 1) / n

        if p2 == p1:
            p2 += 1.0 / n
            index += 1
            if p2 == 1.0:
                return ((np.log((1 - p) / (1 - p1)) + 1) * (1 - p1)) * v[n - 1]

        s = 0

        while p2 < 1.0:
            q = v[index]
            s += q * (np.log(1.0 - p) * (p2 - p1) + (1 - p2) * np.log(1 - p2) - (1 - p1) * np.log(1 - p1) + p2 - p1)
            p1 = p2
            p2 += 1.0 / n
            index += 1

        s += ((np.log((1 - p) / (1 - p1)) + 1) * (1 - p1)) * v[n - 1]

        return s

    return _integrate_superquantile(p) / (1.0 - p)
