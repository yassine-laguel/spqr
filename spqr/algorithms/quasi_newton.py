# Author: Yassine Laguel
# License: BSD


from scipy.optimize import line_search, minimize
from time import time
import numpy as np


class BFGS:
    """ Class aimed at running Low memory bfgs method.

            :param oracle: An oracle object among ``OracleSubgradient``, ``OracleSmoothGradient``,
                            ``IntergratedOracleSubgradient``, ``IntegratedOracleSmoothGradient``.
            :param params: A dictionnary containing the parameters of the algorithm
    """

    def __init__(self, oracle, params):

        self.oracle = oracle
        self.params = params
        self.list_iterates = []
        self.list_values = []
        self.list_times = []
        self.w = np.copy(self.params['w_start'])

    def run(self, X, y, verboose_mode=False):
        """ Runs low-memory BFGS algorithm

                :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
                :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
                :param ``bool`` verboose_mode: If ``True``, saves function values during iterations of selected
                        algorithm as well as time since start.
        """

        start_time = time()

        def f(w):
            return self.oracle.f(w, X, y)

        def g(w):
            return self.oracle.g(w, X, y)

        def callback(x_k):
            self.list_iterates.append(x_k)

            if verboose_mode:
                self.list_times.append(time() - start_time)
                self.list_values.append(self.oracle.cost_function(x_k, X, y))

        w_start = self.params['w_start']

        if verboose_mode:
            self.list_values.append(self.oracle.cost_function(w_start, X, y))

        minimizing_options = {
            'disp': None,
            'maxcor': 10,
            'ftol': 2.220446049250313e-12,
            'gtol': 1e-05,
            'eps': 1e-08,
            'maxfun': 15000,
            'maxiter': 15000,
            'iprint': -1,
            'maxls': 20
        }

        result_object = minimize(f, w_start, jac=g, method='L-BFGS-B', callback=callback, options=minimizing_options)

        self.w = result_object.x