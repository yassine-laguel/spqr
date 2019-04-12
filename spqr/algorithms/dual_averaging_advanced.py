# Author: Yassine Laguel
# License: BSD

import sys
import numpy as np
from time import time


class DualAveragingAdvanced:
    """ Class aimed at running Dual Averaging Method.

        :param oracle: An oracle object among ``OracleSubgradient``, ``OracleSmoothGradient``,
                        ``IntergratedOracleSubgradient``, ``IntegratedOracleSmoothGradient``.
        :param params: A dictionnary containing the parameters of the algorithm
    """

    def __init__(self, oracle, params):

        self.oracle = oracle
        self.params = params
        self.w = np.copy(self.params['w_start'])
        self.list_iterates = []
        self.list_values = []
        self.list_times = []

    def run(self, x, y, verboose_mode=False):
        """ Runs Dual Averaging Algorithm

                :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
                :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
                :param ``bool`` verboose_mode: If ``True``, saves function values during iterations of selected
                        algorithm as well as time since start.
        """

        print(1.0/self.params['alpha'])

        if verboose_mode:
            start_time = time()

        beta_hat = 1.0
        g = self.oracle.g(self.w, x, y)
        norm_gradient = np.linalg.norm(g)
        s = 1.0/np.linalg.norm(g) * g  # We use Weighted Dual Averaging version

        sum_gradients = norm_gradient

        new_iterate = self.w

        counter = 0
        while counter < self.params['dual_averaging_nb_iterations']:

            if verboose_mode:
                self.list_times.append(time() - start_time)
                self.list_values.append(self.oracle.cost_function(self.w, x, y))
                sys.stdout.write('%d / %d  iterations '
                                 'completed \r' % (counter, self.params['dual_averaging_nb_iterations']))
                sys.stdout.flush()

            g = self.oracle.g(new_iterate, x, y)
            norm_gradient = np.linalg.norm(g)

            s = s + 1.0/norm_gradient * g

            beta_hat = beta_hat + 1.0/beta_hat
            beta = (1.0/self.params['alpha']) * beta_hat

            new_iterate = -1.0/beta * s

            self.w = (1.0/(sum_gradients + norm_gradient)) * (sum_gradients * self.w + norm_gradient * new_iterate)
            sum_gradients += norm_gradient

            self.list_iterates.append(self.w)

            counter += 1
