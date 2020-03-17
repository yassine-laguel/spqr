# Author: Yassine Laguel
# License: BSD

import sys
import numpy as np
from time import time


class NesterovMethod:
    """ Class aimed at running Accelerated Gradient Method.

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

        # Assumed smoothness of the function
        self.beta = np.copy(self.params['beta_smoothness'])

        # Constant related to Nesterov Algorithm
        self.lmbda = 0.
        self.x = np.copy(self.w)

    def run(self, x, y, verbose_mode=False):
        """ Runs Nesterov Algorithm

                :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
                :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
                :param ``bool`` verbose_mode: If ``True``, saves function values during iterations of selected
                                algorithm as well as time since start.
        """

        # Nesterov Iterates
        self.w = np.copy(self.params['w_start'])
        self.x = np.copy(self.w)
        self.lmbda = 0.

        counter = 0
        start_time = time()

        while counter < self.params['nesterov_nb_iterations']:

            if verbose_mode:
                self.list_times.append(time() - start_time)
                self.list_values.append(self.oracle.cost_function(self.w, x, y))
                sys.stdout.write('%d / %d  iterations completed \r' % (counter, self.params['nesterov_nb_iterations']))
                sys.stdout.flush()

            self._update_iterates(x, y)
            self.list_iterates.append(self.w)

            counter += 1

    def _update_iterates(self, x, y):

        lmbda_next = (1. + np.sqrt(1. + 4 * self.lmbda ** 2))/2
        gamma = (1. - self.lmbda)/lmbda_next
        w_next = self.x - (1./self.beta) * self.oracle.g(self.x, x, y)

        self.x = (1. - gamma) * w_next + gamma * self.w

        self.w = w_next
        self.lmbda = lmbda_next