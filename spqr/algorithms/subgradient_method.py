# Author: Yassine Laguel
# License: BSD

import sys
import numpy as np
from time import time


class SubgradientMethod:
    """ Class aimed at running subgradient method.

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

    def run(self, x, y, verbose_mode=False):
        """ Runs the subgradient method

                :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
                :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
                :param ``bool`` verbose_mode: If ``True``, saves function values during iterations of selected
                        algorithm as well as time since start.
        """

        counter = 0
        start_time = time()

        while counter < self.params['subgradient_nb_iterations']:

            if verbose_mode:
                self.list_times.append(time() - start_time)
                self.list_values.append(self.oracle.cost_function(self.w, x, y))
                sys.stdout.write('%d / %d  iterations completed \r'
                                 % (counter, self.params['subgradient_nb_iterations']))
                sys.stdout.flush()

            self.w = self.w - self._stepsize(counter) * self.oracle.g(self.w, x, y)
            self.list_iterates.append(self.w)

            counter += 1

    def _stepsize(self, counter):
        return self.params['alpha'] * self.params['subgradient_stepsize_decrease'](counter)
