# Author: Yassine Laguel
# License: BSD

import sys
import numpy as np
from time import time


class SGD:
    """ Class aimed at running Accelerated Gradient Method.

            :param oracle: An oracle object among ``OracleSubgradient``, ``OracleSmoothGradient``,
                            ``IntergratedOracleSubgradient``, ``IntegratedOracleSmoothGradient``.
            :param params: A dictionnary containing the parameters of the algorithm
    """

    def __init__(self, oracle, params):

        self.oracle = oracle
        self.params = params
        self.w = np.copy(self.params['w_start'])
        self.list_iterates = None
        self.list_values = np.zeros(self.params['sgd_nb_iterations'])
        self.list_times = np.zeros(self.params['sgd_nb_iterations'])

        self.mass = np.copy(self.params['sgd_mass'])
        self.velocity = np.zeros_like(self.w)

    def run(self, x, y, logs=False, verbose_mode=False, logs_freq=1):
        """ Runs Stochastic Gradient Descent Algorithm

                :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
                :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
                :param ``bool`` logs: If ``True``, saves function values during iterations of selected
                        algorithm as well as times since start.
                :param ``bool`` verbose_mode: If ``True``, prints out while running the algorithm the number of
                iterations being completed.
        """

        self.w = np.copy(self.params['w_start'])
        self.velocity = np.zeros_like(self.w)
        self.list_iterates = np.zeros((self.params['sgd_nb_iterations'], *self.w.shape))

        counter = 0
        start_time = time()

        while counter < self.params['sgd_nb_iterations']:

            if logs and counter % logs_freq == 0:
                self.list_times[counter] = time() - start_time
                self.list_values[counter] = self.oracle.cost_function(self.w, x, y)
                self.list_iterates[counter] = self.w

            if verbose_mode:
                sys.stdout.write('%d / %d  iterations completed \r' % (counter, self.params['sgd_nb_iterations']))
                sys.stdout.flush()

            self._update_iterates(x, y, counter+1)

            counter += 1

    def _update_iterates(self, x, y, counter):
        g = self.oracle.g(self.w, x, y)
        self.velocity = self.mass * self.velocity + self.params['sgd_stepsize'](counter) * g
        self.w = self.w - self.velocity
