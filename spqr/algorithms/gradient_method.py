# Author: Yassine Laguel
# License: BSD

import sys
import numpy as np
from time import time

class GradientMethod():

    def __init__(self, oracle, params):
        """ Class aimed at running Gradient method.

                :param oracle: An oracle object among ``OracleSubgradient``, ``OracleSmoothGradient``,
                                ``IntergratedOracleSubgradient``, ``IntegratedOracleSmoothGradient``.
                :param params: A dictionnary containing the parameters of the algorithm
        """

        self.oracle = oracle
        self.params = params
        self.w = np.copy(self.params['w_start'])
        self.list_iterates = []
        self.list_values = []
        self.list_times = []
        self.alpha = self.params['alpha']

    def run(self, x, y, verboose_mode=False):
        """ Runs Gradient Method

            :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
            :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
            :param ``bool`` verboose_mode: If ``True``, saves function values during iterations of selected
                                algorithm as well as time since start.
        """

        self.alpha = self.params['alpha']
        start_time = time()

        counter = 0
        while counter < self.params['gradient_nb_iterations']:

            if verboose_mode:
                self.list_times.append(time() - start_time)
                self.list_values.append(self.oracle.cost_function(self.w, x, y))
                sys.stdout.write('%d / %d  iterations completed \r' % (counter, self.params['gradient_nb_iterations']))
                sys.stdout.flush()

            self.w = self.w - self.stepsize(self.w, x, y) * self.oracle.g(self.w, x, y)
            self.list_iterates.append(self.w)

            counter += 1

    def stepsize(self, w, x, y):

        return self._find_stepsize(w, x, y)

    def _find_stepsize(self, w, x, y, mode_debug=False):

        alpha = self.params['alpha']

        # Algorithm can be speed up with right choice of these constants
        a = 0.99
        c1 = 0.01
        c2 = 0.01

        gradient = self.oracle.g(w, x, y)
        gradient_norm_square = np.linalg.norm(gradient)**2
        function_value = self.oracle.f(w, x, y)
        condition = (self.oracle.f(w - alpha * gradient, x, y) > function_value - alpha * c1 * gradient_norm_square)

        if condition:
            while condition:
                alpha *= a
                condition = (self.oracle.f(w - alpha * gradient, x, y) >
                             function_value - alpha * c1 * gradient_norm_square)
        else:
            while not condition:
                alpha /= a
                condition = (self.oracle.f(w - alpha * gradient, x,
                                           y) > function_value - alpha * c2 * gradient_norm_square)
            alpha *= a

        if mode_debug:
            print('alpha_value ' + str(alpha))

        self.params['alpha'] = alpha
        return alpha

