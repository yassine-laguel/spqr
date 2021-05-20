# Author: Yassine Laguel
# License: BSD

import sys
import numpy as np
from time import time


class SVRG:
    """ Class aimed at running Accelerated Gradient Method.

            :param oracle: An oracle object among ``OracleSubgradient``, ``OracleSmoothGradient``,
                            ``IntergratedOracleSubgradient``, ``IntegratedOracleSmoothGradient``.
            :param params: A dictionnary containing the parameters of the algorithm
    """

    def __init__(self, oracle, whole_oracle, params):

        self.oracle = oracle
        self.whole_oracle = whole_oracle
        self.params = params

        self.w = np.copy(self.params['w_start'])
        self.w_s = np.copy(self.w)
        self.g = np.copy(self.w)

        self.list_iterates = None
        self.list_values = np.zeros(self.params['svrg_nb_iterations'])
        self.list_times = np.zeros(self.params['svrg_nb_iterations'])

        # Assumed smoothness of the function
        self.initial_stepsize = self.params['svrg_initial_stepsize']

    def run(self, x, y, logs=False, verbose_mode=False, logs_freq=1):
        """ Runs Variance reduced Stochastic Gradient Descent Algorithm

                :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
                :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
                :param ``bool`` logs: If ``True``, saves function values during iterations of selected
                        algorithm as well as times since start.
                :param ``bool`` verbose_mode: If ``True``, prints out while running the algorithm the number of
                iterations being completed.
        """

        self.w = np.copy(self.params['w_start'])
        self.w_s = np.copy(self.w)

        self.list_iterates = np.zeros((self.params['svrg_nb_iterations'], len(self.w)))

        counter = 0
        start_time = time()

        while counter < self.params['svrg_nb_iterations']:

            self.g = self.whole_oracle.g(self.w, x, y)

            for i in range(self.params['svrg_nb_sub_iterations']):
                self._update_iterates(x, y)
                self.list_iterates[counter] = self.w

                if logs:
                    self.list_times[counter] = time() - start_time
                    self.list_values[counter] = self.oracle.cost_function(self.w, x, y)
                if verbose_mode:
                    sys.stdout.write('%d / %d  iterations completed \r' % (counter, self.params['sgd_nb_iterations']))
                    sys.stdout.flush()

            self.w_s = self.w
            counter += 1

    def _update_iterates(self, x, y):

        sample_index = np.random.choice(len(y), size=self.params['svrg_mini_batch_size'])

        g_i = self.oracle.g(self.w, x, y, sample_index=sample_index)
        g_i_s = self.oracle.g(self.w_s, x, y, sample_index=sample_index)
        direction = -1.0 * (self.g + (g_i - g_i_s))
        stepsize = self.compute_stepsize(self.w, direction, g_i, x, y, sample_index)
        # print(stepsize)
        self.w = self.w + stepsize * direction

    def compute_stepsize(self, w, direction, g_i, x, y, sample_index):

        alpha = 1.1
        stepsize = self.initial_stepsize

        def evaluate_cond(s):
            next_value = self.oracle.f(w + s * direction, x, y, sample_index=sample_index)
            current_value = self.oracle.f(w, x, y, sample_index=sample_index)
            cond = next_value < current_value - 0.5 * s * np.linalg.norm(g_i)**2
            return cond

        cond = evaluate_cond(stepsize)

        if cond:
            while evaluate_cond(stepsize) and stepsize < 1:
                stepsize *= alpha
            self.initial_stepsize = stepsize / alpha
            return stepsize / alpha
        else:
            while (not evaluate_cond(stepsize)) and stepsize > 10e-16:
                stepsize /= alpha
            self.initial_stepsize = stepsize * alpha
            return stepsize * alpha

