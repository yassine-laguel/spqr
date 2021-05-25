# Author: Yassine Laguel
# License: BSD


from scipy.optimize import line_search, minimize
from time import time
import numpy as np
import sys


class LBFGS:
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
        self.bfgs_result_object = None

        self.tot_nb_fun_eval = 0
        self.tot_nb_grad_eval = 0
        self.nb_fun_eval = []
        self.nb_grad_eval = []
        self.counter = 0

    def run(self, X, y, logs=False, verbose_mode=False, logs_freq=1):
        """ Runs low-memory BFGS algorithm

                :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
                :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
                :param ``bool`` logs: If ``True``, saves function values during iterations of selected
                        algorithm as well as times since start.
                :param ``bool`` verbose_mode: If ``True``, prints out while running the algorithm the number of
                iterations being completed.
        """

        f = LoggedOracleFun(self.oracle, X, y)
        g = LoggedOracleGrad(self.oracle, X, y)

        start_time = time()

        # def f(w):
        #     return self.oracle.f(w, X, y)
        #
        # def g(w):
        #     return self.oracle.g(w, X, y)

        def callback(x_k):
            if verbose_mode:
                sys.stdout.write('%d epochs completed \r' % self.counter)
                sys.stdout.flush()
            if logs and self.counter % logs_freq == 0:
                self.nb_fun_eval.append(f.nb_calls)
                self.nb_grad_eval.append(g.nb_calls)
                self.list_times.append(time() - start_time)
                self.list_values.append(self.oracle.cost_function(x_k, X, y))
                self.list_iterates.append(x_k)

            if verbose_mode or logs:
                self.counter += 1

        w_start = self.params['w_start']

        if logs:
            self.list_values.append(self.oracle.cost_function(w_start, X, y))

        minimizing_options = {
            'disp': None,
            'maxcor': 100,
            'ftol': 2.220446049250313e-12,
            'gtol': 1e-12,
            'eps': 1e-08,
            'maxfun': 100000,
            'maxiter': 100000,
            'iprint': -1,
            'maxls': 10  # 10
        }

        # minimizing_options = {
        #     'disp': None,
        #     'gtol': 1e-12,
        #     'maxiter': 100000,
        # }

        # minimizing_options = {
        #     'disp': None,
        #     'maxcor': 10,
        #     'ftol': 2.220446049250313e-09,
        #     'gtol': 1e-05,
        #     'eps': 1e-08,
        #     'maxfun': 15000,
        #     'maxiter': 15000,
        #     'iprint': -1,
        #     'maxls': 20
        # }


        result_object = minimize(f, w_start, jac=g, method='L-BFGS-B', callback=callback, options=minimizing_options)
        # result_object = minimize(f, w_start, jac=g, method='BFGS', callback=callback, options=minimizing_options)

        self.bfgs_result_object = result_object

        self.tot_nb_fun_eval = result_object.nfev
        self.tot_nb_grad_eval = result_object.nfev

        self.w = result_object.x

class LoggedOracleFun(object):

    def __init__(self, oracle, x_train, y_train):

        self.oracle = oracle
        self.x_train = x_train
        self.y_train = y_train
        self.nb_calls = 0

    def __call__(self, w):
        self.nb_calls += 1
        return self.oracle.f(w, self.x_train, self.y_train)


class LoggedOracleGrad(object):

    def __init__(self, oracle, x_train, y_train):
        self.oracle = oracle
        self.nb_calls = 0
        self.x_train = x_train
        self.y_train = y_train

    def __call__(self, w):
        self.nb_calls += 1
        return self.oracle.g(w, self.x_train, self.y_train)





