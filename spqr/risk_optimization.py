"""Module containing the main object to run superquantile-based learning methods.


.. moduleauthor:: Yassine LAGUEL
"""

from sklearn.base import BaseEstimator
from .oracle import OracleSubgradient, OracleSmoothGradient, OracleStochasticGradient
from .algorithms.subgradient_method import SubgradientMethod
from .algorithms.gradient_method import GradientMethod
from .algorithms.dual_averaging_advanced import DualAveragingAdvanced
from .algorithms.quasi_newton import LBFGS
from .algorithms.nesterov import NesterovMethod
from .algorithms.sgd import SGD
from .algorithms.svrg import SVRG
import numpy as np


class RiskOptimizer(BaseEstimator):
    """ Base class for optimization of superquantile-based losses.
            For an input oracle :math:`L` given through two functions ``function_l`` and ``gradient_l``,
            this class is an interface to run optimization procedures aimed at minimizing
            :math:`w \\mapsto Cvar \\circ L(w)`. Given the regularity of the loss, the algorithm chosen for
            optimization should be carefully chosen.

            :param loss: function associated to the oracle
            :param loss_grad: gradient associated to the oracle
            :param p: probability level (by default 0.8)
            :param algorithm: chosen algorithm for optimization. Allowed inputs are ``'subgradient'``,
                ``'dual_averaging'``, ``'gradient'``, ``'nesterov'`` and ``'l-bfgs'``. Default is ``'subgradient'``
            :param w_start: starting point of the algorithm
            :param alpha: scale parameter for the direction descent (by default computed through a line search)
            :param mu: smoothing parameter associated to the CVar
            :param beta_smoothness: estimation of the smoothness of :math:`L` (used for accelerated gradient method).
    """

    def __init__(self, loss, loss_grad, algorithm=None, w_start=None, p=None,
                 alpha=None, mu=None, max_iter=None, dual_averaging_lmbda=None, beta_smoothness=None, params=None):

        self._treat_parameters(algorithm, w_start, p, alpha, mu, max_iter,
                               dual_averaging_lmbda, beta_smoothness, params)

        if self.params['algorithm'] == 'subgradient':
            self.oracle = OracleSubgradient(loss, loss_grad, self.params['p'])
            self.algorithm = SubgradientMethod(self.oracle, self.params)

        elif self.params['algorithm'] == 'dual_averaging':
            self.oracle = OracleSubgradient(loss, loss_grad, self.params['p'])
            self.algorithm = DualAveragingAdvanced(self.oracle, self.params)

        elif self.params['algorithm'] == 'gradient':
            self.oracle = OracleSmoothGradient(loss, loss_grad, self.params['p'],
                                               smoothing_parameter=self.params['mu'])
            self.algorithm = GradientMethod(self.oracle, self.params)

        elif self.params['algorithm'] == 'nesterov':
            self.oracle = OracleSmoothGradient(loss, loss_grad, self.params['p'],
                                               smoothing_parameter=self.params['mu'])
            self.algorithm = NesterovMethod(self.oracle, self.params)

        elif self.params['algorithm'] == 'l-bfgs':
            self.oracle = OracleSmoothGradient(loss, loss_grad, self.params['p'],
                                               smoothing_parameter=self.params['mu'])
            self.algorithm = LBFGS(self.oracle, self.params)

        elif self.params['algorithm'] == 'sgd':
            self.oracle = OracleStochasticGradient(loss, loss_grad, self.params['p'],
                                                         smoothing_parameter=self.params['mu'],
                                                         mini_batch_size=self.params['sgd_mini_batch_size'])
            self.algorithm = SGD(self.oracle, self.params)

        elif self.params['algorithm'] == 'svrg':

            whole_oracle = OracleSmoothGradient(loss, loss_grad, self.params['p'], smoothing_parameter=self.params['mu'])

            self.oracle = OracleStochasticGradient(loss, loss_grad, self.params['p'],
                                                         smoothing_parameter=self.params['mu'],
                                                         mini_batch_size=self.params['svrg_mini_batch_size'])
            self.algorithm = SVRG(self.oracle, whole_oracle, self.params)

        # # Only for testing TODO: Remove this condition block after testing
        # elif self.params['algorithm'] == 'bfgs_test':
        #     self.oracle = OracleSubgradient(loss, loss_grad, self.params['p'])
        #     self.algorithm = LBFGS(self.oracle, self.params)

        self.solution = 0.
        self.list_iterates = []

    def fit(self, x, y, logs=False, verbose_mode=False, logs_freq=1):
        """ Runs the optimization of the model

                :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable :math:`X`
                :param ``numpy.array`` y: vector whose coefficients are realizations of random variable :math:`y`
                :param ``bool`` verbose_mode: If ``True``, saves function values during iterations of selected algorithm as well as time since start.
        """

        self._complete_params_and_scale_X(x, y)
        self.algorithm.run(x, y, logs=logs, verbose_mode=verbose_mode, logs_freq=logs_freq)
        self.solution = self.algorithm.w
        self.list_iterates = self.algorithm.list_iterates

    def _treat_parameters(self, algorithm, w_start, p, alpha, mu, max_iter,
                          dual_averaging_lmbda, beta_smoothness, params):

        arguments = locals()
        del arguments['self']

        self.params = self._create_params()

        for given_parameter in arguments:
            if arguments[given_parameter]is not None:
                self.params[str(given_parameter)] = arguments[given_parameter]

        if params is not None:
            for key in params:
                self.params[key] = params[key]

        if self.params['max_iter'] is not None:
            s = self.params['algorithm'] + '_nb_iterations'
            self.params[s] = self.params['max_iter']

    def _create_params(self):

        params = {
            # General Parameters
            'algorithm': 'subgradient',
            'w_start': None,
            'p': 0.8,
            'alpha': None,
            'alpha_start': 100.0,
            'max_iter': None,

            # Smoothing_parameters
            'mu': 1000.0,

            # Subgradient Parameters
            'subgradient_stepsize_decrease': lambda k: 1.0 / (1.0 + np.sqrt(k/10) ** 0.6),
            'subgradient_nb_iterations': 1000,

            # Dual Averaging Parameters
            'dual_averaging_basic_stepsize_decrease': lambda k: 1.0 / (1.0 + np.sqrt(k / 10) ** 0.6),
            'dual_averaging_basic_nb_iterations': 100,
            'dual_averaging_basic_gamma': 100,
            'dual_averaging_basic_lmbda': 0.00000001,

            # Dual Averaging Advanced Parameters
            'dual_averaging_stepsize_decrease': lambda k: 1.0 / (1.0 + np.sqrt(k / 10) ** 0.6),
            'dual_averaging_nb_iterations': 100,
            'dual_averaging_gamma': 100.0,
            'dual_averaging_alpha': 0.5,

            # Gradient Parameters
            'gradient_stepsize_decrease': lambda k: 1.0,
            'gradient_nb_iterations': 100,

            # Nesterov Parameters
            'nesterov_nb_iterations': 100,
            'beta_smoothness': 1000.0,

            # LBFGS Parameters
            'l-bfgs_nb_iterations': 1000,

            # SGD Parameters
            'sgd_nb_iterations': 10000,
            'sgd_stepsize': lambda k: 0.1 / (1.0 + (k-1.0/10) ** 0.5),
            'sgd_mass': 0.9,
            'sgd_mini_batch_size': 50,

            # SVRG Parameters
            'svrg_nb_iterations': 100,
            'svrg_nb_sub_iterations': 100,
            'svrg_initial_stepsize': 0.1,
            'svrg_mini_batch_size': 50,
        }

        return params

    def _complete_params_and_scale_X(self, x, y):

        if self.params['w_start'] is None:
            self.params['w_start'] = np.zeros(x.shape[1])
            self.algorithm.w = self.params['w_start']
            self._update_algoritm_params()

        if not self.params['algorithm'] in ['l-bfgs', 'sgd']:
            self._find_alpha(x, y)

    def _find_alpha(self, x, y):

        if self.params['alpha'] is not None:
            return

        alpha = self.params['alpha_start']
        w = self.params['w_start']

        a = 0.99
        c1 = 0.0001

        gradient = self.oracle.g(w, x, y)
        gradient_norm_square = np.linalg.norm(gradient) ** 2
        function_value = self.oracle.f(w, x, y)

        condition = self.oracle.f(w - alpha * gradient, x, y) > function_value - alpha * c1 * gradient_norm_square

        while condition:
            alpha *= a
            condition = self.oracle.f(w - alpha * gradient, x, y) > function_value - alpha * c1 * gradient_norm_square

        self.params['alpha'] = alpha
        self._update_algoritm_params()

    def _update_algoritm_params(self):
        self.algorithm.params = self.params
