# Author: Yassine Laguel
# License: BSD

import sys
import numpy as np
from time import time


class DualAveraging:

    def __init__(self, oracle, params):

        self.oracle = oracle
        self.params = params
        self.w = np.copy(self.params['w_start'])
        self.list_iterates = []
        self.list_values = []
        self.list_times = []

    def run(self, x, y, verbose_mode=False):
        """ Runs a bacic version of Dual Averaging Algorithm

                :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
                :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
                :param ``bool`` verbose_mode: If ``True``, saves function values during iterations of selected
                        algorithm as well as time since start.
        """

        if verbose_mode:
            start_time = time()

        d = len(self.w)
        n = len(y)
        q = np.ones(n) / n

        lmbda = self.params['dual_averaging_basic_lmbda']
        beta_hat = 1.0
        g = self.oracle.g(self.w, q, x, y)

        norm_gradient = np.sqrt((1.0/lmbda) * np.linalg.norm(g[:d])**2 +
                             (1.0 / (1.0 - lmbda)) * np.linalg.norm(g[d:])**2)

        s = 1.0 / norm_gradient * g  # Weighted Dual averaging

        sum_gradients = norm_gradient

        new_iterate_w = self.w

        counter = 0
        while counter < self.params['dual_averaging_basic_nb_iterations']:

            if verbose_mode:
                self.list_times.append(time() - start_time)
                self.list_values.append(self.oracle.cost_function(self.w, x, y))
                sys.stdout.write('%d / %d  iterations completed \r' %
                                 (counter, self.params['dual_averaging_basic_nb_iterations']))
                sys.stdout.flush()

            g = self.oracle.g(new_iterate_w, q, x, y)

            norm_gradient = np.sqrt((1.0/lmbda) * np.linalg.norm(g[:d])**2 +
                             (1.0 / (1.0 - lmbda)) * np.linalg.norm(g[d:])**2)

            s = s + 1.0/norm_gradient * g

            beta_hat = beta_hat + 1.0/beta_hat
            beta = (1.0/self.params['alpha'] * 10000) * beta_hat

            s_w = s[:d]
            s_q = s[d:]

            new_iterate_w = -1.0/(beta * lmbda) * s_w
            q = self.euclidean_proj_simplex(-1.0/((1.0 - lmbda) * beta) * np.array(s_q))

            self.w = (1.0 / (sum_gradients + norm_gradient)) * (sum_gradients * self.w + norm_gradient * new_iterate_w)

            sum_gradients += norm_gradient

            self.list_iterates.append(self.w)

            counter += 1

    def euclidean_proj_simplex(self, v, s=1):

        # """ Compute the Euclidean projection on a positive simplex

        # Solves the optimisation problem (using the algorithm from [1]):
        #    min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0

        # :param: v: n-dimensional vector to project
        # :param: s: radius of the simplex

        # :return: Euclidean projection of v on the simplex
        #
        # Notes
        # -----
        # The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
        # Better alternatives exist for high-dimensional sparse vectors (cf. [1])
        # However, this implementation still easily scales to millions of dimensions.
        # Author
        # -----
        # Ardrien Gaidon, (see https://gist.github.com/daien/1272551)
        # References
        # ----------
        # [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        #     John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        #     International Conference on Machine Learning (ICML 2008)
        #     http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
        # """

        assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
        n, = v.shape  # will raise ValueError if v is not 1-D
        # check if we are already on the simplex
        if v.sum() == s and np.alltrue(v >= 0):
            # best projection: itself!
            return v
        # get the array of cumulative sums of a sorted (decreasing) copy of v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        # get the number of > 0 components of the optimal solution
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
        # compute the Lagrange multiplier associated to the simplex constraint
        theta = float(cssv[rho] - s) / rho
        # compute the projection by thresholding v using theta
        w = (v - theta).clip(min=0)

        return w
