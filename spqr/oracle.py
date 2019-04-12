"""
.. module:: oracle
   :synopsis: Module with definitions of first order oracles class and their smoothed version.


.. moduleauthor:: Yassine LAGUEL
"""

from .measures import *


class OracleDualAveraging:

    def __init__(self, loss, loss_grad, p):

        self.L_f = loss
        self.L_f_prime = loss_grad
        self.p = p

    def cost_function(self, w, x, y):
        # """ Computes the value of :math:`w \mapsto Cvar \circ L(w)` for the dataset :math:`(x,y)`

        #            :param  w: points for with the gradient is to be computed
        #            :param  x: matrix whose lines are realizations of random variable X
        #            :param  y: vector whose coefficients are realizations of random variable y

        #            :return: :math:`Cvar \circ L(w)`
        # """
        n = len(y)
        u = [self.L_f(w, x[i], y[i]) for i in range(n)]
        return superquantile(self.p, u)

    def f(self, w, q, x, y):

        n = len(y)
        u = [q[i] * self.L_f(w, x[i], y[i]) for i in range(n)]

        return np.sum(u)

    def g(self, w, q, x, y):
        n = len(y)

        u_prime = [self.L_f_prime(w, x[i], y[i]) for i in range(n)]
        u_prime = np.dot(np.transpose(u_prime), q)
        v_prime = [-1.0 * self.L_f(w, x[i], y[i]) for i in range(n)]

        return np.concatenate((u_prime, v_prime))


class OracleSubgradient:
    """Base class that instantiate the superquantile oracle for a non differentiable loss

        For an input oracle :math:`L` given through two functions ``loss`` and ``loss_grad``,
        this class is an interface to compute the value and a subgradient of the function
        :math:`w \mapsto Cvar \circ L(w)` over a specified dataset

        :param loss: function associated to the oracle
        :param loss_grad: gradient associated to the oracle
        :param p: probability level (by default 0.8)

    """

    def __init__(self, loss, loss_grad, p):

        self.L_f = loss
        self.L_f_prime = loss_grad
        self.p = p

    def cost_function(self, w, x, y):
        """Computes the value of :math:`w \mapsto Cvar \circ L(w)` for the dataset :math:`(x,y)`
        """
        # :param ``numpy.array`` w: evaluation point
        # :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
        # :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
        #
        # :return: :math:`Cvar \circ L(w)`

        n = len(y)
        u = [self.L_f(w, x[i], y[i]) for i in range(n)]

        return superquantile(self.p, u)

    def f(self, w, x, y):
        """ Does the exact same job as OracleSubgradient.cost_function
        """

        return self.cost_function(w, x, y)

    def g(self, w, x, y):
        """Computes a subgradient of :math:`w \mapsto Cvar \circ L(w)` for the dataset :math:`(x,y)`
        """
        # :param ``numpy.array`` w: points for with the gradient is to be computed
        # :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
        # :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
        #
        # :return: a subgradient of :math:`w \mapsto Cvar \circ L(w)` for the dataset :math:`(x,y)`

        n = len(y)
        u = [self.L_f(w, x[i], y[i]) for i in range(n)]
        u_prime = [self.L_f_prime(w, x[i], y[i]) for i in range(n)]

        q_p_u = quantile(self.p, u)

        i_p = [i for i, x in enumerate(u) if x == q_p_u]
        j_p = [i for i, x in enumerate(u) if x > q_p_u]

        alpha = (np.ceil(self.p*n)/n - self.p)/len(i_p)

        res = 1.0/(1.0-self.p) * ((1.0/n) * np.sum([u_prime[i] for i in j_p], axis=0)
                                  + alpha * np.sum([u_prime[i] for i in i_p], axis=0))

        return res


class OracleSmoothGradient:
    """Base class that instantiate the superquantile oracle for a differentiable loss

            For an input oracle :math:`L` given through two functions ``loss`` and ``grad_loss``,
            this class is an interface to compute the value and the gradient of the function
            :math:`w \mapsto Cvar \circ L(w)` over a specified dataset.

            :param loss: function associated to the oracle
            :param loss_grad: gradient associated to the oracle
            :param p: probability level (by default 0.8)
            :param smoothing_parameter: specified smoothing parameter according to Nesterov's smoothing.

    """

    def __init__(self, loss, loss_grad, p, smoothing_parameter=1000.0):

        self.L_f = loss
        self.L_f_prime = loss_grad
        self.p = p
        self.smoothing_parameter = smoothing_parameter

    def cost_function(self, w, x, y):
        """Computes the value of :math:`w \mapsto Cvar \circ L(w)` for the dataset :math:`(x,y)`
        """
        # :param ``numpy.array`` w: points for with the gradient is to be computed
        # :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
        # :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
        #
        # :return: :math:`Cvar \circ L(w)`

        n = len(y)
        u = [self.L_f(w, x[i], y[i]) for i in range(n)]

        return superquantile(self.p, u)

    def f(self, w, x, y):
        """Computes the value of the smooth approximation :math:`w \mapsto Cvar \circ L(w)`
        """
        # :param ``numpy.array`` w: points for with the gradient is to be computed
        # :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable :math:`X`
        # :param ``numpy.array`` y: vector whose coefficients are realizations of random variable :math:`y`
        #
        # :return:  Value of the gradient

        f, g = self._smooth_superquantile(w, x, y)

        return f

    def g(self, w, x, y):
        """Computes the gradient of the smooth approximation of :math:`w \mapsto Cvar \circ L(w)`
        """
        # :param ``numpy.array`` w: points for with the gradient is to be computed
        # :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
        # :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
        #
        # :return:  The gradient of the smooth approximation of :math:`w \mapsto Cvar \circ L(w)`

        f, g = self._smooth_superquantile(w, x, y)

        return g

    def _smooth_superquantile(self, w, x, y):

        # """ Computes the value of :math:`w \mapsto Cvar \circ L(w)` for the dataset :math:`(x,y)`
        #
        #    :param ``numpy.array`` w: points for with the gradient is to be computed
        #    :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
        #    :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
        #
        #    :return: :math:`Cvar \circ L(w)`
        # """

        n = len(y)

        simplex_center = 1.0 / n * np.ones(n)
        n = len(y)
        u = [self.L_f(w, x[i], y[i]) for i in range(n)]
        point_to_project = np.array(u)
        q_mu = self._projection(point_to_project)

        f = np.dot(u, q_mu) - self.smoothing_parameter * np.linalg.norm(q_mu - simplex_center) ** 2

        jacobian_l = np.array([self.L_f_prime(w, x[i], y[i]) for i in range(n)])

        g = np.transpose(jacobian_l).dot(q_mu)

        return f, g

    def _projection(self, u):
        # """

        #   Project the point u on the set K defined as the intersection
        #                of the simplex and the ball centered in 0 and of radius \frac{1}{n(1-p)}
        #                associated to the infinity norm.
        # """

        mu = self.smoothing_parameter

        n = len(u)
        c = 1.0 / (n * (1 - self.p))
        v = u + (2.0 * mu / n) * np.ones(n)

        # sorts the coordinate of v
        sorted_index = np.argsort(v)

        # finds the zero of the function theta prime
        lmbda = self._find_lmbda(v, sorted_index)

        # instantiate the output
        res = np.zeros(n)

        # fills the coordinate of the output
        counter = n - 1
        while counter >= 0:
            if lmbda > v[sorted_index[counter]]:
                break
            elif lmbda > v[sorted_index[counter]] - 2 * mu * c:
                res[sorted_index[counter]] = (v[sorted_index[counter]] - lmbda) / (2 * mu)
            else:
                res[sorted_index[counter]] = c
            counter -= 1

        return res

    def _theta_prime(self, lmbda, v, sorted_index):
        # """
        #     Derivative of theta at lmbda
        # """

        n = len(v)
        c = 1.0 / (n * (1.0 - self.p))

        res = 1.0
        counter = n - 1
        while counter >= 0:
            if lmbda >= v[sorted_index[counter]]:
                break
            elif lmbda >= v[sorted_index[counter]] - 2 * self.smoothing_parameter * c:
                res -= (v[sorted_index[counter]] - lmbda) / (2 * self.smoothing_parameter)
            else:
                res -= c
            counter -= 1
        return res

    def _find_lmbda(self, v, sorted_index):

        # """
        #
        #    Compute by dichotomy the zero of the function theta prime, auxiliary function used in methods
        #    projection and smooth_hyperquantile.
        #
        # """

        n = len(v)
        c = 1.0 / (n * (1.0 - self.p))
        set_p = np.sort(np.concatenate([v, v - 2 * self.smoothing_parameter * c * np.ones(n)]))

        def aux(a=0, b=2 * n - 1):

            m = (a + b) // 2
            while (b - a) > 1:
                if self._theta_prime(set_p[m], v, sorted_index) > 0:
                    b = m
                elif self._theta_prime(set_p[m], v, sorted_index) < 0:
                    a = m
                else:
                    return set_p[m]
                m = (a + b) // 2

            res = set_p[a] - (self._theta_prime(set_p[a], v, sorted_index) * (set_p[b] - set_p[a])) / (
                    self._theta_prime(set_p[b], v, sorted_index) - self._theta_prime(set_p[a], v, sorted_index))

            return res

        return aux()


class IntergratedOracleSubgradient:
    """ Base class that instantiate the hyperquantile oracle for a non differentiable loss

            For an input oracle :math:`L` given through two functions ``loss`` and ``loss_grad``,
            this class is an interface to compute the value and a subgradient of the function
            :math:`w \mapsto \bar{Cvar} \circ L(w)` over a specified dataset

                    :param loss: function associated to the oracle
                    :param loss_grad: gradient associated to the oracle
                    :param p: probability level (by default 0.8)
    """

    def __init__(self, loss, loss_grad, p):

        self.L_f = loss
        self.L_f_prime = loss_grad
        self.p = p

    def cost_function(self, w, x, y):
        """ Computes the value of :math:`w \mapsto \bar{Cvar} \circ L(w)` for the dataset :math:`(x,y)`
        """
        # :param ``numpy.array`` w: points for with the gradient is to be computed
        # :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
        # :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
        #
        # :return: :math:`\bar{Cvar} \circ L(w)`

        n = len(y)
        u = [self.L_f(w, x[i], y[i]) for i in range(n)]

        return hyperquantile(self.p, u)

    def f(self, w, x, y):
        """ Does the exact same job as OracleSubgradient.cost_function
        """

        return self.cost_function(w, x, y)

    def g(self, w, x, y):
        """ Computes a subgradient of :math:`w \mapsto \bar{Cvar} \circ L(w)` for the dataset :math:`(x,y)`
        """
        # :param ``numpy.array`` w: points for with the gradient is to be computed
        # :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
        # :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
        #
        # :return: :math:`\bar{Cvar} \circ L(w)`

        n = len(y)

        u = [self.L_f(w, x[i], y[i]) for i in range(n)]
        u_prime = [self.L_f_prime(w, x[i], y[i]) for i in range(n)]

        sorting_index = np.argsort(u)
        index_quantile = int(np.ceil(n * self.p) - 1)

        # Auxiliary function
        def _set_counter_1(counter):
            new_counter_1 = counter + 1

            while u[sorting_index[new_counter_1 - 1]] == u[sorting_index[counter]]:
                new_counter_1 -= 1
                if new_counter_1 - 1 == 0:
                    break
            return new_counter_1

        # Auxiliary function
        def _set_counter_2(counter):
            new_counter_2 = counter - 1

            while u[sorting_index[new_counter_2 + 1]] == u[sorting_index[counter]]:
                new_counter_2 += 1
                if new_counter_2 + 1 == n:
                    break
            return new_counter_2

        # Auxiliary function
        def _add(counter1, counter2):
            if counter1 != counter2:
                s = np.sum([u_prime[sorting_index[counter1: counter2 + 1]]])
            else:
                s = u_prime[sorting_index[counter1]]

            next_jump = (counter2 + 1) / n

            if counter1 <= index_quantile:

                if next_jump < 1.0:
                    factor = next_jump - self.p + (1.0 - next_jump) * np.log((1.0-next_jump)/(1.0 - self.p))
                else:
                    factor = 1.0 / n
                return s * factor / (counter2 - counter1 + 1)

            else:
                factor1 = 1.0 / n * np.log((1.0 - self.p)/(1.0 - next_jump + 1.0/n))
                if next_jump < 1.0:
                    factor2 = (1.0 / n + (1.0 - next_jump) * np.log((1.0-next_jump)/(1.0 - next_jump + 1.0/n))) \
                              / (counter2 - counter1 + 1)
                else:
                    factor2 = 1.0 / (n * (counter2 - counter1 + 1))
                return (factor1 + factor2) * s

        counter_1 = _set_counter_1(index_quantile)

        res = np.zeros(len(x[0]))

        # Main Loop
        while counter_1 < n:
            counter_2 = _set_counter_2(counter_1)
            res += _add(counter_1, counter_2)

            counter_1 = counter_2 + 1

        return 1.0/(1 - self.p) * res


class IntegratedOracleSmoothGradient:
    """ Base class that instantiate the hyperquantile oracle for a differentiable loss

                For an input oracle :math:`L` given through two functions ``loss`` and ``loss_grad``,
                this class is an interface to compute the value and the gradient of the function
                :math:`w \mapsto \bar{Cvar} \circ L(w)` over a specified dataset

                        :param loss: function associated to the oracle
                        :param loss_grad: gradient associated to the oracle
                        :param p: probability level (by default 0.8)
                        :param smoothing_parameter: specified smoothing parameter according to Nesterov's smoothing.
    """

    def __init__(self, loss, loss_grad, p, smoothing_parameter=1000.0):

        self.L_f = loss
        self.L_f_prime = loss_grad
        self.p = p
        self.smoothing_parameter = smoothing_parameter

    def cost_function(self, w, x, y):
        """ Computes the value of :math:`w \mapsto \bar{Cvar} \circ L(w)` for the dataset :math:`(x,y)`
        """
        # :param ``numpy.array`` w: points for with the gradient is to be computed
        # :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
        # :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
        #
        # :return: :math:`Cvar \circ L(w)`

        n = len(y)
        u = [self.L_f(w, x[i], y[i]) for i in range(n)]

        return hyperquantile(self.p, u)

    def f(self, w, x, y):
        """ Computes the value of the smooth approximation :math:`w \mapsto \bar{Cvar}  \circ L(w)`
        """
        # :param ``numpy.array`` w: points for with the gradient is to be computed
        # :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
        # :param ``numpy.array`` y: vector whose coefficients are realizations of random variable y
        #
        # :return:  :math:`\bar{Cvar}  \circ L(w)`

        f, g = self.smooth_hyperquantile(w, x, y)
        return f

    def g(self, w, x, y):
        """ Computes the gradient of the smooth approximation :math:`w \mapsto \bar{Cvar}  \circ L(w)`
        """
        # :param ``numpy.array`` w: points for with the gradient is to be computed
        # :param ``numpy.ndarray`` x: matrix whose lines are realizations of random variable X
        # :param ``numpy.array`` y: vector whose coefficients are realizations of a random variable y
        #
        # :return:  the gradient of the smooth approximation :math:`w \mapsto \bar{Cvar}  \circ L(w)`

        f, g = self.smooth_hyperquantile(w, x, y)
        return g

    def smooth_hyperquantile(self, w, x, y):

        # Size of dataset
        n = len(y)

        u = self.L_f(w, x, y) + (self.smoothing_parameter / n) * np.ones(n)
        jacobian_l = np.array([self.L_f_prime(w, x[i], y[i]) for i in range(n)])

        l_0 = 1.0/(n*(1-self.p))

        res_f = 0.0
        integrated_q = np.zeros(n)
        sorted_index = np.argsort(u)

        lmbda_0 = self.find_lmbda(u, sorted_index)

        index_dealer = IndexDealer(u, sorted_index, self.smoothing_parameter, l_0)
        index_dealer.initialize_sets(l_0, lmbda_0)
        index_dealer.initialize_i_j(l_0, lmbda_0)

        lmbda_0 = index_dealer.treat_pathological_case(l_0, lmbda_0)

        res_f += -1.0 * self.smoothing_parameter / (2 * n ** 2) * (1 / l_0)

        while len(index_dealer.J_2) > 0:

            u_i_0 = u[sorted_index[index_dealer.i]]
            u_i_0prime = u[sorted_index[index_dealer.j]]

            l_10 = l_0 + len(index_dealer.J_1)/(self.smoothing_parameter * len(index_dealer.J_2)) * (u_i_0 - lmbda_0)
            l_11 = l_0 + 1.0/(self.smoothing_parameter * (1.0 + len(index_dealer.J_2)/len(index_dealer.J_1))) * \
                   (u_i_0prime - self.smoothing_parameter * l_0 - lmbda_0)
            l_1 = min([l_10, l_11])

            for i in index_dealer.J_1:
                beta = (len(index_dealer.J_2)/len(index_dealer.J_1))
                alpha = (u[i] - lmbda_0 + self.smoothing_parameter * beta * l_0)

                integrated_q[i] += 1.0/self.smoothing_parameter * (alpha * (1.0/l_0 - 1.0/l_1)/n) +\
                          (beta/n) * np.log(l_0/l_1)

                res_f += -1.0 * self.smoothing_parameter/(2*n) * \
                         (alpha**2 * (1/l_0 - 1/l_1) - 2 * alpha * beta * np.log(l_1/l_0) + beta**2 * (l_1 - l_0))

            if l_1 == l_10:
                lmbda_1 = u_i_0
            else:
                lmbda_1 = u_i_0prime - self.smoothing_parameter * l_1

            index_dealer.update_sets(l_1, lmbda_1, l_10, l_11)

            lmbda_0 = lmbda_1
            l_0 = l_1

            lmbda_0 = index_dealer.treat_pathological_case(l_0, lmbda_0)

        for i in range(n):
            l_1 = index_dealer.l_1_when_entering_J1[i]
            if l_1 != -1.0:
                integrated_q[i] += np.log(l_1 / index_dealer.l0_start) / n
                res_f += -1.0 * self.smoothing_parameter / (6 * n) * (1 / (index_dealer.l0_start ** 3) - (1 / l_1 ** 3))

        for i in index_dealer.J_1:
            integrated_q[i] += (u[i] - lmbda_0) / (n * self.smoothing_parameter) * (1.0/l_0)
            res_f += - (u[i] - lmbda_0)**2 / (2 * n * self.smoothing_parameter) * (1.0/l_0)

        res_f += np.dot(u, integrated_q)
        res_f *= 1.0/(1.0-self.p)
        res_g = 1.0/(1-self.p) * np.dot(np.transpose(jacobian_l), integrated_q)

        return res_f, res_g

    def projection(self, u):

        mu = self.smoothing_parameter

        n = len(u)
        c = 1.0 / (n * (1 - self.p))
        v = u + (2.0 * mu / n) * np.ones(n)

        # sorts the coordinate of v
        sorted_index = np.argsort(v)

        # finds the zero of the function theta prime
        lmbda = self.find_lmbda(v, sorted_index)

        # instantiate the output
        res = np.zeros(n)

        # fills the coordinate of the output
        counter = n - 1
        while counter >= 0:
            if lmbda > v[sorted_index[counter]]:
                break
            elif lmbda > v[sorted_index[counter]] - 2 * mu * c:
                res[sorted_index[counter]] = (v[sorted_index[counter]] - lmbda) / (2 * mu)
            else:
                res[sorted_index[counter]] = c
            counter -= 1

        return res

    def _theta_prime(self, lmbda, v, sorted_index):

        n = len(v)
        c = 1.0 / (n * (1.0 - self.p))

        res = 1.0
        counter = n - 1
        while counter >= 0:
            if lmbda >= v[sorted_index[counter]]:
                break
            elif lmbda >= v[sorted_index[counter]] - self.smoothing_parameter * c:
                res -= (v[sorted_index[counter]] - lmbda) / self.smoothing_parameter
            else:
                res -= c
            counter -= 1
        return res

    def find_lmbda(self, v, sorted_index):

        n = len(v)
        c = 1.0 / (n * (1.0 - self.p))
        set_p = np.sort(np.concatenate([v, v - self.smoothing_parameter * c * np.ones(n)]))

        def aux(a=0, b=2 * n - 1):

            m = (a + b) // 2
            while (b - a) > 1:
                if self._theta_prime(set_p[m], v, sorted_index) > 0:
                    b = m
                elif self._theta_prime(set_p[m], v, sorted_index) < 0:
                    a = m
                else:
                    return set_p[m]
                m = (a + b) // 2

            res = set_p[a] - (self._theta_prime(set_p[a], v, sorted_index) * (set_p[b] - set_p[a])) / (
                    self._theta_prime(set_p[b], v, sorted_index) - self._theta_prime(set_p[a], v, sorted_index))

            return res

        return aux()


class IndexDealer:
    # """

    #    Auxiliary class involved in the storage of index sets associated to the computation of the gradient of the
    #    smooth approximation of the hyperquantile.
    # """

    def __init__(self, u, sorted_index, smoothing_parameter, l0_start):

        self.U = u
        self.sorted_index = sorted_index
        self.smoothing_parameter = smoothing_parameter

        self.J_0 = []
        self.J_1 = []
        self.J_2 = []

        self.i = len(u) - 1
        self.j = len(u) - 1

        self.l0_start = l0_start
        self.l_1_when_entering_J1 = np.ones(len(u))

    def initialize_sets(self, l_0, lmbda_0):

        n = len(self.U)

        for counter in range(n-1, -1, -1):
            if lmbda_0 >= self.U[self.sorted_index[counter]]:
                self.J_0.append(self.sorted_index[counter])
                self.l_1_when_entering_J1[self.sorted_index[counter]] = -1.0
            elif lmbda_0 < self.U[self.sorted_index[counter]] - self.smoothing_parameter * l_0:
                self.J_2.append(self.sorted_index[counter])
            else:
                self.J_1.append(self.sorted_index[counter])
                self.l_1_when_entering_J1[self.sorted_index[counter]] = -1.0

    def initialize_i_j(self, l_0, lmbda_0):

        n = len(self.U)

        sorted_i_0 = n - 1
        sorted_i_0_prime = n - 1
        while self.U[self.sorted_index[sorted_i_0 - 1]] > lmbda_0:
            sorted_i_0 -= 1
            if sorted_i_0 == 0:
                break
        while self.U[self.sorted_index[sorted_i_0_prime - 1]] - self.smoothing_parameter * l_0 > lmbda_0:
            sorted_i_0_prime -= 1
            if sorted_i_0_prime == 0:
                break

        self.i = sorted_i_0
        self.j = sorted_i_0_prime

    def update_sets(self, l_1, lmbda_1, l_10, l_11):

        if l_1 == l_10:
            while self.U[self.sorted_index[self.i]] == lmbda_1:
                index = self.J_1.pop()
                self.J_0.append(index)
                self.i += 1
                if self.i == len(self.U):
                    break

        if l_1 == l_11:
            while self.U[self.sorted_index[self.j]] - self.smoothing_parameter * l_1 == lmbda_1:
                index = self.J_2.pop()
                self.J_1.append(index)
                self.l_1_when_entering_J1[self.sorted_index[self.j]] = l_1
                self.j += 1
                if self.j == len(self.U):
                    break

    def treat_pathological_case(self, l_0, lmbda_0):
        if len(self.J_1) == 0:
            a = self.U[self.J_2[-1]]

            lmbda_0_prime = a - self.smoothing_parameter * l_0

            while self.U[self.sorted_index[self.j]] - self.smoothing_parameter * l_0 == lmbda_0_prime:
                index = self.J_2.pop()
                self.J_1.append(index)
                self.l_1_when_entering_J1[self.sorted_index[self.j]] = l_0
                self.j += 1
                if self.j == len(self.U):
                    break

            return lmbda_0_prime
        else:
            return lmbda_0
