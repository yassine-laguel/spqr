"""
.. module:: oracle
   :synopsis: Module with definitions of first order oracles class and their smoothed version.


.. moduleauthor:: Yassine LAGUEL
"""

from .measures import *
from .oracles_utils import *


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
        sequence_losses = self.seq_loss(w, x, y)
        return superquantile(self.p, sequence_losses)

    def f(self, w, x, y):
        """ Does the exact same job as OracleSubgradient.cost_function
        """
        return self.cost_function(w, x, y)

    def g(self, w, x, y):
        """Computes a subgradient of :math:`w \mapsto Cvar \circ L(w)` for the dataset :math:`(x,y)`
        """
        sequence_losses = self.seq_loss(w, x, y)
        sequence_gradients = self.seq_grad(w, x, y)
        return compute_subgradient(sequence_losses, sequence_gradients, self.p)

    def seq_loss(self, w, x, y):
        return np.asarray(self.L_f(w, x, y), dtype=np.float64)

    def seq_grad(self, w, x, y):
        return np.asarray(self.L_f_prime(w, x, y), dtype=np.float64)


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
        sequence_losses = self.seq_loss(w, x, y)
        return superquantile(self.p, sequence_losses)

    def f(self, w, x, y):
        """Computes the value of the smooth approximation :math:`w \mapsto Cvar \circ L(w)`
        """
        f, g = self._smooth_superquantile(w, x, y)

        return f

    def g(self, w, x, y):
        """Computes the gradient of the smooth approximation of :math:`w \mapsto Cvar \circ L(w)`
        """
        f, g = self._smooth_superquantile(w, x, y)

        return g

    def seq_loss(self, w, x, y):
        return np.asarray(self.L_f(w, x, y), dtype=np.float64)

    def seq_grad(self, w, x, y):
        return np.asarray(self.L_f_prime(w, x, y), dtype=np.float64)

    def _smooth_superquantile(self, w, x, y):

        simplex_center = 1.0 / len(y) * np.ones(len(y), dtype=np.float64)
        sequence_losses = self.seq_loss(w, x, y)
        q_mu = self._projection(sequence_losses)

        f = np.dot(sequence_losses, q_mu) - self.smoothing_parameter * np.linalg.norm(q_mu - simplex_center) ** 2

        jacobian_l = self.seq_grad(w, x, y)
        g = np.transpose(jacobian_l).dot(q_mu)

        return f, g

    def _projection(self, u):
        return fast_projection(u, self.p, self.smoothing_parameter)

    def _theta_prime(self, lmbda, v, sorted_index):
        return fast_theta_prime(lmbda, v, sorted_index, self.p, self.smoothing_parameter)

    def _find_lmbda(self, v, sorted_index):
        return fast_find_lmbda(v, sorted_index, self.p, self.smoothing_parameter)


class OracleStochasticSubgradient:
    """Base class that instantiate the stochastic oracle for a non differentiable loss

            For an input oracle :math:`L` given through two functions ``loss`` and ``loss_grad``,
            this class is an interface to compute the value and a subgradient of the function
            :math:`w \mapsto Cvar \circ L(w)` over a random subsample of the specified dataset

            :param loss: function associated to the oracle
            :param loss_grad: gradient associated to the oracle
            :param p: probability level (by default 0.8)

        """

    def __init__(self, loss, loss_grad, p, mini_batch_size=50):
        self.L_f = loss
        self.L_f_prime = loss_grad
        self.p = p

        self.mini_batch_size = mini_batch_size
        self.mini_batch_indices = np.arange(mini_batch_size)

    def cost_function(self, w, x, y):
        """Computes the value of :math:`w \mapsto Cvar \circ L(w)` for the dataset :math:`(x,y)`
        """
        sequence_losses = self.seq_loss(w, x, y)
        return superquantile(self.p, sequence_losses)

    def f(self, w, x, y, sample_index=None):
        """ Does the exact same job as OracleStochasticGradient.cost_function
        """
        stoch_seq_losses = self.stoch_seq_loss(w, x, y, sample_index=sample_index)
        return superquantile(self.p, stoch_seq_losses)

    def g(self, w, x, y, sample_index=None):
        """Computes a subgradient of :math:`w \mapsto Cvar \circ L(w)` for the dataset :math:`(x,y)`
        """

        sequence_losses = self.stoch_seq_loss(w, x, y, sample_index=sample_index)
        sequence_gradients = self.stoch_seq_grad(w, x, y, sample_index=self.mini_batch_indices)
        return compute_subgradient(sequence_losses, sequence_gradients, self.p)

    def stoch_seq_loss(self, w, x, y, sample_index=None):
        if sample_index is None:
            self.mini_batch_indices = np.random.choice(len(y), size=self.mini_batch_size)
        else:
            self.mini_batch_indices = sample_index
        sample_x = x[self.mini_batch_indices]
        sample_y = y[self.mini_batch_indices]
        return np.asarray(self.L_f(w, sample_x, sample_y), dtype=np.float64)

    def stoch_seq_grad(self, w, x, y, sample_index=None):
        if sample_index is None:
            self.mini_batch_indices = np.random.choice(len(y), size=self.mini_batch_size)
        else:
            self.mini_batch_indices = sample_index
        sample_x = x[self.mini_batch_indices]
        sample_y = y[self.mini_batch_indices]
        return np.asarray(self.L_f_prime(w, sample_x, sample_y), dtype=np.float64)

    def seq_loss(self, w, x, y):
        return np.asarray(self.L_f(w, x, y), dtype=np.float64)


class OracleStochasticGradient:
    """Base class that instantiate the stochastic superquantile oracle for a differentiable loss

            For an input oracle :math:`L` given through two functions ``loss`` and ``grad_loss``,
            this class is an interface to compute the value and the gradient of the function
            :math:`w \mapsto Cvar \circ L(w)` over a random subsample of the specified dataset.

            :param loss: function associated to the oracle
            :param loss_grad: gradient associated to the oracle
            :param p: probability level (by default 0.8)
            :param smoothing_parameter: specified smoothing parameter according to Nesterov's smoothing.

    """

    def __init__(self, loss, loss_grad, p, mini_batch_size=50, smoothing_parameter=1000.0):

        self.L_f = loss
        self.L_f_prime = loss_grad
        self.p = p
        self.smoothing_parameter = smoothing_parameter

        self.mini_batch_size = mini_batch_size
        self.mini_batch_indices = np.arange(mini_batch_size)

    def cost_function(self, w, x, y):
        """Computes the value of :math:`w \mapsto Cvar \circ L(w)` for the dataset :math:`(x,y)`
        """
        sequence_losses = self.seq_loss(w, x, y)
        return superquantile(self.p, sequence_losses)

    def f(self, w, x, y, sample_index=None):
        """Computes the value of the smooth approximation :math:`w \mapsto Cvar \circ L(w)`
        """
        f, g = self._stoch_smooth_superquantile(w, x, y, sample_index=sample_index)

        return f

    def g(self, w, x, y, sample_index=None):
        """Computes the gradient of the smooth approximation of :math:`w \mapsto Cvar \circ L(w)`
        """
        f, g = self._stoch_smooth_superquantile(w, x, y, sample_index=sample_index)

        return g

    def seq_loss(self, w, x, y):
        return np.asarray(self.L_f(w, x, y), dtype=np.float64)

    def stoch_seq_loss(self, w, x, y, sample_index=None):
        if sample_index is None:
            self.mini_batch_indices = np.random.choice(len(y), size=self.mini_batch_size)
        else:
            self.mini_batch_indices = sample_index
        sample_x = x[self.mini_batch_indices]
        sample_y = y[self.mini_batch_indices]
        return np.asarray(self.L_f(w, sample_x, sample_y), dtype=np.float64)

    def stoch_seq_grad(self, w, x, y, sample_index=None):
        if sample_index is None:
            self.mini_batch_indices = np.random.choice(len(y), size=self.mini_batch_size)
        else:
            self.mini_batch_indices = sample_index
        sample_x = x[self.mini_batch_indices]
        sample_y = y[self.mini_batch_indices]
        return np.asarray(self.L_f_prime(w, sample_x, sample_y), dtype=np.float64)

    def _stoch_smooth_superquantile(self, w, x, y, sample_index=None):

        simplex_center = 1.0 / self.mini_batch_size * np.ones(self.mini_batch_size, dtype=np.float64)
        sequence_losses = self.stoch_seq_loss(w, x, y, sample_index=sample_index)
        q_mu = self._projection(sequence_losses)

        f = np.dot(sequence_losses, q_mu) - self.smoothing_parameter * np.linalg.norm(q_mu - simplex_center) ** 2

        jacobian_l = self.stoch_seq_grad(w, x, y, sample_index=self.mini_batch_indices)
        g = np.transpose(jacobian_l).dot(q_mu)

        return f, g

    def _projection(self, u):
        return fast_projection(u, self.p, self.smoothing_parameter)

    def _theta_prime(self, lmbda, v, sorted_index):
        return fast_theta_prime(lmbda, v, sorted_index, self.p, self.smoothing_parameter)

    def _find_lmbda(self, v, sorted_index):
        return fast_find_lmbda(v, sorted_index, self.p, self.smoothing_parameter)
