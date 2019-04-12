# Author: Yassine Laguel
# License: BSD

import numpy as np
import sys, time


class Catalyst:

    def __init__(self, oracle, params):

        self.oracle = oracle
        self.params=params

        self.w = self.params['w_start']

        self.kappa = self.params['catalyst_kappa']

    class OracleSubProblem:

        def __init__(self, smooth_oracle, X, y, kappa,y_var):
            self.smooth_oracle = smooth_oracle
            self.kappa = kappa
            self.y_var = y_var
            self.X = X
            self.y = y

        def f(self, x):
            return self.smooth_oracle.f(x, self.X, self.y) + 0.5 * self.kappa * np.linalg.norm(x - self.y_var) ** 2

        def g(self, x):
            return self.smooth_oracle.g(x, self.X, self.y) + self.kappa * (x - self.y_var)

    def solve_sub_problem(self, X, y, y_var, x_start, accuracy):
        '''
        Input : y, proximal center in the subproblem of Catalyst
                x_start, starting point for the subproblem
                accuracy, Function that returns a boolean to state if
                          desired accuracy is achieved
        Output : approximate solution of Subproblem in Catalyst
        Description : Solves approximately the subproblem in Catalyst, using the method M
        '''

        oracle_subproblem = Catalyst.OracleSubProblem(self.oracle, X, y, self.kappa, y_var)
        optimizer = FullGradient(oracle_subproblem, accuracy, x_start, self.params['alpha_start'])

        optimizer.run()
        return optimizer.x

    def run(self, X, y, verboose_mode=False):

        self.list_iterates = [self.w]

        # STEP 1

        y_var = np.copy(self.w)
        q = 0.0
        alpha = 1.0

        # STEP 2
        counter = 0

        while counter < self.params['catalyst_nb_iterations']:

            # STEP 2.1 : Warm Start
            z0 = np.copy(y_var)

            # Accuracy associated to criterion (C2)
            sqrt_delta = 1.0 / (counter + 1)
            # TODO : adjust here so that the accuracy is not a problem

            def accuracy(g, u, mode_debug=False):

                if mode_debug:
                    print((sqrt_delta / 2.0) * self.kappa * np.linalg.norm(u - y_var))
                    print(np.linalg.norm(g))
                return np.linalg.norm(g) <= (sqrt_delta / 2.0) * self.kappa * np.linalg.norm(u - y_var)

            w_next = self.solve_sub_problem(X, y, y_var, z0, accuracy)

            # STEP 2.2
            aux_1 = np.sqrt((alpha ** 2 - q) ** 2 + 4 * alpha ** 2)
            aux_2 = -1.0 * (alpha ** 2 - q)
            alpha_next = (aux_1 + aux_2) / 2.0
            if not (alpha_next > 0 and alpha_next < 1):
                alpha_next = (aux_1 - aux_2) / 2.0

            # STEP 2.3
            beta = (alpha * (1.0 - alpha)) / (alpha ** 2 + alpha_next)
            y_var = w_next + beta * (w_next - self.w)
            if np.linalg.norm(self.w - w_next) < 10 ** (-10):  # To avoid infinite loop in subproblem due to precision machine
                while counter < self.params['catalyst_nb_iterations']:
                    self.list_iterates.append(self.w)
                    counter += 1
                    if verboose_mode:
                        sys.stdout.write('%d / %d  iterations completed \r' % (counter, self.params['catalyst_nb_iterations']))
                        sys.stdout.flush()
                break

            # STEP 2.4
            self.w = w_next
            self.list_iterates.append(self.w)
            alpha = alpha_next
            counter += 1

            if verboose_mode:
                sys.stdout.write('%d / %d  iterations completed \r' % (counter, self['catalyst_nb_iterations']))
                sys.stdout.flush()


class FullGradient:

    def __init__(self, oracle, accuracy, x_start, alpha_start):
        # '''
        #    Input : oracle, method that retrieve value and gradient of a smooth and
        #                    strongly convex objective function
        #            alpha, smoothness of the objective function
        #            beta, strong convexity objective function
        #            accuracy, Function that returns a boolean to state if
        #                      desired accuracy is achieved
        #            x_start, starting point
        #    Output : approximate minimum of the oracle with epsilon accuracy
        #    Description : Solves approximately the problem of minimization of the oracle
        #                  function with full gradient method. The stopping criterion is
        #                  based on norm of the gradient
        # '''

        self.oracle = oracle
        self.accuracy = accuracy
        self.alpha_start = alpha_start
        self.x = np.copy(x_start)

    def run(self):

        self._find_stepsize()

        g = self.oracle.g(self.x)
        counter = 0

        while not self.accuracy(g, self.x):

            self.x = self.x - self.alpha * g
            g = self.oracle.g(self.x)
            counter += 1
            if counter > 20:
                self._find_stepsize()
                if self.alpha < 10 ** (-5):
                    return

    def _find_stepsize(self, mode_debug=False):

        alpha = self.alpha_start
        w = self.x

        gradient = self.oracle.g(w)
        norm_gradient_square = np.linalg.norm(gradient)**2

        condition = (self.oracle.f(w - alpha * gradient) > self.oracle.f(w) - alpha * 0.01 * norm_gradient_square)

        while condition:
            alpha *= 0.5
            condition = (self.oracle.f(w - alpha * gradient) > self.oracle.f(w) - alpha * 0.01 * norm_gradient_square)
        if mode_debug:
            print('alpha_value ' + str(alpha))

        self.alpha = alpha
