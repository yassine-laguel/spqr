"""Algorithm module
.. moduleauthor:: Yassine LAGUEL
"""

from .subgradient_method import SubgradientMethod
from .dual_averaging import DualAveraging
from .dual_averaging_advanced import DualAveragingAdvanced
from .gradient_method import GradientMethod
from .nesterov import NesterovMethod
from .quasi_newton import LBFGS

__all__ = ['SubgradientMethod', 'DualAveraging', 'DualAveragingAdvanced', 'GradientMethod', 'NesterovMethod', 'LBFGS']
