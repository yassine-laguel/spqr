"""Core module
.. moduleauthor:: Yassine LAGUEL
"""

from ._template import TemplateEstimator
from ._template import TemplateClassifier
from ._template import TemplateTransformer
from .oracle import OracleSubgradient, OracleSmoothGradient, IntergratedOracleSubgradient, \
    IntegratedOracleSmoothGradient
from .risk_optimization import RiskOptimizer

from ._version import __version__

__all__ = ['TemplateEstimator', 'TemplateClassifier', 'TemplateTransformer',
           '__version__', 'OracleSubgradient', 'OracleSmoothGradient', 'IntergratedOracleSubgradient',
           'IntegratedOracleSmoothGradient', 'RiskOptimizer']
