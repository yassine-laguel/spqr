SPQR API summary
========================

.. currentmodule:: toolbox

Oracles
-------

First Order Oracles for superquantile optimization:

.. autosummary::

    spqr.OracleSubgradient
    spqr.OracleSmoothGradient

First Order Oracles for hyperquantile optimization:

.. autosummary::

    spqr.IntergratedOracleSubgradient
    spqr.IntegratedOracleSmoothGradient

Optimization Algorithms
-----------------------

Algorithms for a non-smooth loss function:

.. autosummary::

    spqr.algorithms.SubgradientMethod
    spqr.algorithms.DualAveraging
    spqr.algorithms.DualAveragingAdvanced

Algorithms for a smooth loss function:

.. autosummary::

    spqr.algorithms.GradientMethod
    spqr.algorithms.NesterovMethod
    spqr.algorithms.BFGS

Risk Optimization Framework
---------------------------

.. autosummary::

    spqr.RiskOptimizer
