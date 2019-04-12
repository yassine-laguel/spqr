.. SPQR documentation master file, created by
   sphinx-quickstart on Wed Apr  3 17:34:44 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SPQR
=============

SPQR is a python toolbox for optimization of superquantile-based risk measures. It is aimed at providing a robust solution to problems with models exposed
to uncertainty. For theoretical matters, see the attached paper "First Order Algorithms for Minimization of superquantile-based Risk Measures".

Overview
--------

For any dataset of features and labels :math:`(X,y)`,this toolbox is aimed at minimizing functions of the form :

.. math::

  \phi(w) = \text{CVAR}_{p} \circ L_{X,y}(w),

where :math:`\text{CVAR}` denotes the superquantile, also called conditional value at risk, average value at risk or expected shortfall and loss function :math:`L` is assumed to be provided by the user together with the dataset :math:`(X,y)`.

A first order oracle is built thanks to it. Nature of the oracle depend on the regularity of function :math:`L`, as well as the algorithms proposed to run the optimization. Among these first order algorithms, one can find the Dual Averaging Method, Nesterov Accelerated Method or BFGS. For instance, quantile regression and superquantile regression can be performed with this toolbox :

.. image:: img/quantile_superquantile_reg-1.png
   :scale: 50 %

A deeper insight of the toolbox is made possible through a jupyter notebook available at ``https://github.com/yassine-laguel/spqr/blob/master/docs/toolbox_demonstration.ipynb``

Table of Contents
-----------------
.. toctree::
   :caption: Table of Contents
   :maxdepth: 2

   Getting Started <start.rst>
   API Summary <api.rst>
   API Oracles <api_detailed/oracles.rst>
   API Algorithms <api_detailed/algorithms.rst>
   API Optimization Framework <api_detailed/risk_optimization.rst>

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Authors
-------
* `Yassine Laguel`
* `Jerome Malick <https://ljk.imag.fr/membres/Jerome.Malick/>`_
* `Zaid Harchaoui <http://faculty.washington.edu/zaid/>`_
