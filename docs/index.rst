.. SPQR documentation master file, created by
   sphinx-quickstart on Wed Apr  3 17:34:44 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SPQR
=============

SPQR is a python toolbox for optimization of superquantile-based risk measures.For more details, we refer to the companion paper “First Order Algorithms for Minimization of superquantile-based Risk Measures”.

Overview
--------

For a couple of features and labels :math:`(X,y)`,this toolbox is aimed at minimizing functions of the form :

.. math::

  \phi(w) = \text{CVAR}_{p} \circ L_{X,y}(w),

where :math:`\text{CVAR}` denotes the superquantile, also called "conditional value at risk", "average value at risk" or "expected shortfall" and loss function :math:`L` is assumed to be provided by the user together with the dataset :math:`(X,y)`.

We build oracles for the nonsmooth function phi and for a smoothed counterpart :math:`phi_mu`. Various first-order algorithms are proposed to minimise these 2 functions. Among these first order algorithms, one can find the Dual Averaging Method, Nesterov Accelerated Method or LBFGS. For instance, quantile regression and superquantile regression can be performed with this toolbox :

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
