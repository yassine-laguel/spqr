Getting Started
===============

Download
----------
Clone the repository available at ::

  $ git clone https://github.com/yassine-laguel/spqr.git
  $ cd toolbox/

Installation
-------------
SPQR requires several packages one can install through ``conda`` and provided yaml file ``spqr_env.yml``::

  $ conda env create --file spqr_env.yml
  $ source activate spqr_env

Simple Case Demo
----------------
Let ``X`` be a ``numpy ndarray`` of dimensions :math:`n \times p` representing the features and ``y``,
is one dimensional array of size :math:`n` representing the targets :

.. code-block:: python

  import numpy as np
  X = np.random.rand(10,2)
  w = np.array([1.,2.])
  y = np.dot(X,w) + np.random.rand(10)

We propose to minimize the conditional value at risk of :math:`L_2`-loss :math:`\frac{1}{2}\|Y-Xw\|^2`:

.. code-block:: python

  def loss(w,x,y):
    return (y - np.dot(x,w))**2
  def loss_prime(w,x,y):
    return -2.0 * (y - np.dot(x,w)) * x

For that purpose, a RiskOptimizer object with probability level for CVar :math:`p=0.8` is instantiated:

.. code-block:: python

  from spqr import RiskOptimizer
  optimizer = RiskOptimizer(loss, loss_prime, p=0.8)

One can run selected descent algorithm (by default ``'subgradient'``) with:

.. code-block:: python

  optimizer.fit(X,y)
  sol = optimizer.solution

A deeper insight of the toolbox is available through ``https://github.com/yassine-laguel/spqr/blob/master/docs/toolbox_demonstration.ipynb``
