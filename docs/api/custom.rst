Custom models
=============

Define an additive contribution using NumPyro/JAX code with ``CustomTerm``.
Its function receives a ``ModelContext`` containing data and dimensions.
The context provides ``design`` and ``index`` helpers for covariates.

CustomTerm
----------

.. autoclass:: dynamirt.CustomTerm

ModelContext
------------

.. autoclass:: dynamirt.ModelContext
   :members: design, index

