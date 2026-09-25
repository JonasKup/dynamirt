Fitting and results
===================

Fit a model with MCMC or variational inference, then call
:meth:`dynamirt.fit.FitResult.to_idata` to obtain labeled posterior results.
The returned result exposes the native inference object through ``inference``.

fit_mcmc
--------

.. autofunction:: dynamirt.fit_mcmc

fit_svi
-------

.. autofunction:: dynamirt.fit_svi

FitResult
---------

.. autoclass:: dynamirt.fit.FitResult
   :members: to_idata

SVIState
--------

.. autoclass:: dynamirt.fit.SVIState

