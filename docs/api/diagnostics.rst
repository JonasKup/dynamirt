Diagnostics and plots
=====================

Inspect latent trajectories, loadings, and item response curves after fitting.
Plotting helpers accept an optional Matplotlib ``ax`` and return the Axes.

plot_trajectories
-----------------

.. autofunction:: dynamirt.plot_trajectories

plot_loadings
-------------

.. autofunction:: dynamirt.plot_loadings

item_curves
-----------

.. autofunction:: dynamirt.item_curves

Supply the covariates needed by the original measurement model, including DIF.
For example, if the model has clinic and age effects:

.. code-block:: python

   curves = item_curves(result, covariates={"clinic": 7, "age": 40.0})
   plot_item_curves(curves, item=0)

Use the same group codes and covariate scaling as during fitting. Scalar values
broadcast across the latent grid. All DIF terms are included, so missing
required covariates raise rather than being filled from the training data.
The supplied profile is not stored in the returned dataset.

plot_item_curves
----------------

.. autofunction:: dynamirt.plot_item_curves

Experimental calibration
------------------------

Calibration helpers are under development and currently use training rows.
They are available from ``dynamirt.calibration_curves``.

.. autofunction:: dynamirt.calibration_curves.calibration

.. autofunction:: dynamirt.calibration_curves.plot_calibration
