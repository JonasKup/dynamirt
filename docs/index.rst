dynamirt
========

**dynamirt** is a Python/NumPyro package for static and longitudinal
multidimensional item response theory (MIRT). It is built for repeated binary
or ordinal item responses collected at irregular times, including ragged
panel and EMA data where respondents have different numbers and schedules of
observations.

Models are assembled from two parts: a *measurement model* (item intercepts or
thresholds, loadings, asymptotes and optional DIF) and a *latent model* made of
additive terms such as linear effects, random walks, AR(1) processes and
Gaussian processes. Models are fitted with MCMC or SVI.

.. toctree::
   :maxdepth: 1

   getting_started

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/tutorial_static
   tutorials/tutorial_dynamic
   tutorials/tutorial_NODE

.. toctree::
   :maxdepth: 2
   :caption: API reference

   modules
