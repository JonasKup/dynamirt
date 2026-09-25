Parameters and pooling
======================

Use ``Param`` to specify fixed values or priors and control sharing across
groups and variables. Use ``Pool`` for non-centered partial pooling.

For example, estimate a separate parameter for each latent dimension while
sharing it across groups::

    import numpyro.distributions as dist
    from dynamirt import Param

    scale = Param(dist.HalfNormal(1.0), by_variable="free")

Param
-----

.. autoclass:: dynamirt.Param

Pool
----

.. autoclass:: dynamirt.Pool

