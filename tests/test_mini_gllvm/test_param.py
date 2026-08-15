# Test to check if Param helper broadcasts correctly
import pytest

from dynamirt.mini_gllvm.parameters import Param

import numpyro
import numpyro.distributions as dist

@pytest.mark.parametrize("by_group, n_g", [("shared", 1), ("free", 3)])
@pytest.mark.parametrize("by_variable, n_v", [("shared", 1), ("free", 4)])
def test_param_shapes(by_group, n_g, by_variable, n_v):
    param = Param(dist.Normal(0, 1), by_group=by_group, by_variable=by_variable)
    with numpyro.handlers.trace() as tr, numpyro.handlers.seed(rng_seed=0):
        out = param("p", 3, 4)

    assert tr["p"]["value"].shape == (n_g, n_v)   # sampled compactly
    assert out.shape == (3, 4)                    # returned broadcast