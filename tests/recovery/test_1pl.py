import numpy as np
import pytest

from .dichotomous import fit_simulated
from .helpers import REPLICATIONS, rmse


@pytest.mark.slow
@pytest.mark.parametrize("replication", range(REPLICATIONS))
def test_1pl_recovery(replication):
    truth, draws, divergences = fit_simulated("1PL", replication)
    assert all(np.isfinite(values).all() for values in draws.values())
    assert divergences == 0
    assert rmse(draws["intercept"], truth["intercept"]) < 0.6
    assert rmse(draws["theta"], truth["theta"]) < 0.9
