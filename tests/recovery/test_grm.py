import numpy as np
import pytest

from .helpers import REPLICATIONS, rmse
from .polytomous import VALID, fit_simulated


@pytest.mark.slow
@pytest.mark.parametrize("replication", range(REPLICATIONS))
def test_grm_recovery(replication):
    truth, draws, divergences = fit_simulated("GRM", replication)
    assert all(np.isfinite(values).all() for values in draws.values())
    assert divergences == 0
    assert rmse(draws["cutpoints"][:, VALID], truth["cutpoints"][VALID]) < 0.6
    assert rmse(draws["discrimination"], truth["discrimination"]) < 0.5
    assert rmse(draws["theta"], truth["theta"]) < 0.9
