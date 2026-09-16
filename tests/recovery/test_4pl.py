import numpy as np
import pytest

from .dichotomous import fit_simulated
from .helpers import REPLICATIONS, rmse


@pytest.mark.slow
@pytest.mark.parametrize("replication", range(REPLICATIONS))
def test_4pl_recovery(replication):
    truth, draws, divergences = fit_simulated("4PL", replication)
    assert all(np.isfinite(values).all() for values in draws.values())
    assert divergences == 0
    assert rmse(draws["intercept"], truth["intercept"]) < 0.6
    assert rmse(draws["discrimination"], truth["discrimination"]) < 0.5
    # Assess the actual upper asymptote, not the internal gap parameter.
    assert rmse(draws["lower_asymptote"], truth["lower_asymptote"]) < 0.15
    assert rmse(draws["upper_asymptote"], truth["upper_asymptote"]) < 0.15
    assert (draws["lower_asymptote"] > 0).all()
    assert (draws["upper_asymptote"] < 1).all()
    assert (draws["upper_asymptote"] > draws["lower_asymptote"]).all()
    assert rmse(draws["theta"], truth["theta"]) < 0.9
