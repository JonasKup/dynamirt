import numpy as np
import pytest

from .helpers import REPLICATIONS, rmse
from .polytomous import VALID, fit_simulated


@pytest.mark.slow
@pytest.mark.parametrize("model_type", ["PCM", "GPCM"])
@pytest.mark.parametrize("replication", range(REPLICATIONS))
def test_pcm_recovery(model_type, replication):
    truth, draws, divergences = fit_simulated(model_type, replication)
    assert all(np.isfinite(values).all() for values in draws.values())
    assert divergences == 0
    assert rmse(draws["steps"][:, VALID], truth["steps"][VALID]) < 0.6
    if model_type == "GPCM":
        assert rmse(draws["discrimination"], truth["discrimination"]) < 0.5
    assert rmse(draws["theta"], truth["theta"]) < 0.9
