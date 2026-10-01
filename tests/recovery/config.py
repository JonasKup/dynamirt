"""Edit these two presets to change the recovery design."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    n_respondents: int = 300
    n_items: int = 12
    warmup: int = 200
    samples: int = 200
    chains: int = 1
    chain_method: str = "sequential"
    max_tree_depth: int = 8


SMALL = Settings()
PAPER = Settings(n_respondents=1000, warmup=1500, samples=1000, chains=4, chain_method="parallel", max_tree_depth=10)
PRESETS = {"small": SMALL, "paper": PAPER}
MODELS = ("1PL", "2PL", "3PL", "4PL", "GRM", "PCM", "GPCM")
SEED = 0
