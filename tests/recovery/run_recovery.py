"""Fit one model/replication without pytest or recovery pass/fail assertions."""
import argparse
from pathlib import Path
import sys

# Permit direct execution while keeping shared helpers in the recovery package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery.config import MODELS, PRESETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--preset", choices=PRESETS, default="paper")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rep < 0:
        parser.error("--rep must be nonnegative")

    settings = PRESETS[args.preset]
    import numpyro
    if settings.chain_method == "parallel":
        # Expose one CPU device per chain before JAX initializes its backend.
        numpyro.set_host_device_count(settings.chains)
    import jax
    if args.preset == "paper":
        jax.config.update("jax_enable_x64", True)
    from recovery import dichotomous, polytomous

    fit = dichotomous.fit_simulated if args.model in MODELS[:4] else polytomous.fit_simulated
    fit(args.model, args.rep, settings=settings, output=args.output)
    print(f"Saved {args.model} replication {args.rep} to {args.output}")


if __name__ == "__main__":
    main()
