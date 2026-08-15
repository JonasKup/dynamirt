"""Internal GLLVM module for dynamirt. API not stable."""

from .model import gllvm
from .terms.linear import Linear

__all__ = ["gllvm", "Linear"]