# wrapper around internal gllvm class for convenience.
# Lets users call dynamirt.terms.Linear instead of 
# dynamirt.gllvm.terms.linear.Linear or top level 
# dynamirt.Linear (also possible)

from .gllvm.terms.linear import Linear
from .gllvm.terms.discrete import GRW, AR1
from .gllvm.terms.continuous import ExactGP, HSGP
