# wrapper around internal gllvm class for convenience.
# Lets users call dynamirt.loadings.Full instead of 
# dynamirt.gllvm.loadings.Full or top level 
# dynamirt.Full (also possible)

from .gllvm.loadings import Full, Fixed, Confirmatory, Sparsity, q_matrix