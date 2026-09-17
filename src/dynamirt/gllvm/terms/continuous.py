from dataclasses import dataclass
from typing import Sequence, Callable, Mapping, Literal

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_matern, diag_spectral_density_squared_exponential
import jax.numpy as jnp
import jax

from tinygp import kernels

import numpy as np

from .._context import _Context
from ..parameters import Param

# To do: Predictive

#------------------------------------------ Exact GPs

def _pad_by_group(X, idx, n_groups):
    """create padded (n_max_points, n_groups, dim) array from (n_obs, dim)"""
    order = np.argsort(idx, kind='stable')
    sorted_idx = idx[order]

    n_max_points = np.bincount(idx, minlength=n_groups).max()

    group_pos = np.arange(idx.size) - np.searchsorted(sorted_idx, sorted_idx, side='left')

    padded = jnp.zeros((n_groups, n_max_points, X.shape[1]), dtype=X.dtype)
    valid = np.zeros((n_groups, n_max_points), bool)
    slot = np.empty_like(idx)

    padded = padded.at[sorted_idx, group_pos].set(X[order])
    valid[sorted_idx, group_pos] = True
    slot[order] = group_pos
    
    return padded, valid, slot

@dataclass(frozen=True)
class ExactGP:
    """Exact Gaussian process term for a gllvm regression.

    Computes ``f = L @ z`` where L is the Cholesky factor of the kernel
    matrix and z are standard-normal draws. Observations are padded into
    a dense (n_groups, n_max_points, dim) tensor so that group-wise
    covariance matrices can be computed in a single vmap pass.

    Might need manual jax.config.update("jax_enable_x64", True) to prevent underflow.

    Attributes:
        name: Sample-site prefix. The GP realisation is stored under
            a deterministic site with this name.
        predictors: Covariate name(s) used as GP inputs.
        kernel: A callable ``kernel(params) -> tinygp.kernels.Kernel``
            that builds the kernel from a dict of sampled parameters.
        params: Mapping from parameter name to ``Param`` instance. Each
            entry is sampled and passed to `kernel`.
        group_by: Optional covariate name giving a grouping factor. A
            separate GP is drawn per level. None means one shared GP.
            Defaults to None.
        varies_over_variables: If True, an independent GP realisation is
            drawn for each response variable; if False, a single
            realisation is broadcast. Defaults to True.
        jitter: Small constant added to the kernel diagonal before
            Cholesky decomposition for numerical stability.
            Defaults to 1e-6.
    """
    
    name: str
    predictors: str | Sequence[str]
    kernel: Callable[[dict], kernels.Kernel]
    params: Mapping[str, Param]
    group_by: str | None = None
    varies_over_variables: bool = True
    jitter: float = 1e-6

    def __call__(self, ctx: _Context, n_vars: int):
        n_target = n_vars if self.varies_over_variables else 1
        
        # use train set covariates if we're doing prediction
        group_idx, n_groups = ctx._factorize(self.group_by, use_train=ctx.is_predictive)
        X = ctx._design(self.predictors, use_train=ctx.is_predictive)
        # pad ragged group points into common shape of n_max_points for efficient batching
        padded_train, valid_train, slot = _pad_by_group(X, group_idx, n_groups)
        n_train_pts = padded_train.shape[1]
        identity = jnp.eye(n_train_pts)

        # special handling for case in which no parameters vary over any variables -> no need to vmap over variables
        n_kernels = n_target if any(q.by_variable != "shared" for q in self.params.values()) else 1
        
        kernel_params = {k: q(f"{self.name}_{k}", n_groups, n_kernels) for k, q in self.params.items()} # (n_groups, n_kernels)

        # GP draws. Multiplied with kernel Cholesky
        z = numpyro.sample(f"{self.name}_z", dist.Normal(0, 1).expand((n_train_pts, n_groups, n_target)).to_event(3))

        def cholesky(kernel_params, padded_input, valid):
            covariance = self.kernel(kernel_params)(padded_input, padded_input)
            covariance = jnp.where(valid[:, None] & valid[None, :], covariance, identity) # set padded regions to identity
            return jnp.linalg.cholesky(covariance + self.jitter * identity) # jitter but no measurement noise because we're latent

        # vmap over groups (outer) and kernels/variables (inner)
        L_train = jax.vmap(jax.vmap(cholesky, (0, None, None)))(kernel_params, padded_train, valid_train) # (n_groups, n_kernels, n_points, n_points)

        #spec = "gts,sgv->tgv" if n_kernels == 1 else "gvts,sgv->tgv"
        #f = jnp.einsum(spec, L[:, 0] if n_kernels == 1 else L, z)
        f_train = jnp.einsum("gvts,sgv->tgv", L_train, z)
        
        numpyro.deterministic(self.name, f_train)       # (n_points, n_groups, n_target)
        
        if not ctx.is_predictive:
            return f_train[slot, group_idx]                     # (n_obs, n_target)
    
        # --- predictive branch: GP conditional f* | f_train ---
        test_idx, _ = ctx._factorize(self.group_by, use_train=False)
        X_test = ctx._design(self.predictors, use_train=False)
        padded_test, valid_test, slot_test = _pad_by_group(X_test, test_idx, n_groups)
        n_test_pts = padded_test.shape[1]
        I_test = jnp.eye(n_test_pts)

        # fresh standard-normal draws for the conditional (sampled from prior during Predictive)
        z_star = numpyro.sample(f"{self.name}_z_star",
            dist.Normal(0, 1).expand((n_test_pts, n_groups, n_target)).to_event(3))

        # broadcast shared kernel to n_target so vmap axes are uniform
        L_bc = jnp.broadcast_to(L_train, (n_groups, n_target, n_train_pts, n_train_pts))
        kp_bc = {k: jnp.broadcast_to(v, (n_groups, n_target)) for k, v in kernel_params.items()}

        def gp_conditional(kp, L_tr, f_tr, x_tr, v_tr, x_te, v_te, z_te):
            """Conditional for a single (group, variable) slice."""
            kern = self.kernel(kp)
            K_cross = kern(x_te, x_tr)                                     # (n_test, n_train)
            K_test  = kern(x_te, x_te)                                     # (n_test, n_test)
            K_cross = jnp.where(v_te[:, None] & v_tr[None, :], K_cross, 0.0)
            K_test  = jnp.where(v_te[:, None] & v_te[None, :], K_test, I_test)

            # conditional mean: K_cross @ K_train^{-1} @ f_train
            alpha = jax.scipy.linalg.cho_solve((L_tr, True), f_tr)
            mean = K_cross @ alpha

            # conditional covariance: K_test - K_cross @ K_train^{-1} @ K_cross^T
            w = jax.scipy.linalg.solve_triangular(L_tr, K_cross.T, lower=True)
            cond_cov = K_test - w.T @ w + self.jitter * I_test
            L_cond = jnp.linalg.cholesky(cond_cov)

            return mean + L_cond @ z_te                                    # reparameterized sample

        kp_axes = {k: 0 for k in kp_bc}
        f_tr_gv    = f_train.transpose(1, 2, 0)                            # (n_groups, n_target, n_train_pts)
        z_star_gv  = z_star.transpose(1, 2, 0)                             # (n_groups, n_target, n_test_pts)

        # inner vmap: over variables/kernels; outer vmap: over groups
        f_star = jax.vmap(jax.vmap(gp_conditional,
            in_axes=(kp_axes, 0, 0, None, None, None, None, 0)),
            in_axes=(kp_axes, 0, 0, 0, 0, 0, 0, 0))(
            kp_bc, L_bc, f_tr_gv,
            padded_train, valid_train,
            padded_test, valid_test,
            z_star_gv
        )                                                                   # (n_groups, n_target, n_test_pts)

        f_star = f_star.transpose(2, 0, 1)                                  # (n_test_pts, n_groups, n_target)
        numpyro.deterministic(f"{self.name}_star", f_star)

        return f_star[slot_test, test_idx]                                  # (n_obs_test, n_target)        
        

        
#------------------------------ HSGPs
# hsgp adapted from https://num.pyro.ai/en/stable/_modules/numpyro/contrib/hsgp/approximation.html
@dataclass(frozen=True)
class HSGP:
    """Hilbert-space approximate Gaussian process term for a gllvm regression.

    Replaces the exact kernel matrix with a truncated basis-function
    expansion following the HSGP method, trading a small approximation
    error for O(n * m) rather than O(n^3) cost. Supports Matérn and
    squared-exponential kernels.

    Attributes:
        name: Sample-site prefix. The GP realisation is stored under a
            deterministic site with this name.
        predictors: Covariate name(s) used as GP inputs.
        kernel: Kernel family, either ``"Matern"`` or ``"ExpSquared"``.
        ell: Boundary factor(s) for the Laplacian eigenfunctions. A
            scalar applies to all input dimensions; a sequence specifies
            per-dimension values.
        m: Number of basis functions. A scalar applies to all input
            dimensions; a sequence specifies per-dimension counts.
        nu: Smoothness parameter for the Matérn kernel. Ignored when
            kernel is ``"ExpSquared"``. Defaults to 1.5.
        group_by: Optional covariate name giving a grouping factor.
            Defaults to None.
        varies_over_variables: If True, an independent GP realisation is
            drawn for each response variable. Defaults to True.
        amplitude: ``Param`` for the kernel amplitude (marginal standard
            deviation). Defaults to a fixed value of 1.0.
        length: ``Param`` for the kernel lengthscale. Defaults to
            InverseGamma(5, 5) sampled independently per variable.
    """
    
    name: str
    predictors: str | Sequence[str]
    kernel: Literal["Matern", "ExpSquared"]
    ell: float | Sequence[float]
    m: int | Sequence[int]
    nu: float = 1.5


    group_by: str | None = None
    varies_over_variables: bool = True

    amplitude: Param = Param(1.0)
    length: Param = Param(dist.InverseGamma(5, 5), by_variable="free")
    
    def _sqrt_spectral_density(self, alpha, length, dim):
        # numpyro's hsgp helpers are scalar in alpha/length so manual vmap is required
        def _spd(alpha_k, length_k):
            if self.kernel == "Matern":
                return diag_spectral_density_matern(nu=self.nu, alpha=alpha_k, length=length_k, ell=self.ell, m=self.m, dim=dim)
            elif self.kernel == "ExpSquared":
                return diag_spectral_density_squared_exponential(alpha=alpha_k, length=length_k, ell=self.ell, m=self.m, dim=dim)

        spd = jax.vmap(_spd)(alpha.reshape(-1), length.reshape(-1))
        return jnp.sqrt(spd).reshape(*alpha.shape, -1)

    def __call__(self, ctx: _Context, n_vars: int):
        
        if self.kernel not in ["Matern", "ExpSquared"]:
            raise ValueError(f"kernel must be 'Matern' or 'ExpSquared'. Got {self.kernel}.")
        
        idx, n_groups = ctx._factorize(self.group_by)
        n_target = n_vars if self.varies_over_variables else 1
        X = ctx._design(self.predictors)

        alpha = self.amplitude(f"{self.name}_amplitude", n_groups, n_target)
        length = self.length(f"{self.name}_length", n_groups, n_target)
        
        phi = eigenfunctions(x=X, ell=self.ell, m=self.m) # (n_obs, n_basis)
        spd = self._sqrt_spectral_density(alpha, length, X.shape[-1])  # (n_groups, n_target, n_basis)
        beta = numpyro.sample(f"{self.name}_beta", dist.Normal(0, 1).expand((n_groups, n_target, phi.shape[-1])).to_event(3))

        weights = spd * beta
        f = jnp.einsum("nb,nvb->nv", phi, weights[idx])
        return numpyro.deterministic(self.name, f)
