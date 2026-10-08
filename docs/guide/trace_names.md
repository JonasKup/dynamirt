# Scoped Parameter Names

Depending on how the model was specified, the parameter names that end up in the posterior trace can vary quite a bit which might be confusing when looking for a specific quantity of interest. This is a small guide to help identify the correct parameter names.

On the top level there are two reserved names recorded by every model that was constructed using {py:func}`dynamirt() <dynamirt.dynamirt>`. The first is `"theta"` which is the parameter name for the full latent construct (e.g., the sum of all latent terms supplied to the `latent_terms` argument of {py:func}`dynamirt() <dynamirt.dynamirt>`). The second is `"loadings"` which is the full loading structure. Below that, parameters are scoped by whichever aspect of the model they belong to. Deterministic sites are omitted when fitting with `return_deterministic=False`. Shapes below exclude posterior sampling dimensions.

## Loadings

All parameters related to the loading structure are scoped under `"loadings."`. E.g., for a loading structure specified through {py:func}`Confirmatory() <dynamirt.Confirmatory>`, the full matrix will be in `"loadings"`. Additionally, the posterior trace will contain separate parameter recordings for the unconstrained (`"loadings.confirmatory_free"`) and positive constrained (`"loadings.confirmatory_positive"`) item loadings.

Site names below omit the `loadings.` prefix.

| Factory | Sites (shape) |
|---|---|
| {py:func}`Fixed <dynamirt.Fixed>` | No additional sites |
| {py:func}`Full <dynamirt.Full>` | `full` `(n_items, n_latent)` |
| {py:func}`Confirmatory <dynamirt.Confirmatory>` | `confirmatory_free` `(n_free,)`, `confirmatory_positive` `(n_positive,)` |
| {py:func}`Sparsity <dynamirt.Sparsity>` | `beta`, `lambda`, `lambda_aux` `(n_items, n_latent)`; `tau`, `tau_aux`, `c_sq` scalar |

`n_free` and `n_positive` count the corresponding estimated entries in Q; empty sets create no site. {py:func}`Sparsity <dynamirt.Sparsity>` sites are raw coefficients and shrinkage variables; the assembled matrix is `loadings`.

## Measurements

All parameters related to the measurement model (e.g. 2PL, GRM, etc) are scoped under `"measurement."`. This is why the dichotomous item intercepts can be found in `"measurement.intercept"`. 3PL models have an additional `"measurement.lower_asymptote"` parameter and 4PL models also include the `"measurement.upper_asymptote"`, etc.

Site names below omit the `measurement.` prefix; `K` is the largest category count.

| Model | Sites (shape) |
|---|---|
| `1PL`, `2PL`, `3PL`, `4PL` | `intercept` `(n_items,)` |
| `3PL`, `4PL` | `lower_asymptote` `(n_items,)` |
| `4PL` | `upper_asymptote_fraction`, `upper_asymptote` `(n_items,)` |
| `GRM` | `baseline_probs` `(n_items, K)`, `cutpoints` `(n_items, K - 1)` |
| `PCM`, `GPCM` | `steps` `(n_items, K - 1)` |

With per-item category counts, GRM instead records `baseline_probs_<k>` `(n_items_with_k_categories, k)` for each distinct count. PCM/GPCM additionally record `steps.raw` `(sum(n_cat - 1),)`. The assembled cutpoints/steps are zero-padded. Cutpoints and upper asymptotes are deterministic; steps are deterministic only with per-item counts.

## Latent & Item Terms

Parameters relating to custom and built-in latent terms get the `"latent.<term_name>."` prefix. The  `(n_obs, n_latent)` contribution that each term makes to `"theta"` is available as `"latent.<term_name>.contribution"`. The length scale parameter for an HSGP term called "trajectory" will be called `"latent.trajectory.lengthscale"`.

Similarly, item terms get the `"item.<term_name>."` prefix, with contributions of shape `(n_obs, n_items)`.

Site names below omit the `latent.<term_name>.` or `item.<term_name>.` prefix. `T` is the number of targets (latents or items), `G` the number of groups, and `P` the number of predictors. Shared realizations use `T = 1`.

| Term | Sites (shape) |
|---|---|
| {py:class}`Linear <dynamirt.Linear>` | `coef.raw`, `coef` `(G, T, P)`; optional `correlation_cholesky` |
| {py:class}`GRW <dynamirt.GRW>` | `scale`; `innovations.raw`, `state` `(n_time, G, T)` |
| {py:class}`AR1 <dynamirt.AR1>` | `scale`, `phi`; `innovations.raw`, `state` `(n_time, G, T)` |
| {py:class}`GP <dynamirt.GP>` | User-named kernel parameters; `state.raw`, `state` `(n_max_points_per_group, G, T)` |
| {py:class}`HSGP <dynamirt.HSGP>` | `amplitude`, `lengthscale`; `basis_weights.raw` `(G, T, n_basis)` |

Every term also records `contribution` `(n_obs, T)`. `coef`, `state`, and `contribution` are deterministic. For {py:class}`Linear <dynamirt.Linear>`, reference coding reduces the raw group axis to `G - 1`; sharing reduces raw axes to 1. `correlation_cholesky` is square, sized by the correlated axis: `P`, `T`, or `P * T`.

Distribution-valued {py:class}`Param <dynamirt.Param>` sites use `(G, T)` with shared axes reduced to 1 (and a trailing `P` for linear coefficients); fixed scalar parameters have shape `()`. Pooling adds `<parameter>.loc` and `<parameter>.scale`, shared along the pooled axis. For {py:class}`Linear <dynamirt.Linear>` these are `coef.loc` and `coef.scale`. Pooled sites contain raw draws; the transformed values are used internally.

[Building Models](models.md) · [Covariates & Built-In Terms](terms.md) · [Plots](plots.md)
