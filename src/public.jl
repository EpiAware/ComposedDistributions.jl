# Public API declarations for Julia 1.11+ (public but not exported).

# The composer node/leaf extension contract: a new node implements
# `child_nleaves` / `child_logpdf` / `child_rand!`, and a new leaf wrapper
# `free_leaf` / `rewrap_leaf`. `component_names` reads a node's child names.
public child_nleaves, child_logpdf, child_rand!
public free_leaf, rewrap_leaf, component_names

# The published leaf protocol a downstream leaf-wrapper package (censoring in
# CensoredDistributions, the modifiers in ModifiedDistributions) extends
# alongside `free_leaf`/`rewrap_leaf`. `uncertain_specs` routes a leaf's
# attached prior specs through to `params_table`/`build_priors`;
# `leaf_detail_lines` routes a leaf's `inspect` rendering; `shared_tag` sees a
# shared tie through a wrapper; `leaf_param_names` names a leaf's estimable
# parameters; `leaf_mean`/`leaf_var` give a leaf's per-moment values; and
# `extra_leaf_params`/`set_extra_leaf_params` carry any modifier-owned free
# parameters (the thinning factor is the first instance). A leaf-wrapper package
# that extends only `free_leaf`/`rewrap_leaf` but not these silently drops an
# attached prior on a wrapped leaf (`build_priors` then treats it as fixed). See
# `docs/src/developer/leaf-protocol.md`.
public uncertain_specs, leaf_detail_lines, shared_tag, leaf_param_names
public leaf_mean, leaf_var, extra_leaf_params, set_extra_leaf_params

# `Pool`'s group name and non-centred flag live in type parameters (like
# `Shared`'s tag); `pool_group`/`pool_noncentred` are the accessors, mirroring
# `shared_tag` above.
public pool_group, pool_noncentred

# The parameter-coordinate contract. A leaf's free parameters are named by
# `param_names` and rebuilt by `leaf_ctor`; together they fix the coordinates
# `params_table`, `uncertain`, `build_priors` and the flat codec work in. A leaf
# whose free parameters are its native constructor arguments needs neither. A
# leaf that reports different parameters — a moment-parameterised wrapper naming
# a mean and a standard deviation rather than a shape and a scale — overrides
# both, so a prior lands on the moment rather than on the native parameter that
# only implies it.
public param_names, leaf_ctor

# The composer abstract-type hierarchy. `AbstractComposedDistribution` is the
# root the composer nodes subtype; `AbstractMultiChild` groups the positional
# multi-child composers (`Sequential` / `Parallel`); `AbstractOneOf` is the
# univariate one_of arm (`Resolve` / `Compete`). Downstream extension packages
# dispatch on these.
public AbstractComposedDistribution, AbstractMultiChild, AbstractOneOf

# The reusable interface-conformance harness (`TestUtils.test_interface` and
# the per-family `test_*` checks).
public TestUtils

# The Turing-free core codec: the flat-vector <-> nested-NamedTuple bijection
# (`flat_dimension`/`flatten`/`unflatten`), the fused flat-vector ->
# rebuilt-distribution primary (`reconstruct`, #178 PR 2), and the assembled
# PPL-neutral log-density (`ComposedLogDensity`/`as_logdensity`/`logdensity`).
# No LogDensityProblems/DynamicPPL dependency here or anywhere in this
# package (#220, #233): DistributionsInference.jl hosts the PPL-facing
# extensions (its own `as_logdensity`/`as_turing`) generically over this
# core via the fit protocol (`parameter_rows`/`reconstruct`).
public flat_dimension, flatten, unflatten, reconstruct
public ComposedLogDensity, as_logdensity, logdensity

# The load-order-independent leaf-wrapper registry (#189, #178 PR 4): a
# leaf-wrapper package extension (censoring, modifiers) registers its type-level
# codec hooks here (in its own `__init__`) instead of adding a direct dispatch
# method to `_leaf_free_type`/`_extra_names_of`, which the generated codec's
# `@generated` generator cannot see reliably once loaded after the fact.
public register_leaf_wrapper!

# The prior-driven unconstrained -> constrained transform (public but not
# exported, like the rest of the codec): `to_constrained(prob, z)` has no
# method until `Bijectors` is loaded, when `ComposedDistributionsBijectorsExt`
# supplies it.
public to_constrained
