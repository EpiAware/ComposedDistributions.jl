# Public API declarations for Julia 1.11+ (public but not exported).

# `update`: the tree-edit / value-replace / uncertain-collapse verb (see the
# umbrella docstring on `function update end` in `composers/introspection.jl`).
# `public`, not `export`ed (#221): several ecosystem packages (and plenty
# outside it) have their own `update`-shaped verb, and exporting a name this
# generic risks the same ambiguous-binding clash #233 hit with `as_turing`
# when two packages both export a same-named generic function.
public update

# The composer node/leaf extension contract: a new node implements
# `child_nleaves` / `child_logpdf` / `child_rand!`, and a new leaf wrapper
# `free_leaf` / `rewrap_leaf`. `component_names` reads a node's child names.
public child_nleaves, child_logpdf, child_rand!
public free_leaf, rewrap_leaf, component_names

# The published leaf protocol a downstream leaf-wrapper package (censoring in
# CensoredDistributions, the modifiers in ModifiedDistributions) extends
# alongside `free_leaf`/`rewrap_leaf`. `uncertain_specs` routes a leaf's
# attached prior specs through to `composed_to_table`/`build_priors`;
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

# The centred-pooling extra-prior surface (#212): DistributionsInference.jl's
# `extra_logprior` for a composed tree scores a centred-pooled parameter's
# population-dependent term by calling these three directly. `CentredPoolPrior`
# is the marker `parameter_rows` pattern-matches on to translate a
# centred-pooled row's prior to `nothing`; `centred_pool_rows` collects the
# rows carrying that marker; `pool_centred_logprior` scores them against the
# unflattened parameter tree. No behaviour change — declaring what was already
# reachable, unrestricted, by qualified name. `_centred_pool_rows` and
# `_pool_centred_logprior` are transitional aliases for the two functions'
# original (leading-underscore) public names, kept `public` themselves — like
# `EpiAwarePackageTools`' own `scaffold_update`/`update` alias — until
# DistributionsInference.jl's fit-protocol extension moves onto the renamed
# functions, then removed.
public CentredPoolPrior, centred_pool_rows, pool_centred_logprior,
       _centred_pool_rows, _pool_centred_logprior

# The parameter-coordinate contract. A leaf's free parameters are named by
# `param_names` and rebuilt by `leaf_ctor`; together they fix the coordinates
# `composed_to_table`, `uncertain`, `build_priors` and the flat codec work in. A leaf
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
# (`flat_dimension`/`flatten`/`unflatten`), and the fused flat-vector ->
# rebuilt-distribution primary (`reconstruct`, #178 PR 2). No
# LogDensityProblems/DynamicPPL dependency here or anywhere in this
# package (#220, #233): DistributionsInference.jl hosts the PPL-facing
# log-density/extensions (`as_logdensity`/`as_turing`) generically over this
# core via the fit protocol (`parameter_rows`/`reconstruct`).
public flat_dimension, flatten, unflatten, reconstruct

# The load-order-independent leaf-wrapper registry (#189, #178 PR 4): a
# leaf-wrapper package extension (censoring, modifiers) registers its type-level
# codec hooks here (in its own `__init__`) instead of adding a direct dispatch
# method to `_leaf_free_type`/`_extra_names_of`, which the generated codec's
# `@generated` generator cannot see reliably once loaded after the fact.
public register_leaf_wrapper!

# The node-emission half of the single-table contract (#227 slice 1): a
# composer node or leaf (wrapper) type overrides these to control its own
# `composed_to_table` rows. `node_kind` labels a node/layer; `node_children`
# reads a composer node's children uniformly; `node_attributes` reports a
# node/layer's own fixed-structure attributes; `leaf_layers` lists a leaf's
# wrapper layers outermost to innermost. The rebuild half of the contract
# (`node_rebuild`, `register_node_kind!`, `set_node_params`, for
# `compose(table)` and friends) is deferred to a later slice.
public node_kind, node_children, node_attributes, leaf_layers
