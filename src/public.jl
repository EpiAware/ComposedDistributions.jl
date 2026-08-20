# Public API declarations for Julia 1.11+ (public but not exported).

# `update`: the tree-edit / value-replace / uncertain-collapse verb (see the
# umbrella docstring on `function update end` in `composers/introspection.jl`).
# `public`, not `export`ed (#221): several ecosystem packages (and plenty
# outside it) have their own `update`-shaped verb, and exporting a name this
# generic risks the same ambiguous-binding clash #233 hit with `as_turing`
# when two packages both export a same-named generic function.
public update

# The composer node-extension contract (see
# `docs/src/developer/extending.md`, "Writing a new composer node"): a new
# node implements `node_children` / `node_rebuild` / `component_names`, which
# alone is enough to compose, table, `update` and (given the codec's
# type-parameter layout convention) flatten/fit, and gets
# `has_varying`/`has_uncertain` for free over its own children; a node with
# novel combination semantics (not a plain concatenation of its children, e.g.
# a disjunction or a mixture) additionally overrides `child_nleaves` /
# `child_logpdf` / `child_rand!`, which otherwise default generically off
# `node_children`. A node subtyping `AbstractComposedDistribution` composes as
# a named child of the built-ins with no registration once these are defined.
# A new leaf wrapper implements `free_leaf` / `rewrap_leaf`.
public child_nleaves, child_logpdf, child_rand!

# The composer child-admission gate. A leaf needs no method here: any object
# implementing the univariate Distributions interface (`params` / `logpdf` /
# `rand` / `minimum` / `maximum`) composes without subtyping
# `UnivariateDistribution`, and only the shapes the `compose` front ends already
# spend on something else (a branch probability, a chain to lower, a row label)
# are rejected. `public` so a leaf that genuinely subtypes one of those -- a
# `Number`-backed leaf, say -- can opt back in with one method rather than
# reaching for an internal name.
public is_composable

# The type-domain companion to `is_composable`: the `ValueSupport` a leaf's
# draws take, which fixes the parameter an `Uncertain` leaf reports on its own
# supertype. Defaulted off `Base.eltype`, so a duck-typed leaf needs no method
# unless it wants a support its element type does not imply.
public value_support
public free_leaf, rewrap_leaf, component_names
public node_children, node_rebuild

# `inner_dist` is the single-layer peel hook the read-through leaf-wrapper hooks
# recurse through: a wrapper defines one method returning its inner distribution
# and `free_leaf`/`uncertain_specs`/`extra_leaf_params`/`shared_tag` forward
# through it for free.
public inner_dist

# The published leaf protocol a downstream leaf-wrapper package (censoring in
# CensoredDistributions, the modifiers in ModifiedDistributions) extends
# alongside `free_leaf`/`rewrap_leaf`. `uncertain_specs` routes a leaf's
# attached specs (priors, `pool(...)`, `no_prior()`) through to
# `composed_to_table`; `leaf_detail_lines` routes a leaf's `inspect` rendering;
# `shared_tag` sees a shared tie through a wrapper; `leaf_param_names` names a
# leaf's estimable parameters; `leaf_mean`/`leaf_var` give a leaf's per-moment
# values; and `extra_leaf_params`/`set_extra_leaf_params` carry any
# modifier-owned free parameters (the thinning factor is the first instance).
# A leaf-wrapper package that extends only `free_leaf`/`rewrap_leaf` but not
# these silently drops an attached spec on a wrapped leaf (`composed_to_table`
# then treats it as fixed). See `docs/src/developer/extending.md`.
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

# The pooling gate surface (#325): DistributionsInference.jl's fit-protocol
# extension calls `validate_pool_groups` and `validate_tree_names` directly
# to gate a composed tree once at `distribution_to_logdensity` construction,
# before `composed_to_table` walks it per gradient evaluation.
# `validate_pool_groups` checks every leaf of a pool group agrees on
# population and parameterisation; `validate_tree_names` checks that no pool
# group, shared tag, or top-level edge name collides with one from another of
# those three roles. No behaviour change — declaring what was already
# reachable, unrestricted, by qualified name. `_validate_pool_groups` and
# `_validate_tree_names` are transitional aliases for the two functions'
# original (leading-underscore) public names, kept `public` themselves — like
# `_centred_pool_rows`/`_pool_centred_logprior` above — until
# DistributionsInference.jl's fit-protocol extension moves onto the renamed
# functions, then removed.
public validate_tree
public validate_pool_groups, validate_tree_names,
       _validate_pool_groups, _validate_tree_names

# The parameter-coordinate contract. A leaf's free parameters are named by
# `param_names` and rebuilt by `leaf_ctor`; together they fix the coordinates
# `composed_to_table`, `uncertain` and the flat codec work in. A leaf
# whose free parameters are its native constructor arguments needs neither. A
# leaf that reports different parameters — a moment-parameterised wrapper naming
# a mean and a standard deviation rather than a shape and a scale — overrides
# both, so a prior lands on the moment rather than on the native parameter that
# only implies it.
public param_names, leaf_ctor

# The reconstruction/tie-identity split (#332 follow-up): `_update_leaf` (the
# value-tuple rebuild `update`, `reconstruct` and `uncertain`'s pinning path
# all route through) calls `rebuild_leaf`, an ordinary method with no
# constructor-identity contract; `tie` groups leaves by `leaf_signature`
# instead, which is where the egal-stability contract actually belongs. A
# non-native leaf overrides `rebuild_leaf` (and usually `leaf_signature`)
# rather than `leaf_ctor`, so it never needs a callable-struct constructor
# just to satisfy `tie`.
public rebuild_leaf, leaf_signature

# The table round trip's reconstruction half (`compose(table)`, the inverse
# of `composed_to_table`): one generic, three arities — leaf from values,
# wrapper layer around a rebuilt inner, node around rebuilt children — each
# the type-dispatched counterpart of an instance hook (`rebuild_leaf`,
# `inner_dist`'s peel, `node_rebuild`), because when a table is read there is
# no instance yet, only the type its `node` column carries. Dispatching on
# the type is what lets the round trip work with no name-to-type registry.
# The leaf and node arities are defaulted so the extension cost does not
# move; the wrapper arity is not, a wrapper's fixed structure being its own.
public from_table

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
# log-density/extensions (`distribution_to_logdensity` and
# `distribution_to_turing`) generically over this core via the fit protocol
# (`parameter_rows`/`reconstruct`).
public flat_dimension, flatten, unflatten, reconstruct

# The node-emission half of the single-table contract (#227 slice 1): the one
# method a composer node or leaf (wrapper) type defines to control its own
# `composed_to_table` rows, reporting the fixed, non-parameter structure it
# carries. A row's `node` label is the type itself and a wrapped leaf's
# layers are peeled through `inner_dist`, so neither is asked of a downstream
# type. `node_rebuild` (the round-trip half, `public` above alongside
# `node_children`) is now part of the node-extension contract; `compose(table)`
# reconstructing a tree straight from a `composed_to_table`-shaped source is
# still deferred to a later slice.
public node_attributes
