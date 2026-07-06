# PPL-neutral flat-vector <-> nested-NamedTuple codec, plus the assembled
# `ComposedLogDensity` spec. Turing-free: no DynamicPPL/LogDensityProblems
# dependency here, only `params_table`, `build_priors` and `update`. A thin
# `LogDensityProblems` weakdep extension (deferred; see the package tracker)
# wraps `ComposedLogDensity` for AdvancedHMC/DynamicHMC/Pathfinder-style
# samplers on top of this.

# --- flat <-> nested codec ---------------------------------------------------
#
# Ordering is fixed by `params_table`'s pre-order row walk: row `i` of the
# table is flat index `i`. The nested `NamedTuple` is keyed exactly as
# `build_priors`/`update` expect (a `(:edge, :param)` row nests at
# `_split_edge(edge)` then `param`), so a flat vector round-trips to a named,
# `update`-able `NamedTuple` and back without re-deriving any structure.
#
# An `Uncertain` leaf's row inventories its template's free parameters like any
# other leaf (its spec only rides the row's `prior` entry), so a flat vector
# collapses it via `update` exactly as intended — the codec needs no special
# case for it. A `Varying` leaf has no fixed value until it is resolved against
# a context (its `params`/`params_table` row silently reports its `reference`,
# matching `Varying`'s own delegation elsewhere), so flattening one would
# quietly score the reference and ignore the covariate it is meant to vary
# with; the codec refuses that eagerly instead (`_reject_varying` below).

# Refuse eagerly when `d` still carries a `Varying` leaf: unlike `Uncertain`,
# whose row already tracks concrete template values, a `Varying` leaf's row
# reports its `reference` only, so the codec would otherwise silently ignore
# the covariate dependence rather than score it.
function _reject_varying(d, what)
    has_varying(d) && throw(ArgumentError(
        "cannot $what a tree with varying leaves; resolve them with " *
        "`instantiate(tree, context)` first"))
    return nothing
end

# The flat layout: a vector of `(path, param)` keys, one per table row, in row
# order. `path` is the `_split_edge` tuple of the row's edge; `param` the leaf
# key. This list is the bijection between flat index and named parameter.
function _flat_layout(table)
    edges = Tables.getcolumn(table, :edge)
    params_col = Tables.getcolumn(table, :param)
    return [(_split_edge(edges[i]), params_col[i]) for i in eachindex(edges)]
end

# Read the value at `(path..., param)` of a nested NamedTuple.
function _read_path(nt::NamedTuple, path::Tuple, param::Symbol)
    node = nt
    for k in path
        node = getproperty(node, k)
    end
    return getproperty(node, param)
end

@doc "

The flat parameter dimension of a composed distribution.

`flat_dimension(d)` is the number of scalar free parameters, i.e. the row
count of [`params_table`](@ref)`(d)`. It is the length of the flat vector
[`flatten`](@ref) produces and [`unflatten`](@ref) consumes.

# Arguments
- `d`: a composed distribution.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
# Public but not exported; reach it by the qualified name.
ComposedDistributions.flat_dimension(tree)
```

# See also
- [`flatten`](@ref), [`unflatten`](@ref): the flat <-> nested codec.
"
function flat_dimension(d::AbstractComposedDistribution)
    _reject_varying(d, "compute the flat dimension of")
    return length(Tables.getcolumn(params_table(d), :edge))
end

@doc "

Flatten a nested parameter `NamedTuple` to a flat vector in table-row order.

`flatten(d, nt)` reads `nt` (keyed like [`build_priors`](@ref)`(d)` /
[`params`](@ref)`(d)`, the shape [`update`](@ref) consumes) at each
[`params_table`](@ref) row and returns the values as a `Vector`, ordered by
the table's pre-order row walk. It is the inverse of [`unflatten`](@ref):
`flatten(d, unflatten(d, x)) == x`.

# Arguments
- `d`: the composed distribution whose table fixes the order.
- `nt`: a nested parameter `NamedTuple` keyed like `params(d)`.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
x = collect(params_table(tree).value)
# Public but not exported; reach the codec by the qualified name.
nt = ComposedDistributions.unflatten(tree, x)
ComposedDistributions.flatten(tree, nt)
```

# See also
- [`unflatten`](@ref): the inverse, flat vector -> nested NamedTuple.
- [`flat_dimension`](@ref): the flat length.
"
function flatten(d::AbstractComposedDistribution, nt::NamedTuple)
    _reject_varying(d, "flatten")
    layout = _flat_layout(params_table(d))
    return [_read_path(nt, path, param) for (path, param) in layout]
end

@doc "

Rebuild a nested parameter `NamedTuple` from a flat vector in table-row order.

`unflatten(d, x)` maps the flat vector `x` (laid out by
[`params_table`](@ref)'s row walk, e.g. a draw from a sampler) back to the
nested `NamedTuple` [`update`](@ref) consumes, keyed like
[`build_priors`](@ref)`(d)`. It is the inverse of [`flatten`](@ref), so
`update(d, unflatten(d, x))` reconstructs the distribution at `x`.

# Arguments
- `d`: the composed distribution whose table fixes the layout.
- `x`: a flat vector of length [`flat_dimension`](@ref)`(d)`.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
x = collect(params_table(tree).value)
# Public but not exported; reach it by the qualified name.
update(tree, ComposedDistributions.unflatten(tree, x))
```

# See also
- [`flatten`](@ref): the inverse, nested NamedTuple -> flat vector.
- [`update`](@ref): rebuild the distribution from the result.
"
function unflatten(d::AbstractComposedDistribution, x::AbstractVector)
    _reject_varying(d, "unflatten")
    layout = _flat_layout(params_table(d))
    length(x) == length(layout) || throw(DimensionMismatch(
        "flat vector has length $(length(x)) but $d has " *
        "$(length(layout)) free parameters"))
    tree = Dict{Symbol, Any}()
    for (i, (path, param)) in enumerate(layout)
        _nest_insert!(tree, path, param, x[i])
    end
    return _freeze_tree(tree)
end

# --- assembled log-density spec ---------------------------------------------

# Default likelihood: sum `logpdf(d, record)` over the observed records, the
# per-record contribution a downstream Turing model scores with
# `@addlogprob! logpdf(d, record)`.
_default_loglik(d, data) = sum(record -> logpdf(d, record), data)

@doc "

A PPL-neutral log-density over a composed distribution's flat parameters.

`ComposedLogDensity` carries everything needed to evaluate the (unnormalised)
log-posterior of a composed distribution over a flat parameter vector, with no
DynamicPPL/Turing dependency: the template `dist`, the per-parameter `priors`
(a nested `NamedTuple` from [`build_priors`](@ref)), the observed `data`, and a
`loglik` reducer scoring `data` against the reconstructed distribution. Build
it with [`as_logdensity`](@ref); evaluate it on a flat vector with
[`logdensity`](@ref).

It is the spec a `LogDensityProblems` weakdep extension wraps as a standard
problem (sampleable by AdvancedHMC / DynamicHMC / Pathfinder); the flat layout
is [`params_table`](@ref)`(dist)`'s row order throughout.

# Fields
- `dist`: the template composed distribution (the structure to reconstruct).
- `priors`: nested prior `NamedTuple` keyed like [`build_priors`](@ref)`(dist)`.
- `data`: the observed records scored by `loglik`.
- `loglik`: a reducer `(d, data) -> Real` (default sums `logpdf(d, record)`).
- `flat_priors`: `priors` flattened once at construction, in
  [`params_table`](@ref) row order, so [`logdensity`](@ref) does not re-derive
  it on every evaluation.

# See also
- [`as_logdensity`](@ref): the assembler.
- [`logdensity`](@ref): evaluate on a flat vector.
- [`flatten`](@ref), [`unflatten`](@ref): the flat <-> nested codec.
"
struct ComposedLogDensity{D <: AbstractComposedDistribution, P, T, L, FP}
    dist::D
    priors::P
    data::T
    loglik::L
    flat_priors::FP
end

function ComposedLogDensity(
        dist::AbstractComposedDistribution, priors, data, loglik)
    return ComposedLogDensity(dist, priors, data, loglik,
        flatten(dist, priors))
end

@doc "

Assemble a [`ComposedLogDensity`](@ref) from a composed distribution and data.

`as_logdensity(dist, priors, data; loglik)` packages the template `dist`, the
per-parameter `priors` (a nested `NamedTuple`, usually from
[`build_priors`](@ref)`(dist)`) and the observed `data` into the PPL-neutral
log-density spec. The result evaluates the (unnormalised) log-posterior over
the flat parameter vector via [`logdensity`](@ref).

`priors` defaults to [`build_priors`](@ref)`(params_table(dist))` (support-
derived defaults), so the two-argument form needs only `dist` and `data`.
`loglik` defaults to summing `logpdf(dist, record)` over `data`; pass a custom
reducer for record-aware scoring.

# Arguments
- `dist`: the template composed distribution.
- `priors`: nested prior `NamedTuple` keyed like [`build_priors`](@ref)`(dist)`
  (default: [`build_priors`](@ref)`(params_table(dist))`).
- `data`: the observed records.

# Keyword Arguments
- `loglik`: a reducer `(d, data) -> Real` scoring `data` against the
  reconstructed distribution (default: sum of `logpdf(d, record)`).

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
data = [[0.5, 2.0], [1.0, 3.0]]
priors = build_priors(params_table(tree))
# Public but not exported; reach the spec by the qualified name.
prob = ComposedDistributions.as_logdensity(tree, priors, data)
# The table's `value` column is the flat layout; score at those values.
x = collect(params_table(tree).value)
ComposedDistributions.logdensity(prob, x)
```

# See also
- [`logdensity`](@ref): evaluate the assembled spec on a flat vector.
- [`flatten`](@ref), [`unflatten`](@ref): the flat <-> nested codec.
- [`build_priors`](@ref): assemble `priors` from the tree.
"
function as_logdensity(dist::AbstractComposedDistribution, priors, data;
        loglik = _default_loglik)
    return ComposedLogDensity(dist, priors, data, loglik)
end

function as_logdensity(dist::AbstractComposedDistribution, data;
        loglik = _default_loglik)
    return as_logdensity(dist, build_priors(params_table(dist)), data;
        loglik = loglik)
end

@doc "

Evaluate a [`ComposedLogDensity`](@ref) on a flat parameter vector.

`logdensity(prob, x)` is the (unnormalised) log-posterior at the flat vector
`x` (in [`params_table`](@ref)`(prob.dist)` row order): the sum of the priors'
log-densities at `x` plus the data log-likelihood of the distribution
reconstructed there. `x` is [`flat_dimension`](@ref)`(prob.dist)` long.

# Arguments
- `prob`: the assembled [`ComposedLogDensity`](@ref).
- `x`: a flat parameter vector of length [`flat_dimension`](@ref).

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
priors = build_priors(params_table(tree))
prob = ComposedDistributions.as_logdensity(
    tree, priors, [[0.5, 2.0], [1.0, 3.0]])
x = collect(params_table(tree).value)
ComposedDistributions.logdensity(prob, x)
```

# See also
- [`as_logdensity`](@ref): assemble `prob`.
- [`flatten`](@ref), [`unflatten`](@ref): the flat <-> nested codec.
"
function logdensity(prob::ComposedLogDensity, x::AbstractVector)
    fp = prob.flat_priors
    length(x) == length(fp) || throw(DimensionMismatch(
        "flat parameter vector has length $(length(x)) but " *
        "$(prob.dist) has $(length(fp)) free parameters"))
    lp = isempty(x) ? 0.0 : sum(i -> logpdf(fp[i], x[i]), eachindex(x))
    d = update(prob.dist, unflatten(prob.dist, x))
    return lp + prob.loglik(d, prob.data)
end
