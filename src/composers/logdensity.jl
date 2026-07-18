# PPL-neutral assembled `ComposedLogDensity` spec, over the generated flat
# <-> nested codec (`unflatten`/`flatten`/`flat_dimension`/`reconstruct`, now
# in `codec_gen.jl`). Turing-free: no DynamicPPL/LogDensityProblems dependency
# here, only `params_table`, the uncertain specs and `update`. A thin
# `LogDensityProblems` weakdep extension (deferred; see the package tracker)
# wraps `ComposedLogDensity` for AdvancedHMC/DynamicHMC/Pathfinder-style
# samplers on top of this.

# The estimated rows of a params table: those whose `prior` column carries an
# uncertain spec. Under uncertain-first these are the free (estimated)
# parameters; a fixed leaf's rows hold `nothing` and are excluded, so a tree
# with no uncertain leaves has no estimated rows. Still used by `as_turing`'s
# `_flat_layout` (the estimated `(path, param)` keys, for VarName construction
# at model-build time, not per-gradient) and by `_spec_priors` below.
_estimated_rows(table) = findall(!isnothing, Tables.getcolumn(table, :prior))

# The flat layout: a vector of `(path, param)` keys, one per ESTIMATED row, in
# table order. `path` is the `_split_edge` tuple of the row's edge; `param` the
# leaf key. This list is the bijection between flat index and estimated
# parameter, used only at `as_turing` model-build time (not per gradient).
function _flat_layout(table)
    edges = Tables.getcolumn(table, :edge)
    params_col = Tables.getcolumn(table, :param)
    return [(_split_edge(edges[i]), params_col[i]) for i in _estimated_rows(table)]
end

@noinline function _throw_logdensity_dimmismatch(x, fp, dist)
    throw(DimensionMismatch(
        "flat parameter vector has length $(length(x)) but " *
        "$dist has $(length(fp)) estimated parameters"))
end

# --- assembled log-density spec ---------------------------------------------

# Default likelihood: sum `logpdf(d, record)` over the observed records, the
# per-record contribution a downstream Turing model scores with
# `@addlogprob! logpdf(d, record)`.
_default_loglik(d, data) = sum(record -> logpdf(d, record), data)

@doc "

A PPL-neutral log-density over a composed distribution's flat parameters.

`ComposedLogDensity` carries everything needed to evaluate the (unnormalised)
log-posterior of a composed distribution over its ESTIMATED flat parameter
vector, with no DynamicPPL/Turing dependency: the template `dist`, the
per-parameter `priors` (a nested `NamedTuple`; the uncertain specs read off the
object by default), the observed `data`, and a `loglik` reducer scoring `data`
against the reconstructed distribution. Build it with [`as_logdensity`](@ref);
evaluate it on a flat vector with [`logdensity`](@ref).

It is the spec a `LogDensityProblems` weakdep extension wraps as a standard
problem (sampleable by AdvancedHMC / DynamicHMC / Pathfinder); the flat layout
is [`params_table`](@ref)`(dist)`'s row order restricted to the estimated
(spec'd) parameters throughout.

# Fields
- `dist`: the template composed distribution (the structure to reconstruct).
- `priors`: nested prior `NamedTuple` keyed like [`params`](@ref)`(dist)`, read
  at the estimated (spec'd) parameters.
- `data`: the observed records scored by `loglik`.
- `loglik`: a reducer `(d, data) -> Real` (default sums `logpdf(d, record)`).
- `flat_priors`: `priors` flattened once at construction, in estimated-row
  order, so [`logdensity`](@ref) does not re-derive it on every evaluation.
- `centred_pools`: the centred pooled parameters' `(path, param, pool)` rows,
  collected once at construction, so their population-dependent prior term is
  added without a per-evaluation table walk (empty when nothing pools centred).

# See also
- [`as_logdensity`](@ref): the assembler.
- [`logdensity`](@ref): evaluate on a flat vector.
- [`flatten`](@ref), [`unflatten`](@ref): the flat <-> nested codec.
"
struct ComposedLogDensity{D <: AbstractComposedDistribution, P, T, L, FP, CP}
    dist::D
    priors::P
    data::T
    loglik::L
    flat_priors::FP
    centred_pools::CP
end

function ComposedLogDensity(
        dist::AbstractComposedDistribution, priors, data, loglik)
    return ComposedLogDensity(dist, priors, data, loglik,
        flatten(dist, priors), _centred_pool_rows(dist))
end

# The nested prior `NamedTuple` of a tree's uncertain specs, keyed like the
# tree (spec'd parameters only). Under uncertain-first this is the estimated
# subset's prior source, read straight off the object's `params_table` `prior`
# column, so a fixed leaf contributes neither a prior nor an estimated
# dimension. A shared spec'd leaf rides its tag edge, matching the codec layout.
function _spec_priors(dist::AbstractComposedDistribution)
    table = params_table(dist)
    edges = Tables.getcolumn(table, :edge)
    params_col = Tables.getcolumn(table, :param)
    priors = Tables.getcolumn(table, :prior)
    tree = Dict{Symbol, Any}()
    for i in eachindex(edges)
        priors[i] === nothing && continue
        _nest_insert!(tree, _split_edge(edges[i]), params_col[i], priors[i])
    end
    return _freeze_tree(tree)
end

@doc "

Assemble a [`ComposedLogDensity`](@ref) from a composed distribution and data.

`as_logdensity(dist, data; loglik)` packages the template `dist` and the
observed `data` into the PPL-neutral log-density spec, reading the priors off
the object's [`uncertain`](@ref) specs (the estimation boundary). The result
evaluates the (unnormalised) log-posterior over the ESTIMATED flat parameter
vector — the spec'd parameters — via [`logdensity`](@ref). A tree with no
uncertain leaves estimates nothing: the flat vector is empty and `logdensity`
is the data likelihood at the fixed tree. Promote a tree to estimate its free
parameters with default priors through `update(tree, param_priors(tree))`.

`as_logdensity(dist, priors, data)` overrides the on-object specs with an
explicit nested prior `NamedTuple`, read at the same estimated (spec'd) rows.
`loglik` defaults to summing `logpdf(dist, record)` over `data`; pass a custom
reducer for record-aware scoring.

# Arguments
- `dist`: the template composed distribution, carrying its uncertain specs.
- `priors`: (optional) a nested prior `NamedTuple` keyed like
  [`params`](@ref)`(dist)`, overriding the on-object specs at the estimated
  rows (default: the object's specs).
- `data`: the observed records.

# Keyword Arguments
- `loglik`: a reducer `(d, data) -> Real` scoring `data` against the
  reconstructed distribution (default: sum of `logpdf(d, record)`).

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((
    onset_admit = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
    admit_death = LogNormal(0.5, 0.4)))
data = [[0.5, 2.0], [1.0, 3.0]]
# Public but not exported; reach the spec by the qualified name. Priors read
# off the object's specs; the one estimated parameter is onset_admit.shape.
prob = ComposedDistributions.as_logdensity(tree, data)
ComposedDistributions.logdensity(prob, [2.0])
```

# See also
- [`logdensity`](@ref): evaluate the assembled spec on a flat vector.
- [`flatten`](@ref), [`unflatten`](@ref): the flat <-> nested codec.
- [`param_priors`](@ref): default priors for `update(tree, param_priors(tree))`.
"
function as_logdensity(dist::AbstractComposedDistribution, priors, data;
        loglik = _default_loglik)
    # Gate the pooling groups' consistency once here (not per gradient
    # evaluation): every member of a group must share one population.
    _validate_pool_groups(dist)
    # Gate the pool/shared/root-edge name namespaces once here too (#177):
    # a pool group, a shared tag, and a root edge name must not collide.
    _validate_tree_names(dist)
    return ComposedLogDensity(dist, priors, data, loglik)
end

function as_logdensity(dist::AbstractComposedDistribution, data;
        loglik = _default_loglik)
    return as_logdensity(dist, _spec_priors(dist), data; loglik = loglik)
end

@doc "

Evaluate a [`ComposedLogDensity`](@ref) on its estimated flat parameter vector.

`logdensity(prob, x)` is the (unnormalised) log-posterior at the estimated flat
vector `x` (the spec'd parameters, in [`params_table`](@ref)`(prob.dist)` row
order): the sum of the specs' log-densities at `x` plus the data log-likelihood
of the distribution reconstructed there (each uncertain leaf collapsed at its
draw, fixed parameters held at the template). `x` is
[`flat_dimension`](@ref)`(prob.dist)` long — empty for a tree with no uncertain
leaves, where `logdensity` is just the data likelihood.

# Arguments
- `prob`: the assembled [`ComposedLogDensity`](@ref).
- `x`: an estimated flat parameter vector of length [`flat_dimension`](@ref).

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((
    onset_admit = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
    admit_death = LogNormal(0.5, 0.4)))
prob = ComposedDistributions.as_logdensity(
    tree, [[0.5, 2.0], [1.0, 3.0]])
ComposedDistributions.logdensity(prob, [2.0])
```

# See also
- [`as_logdensity`](@ref): assemble `prob`.
- [`flatten`](@ref), [`unflatten`](@ref): the flat <-> nested codec.
"
function logdensity(prob::ComposedLogDensity, x::AbstractVector)
    fp = prob.flat_priors
    length(x) == length(fp) || _throw_logdensity_dimmismatch(x, fp, prob.dist)
    # The fixed per-row priors (hyperparameters, non-centred latents, ordinary
    # uncertain parameters). A centred pooled parameter's row carries a
    # `CentredPoolPrior` marker instead — its prior is the population at the
    # current hyperparameters, so it is scored below, not here.
    lp = _fixed_row_logprior(fp, x)
    nt = unflatten(prob.dist, x)
    lp += _pool_centred_logprior(prob.centred_pools, nt)
    d = update(prob.dist, nt)
    return lp + prob.loglik(d, prob.data)
end

# Sum the fixed per-row prior log-densities, skipping centred-pool marker rows
# (scored against the population in `_pool_centred_logprior`).
function _fixed_row_logprior(fp, x)
    isempty(x) && return 0.0
    return sum(eachindex(x)) do i
        fp[i] isa CentredPoolPrior ? zero(eltype(x)) : logpdf(fp[i], x[i])
    end
end

@doc "

Map an unconstrained vector to the constrained scale and its log-Jacobian.

`to_constrained(prob, z)` returns `(x, logjac)`: the constrained ESTIMATED flat
parameters `x` corresponding to the unconstrained vector `z`, and the
log-determinant Jacobian of that (inverse) transform. The transform is built
per row from [`ComposedLogDensity`](@ref)'s stored `flat_priors` (each row's
`Bijectors.bijector(prior)` — a positive-support prior pushes through an
exp-type link, a stick-breaking `Beta` row through a logit-type link, and so
on); a centred-pooled row (see [`pool`](@ref)) carries no fixed prior of its
own — its row holds a `CentredPoolPrior` marker instead, since its population
is hyperparameter-dependent — so its transform is read off its population's
family instead. The unconstrained log-density a sampler works with is
`logdensity(prob, x) + logjac`.

This has no method until `Bijectors` is loaded; the prior-driven transform
lives in the `ComposedDistributionsBijectorsExt` extension, so the core codec
stays free of a `Bijectors` dependency.

# Arguments
- `prob`: the assembled [`ComposedLogDensity`](@ref).
- `z`: an unconstrained flat vector of length
  [`flat_dimension`](@ref)`(prob.dist)`.

# Examples
```@example
using ComposedDistributions, Distributions, Bijectors

tree = compose((
    onset_admit = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
    admit_death = LogNormal(0.5, 0.4)))
prob = ComposedDistributions.as_logdensity(tree, [[0.5, 2.0]])
# An unconstrained draw maps to constrained parameters plus a log-Jacobian.
z = zeros(ComposedDistributions.flat_dimension(tree))
x, logjac = ComposedDistributions.to_constrained(prob, z)
x
```

# See also
- [`as_logdensity`](@ref): assemble `prob`.
- [`logdensity`](@ref): the constrained-scale density this transform feeds.
"
function to_constrained end
