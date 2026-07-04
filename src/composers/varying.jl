# ============================================================================
# Varying: a context-indexed leaf and the `instantiate` resolution seam
# ============================================================================
#
# A composed tree is stationary today: every leaf is a fixed
# `UnivariateDistribution`. Non-stationarity — a delay that changes with
# calendar time, a stratum, a region — is modelled here NOT by a new composer
# verb but by generalising the LEAF. A [`Varying`](@ref) leaf carries a map
# `covariate value -> UnivariateDistribution` and a `reference` distribution;
# the seam [`instantiate`](@ref)`(tree, ctx)` walks a composed tree against a
# [`Context`](@ref) and returns the same tree with every varying leaf resolved
# to a concrete distribution. Scoring/sampling/convolving then run unchanged on
# the resolved tree, so non-stationarity is resolved BEFORE those steps rather
# than threaded through their hot paths.
#
# Design note: `design/0001-time-and-covariate-varying-distributions.md`. Time is
# just one covariate; strata are another. The same seam is intended to carry the
# `uncertain distributions` work (a latent, sampled index) by placing sampled
# parameter values in the `Context` — hence `Context` holds an open NamedTuple of
# covariates rather than a fixed `time` field.

# --- the per-record / per-step context -------------------------------------

@doc "

Supertype of the covariate contexts a composed tree is resolved against.

A subtype carries the covariates a [`Varying`](@ref) leaf reads. [`Context`](@ref)
is the concrete open-`NamedTuple` implementation; the abstract type is the seam
the `uncertain distributions` work can extend with its own sampled-parameter
context. [`instantiate`](@ref) dispatches on `AbstractContext`.

# See also
- [`Context`](@ref): the concrete covariate bag.
- [`instantiate`](@ref): the resolution seam.
"
abstract type AbstractContext end

@doc "

The covariate context a [`Varying`](@ref) leaf is resolved against.

A `Context` is an open bag of covariates (a `NamedTuple`) — calendar `time`, a
`region`/`stratum`, or (for the uncertain-distributions work) sampled parameter
values. [`instantiate`](@ref) reads the covariate a leaf names from it. Build one
with keyword covariates.

# Examples
```@example
using ComposedDistributions

ctx = Context(time = 4.0)
ctx.covariates.time
```

# See also
- [`instantiate`](@ref): resolve a tree/leaf against a context.
- [`Varying`](@ref): the context-indexed leaf.
"
struct Context{NT <: NamedTuple} <: AbstractContext
    "The covariates keyed by name (`time`, `region`, sampled params, ...)."
    covariates::NT
end

Context(; covariates...) = Context(NamedTuple(covariates))

# Fetch a named covariate, with a clear error naming what was asked for and what
# the context actually carries (a missing covariate is a caller mistake, not a
# silent fallback).
function _covariate(ctx::Context, name::Symbol)
    haskey(ctx.covariates, name) || throw(ArgumentError(
        "context has no covariate $(repr(name)); it carries " *
        "$(collect(keys(ctx.covariates)))"))
    return ctx.covariates[name]
end

# --- the Varying leaf -------------------------------------------------------

@doc "

A context-indexed leaf: a delay whose distribution varies with a covariate.

`Varying` holds a map `f` from a covariate value to a `UnivariateDistribution`
(e.g. `t -> Gamma(shape(t), scale)`), the `covariate` name it reads from a
[`Context`](@ref) (default `:time`), and a `reference` distribution used whenever
the leaf is queried WITHOUT a context. Because `Varying <: UnivariateDistribution`
it drops into [`Sequential`](@ref) / [`Parallel`](@ref) / [`compose`](@ref) as an
ordinary leaf, and every `Distributions` method (`logpdf`, `cdf`, `mean`, `rand`,
`params`, ...) delegates to the `reference`, so a tree with varying leaves still
scores and samples at its reference by default.

[`instantiate`](@ref)`(d, ctx)` is the seam that swaps the reference for `f`
evaluated at the context's covariate: `instantiate(leaf, Context(time = 4.0))`
returns `f(4.0)`. Resolve a whole tree at a context and then score / sample /
convolve the concrete result.

The varying map `f` is FIXED STRUCTURE (like a truncation bound or a censoring
window), so the introspection interface ([`params_table`](@ref), [`update`](@ref))
treats the `reference`'s parameters as the free parameters and peels/rewraps
through the wrapper; the coefficients of `f` are not (yet) inventoried (see the
design note's open questions).

# Fields
- `f`: map from a covariate value to a `UnivariateDistribution`.
- `covariate`: the [`Context`](@ref) field name to read (`Symbol`, default `:time`).
- `reference`: the distribution used when no context is supplied.

# See also
- [`varying`](@ref): friendly constructor.
- [`instantiate`](@ref): the resolution seam.
- [`Context`](@ref): the covariate bag.
"
struct Varying{F, D <: UnivariateDistribution} <: UnivariateDistribution{Continuous}
    "Map from a covariate value to a `UnivariateDistribution`."
    f::F
    "The `Context` field name this leaf reads (default `:time`)."
    covariate::Symbol
    "The distribution used when no context is supplied."
    reference::D
end

@doc "

Build a [`Varying`](@ref) context-indexed leaf.

`varying(f; covariate = :time, reference = f(0.0))` wraps a map `f` (a covariate
value to a `UnivariateDistribution`) as a leaf that varies with the named
`covariate`. The `reference` distribution is used when the leaf is queried
without a context; it defaults to `f(0.0)` (the map at the origin), which suits a
`:time` covariate — pass an explicit `reference` for a categorical covariate
where `f(0.0)` is not meaningful.

# Arguments
- `f`: map from a covariate value to a `UnivariateDistribution`.

# Keyword Arguments
- `covariate`: the [`Context`](@ref) field name to read (default `:time`).
- `reference`: the distribution used without a context (default `f(0.0)`).

# Examples
```@example
using ComposedDistributions, Distributions

# An onset->admit delay whose mean shortens over calendar time.
d = varying(t -> Gamma(2.0, 1.0 + 0.1 * t); covariate = :time)
mean(d)                                   # the reference (t = 0)
mean(instantiate(d, Context(time = 5.0))) # the delay at t = 5
```

# See also
- [`Varying`](@ref): the leaf type.
- [`instantiate`](@ref): resolve against a context.
"
function varying(f; covariate::Symbol = :time, reference = f(0.0))
    reference isa UnivariateDistribution || throw(ArgumentError(
        "varying `reference` must be a UnivariateDistribution; got " *
        "$(typeof(reference))"))
    return Varying(f, covariate, reference)
end

# --- Distributions delegation (a Varying leaf behaves as its reference) -----
#
# Without a context a Varying leaf IS its reference distribution, so every
# scalar Distributions query delegates. This keeps a tree with varying leaves
# fully usable (score, sample, moments) at the reference, and is what makes the
# `instantiate` seam opt-in rather than mandatory.
params(d::Varying) = params(d.reference)
minimum(d::Varying) = minimum(d.reference)
maximum(d::Varying) = maximum(d.reference)
insupport(d::Varying, x::Real) = insupport(d.reference, x)
logpdf(d::Varying, x::Real) = logpdf(d.reference, x)
pdf(d::Varying, x::Real) = pdf(d.reference, x)
cdf(d::Varying, x::Real) = cdf(d.reference, x)
logcdf(d::Varying, x::Real) = logcdf(d.reference, x)
ccdf(d::Varying, x::Real) = ccdf(d.reference, x)
logccdf(d::Varying, x::Real) = logccdf(d.reference, x)
quantile(d::Varying, q::Real) = quantile(d.reference, q)
mean(d::Varying) = mean(d.reference)
var(d::Varying) = var(d.reference)
std(d::Varying) = std(d.reference)
sampler(d::Varying) = sampler(d.reference)
Base.rand(rng::AbstractRNG, d::Varying) = rand(rng, d.reference)
Base.rand(d::Varying) = rand(default_rng(), d.reference)

# The varying map is fixed structure; the reference carries the free parameters.
# Peel/rewrap through the wrapper so `params_table` / `update` see the inner free
# delay and a parameter update rebuilds the same varying leaf around it.
free_leaf(d::Varying) = free_leaf(d.reference)
function rewrap_leaf(d::Varying, inner)
    return Varying(d.f, d.covariate, rewrap_leaf(d.reference, inner))
end
_shared_tag(d::Varying) = _shared_tag(d.reference)

function Base.show(io::IO, d::Varying)
    print(io, "Varying(", d.covariate, " -> ", d.reference, ")")
    return nothing
end

# --- the resolution seam ----------------------------------------------------

@doc "

Resolve a composed tree (or a leaf) against a [`Context`](@ref).

`instantiate(d, ctx)` walks a composed distribution and returns the SAME tree
with every [`Varying`](@ref) leaf replaced by its distribution at the context's
covariate; a fixed leaf is returned unchanged (the identity default), so a
stationary tree is untouched and passing `nothing` is always a no-op. The result
is a fully concrete composer that scores, samples and convolves exactly as a
hand-built stationary tree would — non-stationarity is resolved HERE, before
those steps, so the convolution / renewal layer receives a concrete kernel per
context.

# Arguments
- `d`: a composer, a leaf, or a `Varying` leaf.
- `ctx`: a [`Context`](@ref) of covariates, or `nothing` (a no-op).

# Examples
```@example
using ComposedDistributions, Distributions

chain = sequential(:onset_admit => varying(t -> Gamma(2.0, 1.0 + 0.1t)),
    :admit_death => LogNormal(0.5, 0.4))
at_day5 = instantiate(chain, Context(time = 5.0))  # concrete, stationary chain
observed_distribution(at_day5)                     # the convolution kernel at t = 5
```

# See also
- [`Varying`](@ref), [`Context`](@ref): the leaf and the covariate bag.
- [`observed_distribution`](@ref): collapse the resolved chain to its kernel.
"
instantiate(d, ::Nothing) = d
instantiate(d::UnivariateDistribution, ::AbstractContext) = d
instantiate(d::Varying, ctx::AbstractContext) = d.f(_covariate(ctx, d.covariate))

# Composers rebuild themselves with each child resolved, so the tree shape and
# names are preserved and only the leaves change. `Resolve` / `Compete` are
# `UnivariateDistribution`s, so their more specific methods below win over the
# leaf identity.
function instantiate(d::Sequential, ctx::AbstractContext)
    return Sequential(map(c -> instantiate(c, ctx), d.components), d.names)
end
function instantiate(d::Parallel, ctx::AbstractContext)
    return Parallel(map(c -> instantiate(c, ctx), d.components), d.names)
end
function instantiate(d::Choose, ctx::AbstractContext)
    return Choose(d.names, map(c -> instantiate(c, ctx), d.alternatives),
        d.selector)
end
function instantiate(c::Resolve, ctx::AbstractContext)
    delays = map(x -> instantiate(x, ctx), c.delays)
    return Resolve(c.names, delays, c.branch_probs)
end
function instantiate(c::Compete, ctx::AbstractContext)
    delays = map(x -> instantiate(x, ctx), c.delays)
    return Compete(c.names, delays)
end

# A tagged shared leaf keeps its tag through resolution (the resolved value is
# still the same shared parameter group).
instantiate(d::Shared, ctx::AbstractContext) = Shared(d.tag, instantiate(d.dist, ctx))
