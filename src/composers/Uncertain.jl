# ============================================================================
# Uncertain: a leaf whose parameters are themselves distributions
# ============================================================================
#
# An `Uncertain` leaf pairs a concrete `template` with `specs`, priors attached
# to the template's FREE parameters. The user-facing story (the hierarchical
# model, the marginal `rand`, the collapse-by-`update`, the truncation
# push-inside) lives in the docstrings below.
#
# `Uncertain` is the LATENT case of a DEFERRED LEAF — a leaf that is not yet a
# concrete distribution but a map to one, delegating silently to a fallback
# until resolved, and guarded by a `has_*` predicate. Its sibling is the
# OBSERVED case, `Varying` (`varying.jl`): `Varying` maps an observed covariate
# read from a `Context` and is resolved by `instantiate`; `Uncertain` maps a
# latent parameter draw with a prior and is resolved by `rand` (the marginal)
# or collapsed by `update`. The two share the `_node_children` guard walk (see
# `has_uncertain` below) and `Varying`'s `instantiate` shares the `_rebuild`
# reconstruction machinery. Maintenance notes:
#
#   - The specs are priors attached to the template's free parameters, so the
#     leaf protocol treats the uncertainty like a wrapper: `free_leaf` peels to
#     the template's free delay and `rewrap_leaf`/`_update_leaf` rebuild the
#     CONCRETE leaf WITHOUT the specs. Pinning values with `update` therefore
#     collapses the uncertainty by design.
#   - `_uncertain_specs` is the routing hook (default `nothing`,
#     introspection.jl); wrapper types forward it exactly like `_shared_tag`
#     (`Shared` here, `Truncated` in introspection.jl, the modifiers in the
#     ModifiedDistributions extension).
#   - The ONE special behaviour is `rand`, which draws the marginal by drawing
#     each spec and rebuilding the concrete leaf through `_uncertain_leaf`. The
#     rest of the univariate surface (scalar `logpdf`/`cdf`/`quantile`/..., the
#     moments) delegates to the template, reporting the leaf at its central
#     values, NOT the marginal — silently, by design, matching `Varying`'s
#     reference-delegation. `has_uncertain` (below) is the loud guard: check it
#     in a fitting loop before scoring, mirroring `Varying`'s `has_varying`.
#   - `Uncertain` is parameterised on the template's `ValueSupport` (`VS`), not
#     the abstract `ValueSupport` itself: `Distributions.MixtureModel` (built by
#     `Resolve`'s `as_mixture`, so `rand`/`mean`/`var`/`logpdf` on a `Resolve`
#     branch) dispatches on `value_support(eltype(components))`, which has no
#     method for the abstract type. An `Uncertain` branch sitting alongside a
#     differently-typed sibling (the ordinary case) would otherwise widen
#     `collect(c.delays)` to that abstract eltype and crash every `Resolve`
#     verb with a confusing `MethodError`, not just the moments this file
#     otherwise guards.

@doc raw"

A leaf distribution whose parameters are themselves distributions.

`Uncertain` pairs a concrete `template` leaf with `specs`, a `NamedTuple`
mapping parameter names (as in [`params_table`](@ref)'s `param` column) to
distributions. A spec entry may itself be an `Uncertain`, so parameter
uncertainty nests. Parameters without a spec stay fixed at the template's
values, and the template's fixed wrapper structure (truncation, censoring) is
carried through every draw via [`free_leaf`](@ref)/[`rewrap_leaf`](@ref).

The generative model is hierarchical:

```math
\theta_j \sim \text{spec}_j, \qquad x \sim D(\theta),
```

with fixed parameters taken from the template. `rand` draws the marginal
(parameters drawn internally). The rest of the univariate surface (scalar
`logpdf`/`pdf`/`cdf`/`quantile`, the moments) delegates to the template, so it
reports the leaf AT the template's central parameter values, NOT the marginal.
Collapse an uncertain leaf to a concrete distribution by pinning its parameters
with [`update`](@ref)`(tree, params)`.

!!! warning \"Only `rand` is marginal\"
    Every other method — `logpdf`/`cdf`/`quantile`/... AND the moments
    `mean`/`var`/`std` — silently reports the template's central values, not
    the marginal. Scoring or summarising a raw `Uncertain` leaf therefore
    answers \"as if\" its parameters were fixed at the template. Guard a
    scoring/fitting loop with [`has_uncertain`](@ref), and collapse to
    concrete values first with [`update`](@ref)`(tree, params)`.

# Fields
- `template`: the concrete (possibly wrapped) leaf supplying the family, the
  fixed parameter values, and the fixed wrapper structure.
- `specs`: `NamedTuple` of the uncertain parameters, each value a distribution
  (possibly itself an `Uncertain`).

# See also
- [`uncertain`](@ref): the public constructor.
- [`update`](@ref): collapse an uncertain leaf to a concrete distribution.
"
struct Uncertain{VS <: ValueSupport, L <: UnivariateDistribution{VS},
    S <: NamedTuple} <: UnivariateDistribution{VS}
    "The concrete (possibly wrapped) template leaf: family, fixed parameter
    values, and fixed wrapper structure (truncation / censoring)."
    template::L
    "`NamedTuple` of the uncertain parameters: each key a parameter name of the
    template's free delay, each value a distribution (possibly `Uncertain`)."
    specs::S

    function Uncertain(template::L,
            specs::S) where {
            VS <: ValueSupport, L <: UnivariateDistribution{VS}, S <: NamedTuple}
        template isa Uncertain && throw(ArgumentError(
            "the template of an Uncertain must be a concrete distribution; " *
            "nest uncertainty in the parameter specs instead"))
        isempty(specs) && throw(ArgumentError(
            "Uncertain needs at least one distribution-valued parameter; " *
            "use the plain distribution for a fully fixed leaf"))
        tvals = params(free_leaf(template))
        all(v -> v isa Real, tvals) || throw(ArgumentError(
            "the template of an Uncertain must have scalar parameters; " *
            "$(template) has composite (non-scalar) parameters $(tvals) " *
            "(e.g. a Convolved/Difference leaf, whose parameters are its " *
            "components' own parameter tuples); make an individual COMPONENT " *
            "uncertain instead — build the composite from uncertain components, " *
            "or target one via `update(tree, (leaf = (component_1 = " *
            "(param = prior,),),))`"))
        pnames = _leaf_param_names(template)
        for (k, v) in pairs(specs)
            k in pnames || throw(ArgumentError(
                "unknown uncertain parameter $(repr(k)); the template " *
                "$(template) has parameters $(collect(pnames))"))
            v isa Union{UnivariateDistribution, Pool} || throw(ArgumentError(
                "the spec for $(repr(k)) must be a UnivariateDistribution " *
                "(a distribution over the parameter) or a `pool(...)` spec " *
                "(partial pooling across a group); got $(typeof(v))"))
        end
        return new{VS, L, S}(template, specs)
    end
end

@doc raw"

Attach parameter uncertainty to a distribution: parameters that are themselves
distributions, nestable to any depth.

`uncertain` has three forms:

- `uncertain(template; kwargs...)` wraps a concrete `template` leaf so the named
  parameters are drawn from the given distributions rather than fixed. Each
  keyword is a parameter name of the template's free delay (as in
  [`params_table`](@ref)'s `param` column); a distribution value makes that
  parameter uncertain, a `Real` value re-pins it to a new fixed value.
- `uncertain(Family, args...)` (a type, e.g. `Gamma`) takes one positional
  argument per parameter, in the family's constructor order: a
  `UnivariateDistribution` makes that parameter uncertain, a `Real` fixes it.
  The template is the family's default instance with the `Real` slots pinned;
  the uncertain slots are driven by their specs.
- `uncertain(Family; kwargs...)` is the keyword form on the family's
  default-constructed template; every parameter must then be given explicitly.

A spec may itself be an [`uncertain`](@ref) distribution, so hyper-uncertainty
nests. The template may be a wrapped leaf (`truncated(...)`, a censoring
wrapper): the wrapper is fixed structure re-applied to every draw. Apply such
wrappers INSIDE the template. `truncated` is the exception: applied outside it
pushes itself into the template automatically.

The result is a `Distributions.UnivariateDistribution` and composes as a leaf
everywhere ([`sequential`](@ref), [`parallel`](@ref), [`resolve`](@ref),
[`compete`](@ref), [`choose`](@ref), [`shared`](@ref)): `rand` draws the
marginal, and [`update`](@ref)`(tree, params)` collapses an uncertain leaf to
its concrete template. In [`params_table`](@ref) an uncertain parameter's spec
rides the row's `prior` column, so [`build_priors`](@ref) picks it up without an
explicit override.

!!! warning \"Only `rand` is marginal\"
    Every other query on the result — `logpdf`/`cdf`/`quantile`/... and the
    moments `mean`/`var`/`std` — silently reports the template's central
    values, not the marginal. Guard a scoring/fitting loop with
    [`has_uncertain`](@ref) before assuming a tree is fully concrete.

A `template` whose parameters are themselves composite (e.g. a `Convolved`/
`Difference` node from the ConvolvedDistributions interop, whose parameters
are its components' own parameter tuples rather than scalars) is refused with
an informative `ArgumentError`: attach uncertainty to an individual COMPONENT
instead, either by building the composite from uncertain components or by
targeting one through [`update`](@ref) at its `component_i` path (that interop
sees through a composite leaf to its component parameters).

# Arguments
- `template`: the concrete (possibly wrapped) leaf distribution, or a
  distribution type (e.g. `Gamma`).
- `args...`: for the positional family form, one value per parameter (a
  distribution for an uncertain parameter, a `Real` for a fixed one).

# Keyword Arguments
- `kwargs...`: parameter name `=` spec pairs. A distribution spec makes the
  parameter uncertain; a `Real` re-pins the template's fixed value.

# Examples
```@example
using ComposedDistributions, Distributions

# A literature-reported Gamma delay with an uncertain shape.
u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
rand(u)

# The positional family form: shape uncertain, scale fixed at 1.0.
uncertain(Gamma, LogNormal(log(2.0), 0.2), 1.0)

# Nested: the shape's prior location is itself uncertain.
uncertain(Gamma(2.0, 1.0);
    shape = uncertain(LogNormal(log(2.0), 0.2); mu = Normal(log(2.0), 0.1)))
```

# See also
- [`Uncertain`](@ref): the wrapper type.
- [`update`](@ref): collapse an uncertain leaf to a concrete distribution.
"
function uncertain(template::UnivariateDistribution; kwargs...)
    nt = values(kwargs)
    pnames = _leaf_param_names(template)
    for (k, v) in pairs(nt)
        k in pnames || throw(ArgumentError(
            "unknown parameter $(repr(k)) for $(template); expected one of " *
            "$(collect(pnames))"))
        v isa Union{Real, UnivariateDistribution, Pool} || throw(ArgumentError(
            "the value for $(repr(k)) must be a Real (a fixed value), a " *
            "UnivariateDistribution (an uncertain parameter), or a `pool(...)` " *
            "spec (partial pooling); got $(typeof(v))"))
    end
    spec_keys = Tuple(k
    for (k, v) in pairs(nt) if v isa Union{UnivariateDistribution, Pool})
    specs = NamedTuple{spec_keys}(Tuple(nt[k] for k in spec_keys))
    fixed_keys = Tuple(k for (k, v) in pairs(nt) if v isa Real)
    pinned = if isempty(fixed_keys)
        template
    else
        tvals = params(free_leaf(template))
        newvals = ntuple(length(pnames)) do i
            pnames[i] in fixed_keys ? nt[pnames[i]] : tvals[i]
        end
        _update_leaf(template, newvals)
    end
    return Uncertain(pinned, specs)
end

# The positional family form: one argument per parameter, in the family's
# constructor order. A `UnivariateDistribution` argument makes that positional
# parameter uncertain, a `Real` fixes it. The family's default instance supplies
# a valid concrete placeholder for the uncertain slots (the specs then drive the
# draws), and the `Real` slots re-pin it, so this reduces to the keyword form.
function uncertain(::Type{D}, arg1::Union{Real, UnivariateDistribution},
        args::Union{Real, UnivariateDistribution}...) where {
        D <: UnivariateDistribution}
    probe = try
        D()
    catch
        throw(ArgumentError(
            "$(D) has no default template; pass a concrete template " *
            "instead, e.g. uncertain($(nameof(D))(...); ...)"))
    end
    positional = (arg1, args...)
    pnames = _leaf_param_names(probe)
    length(positional) == length(pnames) || throw(ArgumentError(
        "uncertain($(nameof(D)), ...) needs one positional argument per " *
        "parameter; $(D) has parameters $(collect(pnames)) but got " *
        "$(length(positional)) argument(s)"))
    kwargs = NamedTuple{pnames}(positional)
    return uncertain(probe; kwargs...)
end

function uncertain(::Type{D}; kwargs...) where {D <: UnivariateDistribution}
    probe = try
        D()
    catch
        throw(ArgumentError(
            "$(D) has no default template; pass a concrete template " *
            "instead, e.g. uncertain($(nameof(D))(...); ...)"))
    end
    nt = values(kwargs)
    pnames = _leaf_param_names(probe)
    for n in pnames
        haskey(nt, n) || throw(ArgumentError(
            "uncertain($(nameof(D)); ...) needs every parameter given " *
            "explicitly; missing $(repr(n)) (expected $(collect(pnames)))"))
    end
    return uncertain(probe; kwargs...)
end

# --- the leaf-protocol hooks -------------------------------------------------
#
# The specs are priors ATTACHED to the template's free parameters, so the
# prior/params interface sees through an `Uncertain` exactly like a fixed
# wrapper: `free_leaf` peels to the template's free delay (its parameters ARE
# the leaf's free parameters), and `rewrap_leaf` re-applies the template's
# fixed wrapper structure around a rebuilt delay WITHOUT the uncertainty —
# pinning definite values (an `update` from fitted draws) collapses the
# uncertain leaf to its concrete distribution.

free_leaf(d::Uncertain) = free_leaf(d.template)
rewrap_leaf(d::Uncertain, inner) = rewrap_leaf(d.template, inner)

# A shared tag survives an uncertain leaf (a `shared(:inc, uncertain(...))`
# is tagged outside, but forward for robustness when nested the other way).
_shared_tag(d::Uncertain) = _shared_tag(d.template)

# The uncertain-spec protocol hook (default `nothing` in introspection.jl):
# an `Uncertain` reports its own specs, and the known wrapper leaves forward so
# a wrapped uncertain leaf still exposes them to `params_table`'s prior column.
# (Extension wrapper types add their own forwarding methods.)
_uncertain_specs(d::Uncertain) = d.specs
_uncertain_specs(d::Shared) = _uncertain_specs(d.dist)

# --- merge mode: `update` introduces or extends uncertainty ------------------
#
# `_merge_leaf` folds a (possibly partial) NamedTuple of specs/values into a
# leaf: a distribution value makes that parameter uncertain (a spec), a `Real`
# value pins it (collapsing any existing spec), and an absent parameter keeps
# the leaf's current spec or fixed value. This is the object-level spelling of
# "distribution in the slot = estimate, value = fix": `update` with distribution
# values is the targeted way to make parameters uncertain, and
# `update(tree, param_priors(tree))` promotes a whole tree to uncertainty over
# its free parameters with default priors (the explicit estimate-everything
# escape hatch). Called from the leaf `_update` in merge mode; the methods live
# here (not introspection.jl) so they can dispatch on `Shared`/`Uncertain`.
# Shared stays OUTERMOST so its tag keeps routing; `Uncertain` wraps the
# concrete (possibly `Truncated`) template so its `ValueSupport` stays concrete
# (see the parameterisation note above).

# Shared stays outermost: peel the tag, merge the inner leaf, re-apply the tag.
function _merge_leaf(leaf::Shared, updates::NamedTuple)
    Shared(leaf.tag, _merge_leaf(leaf.dist, updates))
end

function _merge_leaf(leaf, updates::NamedTuple)
    pnames = _leaf_param_names(leaf)
    _check_merge_keys(updates, pnames, nameof(typeof(leaf)))
    tvals = params(free_leaf(leaf))
    existing = _uncertain_specs(leaf)
    # Re-pin the fixed value at any `Real` update; keep the current value else.
    new_vals = ntuple(length(pnames)) do i
        p = pnames[i]
        (haskey(updates, p) && updates[p] isa Real) ? updates[p] : tvals[i]
    end
    new_template = _update_leaf(leaf, new_vals)
    # The new specs: a distribution update wins, a `Real` drops the spec, else
    # keep any existing spec.
    names = Symbol[]
    vals = Any[]
    for p in pnames
        if haskey(updates, p) && updates[p] isa Union{UnivariateDistribution, Pool}
            push!(names, p)
            push!(vals, updates[p])
        elseif haskey(updates, p) && updates[p] isa Real
            # pinned: collapses any existing spec, so no entry.
        elseif existing !== nothing && haskey(existing, p)
            push!(names, p)
            push!(vals, existing[p])
        end
    end
    isempty(names) && return new_template
    return Uncertain(new_template, NamedTuple{Tuple(names)}(Tuple(vals)))
end

# --- the univariate surface: delegate to the template ------------------------
#
# Every ordinary query is answered at the template's (central) parameter values,
# so an `Uncertain` behaves like a plain distribution there. The scalar
# `logpdf`/`cdf`/... and the moments are therefore NOT the marginal (which
# integrates over the parameter draws); `rand` is the one method that draws the
# marginal, and `update` collapses to a concrete leaf.

# The uncertainty does not change the draw's element type.
Base.eltype(::Type{<:Uncertain{VS, L}}) where {VS, L} = eltype(L)

params(d::Uncertain) = params(d.template)
minimum(d::Uncertain) = minimum(d.template)
maximum(d::Uncertain) = maximum(d.template)
insupport(d::Uncertain, x::Real) = insupport(d.template, x)

logpdf(d::Uncertain, x::Real) = logpdf(d.template, x)
pdf(d::Uncertain, x::Real) = pdf(d.template, x)
cdf(d::Uncertain, x::Real) = cdf(d.template, x)
logcdf(d::Uncertain, x::Real) = logcdf(d.template, x)
ccdf(d::Uncertain, x::Real) = ccdf(d.template, x)
logccdf(d::Uncertain, x::Real) = logccdf(d.template, x)
quantile(d::Uncertain, q::Real) = quantile(d.template, q)

mean(d::Uncertain) = mean(d.template)
var(d::Uncertain) = var(d.template)
std(d::Uncertain) = std(d.template)

sampler(d::Uncertain) = d

@doc "

Draw the marginal of an uncertain distribution: draw every uncertain parameter
from its spec (recursively, so a nested `Uncertain` spec draws via its own
`rand`), rebuild the concrete leaf (fixed wrapper structure re-applied), then
draw the value. Each call draws a fresh parameter set, so repeated draws are iid
from the marginal.

See also: [`Uncertain`](@ref), [`update`](@ref)
"
function Base.rand(rng::AbstractRNG, d::Uncertain)
    return rand(rng, _uncertain_leaf(d.template,
        map(spec -> rand(rng, spec), d.specs)))
end

# Rebuild the concrete leaf implied by drawn scalar parameter values: the
# template's values overridden at each spec'd name by the draw, rebuilt through
# the `free_leaf`/`rewrap_leaf` protocol so the fixed wrapper structure carries
# through. Reused by `rand` (and available for a hand-written two-stage draw).
function _uncertain_leaf(template, drawn::NamedTuple)
    pnames = _leaf_param_names(template)
    tvals = params(free_leaf(template))
    newvals = ntuple(length(pnames)) do i
        haskey(drawn, pnames[i]) ? drawn[pnames[i]] : tvals[i]
    end
    return _update_leaf(template, newvals)
end

@doc "

Print an [`Uncertain`](@ref) leaf as its constructor form: the template and
the `name = spec` pairs.

See also: [`uncertain`](@ref)
"
function Base.show(io::IO, d::Uncertain)
    specs = join(("$(k) = $(v)" for (k, v) in pairs(d.specs)), ", ")
    print(io, "uncertain(", d.template, "; ", specs, ")")
    return nothing
end

# Structural equality/hash (the composers' equality.jl loads before this type
# exists, so the methods live here): same template, same specs.
Base.:(==)(a::Uncertain, b::Uncertain) = a.template == b.template &&
                                         a.specs == b.specs
Base.hash(d::Uncertain, h::UInt) = hash(d.specs, hash(d.template,
    hash(:Uncertain, h)))

# --- truncation pushes inside ------------------------------------------------
#
# The eager `truncated` constructor computes the wrapped distribution's cdf at
# the bounds. Truncating an uncertain distribution instead truncates the
# TEMPLATE, so every draw is from the truncated concrete distribution — the
# conditional (per-parameter-draw) semantics an observation model needs — and
# `Truncated{Uncertain}` never exists. The method set mirrors the upstream
# `truncated` signatures, so both the positional and `lower =`/`upper =` keyword
# forms route here.

function Distributions.truncated(d::Uncertain, lower::T,
        upper::T) where {T <: Real}
    return Uncertain(truncated(d.template, lower, upper), d.specs)
end
function Distributions.truncated(d::Uncertain, lower::Real, ::Nothing)
    return Uncertain(truncated(d.template, lower, nothing), d.specs)
end
function Distributions.truncated(d::Uncertain, ::Nothing, upper::Real)
    return Uncertain(truncated(d.template, nothing, upper), d.specs)
end
Distributions.truncated(d::Uncertain, ::Nothing, ::Nothing) = d

# --- has-uncertainty predicate ----------------------------------------------

# The composer nodes recurse through the shared `_node_children` accessor,
# mirroring `has_varying` (the two deferred-leaf guards share one walk); the
# leaf base case reports a spec via the `_uncertain_specs` routing hook.

@doc "

Whether a composed distribution still contains an [`Uncertain`](@ref) leaf.

An `Uncertain` leaf delegates every `Distributions` method except `rand` to its
template until it is collapsed with [`update`](@ref)`(tree, params)`, so scoring
or summarising a raw tree that still holds an `Uncertain` leaf SILENTLY uses the
template's central values instead of the marginal — a silent wrong answer, not
an error. Guard a scoring/fitting loop with this predicate:

```julia
collapsed = update(tree, fitted_params)
@assert !has_uncertain(collapsed)   # catch a forgotten update before scoring
logpdf(collapsed, x)
```

`has_uncertain` walks the tree (through `Sequential`/`Parallel`/`Choose`/the
one_of composers, and through wrapper leaves via the `_uncertain_specs` routing
hook so a `shared`/modifier-wrapped uncertain leaf is still seen) and returns
`true` as soon as any leaf carries a spec; a fully collapsed tree returns
`false`.

# Arguments
- `d`: the composed distribution, node, or leaf to check.

# Examples
```@example
using ComposedDistributions, Distributions

u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
tree = compose((onset_admit = u, admit_death = LogNormal(0.5, 0.4)))
has_uncertain(tree)   # an uncertain leaf remains

collapsed = update(tree, (onset_admit = (shape = 3.0, scale = 1.5),
    admit_death = (mu = 0.7, sigma = 0.5)))
has_uncertain(collapsed)   # resolved: false
```

# See also
- [`Uncertain`](@ref), [`uncertain`](@ref): the leaf and its constructor.
- [`update`](@ref): collapse an uncertain leaf to a concrete distribution.
- [`has_varying`](@ref): the same guard for the observed (varying) case.
"
function has_uncertain(d::Union{Sequential, Parallel, AbstractOneOf, Choose})
    return any(has_uncertain, _node_children(d))
end
# A `Resolve` is also uncertain when its branch probabilities carry an attached
# simplex prior (a node-level uncertain parameter, not a leaf), so a tree with
# an uncertain branch-probability simplex but fully fixed delays is still seen.
function has_uncertain(c::Resolve)
    return c.branch_prob_prior !== nothing || any(has_uncertain, c.delays)
end
has_uncertain(leaf) = _uncertain_specs(leaf) !== nothing
