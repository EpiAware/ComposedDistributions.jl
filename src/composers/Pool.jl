# ============================================================================
# Pool: partial pooling of a parameter across the leaves of a group
# ============================================================================
#
# A `Pool` spec declares a parameter partially pooled across the leaves that
# share a group tag: each member's parameter is drawn from one common
# *population* distribution whose own free parameters are the estimated
# hyperparameters, so a data-poor stratum shrinks towards the population while a
# data-rich one moves freely. It is the middle of the pooling spectrum between
# `shared`/`tie` (complete pooling, one value everywhere) and independent
# `uncertain` specs (no pooling, K unlinked values). See issue #78.
#
# The population is an ordinary distribution — typically an `uncertain` one, so
# its free parameters carry their priors through the *existing* uncertain-spec
# machinery, exactly like any other uncertain leaf. That distribution encodes
# how the strata relate; the family is not baked in. The codec lowers a pooled
# group to ordinary scalar rows:
#
#   - the population's own spec'd parameters as hyperparameter rows under the
#     `<group>` edge (`<group>.mu`, `<group>.sigma`, ...), each carrying the
#     spec's prior — these simply fall out of running the uncertain machinery on
#     the population; and
#   - one latent row per member.
#
# Two parameterisations, chosen by the population family:
#
#   - a *location-scale* population (`Normal`/`LogNormal`) is reparameterised
#     *non-centred*: the member's latent is `z ~ Normal(0, 1)` and the parameter
#     is reconstructed `loc + scale*z` (`Normal`) or `exp(loc + scale*z)`
#     (`LogNormal`), where `(loc, scale)` are the population's hyperparameters.
#     This keeps the CensoredDistributions-compatible `[hyper..., z...]` flat
#     layout (`mu ~ ...; sigma ~ ...; z ~ filldist(Normal(0, 1), K)`), so a
#     model authored either way is interchangeable.
#   - a *general* population takes the *centred* path: the member's latent *is*
#     its parameter, scored directly against the population distribution
#     reconstructed at the current hyperparameters. The population prior is
#     parameter-dependent, so it is scored separately by
#     `pool_centred_logprior` rather than sitting in the fixed per-row prior
#     vector; the hyperparameters stay ordinary fixed per-row priors.
#
# The shared hyperparameters live at the top level under the group key and are
# threaded to every member exactly like a `shared` tag's value (see `_update`).

@doc raw"
A partial-pooling spec: a parameter drawn, across a group of leaves, from one
shared population distribution.

`Pool` marks a parameter (an entry of an [`uncertain`](@ref) leaf's specs) as
partially pooled across the leaves that name the same `group`: every member's
parameter is drawn from one common `population` distribution whose own free
parameters are the estimated hyperparameters. It is the middle of the pooling
spectrum — [`shared`](@ref)/[`tie`](@ref) is complete pooling (one value
everywhere) and independent [`uncertain`](@ref) specs are no pooling (K unlinked
values).

The `population` is an ordinary distribution — usually an [`uncertain`](@ref)
one, so its free parameters carry their priors through the same machinery as any
uncertain leaf. A location-scale population (`Normal`/`LogNormal`) is
reparameterised non-centred (one `Normal(0, 1)` latent per member); a general
population is scored centred (each member's parameter directly against the
population).

# Fields
- the pooling-group name (`Symbol`) lives in the `group` type parameter (read
  with [`pool_group`](@ref)); leaves naming the same group are one population.
- `population`: the population distribution (its free parameters are the
  hyperparameters).
- whether the non-centred (location-scale) parameterisation is used (only for
  a `Normal`/`LogNormal` population) lives in the `noncentred` type parameter
  (read with [`pool_noncentred`](@ref)).

# See also
- [`pool`](@ref): the public constructor.
- [`shared`](@ref)/[`tie`](@ref): complete pooling (the tied extreme).
- [`uncertain`](@ref): builds a population with hyperparameter priors.
"
struct Pool{group, noncentred, P <: UnivariateDistribution}
    "The population distribution; its free parameters are the hyperparameters."
    population::P
end

# `group`/`noncentred` live in type parameters (like `NamedTuple{names}`); this
# instantiates them directly from the runtime `pool(...)` call, with no
# call-site change from the field-based constructor.
function Pool{group, noncentred}(population::P) where {
        group, noncentred, P <: UnivariateDistribution}
    return Pool{group, noncentred, P}(population)
end

@doc "
The pooling-group name (`Symbol`) of a [`Pool`](@ref) spec.

# Examples
```@example
using ComposedDistributions

spec = pool(:district)
ComposedDistributions.pool_group(spec)
```

See also: [`pool_noncentred`](@ref), [`pool`](@ref)
"
pool_group(::Pool{group}) where {group} = group

@doc "
Whether a [`Pool`](@ref) spec uses the non-centred (location-scale)
parameterisation.

# Examples
```@example
using ComposedDistributions

spec = pool(:district)
ComposedDistributions.pool_noncentred(spec)
```

See also: [`pool_group`](@ref), [`pool`](@ref)
"
pool_noncentred(::Pool{group, noncentred}) where {group, noncentred} = noncentred

# The default population: a `LogNormal` whose location `mu` and scale `sigma`
# are both estimated (weakly-informative priors), reparameterised non-centred.
function _default_pool_population()
    return uncertain(Distributions.LogNormal(0.0, 1.0);
        mu = Distributions.Normal(0.0, 1.0),
        sigma = Distributions.truncated(
            Distributions.Normal(0.0, 1.0); lower = 0.0))
end

@doc raw"
Declare a parameter partially pooled across the leaves of a `group`, drawn from
a shared `population` distribution.

`pool(group, population)` returns a [`Pool`](@ref) spec to place inside an
[`uncertain`](@ref) leaf where a prior would go, e.g.

```julia
uncertain(Gamma(2.0, 1.0);
    shape = pool(:district,
        uncertain(LogNormal(0.0, 1.0); mu = Normal(0.0, 1.0),
            sigma = truncated(Normal(0.0, 1.0); lower = 0.0))))
```

reading as: *`shape` is partially pooled across the `:district` leaves — each
district's `shape` is drawn from one shared `LogNormal` population whose
`(mu, sigma)` are estimated.* The `population` is an ordinary distribution;
build it with [`uncertain`](@ref) so its free parameters carry their priors
through the same machinery as any uncertain leaf (those become the
hyperparameter rows `<group>.mu`, `<group>.sigma`, ...). The leaves that name
the same `group` are one population, grouped by tag the way [`shared`](@ref)
groups tied leaves. `pool(group)` uses a default estimated-`LogNormal`
population.

A location-scale population (`Normal`/`LogNormal`) is reparameterised
*non-centred* — one `Normal(0, 1)` latent per member, reconstructed
`loc + scale*z` (`Normal`) or `exp(loc + scale*z)` (`LogNormal`) — keeping the
CensoredDistributions-compatible `[hyper..., z...]` flat vector. A general
population takes the *centred* path (each member's parameter scored directly
against the population). Pass `noncentred = false` to force the centred form on
a location-scale population; `noncentred = true` is rejected for a general one.

`rand` on a pooled leaf draws that one parameter's marginal from the population;
the joint prior-predictive of a whole pooled tree (population shared across
members) comes from sampling the flat priors and rebuilding with
[`update`](@ref)`(tree, `[`unflatten`](@ref)`(tree, x))`.

# Arguments
- `group`: the pooling-group name (`Symbol`).
- `population`: the shared population distribution (default: an estimated
  `LogNormal`). Its free parameters are the hyperparameters.

# Keyword Arguments
- `noncentred`: force the parameterisation. Defaults to `true` for a
  location-scale (`Normal`/`LogNormal`) population, `false` otherwise.

# Examples
```@example
using ComposedDistributions, Distributions

# Three districts' onset->death delays with a partially pooled shape, drawn
# from a shared estimated-LogNormal population.
model = compose((
    north = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
    east  = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
    south = uncertain(Gamma(2.0, 1.0); shape = pool(:district))))
# 2 hyperparameters + 3 latents = 5 estimated parameters.
ComposedDistributions.flat_dimension(model)
```

# See also
- [`Pool`](@ref): the spec type.
- [`uncertain`](@ref): builds the population with hyperparameter priors.
- [`shared`](@ref)/[`tie`](@ref): the complete-pooling (tied) extreme.
"
function pool(group::Symbol,
        population::UnivariateDistribution = _default_pool_population();
        noncentred::Union{Bool, Nothing} = nothing)
    ls = _is_location_scale(_population_family(population))
    nc = noncentred === nothing ? ls : noncentred
    (nc && !ls) && throw(ArgumentError(
        "non-centred pooling is only available for a location-scale population " *
        "(Normal or LogNormal); a $(nameof(_population_family(population))) " *
        "population must use the centred parameterisation (noncentred = false)"))
    return Pool{group, nc}(population)
end

# The population's template (an `uncertain` population's concrete template, or
# the population itself for a plain distribution) and its family. The family
# drives the link and the location-scale (non-centred) eligibility.
#
# This asks which family the population is, not how to rebuild it, so it reads
# the peeled type directly rather than going through `leaf_ctor`. The two are
# the same for a leaf whose params are its constructor arguments, but a leaf that
# overrides `leaf_ctor` returns a callable that is not a family: routing this
# through the hook would make `_is_location_scale` false for such a leaf and
# silently demote a non-centred pool to a centred one.
_population_template(pop::UnivariateDistribution) = pop
_population_template(pop::Uncertain) = pop.template
function _population_family(pop::UnivariateDistribution)
    return Base.typename(typeof(free_leaf(_population_template(pop)))).wrapper
end

# A location-scale family reparameterises non-centred (a standard-normal latent
# through a `loc + scale*z` / `exp(loc + scale*z)` transform).
function _is_location_scale(fam)
    return fam === Distributions.Normal || fam === Distributions.LogNormal
end

# The non-centred reconstruction: member `k`'s parameter from the population's
# location/scale hyperparameters and its standard-normal latent. `LogNormal` ->
# exp (a positive parameter), `Normal` -> identity (a real parameter). The
# element type flows through (an AD `Dual` differentiates the reconstruction).
function _noncentred_link(spec::Pool, loc, scale, z)
    fam = _population_family(spec.population)
    return fam === Distributions.LogNormal ? exp(loc + scale * z) :
           loc + scale * z
end

# The per-member latent's prior in the non-centred parameterisation: a standard
# normal (the population is carried by the hyperparameters, not the latent).
_pool_z_prior(::Pool) = Distributions.Normal(0.0, 1.0)

# The `seen`-set key marking a pooling group's hyperparameters as already
# emitted by the params-table walk. Namespaced (`pool.<group>`) so it never
# collides with a `shared` tag sharing the group's name.
_pool_seen_key(group::Symbol) = Symbol("pool.", group)

# The population reconstructed at a draw of its hyperparameters: an `uncertain`
# population collapsed at the drawn spec'd values (fixed parameters kept from
# the template); a plain population is already concrete. Reused by both the
# non-centred reconstruction (to read `(loc, scale)`) and the centred prior term.
function _collapse_population(pop::Uncertain, hyper::NamedTuple)
    _uncertain_leaf(pop.template, hyper)
end
_collapse_population(pop::UnivariateDistribution, ::NamedTuple) = pop

@doc "
The centred latent's prior marker, carried on the `prior` column of a centred
pooled parameter's row.

It is not a fixed distribution (the population depends on the estimated
hyperparameters), so DistributionsInference.jl's `as_logdensity`/`logdensity`
scores it separately ([`_pool_centred_logprior`](@ref)) and skips it in the
fixed per-row prior sum.
A non-`nothing` entry, so the row still counts as estimated.

Reached by qualified name from outside this package — DistributionsInference.jl's
fit-protocol extension pattern-matches on this marker to translate a
centred-pooled row's prior to `nothing` (#212).

See also: [`_centred_pool_rows`](@ref), [`_pool_centred_logprior`](@ref),
[`pool`](@ref)
"
struct CentredPoolPrior{P <: Pool}
    pool::P
end

function Base.show(io::IO, p::CentredPoolPrior)
    print(io, "centred-pool(", repr(pool_group(p.pool)), ", ",
        p.pool.population, ")")
    return nothing
end

# The pooled subset of a leaf's uncertain specs (`param => Pool`), or `nothing`
# when the leaf pools nothing. Drives the pooled reconstruction in `_update`.
function _pool_specs(leaf)
    specs = _uncertain_specs(leaf)
    specs === nothing && return nothing
    ks = filter(k -> specs[k] isa Pool, keys(specs))
    isempty(ks) && return nothing
    return NamedTuple{Tuple(ks)}(map(k -> specs[k], Tuple(ks)))
end

# --- params-table rows for a pooled parameter -------------------------------
#
# Emit the population's hyperparameter rows once per group (deduped through the
# walk's `seen` set, so they precede every member's latent and the flat vector
# opens with `[hyper..., ...]`), then this member's latent: a `Normal(0, 1)`
# `z` row (non-centred) or the member's own parameter carrying the centred-pool
# marker (centred). All ordinary scalar rows.
function _pool_rows!(edges, params_col, values, supports, priors, seen,
        p::Pool, leaf_edge, pname, v, s)
    gkey = _pool_seen_key(pool_group(p))
    if !(gkey in seen)
        push!(seen, gkey)
        _pool_hyper_rows!(edges, params_col, values, supports, priors, p)
    end
    if pool_noncentred(p)
        push!(edges, _join_path((_split_edge(leaf_edge)..., pname)))
        push!(params_col, :z)
        push!(values, 0.0)
        push!(supports, (-Inf, Inf))
        push!(priors, _pool_z_prior(p))
    else
        push!(edges, leaf_edge)
        push!(params_col, pname)
        push!(values, v)
        push!(supports, s)
        push!(priors, CentredPoolPrior(p))
    end
    return nothing
end

# The population's hyperparameter rows: its spec'd (estimated) parameters,
# emitted under the `<group>` edge with the specs' priors. A population with no
# uncertain specs (a fully fixed population) contributes no hyperparameters.
# This is exactly the uncertain-leaf param walk restricted to the spec'd rows.
function _pool_hyper_rows!(edges, params_col, values, supports, priors, p::Pool)
    specs = _uncertain_specs(p.population)
    specs === nothing && return nothing
    tmpl = _population_template(p.population)
    inner = free_leaf(tmpl)
    pnames = _leaf_param_names(tmpl)
    vals = params(inner)
    sup = (minimum(inner), maximum(inner))
    for (pname, v) in zip(pnames, vals)
        haskey(specs, pname) || continue
        push!(edges, pool_group(p))
        push!(params_col, pname)
        push!(values, v)
        push!(supports, sup)
        push!(priors, specs[pname])
    end
    return nothing
end

# --- flat codec: interleaved leaf/group runtime resolution (S3 fix) --------
#
# `codec_gen.jl`'s leaf case cannot compute a leaf's NATIVE parameter order at
# generation time (S3 removed the type-level table that used to carry it),
# yet `_pool_rows!`/`_pool_hyper_rows!` above insert a pooled group's
# hyperparameter rows into `params_table` at the pooled parameter's OWN
# native-order position -- not before or after the whole leaf's block, and a
# population's own hyperparameters are walked in ITS native order, not its
# `uncertain(...)` kwargs order. Both need `leaf_param_names` resolved on a
# REAL instance to get right, so -- exactly `_leaf_entry`'s trick
# (introspection.jl), extended to also interleave a group's materialisation
# -- this walk runs from the generated code, at ACTUAL CALL TIME, not the
# generator. `codec_gen.jl` only needs to know, at generation time, how many
# `x` slots a leaf-visit occupies (`speckeys` count plus each newly-seen
# group's population `speckeys` count, `_pool_hyper_count` below): a sum, not
# an order, so it stays generation-time-safe without reviving the removed
# registry.
#
# `speckeys` are this leaf's own estimated names; `materialize` (a `Val` of
# the subset of `speckeys` whose pool group is first-seen ACROSS THE WHOLE
# TREE, decided at generation time via `ctx.seen_groups`) marks which of this
# leaf's pooled names are responsible for consuming their group's
# hyperparameter block here, before their own slot -- mirroring
# `_pool_rows!`'s "hyper rows once, then this member's row" order exactly. A
# leaf touching no first-seen group never reaches these functions; codec_gen
# keeps using the plain `_leaf_entry`/`_leaf_flatten_values` seam for that
# (and for every non-pooled leaf), unchanged.

# The `x`-slot count a `Pool` spec's population contributes when it is this
# leaf-visit's first (group-materialising) occurrence: its own spec'd
# (estimated) parameter count, `0` for a fully fixed population. A count, not
# an order, so (unlike the removed `_pool_hyper_unflatten_expr`) it needs no
# instance and stays safe to call from `codec_gen.jl`'s generation-time walk.
function _pool_hyper_count(specT::Type{<:Pool})
    P = specT.parameters[3]
    P <: Uncertain || return 0
    PS = P.parameters[3]
    return length(PS.parameters[1]::Tuple)
end

# unflatten direction: build this leaf's own `(native..., extra...)` entry
# (like `_leaf_entry`) while interleaving each materialising group's hyper
# entry at the right native-order position, consuming `slots` -- a single
# contiguous `x` block sized `length(speckeys) + sum(_pool_hyper_count(...))`
# -- in that same interleaved order. Returns `(entry, groups)`, `groups` a
# `NamedTuple` of `group => hyper_entry` for every group this leaf-visit
# materialises (empty when it materialises none).
function _leaf_entry_grouped(leaf, ::Val{speckeys}, ::Val{pool_names},
        ::Val{materialize}, slots::Tuple) where {speckeys, pool_names, materialize}
    names = leaf_param_names(leaf)
    vals = leaf_param_values(leaf)
    specs = _uncertain_specs(leaf)
    entry_vals, group_pairs, _ = _leaf_walk_grouped(
        names, vals, speckeys, materialize, specs, slots)
    entry = NamedTuple{names}(entry_vals)
    entry = isempty(pool_names) ? entry : _wrap_pool_entries(entry, Val(pool_names))
    return entry, (; group_pairs...)
end

@inline function _leaf_walk_grouped(::Tuple{}, ::Tuple{}, ::Tuple, ::Tuple,
        specs, slots::Tuple)
    return (), (), slots
end
@inline function _leaf_walk_grouped(names::Tuple, vals::Tuple, speckeys::Tuple,
        materialize::Tuple, specs, slots::Tuple)
    pname = names[1]
    hit = pname in speckeys
    group_pair, slots = if hit && pname in materialize
        spec = specs[pname]
        hyper, rest = _pool_hyper_entry(spec.population, slots)
        (pool_group(spec) => hyper,), rest
    else
        (), slots
    end
    v, slots = hit ? (slots[1], Base.tail(slots)) : (vals[1], slots)
    rest_vals, rest_groups, slots = _leaf_walk_grouped(
        Base.tail(names), Base.tail(vals), speckeys, materialize, specs, slots)
    return (v, rest_vals...), (group_pair..., rest_groups...), slots
end

# A materialising group's hyperparameter entry: the population's own spec'd
# names, walked in the population's NATIVE order (mirroring
# `_pool_hyper_rows!` exactly), each consuming one `slots` value. A fully
# fixed population contributes no names and consumes nothing.
_pool_hyper_entry(pop::UnivariateDistribution, slots::Tuple) = NamedTuple(), slots
function _pool_hyper_entry(pop::Uncertain, slots::Tuple)
    specs = _uncertain_specs(pop)
    pnames = leaf_param_names(_population_template(pop))
    ks, vs, rest = _hyper_walk(pnames, specs, slots)
    return NamedTuple{ks}(vs), rest
end

@inline _hyper_walk(::Tuple{}, specs, slots::Tuple) = (), (), slots
@inline function _hyper_walk(pnames::Tuple, specs, slots::Tuple)
    pname = pnames[1]
    if haskey(specs, pname)
        rest_k, rest_v, rest_slots = _hyper_walk(
            Base.tail(pnames), specs, Base.tail(slots))
        return (pname, rest_k...), (slots[1], rest_v...), rest_slots
    end
    return _hyper_walk(Base.tail(pnames), specs, slots)
end

# flatten direction: the dual of `_leaf_entry_grouped`, reading `entry` (this
# leaf's already-`unflatten`ed sub-tree) and each materialising group's hyper
# `NamedTuple` (read off the root `nt`, where `unflatten` root-lifts it,
# exactly like `_pool_hyper_flatten_reads!`'s removed matching read) back out
# as a flat value sequence in the SAME interleaved order `_leaf_entry_grouped`
# consumes, so the two stay inverse.
function _leaf_flatten_grouped(leaf, ::Val{speckeys}, ::Val{pool_names},
        ::Val{materialize}, entry::NamedTuple, nt::NamedTuple) where
        {speckeys, pool_names, materialize}
    names = leaf_param_names(leaf)
    specs = _uncertain_specs(leaf)
    return _leaf_flatten_walk_grouped(
        names, speckeys, pool_names, materialize, specs, entry, nt)
end

@inline function _leaf_flatten_walk_grouped(::Tuple{}, ::Tuple, ::Tuple, ::Tuple,
        specs, entry::NamedTuple, nt::NamedTuple)
    return ()
end
@inline function _leaf_flatten_walk_grouped(names::Tuple, speckeys::Tuple,
        pool_names::Tuple, materialize::Tuple, specs, entry::NamedTuple,
        nt::NamedTuple)
    pname = names[1]
    hit = pname in speckeys
    hyper_vals = if hit && pname in materialize
        spec = specs[pname]
        _pool_hyper_flatten(spec.population, getproperty(nt, pool_group(spec)))
    else
        ()
    end
    own_vals = if hit
        raw = getproperty(entry, pname)
        (pname in pool_names ? raw.z : raw,)
    else
        ()
    end
    rest = _leaf_flatten_walk_grouped(Base.tail(names), speckeys, pool_names,
        materialize, specs, entry, nt)
    return (hyper_vals..., own_vals..., rest...)
end

# The dual of `_pool_hyper_entry`: the population's spec'd values read back
# off its already-`unflatten`ed hyper `NamedTuple`, in the same native order.
_pool_hyper_flatten(pop::UnivariateDistribution, ::NamedTuple) = ()
function _pool_hyper_flatten(pop::Uncertain, hyper_nt::NamedTuple)
    specs = _uncertain_specs(pop)
    pnames = leaf_param_names(_population_template(pop))
    return _hyper_flatten_walk(pnames, specs, hyper_nt)
end

@inline _hyper_flatten_walk(::Tuple{}, specs, nt::NamedTuple) = ()
@inline function _hyper_flatten_walk(pnames::Tuple, specs, nt::NamedTuple)
    pname = pnames[1]
    rest = _hyper_flatten_walk(Base.tail(pnames), specs, nt)
    haskey(specs, pname) || return rest
    return (getproperty(nt, pname), rest...)
end

# --- pooled leaf reconstruction ---------------------------------------------
#
# Rebuild a pooled leaf at a draw. A non-centred pooled parameter is
# `link(loc + scale*z)` from the population's hyperparameters (read from the
# top-level group entry, threaded like a `shared` tag) and the member's latent;
# a centred pooled parameter *is* its latent directly (its population prior is
# scored by `pool_centred_logprior`). A non-pooled parameter takes its supplied
# value (or the template's). Then rebuild the concrete leaf, collapsing the
# uncertainty.
function _reconstruct_pooled_leaf(leaf, leaf_params, shared, pooled, pnames)
    tvals = params(free_leaf(leaf))
    newvals = ntuple(length(pnames)) do i
        p = pnames[i]
        if haskey(pooled, p)
            spec = pooled[p]
            if pool_noncentred(spec)
                hyper = _pool_hyper(shared, spec)
                pop = _collapse_population(spec.population, hyper)
                loc, scale = params(free_leaf(pop))
                _noncentred_link(spec, loc, scale, _pool_z(leaf_params, p))
            else
                leaf_params[p]
            end
        else
            haskey(leaf_params, p) ? leaf_params[p] : tvals[i]
        end
    end
    return _update_leaf(leaf, newvals)
end

# The population's hyperparameters for a pooled spec, read from the top-level
# group entry (threaded as `shared`, like a `shared` tag's value). A population
# with no estimated hyperparameters carries no group entry, so an empty
# NamedTuple (the population stays at its template) is returned.
function _pool_hyper(shared, p::Pool)
    _uncertain_specs(p.population) === nothing && return NamedTuple()
    group = pool_group(p)
    (shared isa NamedTuple && haskey(shared, group)) || throw(ArgumentError(
        "update(...) is missing the pooled population $(repr(group)); a " *
        "`pool($(repr(group)), ...)` leaf needs a top-level `$(group)` " *
        "entry with the population hyperparameters"))
    return shared[group]
end

# The member's non-centred latent, read from its `(param = (z = ...,),)` slot.
function _pool_z(leaf_params::NamedTuple, pname::Symbol)
    haskey(leaf_params, pname) || throw(ArgumentError(
        "pooled parameter $(repr(pname)) is missing its latent in the update"))
    entry = leaf_params[pname]
    (entry isa NamedTuple && haskey(entry, :z)) || throw(ArgumentError(
        "pooled parameter $(repr(pname)) must be updated with a `(z = ...,)` " *
        "latent; got $(entry)"))
    return entry.z
end

# --- centred population prior term ------------------------------------------
#
# For a centred pooled parameter the member's latent *is* its parameter, scored
# directly against the population reconstructed at the current hyperparameters.
# That prior is parameter-dependent, so `pool_centred_logprior` scores it from
# the reconstructed nested `NamedTuple` rather than the fixed per-row prior
# vector. The `(path, param, pool)` rows are collected once
# (`centred_pool_rows`, at DistributionsInference.jl's `as_logdensity`
# construction), so a tree with only non-centred (or no) pooling adds no
# per-evaluation cost.

@doc raw"

The centred pooled parameters' `(path, param, pool)` triples, in table order.

Collected once per [`params_table`](@ref) walk (typically at `as_logdensity`
construction time), so a tree with only non-centred (or no) pooling adds no
per-evaluation cost. Reached by qualified name from outside this package —
DistributionsInference.jl's fit-protocol extension calls this directly to
find the rows [`pool_centred_logprior`](@ref) needs to score (#212).

# Arguments
- the composed tree whose centred-pooled rows are collected.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((north = uncertain(Gamma(2.0, 1.0);
        shape = pool(:region, Beta(2.0, 3.0))),
    south = uncertain(Gamma(2.0, 1.0); shape = pool(:region, Beta(2.0, 3.0)))))
ComposedDistributions.centred_pool_rows(tree)
```

# See also
- [`CentredPoolPrior`](@ref), [`pool_centred_logprior`](@ref)
"
function centred_pool_rows(dist)
    tbl = params_table(dist)
    prcol = Tables.getcolumn(tbl, :prior)
    edges = Tables.getcolumn(tbl, :edge)
    params_col = Tables.getcolumn(tbl, :param)
    rows = Tuple{Tuple, Symbol, Pool}[]
    for i in eachindex(prcol)
        prcol[i] isa CentredPoolPrior || continue
        push!(rows, (_split_edge(edges[i]), params_col[i], prcol[i].pool))
    end
    return rows
end

@doc raw"

Deprecated alias for [`centred_pool_rows`](@ref); kept transitionally so a
caller already qualifying it (`ComposedDistributions._centred_pool_rows`, or
an explicit `using ComposedDistributions: _centred_pool_rows`) keeps working
across the rename (#212 declared this `public` with the leading underscore,
which the org's naming convention reserves for internal-only names). New code
should call `centred_pool_rows`; this alias is removed in a future cleanup
once DistributionsInference.jl's fit-protocol extension has moved off it.

# Arguments
- the composed tree whose centred-pooled rows are collected; see
  [`centred_pool_rows`](@ref).

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((north = uncertain(Gamma(2.0, 1.0);
        shape = pool(:region, Beta(2.0, 3.0))),
    south = uncertain(Gamma(2.0, 1.0); shape = pool(:region, Beta(2.0, 3.0)))))
ComposedDistributions._centred_pool_rows(tree)
```

# See also
- [`centred_pool_rows`](@ref)
"
const _centred_pool_rows = centred_pool_rows

@doc raw"

Sum each centred member's log-density against its population reconstructed at
the current hyperparameters (read from the flattened draw `nt`).

Reached by qualified name from outside this package — DistributionsInference.jl's
fit-protocol extension calls this to score a composed tree's centred-pooled
population term (#212).

# Arguments
- `rows`: the `(path, param, pool)` triples from [`centred_pool_rows`](@ref).
- `nt`: the nested `NamedTuple` from `unflatten` at the same draw.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((north = uncertain(Gamma(2.0, 1.0);
        shape = pool(:region, Beta(2.0, 3.0))),
    south = uncertain(Gamma(2.0, 1.0); shape = pool(:region, Beta(2.0, 3.0)))))
rows = ComposedDistributions.centred_pool_rows(tree)
x = fill(0.5, ComposedDistributions.flat_dimension(tree))
nt = ComposedDistributions.unflatten(tree, x)
ComposedDistributions.pool_centred_logprior(rows, nt)
```

# See also
- [`centred_pool_rows`](@ref), [`CentredPoolPrior`](@ref)
"
function pool_centred_logprior(rows, nt)
    isempty(rows) && return 0.0
    return sum(rows) do (path, param, pool)
        logpdf(_collapse_population(pool.population, _pool_hyper(nt, pool)),
            _read_path(nt, path, param))
    end
end

@doc raw"

Deprecated alias for [`pool_centred_logprior`](@ref); kept transitionally so a
caller already qualifying it (`ComposedDistributions._pool_centred_logprior`,
or an explicit `using ComposedDistributions: _pool_centred_logprior`) keeps
working across the rename (#212 declared this `public` with the leading
underscore, which the org's naming convention reserves for internal-only
names). New code should call `pool_centred_logprior`; this alias is removed in
a future cleanup once DistributionsInference.jl's fit-protocol extension has
moved off it.

# Arguments
- `rows`: the `(path, param, pool)` triples from [`centred_pool_rows`](@ref).
- `nt`: the nested `NamedTuple` from `unflatten` at the same draw.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((north = uncertain(Gamma(2.0, 1.0);
        shape = pool(:region, Beta(2.0, 3.0))),
    south = uncertain(Gamma(2.0, 1.0); shape = pool(:region, Beta(2.0, 3.0)))))
rows = ComposedDistributions.centred_pool_rows(tree)
x = fill(0.5, ComposedDistributions.flat_dimension(tree))
nt = ComposedDistributions.unflatten(tree, x)
ComposedDistributions._pool_centred_logprior(rows, nt)
```

# See also
- [`pool_centred_logprior`](@ref)
"
const _pool_centred_logprior = pool_centred_logprior

# --- group consistency gate --------------------------------------------------
#
# Every leaf of a pooling group must declare the same population and
# parameterisation (they are one population); the params-table walk emits the
# group's hyperparameters from the first member it meets. `validate_pool_groups`
# (called once at DistributionsInference.jl's `as_logdensity` construction, not
# per gradient evaluation) rejects a mismatch eagerly.

@doc raw"

Check that every leaf of each `pool` group declares the same population
distribution and parameterisation, throwing an `ArgumentError` on a mismatch.

Called once at [`params_table`](@ref) construction time (typically at
`as_logdensity` construction), not per gradient evaluation. Reached by
qualified name from outside this package — DistributionsInference.jl's
fit-protocol extension calls this directly to gate a tree before fitting
(#212).

# Arguments
- `d`: the composed tree whose pool groups are checked.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((north = uncertain(Gamma(2.0, 1.0);
        shape = pool(:region, Beta(2.0, 3.0))),
    south = uncertain(Gamma(2.0, 1.0); shape = pool(:region, Beta(2.0, 3.0)))))
ComposedDistributions.validate_pool_groups(tree)
```

# See also
- [`validate_tree_names`](@ref)
"
function validate_pool_groups(d)
    acc = Dict{Symbol, Pool}()
    _collect_pools!(acc, d)
    return d
end

@doc raw"

Deprecated alias for [`validate_pool_groups`](@ref); kept transitionally so a
caller already qualifying it (`ComposedDistributions._validate_pool_groups`,
or an explicit `using ComposedDistributions: _validate_pool_groups`) keeps
working across the rename (this was declared `public` with the leading
underscore, which the org's naming convention reserves for internal-only
names). New code should call `validate_pool_groups`; this alias is removed in
a future cleanup once DistributionsInference.jl's fit-protocol extension has
moved off it.

# Arguments
- `d`: the composed tree whose pool groups are checked; see
  [`validate_pool_groups`](@ref).

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((north = uncertain(Gamma(2.0, 1.0);
        shape = pool(:region, Beta(2.0, 3.0))),
    south = uncertain(Gamma(2.0, 1.0); shape = pool(:region, Beta(2.0, 3.0)))))
ComposedDistributions._validate_pool_groups(tree)
```

# See also
- [`validate_pool_groups`](@ref)
"
const _validate_pool_groups = validate_pool_groups

function _collect_pools!(acc::Dict,
        d::Union{Sequential, Parallel, AbstractOneOf, Choose})
    for c in _node_children(d)
        _collect_pools!(acc, c)
    end
    return nothing
end

function _collect_pools!(acc::Dict, leaf)
    specs = _uncertain_specs(leaf)
    specs === nothing && return nothing
    for (k, v) in pairs(specs)
        v isa Pool || continue
        group = pool_group(v)
        if haskey(acc, group)
            _assert_pool_compatible(acc[group], v)
        else
            acc[group] = v
        end
    end
    return nothing
end

function _assert_pool_compatible(a::Pool, b::Pool)
    (a.population == b.population && pool_noncentred(a) == pool_noncentred(b)) ||
        throw(ArgumentError(
            "pool($(repr(pool_group(a)))) is declared inconsistently across " *
            "leaves: every member of a pooled group must share the same " *
            "population distribution and parameterisation (one population)"))
    return nothing
end

# --- namespace collision gate -------------------------------------------------
#
# `pool` groups, `shared` tags, and a tree's own top-level (root) edge names
# all end up as sibling entries in the same root-lifted NamedTuple the codec
# builds (`unflatten`'s `_root_merge_expr` in `codec_gen.jl` merges each
# family's top-level entry alongside the tree's own names). A pool group and a
# shared tag sharing a name silently clobber each other in that merge, and so
# does either family sharing a name with a root edge (see #177 and the #178
# risk list). `validate_tree_names` gates all three cross-role collisions
# once at `as_logdensity` construction time, alongside
# `validate_pool_groups`'s pool group consistency check, not per gradient
# evaluation. Reusing the same tag for a deliberate tie (`shared`/`tie`, or a
# pool group with several members) is the intended feature and is not
# flagged; only a name crossing roles is an error.

@doc raw"

Check that no `pool` group, `shared` tag, or top-level edge name in the tree
collides with one from another of those three roles, throwing an
`ArgumentError` on a collision.

The root-lifted codec merge (`unflatten`'s `_root_merge_expr`) puts pool
groups, shared tags, and root edge names into the same flat namespace; a name
crossing roles would silently clobber another entry there. Reusing the same
tag for a deliberate tie (`shared`/`tie`, or a pool group with several
members) is the intended feature and is not flagged; only a name crossing
roles is an error. Called once at [`params_table`](@ref) construction time
(typically at `as_logdensity` construction), not per gradient evaluation.
Reached by qualified name from outside this package —
DistributionsInference.jl's fit-protocol extension calls this directly to
gate a tree before fitting (#212).

# Arguments
- `d`: the composed tree whose pool/shared/root names are checked.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((north = uncertain(Gamma(2.0, 1.0);
        shape = pool(:region, Beta(2.0, 3.0))),
    south = uncertain(Gamma(2.0, 1.0); shape = pool(:region, Beta(2.0, 3.0)))))
ComposedDistributions.validate_tree_names(tree)
```

# See also
- [`validate_pool_groups`](@ref)
"
function validate_tree_names(d)
    pools = Dict{Symbol, Pool}()
    _collect_pools!(pools, d)
    shared_tags = _collect_shared(d)
    roots = Set(_root_edge_names(d))
    for group in keys(pools)
        group in roots && throw(ArgumentError(
            "$(repr(group)) is used as both a pool group and a top-level " *
            "edge name in the same tree; rename one to avoid a readback " *
            "namespace collision"))
    end
    for (tag, _) in shared_tags
        haskey(pools, tag) && throw(ArgumentError(
            "$(repr(tag)) is used as both a shared tag and a pool group in " *
            "the same tree; rename one to avoid a readback namespace " *
            "collision"))
        tag in roots && throw(ArgumentError(
            "$(repr(tag)) is used as both a shared tag and a top-level edge " *
            "name in the same tree; rename one to avoid a readback " *
            "namespace collision"))
    end
    return nothing
end

@doc raw"

Deprecated alias for [`validate_tree_names`](@ref); kept transitionally so a
caller already qualifying it (`ComposedDistributions._validate_tree_names`,
or an explicit `using ComposedDistributions: _validate_tree_names`) keeps
working across the rename (this was declared `public` with the leading
underscore, which the org's naming convention reserves for internal-only
names). New code should call `validate_tree_names`; this alias is removed in
a future cleanup once DistributionsInference.jl's fit-protocol extension has
moved off it.

# Arguments
- `d`: the composed tree whose pool/shared/root names are checked; see
  [`validate_tree_names`](@ref).

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((north = uncertain(Gamma(2.0, 1.0);
        shape = pool(:region, Beta(2.0, 3.0))),
    south = uncertain(Gamma(2.0, 1.0); shape = pool(:region, Beta(2.0, 3.0)))))
ComposedDistributions._validate_tree_names(tree)
```

# See also
- [`validate_tree_names`](@ref)
"
const _validate_tree_names = validate_tree_names

# The direct child names at the root of a composer tree, the level the
# codec's root-lift merge (see `validate_tree_names` above) lifts pool/shared
# entries onto. Every
# `AbstractComposedDistribution` subtype implements `component_names`.
_root_edge_names(d) = component_names(d)

# A `Pool` value in an `update` NamedTuple makes that parameter pooled (a spec),
# like a distribution value makes it uncertain, so an update carrying only a
# `pool(...)` spec switches `update` to *merge* mode (attach) rather than
# *strict* mode (concrete replacement). Extends the `_has_distribution_value`
# router.
_has_distribution_value(::Pool) = true

# --- the spec surface: rand, show, equality ---------------------------------

@doc "
Draw the marginal of one pooled parameter: draw from the population (its own
hyperparameters, then the parameter).

This is the marginal of a single pooled parameter in isolation; the joint
prior-predictive of a whole pooled tree, where the population is shared across
members, comes from sampling the flat priors and rebuilding with
[`update`](@ref)`(tree, `[`unflatten`](@ref)`(tree, x))`.

See also: [`Pool`](@ref), [`pool`](@ref)
"
Base.rand(rng::AbstractRNG, p::Pool) = rand(rng, p.population)

@doc "
Draw `n` independent marginal draws of one pooled parameter.

Mirrors the single-draw [`rand`](@ref)`(::Pool)`: delegates to the
population's own multi-draw `rand(rng, population, n)`. As with the
single-draw form, this is the marginal of one pooled parameter in isolation,
not the joint prior-predictive of a whole pooled tree.

See also: [`Pool`](@ref), [`pool`](@ref)
"
Base.rand(rng::AbstractRNG, p::Pool, n::Int) = rand(rng, p.population, n)
Base.rand(p::Pool, n::Int) = rand(default_rng(), p, n)

@doc "
Print a [`Pool`](@ref) spec as its constructor form.

See also: [`pool`](@ref)
"
function Base.show(io::IO, p::Pool)
    print(io, "pool(", repr(pool_group(p)), ", ",
        _leaf_label(p.population), ")")
    return nothing
end

function Base.:(==)(a::Pool, b::Pool)
    return pool_group(a) == pool_group(b) && a.population == b.population &&
           pool_noncentred(a) == pool_noncentred(b)
end
function Base.hash(p::Pool, h::UInt)
    return hash(pool_group(p), hash(p.population,
        hash(pool_noncentred(p), hash(:Pool, h))))
end
