@doc "

Resolve risks by racing hazards: the dual of `convolved`
under minimum instead of sum.

Given cause-specific delay distributions `D_1, ..., D_n`, `Compete`
represents the first-event time `T = min_k D_k` together with which cause won.
The marginal `any-event` survival is `∏_k S_k(t)` and density
`∑_j f_j(t) ∏_{k≠j} S_k(t)`, so it nests as a univariate leaf. Observing a
resolved `(cause j, time t)` scores `f_j(t) ∏_{k≠j} S_k(t)`. The winning
probability of each cause is *derived* from the hazards
(`P(cause = j) = ∫ f_j ∏_{k≠j} S_k`), *not* a free parameter — this is the key
difference from the fixed-probability mixture [`Resolve`](@ref).

Build it with the [`compete`](@ref) constructor by giving bare delays
(no branch probabilities): `compete(:death => D1, :recover => D2)`.

Three views must agree: [`rand`](@ref) draws a latent time per cause and
returns the winning cause's named event record; [`logpdf`](@ref) is the
one_of-risks likelihood (marginal or cause-resolved); and the forward
[`convolve_series`](@ref) stream is
the per-outcome sub-density, sub-stochastic (not renormalised). `Compete`
ships against plain `Distributions.ccdf`/`logccdf`, so any stock univariate
leaf races without a package-specific interface.

# Fields
- the one_of outcome names (`Symbol`s) live in the `names` type parameter
  (read with [`component_names`](@ref)).
- `delays`: tuple of the cause-specific delay distributions.

# See also
- [`compete`](@ref): the constructor (bare delays; no branch probabilities).
- [`Resolve`](@ref): the fixed-probability mixture sibling.
- `Distributions.probs`: the derived per-cause winning probabilities.
- `convolved`: the sum dual (events in series).
"
struct Compete{names, D <: Tuple} <: AbstractOneOf
    "Tuple of the cause-specific delay distributions."
    delays::D

    function Compete{names}(delays::D) where {names, D <: Tuple}
        length(names) >= 2 ||
            throw(ArgumentError("Compete needs at least two outcomes"))
        length(names) == length(delays) || throw(ArgumentError(
            "Compete names and delays must have equal length; got " *
            "$(length(names)) and $(length(delays))"))
        all(n -> n isa Symbol, names) ||
            throw(ArgumentError("each one_of outcome name must be a Symbol"))
        allunique(names) ||
            throw(ArgumentError("Compete outcome names must be unique"))
        any(_is_no_event, delays) && throw(ArgumentError(
            "a racing-hazard one_of node has no no-event branch: the " *
            "no-event probability is derived as the survival ∏ S_k(horizon). " *
            "Use the fixed-probability `Resolve` for an explicit no-event mass"))
        return new{names, D}(delays)
    end
end

# The names live in the `names` type parameter (like `NamedTuple{names}`); this
# instantiates it directly from the runtime tuple `compete`/`compose` pass, with
# no call-site change from the field-based constructor.
function Compete(names::C, delays::D) where {C <: Tuple, D <: Tuple}
    return Compete{names}(delays)
end

component_names(::Compete{names}) where {names} = names

@doc "

Build a racing-hazard [`Compete`](@ref) node from `name => delay`
outcomes (bare delays, no branch probabilities).

Each outcome is `name => delay`. The winning probability of each cause is derived
from the hazards, so no branch probability is supplied (that is what selects this
type over the fixed-probability mixture [`Resolve`](@ref)). At least two
outcomes are required.

# Examples
```@example
using ComposedDistributions, Distributions

node = Compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
probs(node)
```

# See also
- [`compete`](@ref): the racing-hazard constructor.
- [`Resolve`](@ref): the fixed-probability mixture sibling.
"
function Compete(outcomes::Pair...)
    length(outcomes) >= 2 ||
        throw(ArgumentError("Compete needs at least two outcomes"))
    # `map`, not `Tuple(gen)`: a generator-collect over a heterogeneous outcome
    # tuple lowers to `collect_to!` building a non-concrete `Array` temporary
    # Enzyme's type analysis rejects inside a differentiated build. `map` over a
    # tuple stays type-stable, returning a `Tuple` with no `Array` (see
    # `Resolve`).
    names = map(o -> o.first, outcomes)
    delays = map(o -> o.second, outcomes)
    all(_is_one_of_branch, delays) || throw(ArgumentError(
        "each racing-hazard outcome payload must be a bare delay distribution " *
        "or composer subtree (no branch probability); got a `(delay, prob)` " *
        "tuple? use `Resolve`"))
    return Compete(names, delays)
end

params(c::Compete) = map(params, c.delays)

# The marginal any-event distribution `T = min_k D_k` is univariate: its survival
# is `∏_k S_k(t)` and its support runs from the union floor (the soonest any
# cause can fire, i.e. the earliest cause lower bound) up to the largest cause
# maximum. With staggered onsets the floor must be the earliest cause, not the
# latest: a min over racing causes can fire as soon as the first one's support
# opens, and integrating the cause-resolved split from the latest floor would
# drop the mass where an early cause wins before a later cause's clock starts.
Base.minimum(c::Compete) = minimum(map(minimum, c.delays))
Base.maximum(c::Compete) = maximum(map(maximum, c.delays))
function insupport(c::Compete, x::Real)
    return minimum(c) <= x <= maximum(c)
end

# Survival of the marginal any-event time: `log ∏_k S_k(t) = Σ_k logccdf_k(t)`,
# summed directly so a `Dual`/tracked leaf param propagates (no `float` strip).
# Each term goes through `logccdf_ad_safe` so a `Gamma` survival differentiates
# w.r.t. its shape/scale (the stock `logccdf(::Gamma)` has no `Dual`-shape rule).
function _hazard_logsurvival(c::Compete, t::Real)
    return sum(ntuple(k -> logccdf_ad_safe(c.delays[k], t), _n_branches(c)))
end

# Cause-resolved log sub-density `log f_j(t) ∏_{k≠j} S_k(t) = log f_j(t) +
# Σ_{k≠j} logccdf_k(t)`, the likelihood term for an observed `(cause j, time t)`.
# Equivalently `logpdf_j(t) - logccdf_j(t) + Σ_k logccdf_k(t)` (the hazard form),
# but written as the explicit `≠ j` sum to avoid an `Inf - Inf` when a cause's
# survival underflows. AD-safe (`logccdf_ad_safe` per term; the leaf params flow
# through).
function _hazard_cause_logpdf(c::Compete, j::Int, t::Real)
    n = _n_branches(c)
    return logpdf(c.delays[j], t) +
           sum(ntuple(
        k -> k == j ? zero(logccdf_ad_safe(c.delays[k], t)) :
             logccdf_ad_safe(c.delays[k], t), n))
end

@doc "

Log density of the racing-hazard marginal any-event time `T = min_k D_k`.

The marginal density is `∑_j f_j(t) ∏_{k≠j} S_k(t)`; this is its log via the
log-sum-exp of the cause-resolved sub-densities, AD-safe (the leaf params
propagate, no `float` stripping).

See also: [`Compete`](@ref), `Distributions.probs`
"
function logpdf(c::Compete, t::Real)
    _is_nonterminal(c) && _nonterminal_marginal_error("logpdf")
    n = _n_branches(c)
    terms = ntuple(j -> _hazard_cause_logpdf(c, j, t), n)
    m = maximum(terms)
    isfinite(m) || return m
    s = zero(m)
    @inbounds for term in terms
        s += exp(term - m)
    end
    return m + log(s)
end

pdf(c::Compete, t::Real) = exp(logpdf(c, t))

# Mean / variance of the marginal any-event time `T = min_k D_k`, by AD-safe
# fixed-node Gauss-Legendre quadrature of the survival `∏ S_k` (`E[T] = ∫ S(t)
# dt` for a non-negative `T`, `E[T²] = ∫ 2t S(t) dt`). A racing node's support
# floor may be positive; the integral runs from zero over the survival so the
# `E[T] = ∫ S` identity holds for a non-negative time.
function _hazard_marginal_window(c::Compete)
    hi = _hazard_support_ceiling(c)
    isfinite(hi) && return hi
    return _hazard_quad_window(c)
end

# The shared 64-node Gauss-Legendre solver for the racing-hazard moment /
# winning-probability / cause-cdf quadratures, driven through the public
# `integrate(::GaussLegendre, f, lo, hi)` (no reach into the internal rule). The
# convolution quadrature default (192 nodes) is tuned for the batched-window
# convolution path; these smooth density-weighted integrals over one window
# converge at far fewer nodes, so a fixed 64-node solver threads through them
# for speed with no loss of accuracy.
const _PRIMARY = GaussLegendre(; n = 64)

function mean(c::Compete)
    _is_nonterminal(c) && _nonterminal_marginal_error("mean")
    hi = _hazard_marginal_window(c)
    return integrate(
        _PRIMARY, t -> exp(_hazard_logsurvival(c, t)), zero(hi), hi)
end

function var(c::Compete)
    _is_nonterminal(c) && _nonterminal_marginal_error("var")
    hi = _hazard_marginal_window(c)
    m = mean(c)
    e2 = integrate(
        _PRIMARY, t -> 2 * t * exp(_hazard_logsurvival(c, t)), zero(hi), hi)
    # Two independent quadratures can leave `e2 - m^2` a tiny negative for a
    # near-degenerate node; clamp to keep the variance non-negative.
    diff = e2 - m^2
    return max(zero(diff), diff)
end

@doc "

Survival of the racing-hazard marginal any-event time at `t`: `∏_k S_k(t)`.

See also: [`Compete`](@ref)
"
ccdf(c::Compete, t::Real) = exp(_hazard_logsurvival(c, t))
logccdf(c::Compete, t::Real) = _hazard_logsurvival(c, t)

# A racing-hazard node is a univariate leaf for the survival surface (#465 / the
# forward path): its AD-safe survival is just `_hazard_logsurvival`, so an outer
# `logccdf_ad_safe`/`ccdf_ad_safe` query (e.g. a parent racing node) recurses
# through the already-AD-safe terms rather than the stock `logccdf`.
EpiAwareADTools.logccdf_ad_safe(c::Compete, t::Real) = _hazard_logsurvival(c, t)
EpiAwareADTools.ccdf_ad_safe(c::Compete, t::Real) = exp(_hazard_logsurvival(c, t))
cdf(c::Compete, t::Real) = -expm1(_hazard_logsurvival(c, t))
function logcdf(c::Compete, t::Real)
    return log1mexp(_hazard_logsurvival(c, t))
end

# A bare `rand(c::Compete)` draws the full named event record of the cause that
# won the race, and `rand(c::Compete; outcome = true)` the compact `(name,
# time)` pair; both go through the shared `Base.rand(::AbstractRNG,
# ::AbstractOneOf; outcome)` method in `Resolve.jl`, so this file only supplies
# the `Compete` record/outcome samplers those route into.

# The scalar marginal draw of a terminal Compete (its racing any-event time
# `min_k D_k`, discarding which cause won). Used by the plain flat value path
# (`child_rand!`), where a Compete child is one value slot, and wherever the
# marginal time alone is wanted.
_one_of_marginal_rand(rng::AbstractRNG, c::Compete) = _rand_outcome(rng, c)[2]

@doc "

Score a standalone [`Compete`](@ref) cause record (the shape a bare `rand(c)`
returns): the cause-resolved sub-density `f_j(t) ∏_{k≠j} S_k(t)` of the winning
cause `j` at time `t` (the winning probability is derived from the hazards, so
there is no branch-probability term), so `logpdf(c, rand(c))` round-trips. A
column table (a `NamedTuple` of vectors) is a multi-record source, summed per
row.

See also: [`rand`](@ref), [`event_names`](@ref)
"
function logpdf(c::Compete, x::NamedTuple)
    Tables.istable(x) && return _one_of_table_logpdf(c, x)
    _is_nonterminal(c) && _nonterminal_marginal_error("logpdf")
    events = _row_event_vector_by_name(_flat_event_names(c), x)
    obs_i = _observed_one_of_outcome(c, events)
    obs_i == 0 && return 0.0
    y = events[obs_i + 1]
    o = events[1]
    # `obs_i > 0` guarantees the winning slot and the origin are observed; the
    # guard narrows the `Union{Missing, Float64}` cells to `Float64` for the
    # scorer (the branch is unreachable for a `rand`-shaped record).
    (y === missing || o === missing) && return 0.0
    return _hazard_cause_logpdf(c, obs_i, y - o)
end

# Racing-hazard form of `_rand_outcome`; documented on the internal umbrella
# comment on `function _rand_outcome end` in `Resolve.jl`. Draws a latent time
# per cause and returns the `argmin` cause name with its `min` time.
function _rand_outcome(rng::AbstractRNG, c::Compete)
    n = _n_branches(c)
    best_i = 1
    best_t = rand(rng, c.delays[1])
    @inbounds for k in 2:n
        t = rand(rng, c.delays[k])
        if t < best_t
            best_t = t
            best_i = k
        end
    end
    return component_names(c)[best_i], best_t
end
_rand_outcome(c::Compete) = _rand_outcome(default_rng(), c)

@doc "

The derived per-cause winning probabilities of a racing-hazard
[`Compete`](@ref) node: `P(cause = j) = ∫ f_j(t) ∏_{k≠j} S_k(t) dt`,
returned as a `NamedTuple` keyed by the outcome names.

This is the [`Compete`](@ref) method of `Distributions.probs`, the standard
mixture-weight reader: it gives the same per-outcome split [`Resolve`](@ref)
returns from its declared branch probabilities, but derived here from the
hazards rather than declared.

Computed by AD-safe fixed-node Gauss-Legendre quadrature of the cause-resolved
sub-density over the marginal support. The probabilities are sub-stochastic-free
(they sum to one for proper, eventually-certain causes); a node whose causes can
leave residual survival at `+∞` (a defective cause) sums to less than one, the
deficit being the never-resolved mass.

!!! warning \"Accuracy for a wide quadrature window\"
    The quadrature runs on a fixed 64-node rule over whatever window the
    causes resolve to (see `_hazard_quad_window`). For a genuinely
    heavy-tailed cause, or an ordinary composite/convolved cause whose
    moment-based fallback window is itself large, that window can be wide
    enough that the returned split is badly wrong -- not merely imprecise,
    off by 100% and potentially naming the wrong winning cause -- while
    still looking like a valid split (finite, in `[0, 1]`, summing to
    `<= 1`). Never throws or returns `NaN` for such a cause, but does not
    yet guarantee accuracy either; see #294.

# Arguments
- `c`: the [`Compete`](@ref) node whose derived per-cause winning split to read.

# Examples
```@example
using ComposedDistributions, Distributions

node = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
probs(node)
```

See also: [`Compete`](@ref), [`occurrence_probability`](@ref)
"
function probs(c::Compete)
    _is_nonterminal(c) && _nonterminal_marginal_error("probs")
    lo = _hazard_support_floor(c)
    hi_raw = _hazard_support_ceiling(c)
    # Bind `hi` unconditionally (a ternary, not `isfinite(hi) || (hi = ...)`): the
    # short-circuit-assignment form leaves `hi` only conditionally assigned, and the
    # `ntuple` closure below that captures it then trips JET's `local variable hi is
    # not defined` (the closure cannot prove the assignment branch ran). An
    # unbounded cause support falls back to a finite high-quantile quad window.
    hi = isfinite(hi_raw) ? hi_raw : lo + _hazard_quad_window(c)
    n = _n_branches(c)
    winning = ntuple(n) do j
        integrate(_PRIMARY, t -> exp(_hazard_cause_logpdf(c, j, t)), lo, hi)
    end
    # The winning split is mathematically sub-stochastic: it sums to
    # `1 - ∏ S_k(∞) ≤ 1`, the deficit being a defective cause's never-resolved
    # mass. Gauss-Legendre quadrature can overshoot slightly above one for
    # proper (eventually-certain) causes, so rescale down to a valid probability
    # vector in that case, leaving a genuine sub-one defective deficit intact.
    total = sum(winning)
    winning = total > one(total) ? winning ./ total : winning
    return NamedTuple{component_names(c)}(winning)
end

# Evaluate `f()`, returning `default` instead of propagating an exception. Used
# to make the shared quadrature window construction robust to a cause whose
# `minimum`/`maximum`/`quantile` is unavailable (e.g. a composite/convolved
# cause with no such method, or a defective node) rather than letting one
# awkward cause raise a `MethodError` for the whole node.
function _try_or(f, default)
    try
        return f()
    catch
        return default
    end
end

# Component-robust support floor/ceiling for the shared quadrature domain: a
# per-cause `minimum`/`maximum` that throws (a composite/convolved cause with
# no such method) contributes `0`/`Inf` respectively instead of stopping the
# whole node's `probs`/`mean`/`var` from computing over the other causes.
function _hazard_support_floor(c::Compete)
    return minimum(map(d -> _try_or(() -> float(minimum(d)), 0.0), c.delays))
end
function _hazard_support_ceiling(c::Compete)
    return maximum(map(d -> _try_or(() -> float(maximum(d)), Inf), c.delays))
end

# A finite quadrature window for a cause with unbounded support: a high
# quantile of the soonest-firing marginal, so the tail beyond it carries
# negligible mass for the winning-probability integral. Falls back to a
# moment-based window (`mean + 10*std`) when a cause has no `quantile` method
# (e.g. a composite/convolved cause) or its `0.9999` quantile is itself
# non-finite; a cause with neither a usable `quantile` nor usable moments
# (e.g. a defective node, whose true upper quantile is infinite) is ignored
# rather than letting it poison the shared window. Errors only if no cause
# has any usable bound.
#
# This only guards against a crash (a thrown MethodError or a non-finite
# window). It does *not* guard the shared quadrature's accuracy: `probs`'s
# fixed 64-node Gauss-Legendre rule integrates over whatever window this
# returns, and for a wide window (a genuinely heavy-tailed cause, or an
# ordinary composite/convolved cause whose `mean + 10*std` fallback is
# itself large) the 64-node answer can be badly wrong -- not merely
# imprecise, off by 100% including the wrong cause winning -- while still
# looking like a valid split (finite, in [0, 1], summing to <= 1). See #294
# for worked examples and the tracked fix.
function _hazard_quad_window(c::Compete)
    windows = filter(isfinite, map(_component_quad_window, c.delays))
    isempty(windows) && throw(ArgumentError(
        "no cause of this Compete node has a usable quadrature window " *
        "(a finite `quantile(cause, 0.9999)` and a finite `mean`/`std` are " *
        "both unavailable for every cause); provide at least one cause " *
        "with one of these"))
    return maximum(windows)
end

# The quadrature window bound for one cause: its own `0.9999` quantile when
# finite, else a moment-based `mean + 10*std` window, else `Inf` (no usable
# bound; the caller ignores this cause).
function _component_quad_window(d)
    q = _try_or(() -> float(quantile(d, 0.9999)), Inf)
    isfinite(q) && return q
    return _try_or(() -> float(mean(d) + 10 * std(d)), Inf)
end

@doc "

The probability that any (non-no-event) outcome occurs for a one_of node.

For a racing-hazard [`Compete`](@ref) node `occurrence_probability` is the
sum of the derived per-cause split `Distributions.probs` returns (one for
proper, eventually-certain causes; the resolved mass for a defective node).
For a fixed-probability [`Resolve`](@ref) node it is one minus the no-event
branch mass.

# Arguments
- `c`: the [`Compete`](@ref) node whose any-event probability to read.

# Examples
```@example
using ComposedDistributions, Distributions

node = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
occurrence_probability(node)
```

# See also
- `Distributions.probs`: the per-outcome winning split this sums.
"
function occurrence_probability(c::Compete)
    return sum(values(probs(c)))
end

# ----------------------------------------------------------------------------
# Cause-resolved sub-density leaf (for the forward convolve stream)
# ----------------------------------------------------------------------------
#
# The forward `convolve_series(stack, series)` per-outcome stream of a
# racing-hazard node is `series ⊛ pmf(f_j ∏_{k≠j} S_k)`, sub-stochastic: each
# outcome's mass equals its derived winning probability, the deficit being the
# one_of fraction. `_HazardCauseDelay` is the cause-resolved sub-density of one
# cause `j` of a [`Compete`](@ref) node as a (defective) univariate
# distribution: its `pdf` is `f_j(t) ∏_{k≠j} S_k(t)` and its `cdf` is that
# sub-density integrated from the support floor to `t` (the cause-`j` winning
# probability accumulated by time `t`). The convolve layer discretises it through
# `interval_censored`, so the resulting masses are exactly the sub-stochastic
# per-outcome stream (no renormalise). AD-safe: the leaf params flow through the
# log-sum / quadrature.
struct _HazardCauseDelay{H <: Compete} <: UnivariateDistribution{Continuous}
    node::H
    cause::Int
end

Base.minimum(d::_HazardCauseDelay) = minimum(d.node)
Base.maximum(d::_HazardCauseDelay) = maximum(d.node)
function insupport(d::_HazardCauseDelay, x::Real)
    return insupport(d.node, x)
end
function logpdf(d::_HazardCauseDelay, t::Real)
    return _hazard_cause_logpdf(d.node, d.cause, t)
end
pdf(d::_HazardCauseDelay, t::Real) = exp(logpdf(d, t))

# The cause-`j` winning probability accumulated by `t`: `∫_lo^t f_j ∏_{k≠j} S_k`.
# Fixed-node Gauss-Legendre over the support floor to `t` (AD-safe; the leaf
# params flow through the integrand). A `t` at or below the floor has zero mass.
function cdf(d::_HazardCauseDelay, t::Real)
    lo = float(minimum(d.node))
    t <= lo && return zero(float(t))
    return integrate(
        _PRIMARY, u -> exp(_hazard_cause_logpdf(d.node, d.cause, u)),
        lo, float(t))
end

@doc "

Print a [`Compete`](@ref) node as a recursive indented tree.

See also: [`Compete`](@ref)
"
function Base.show(io::IO, ::MIME"text/plain", c::Compete)
    _show_composer_tree(io, c)
    return nothing
end

function Base.show(io::IO, c::Compete)
    names = component_names(c)
    parts = ["$(names[k])~$(c.delays[k])" for k in 1:_n_branches(c)]
    print(io, "Compete(", join(parts, " | "), ")")
    return nothing
end
