# --- AbstractOneOf: the shared supertype for the one_of-outcome composers ---
@doc raw"

Shared supertype of the one_of-outcome composers.

The two one_of-outcome nodes — the fixed-probability mixture [`Resolve`](@ref)
(cause and timing independent) and the racing-hazard [`Compete`](@ref) (the
winning probability derived from the hazards, timing coupled) — subtype
`AbstractOneOf`. The tree walkers dispatch on it wherever the behaviour is shared
(one event slot per outcome, the shared origin, the per-outcome `rand`) and on
the concrete type only where the scoring arithmetic differs.

`AbstractOneOf` is the univariate arm of the composer hierarchy: it subtypes
[`AbstractComposedDistribution`](@ref)`{Univariate, Continuous}`, so it stays a
`UnivariateDistribution` while sharing the composed supertype the multivariate
`Sequential` / `Parallel` / `Choose` also sit under.

# Examples
```@example
using ComposedDistributions, Distributions

r = resolve(:death => (Gamma(1.5, 1.0), 0.3), :disch => Gamma(2.0, 1.5))
r isa ComposedDistributions.AbstractOneOf
```

# See also
- [`Resolve`](@ref), [`Compete`](@ref): the concrete one_of nodes.
"
abstract type AbstractOneOf <:
              AbstractComposedDistribution{Univariate, Continuous} end

# Outcome names, one per one_of outcome. Both concrete types store `names`.
component_names(c::AbstractOneOf) = c.names
_n_branches(c::AbstractOneOf) = length(c.names)

@doc "

Marker distribution for a NO-EVENT (absorbing) outcome of a [`resolve`](@ref)
node: the outcome where *nothing happens* and no event time is written.

A `none => (NoEvent(), q)` branch carries no delay; its mass `q` is the
probability that no event occurs. On `rand` a no-event win yields `missing` (no
time recorded). On `logpdf` an OBSERVED non-occurrence (an explicit `no event by
the horizon` record) scores the survival term `log q` (mixture) or the
racing-hazard survival `∏ S_k`; a latent non-occurrence (a record whose no-event
slot is simply missing) contributes no one_of term.

`NoEvent` is a degenerate placeholder, not a sampling distribution: it has no
support and errors if asked for a density or a draw. It exists only to MARK the
absorbing branch so the one_of node carries its mass `q`.

# See also
- [`resolve`](@ref): the fixed-probability constructor.
- [`Resolve`](@ref): the mixture one_of node.
"
struct NoEvent <: UnivariateDistribution{Continuous} end

# `NoEvent` is a marker, not a sampleable density: every density / draw / support
# query errors with a clear message so a stray use surfaces immediately rather
# than silently scoring a degenerate term. The one_of scorers special-case the
# no-event branch (its mass is a survival term, never `logpdf(NoEvent(), .)`).
function _no_event_error()
    throw(ArgumentError(
        "NoEvent is a marker for a one_of no-event branch and has no density or " *
        "support; its mass is scored as a survival term by the one_of node"))
end
logpdf(::NoEvent, ::Real) = _no_event_error()
pdf(::NoEvent, ::Real) = _no_event_error()
cdf(::NoEvent, ::Real) = _no_event_error()
Base.minimum(::NoEvent) = _no_event_error()
Base.maximum(::NoEvent) = _no_event_error()
Base.rand(::AbstractRNG, ::NoEvent) = _no_event_error()
# An empty param tuple (no free parameters): the marker carries no delay, so the
# tree's `_param_eltype` / `_tree_core_eltype` promotions skip it cleanly.
params(::NoEvent) = ()

# Whether an outcome's delay payload is the no-event marker. Used by the scorers
# and the tree walkers to skip a no-event slot's density and treat its mass as a
# survival term.
_is_no_event(::NoEvent) = true
_is_no_event(::Any) = false

# Whether a one_of node carries a no-event branch (one of its delays is the
# marker). Such a node is a defective marginal (the observed-time mass is `< 1`),
# so its scalar `logpdf` / `mean` / `as_mixture` error: it is multivariate /
# sub-stochastic, scored only through the event-vector path.
_has_no_event(c::AbstractOneOf) = any(_is_no_event, c.delays)

# `_is_composer_outcome` / `_is_nonterminal` (the non-terminal one_of predicate,
# #466 Feature 3) reference `Sequential` / `Parallel` / `Choose`, which are loaded
# after this file, so they are defined in `nesting.jl` (loaded once all composer
# types exist) rather than here.

@doc "

Resolve outcomes composed from any univariate distributions: exactly one of
several outcomes occurs, governed by branch probabilities summing to one.

`Resolve` names each one_of outcome, its delay distribution, and the branch
probability of that outcome. It lowers to a `Distributions.MixtureModel`
(see [`as_mixture`](@ref)) over the outcome delays weighted by the branch
probabilities, so the realisation is a single time and the type is univariate.
A death-versus-recovery competition makes the death branch probability the
case-fatality ratio.

Being univariate, a `Resolve` nests as a child of [`Sequential`](@ref) or
[`Parallel`](@ref). This is the plain generic composition; per-record outcome
selection and censoring are not part of this type.

The branch probabilities are ordinarily fixed structure. To ESTIMATE them,
attach a simplex-valued `Distributions.Dirichlet` prior with
[`update`](@ref)`(node, (branch_probs = Dirichlet(α),))`: the `Dirichlet` is
what you write, but the codec estimates the node through the `Dirichlet`'s K-1
stick-breaking coordinates (`:stick_1 … :stick_{K-1}`, each a `Beta`, so every
draw lands on the simplex and the gradient is well-defined), and the
probabilities are recovered from any draw (via [`update`](@ref) /
`Distributions.probs`). See [`update`](@ref) for the full story.

# Fields
- `names`: tuple of the one_of outcome names (`Symbol`s).
- `delays`: tuple of the one_of outcome delay distributions.
- `branch_probs`: tuple of the branch probabilities, summing to one.
- `branch_prob_prior`: the attached `Dirichlet` prior when the branch
  probabilities are uncertain, else `nothing` (fixed structure).

# See also
- [`as_mixture`](@ref): the `MixtureModel` lowering
- [`update`](@ref): attach a `Dirichlet` to estimate the branch probabilities
- [`Sequential`](@ref): a chain of additive steps
- [`Parallel`](@ref): independent branches
"
struct Resolve{C <: Tuple, D <: Tuple, P <: Tuple, S} <: AbstractOneOf
    "Tuple of the one_of outcome names (`Symbol`s)."
    names::C
    "Tuple of the one_of outcome delay distributions."
    delays::D
    "Tuple of the branch probabilities, summing to one."
    branch_probs::P
    "The attached simplex-valued prior over the branch probabilities (a
    `Distributions.Dirichlet`), or `nothing` when the probabilities are fixed
    structure. When present the branch probabilities are ESTIMATED through the
    stick-breaking codec: the user writes the `Dirichlet`, K-1 stick
    coordinates are what the sampler estimates, and the probabilities are
    recovered from any draw (see [`update`](@ref))."
    branch_prob_prior::S

    # Validate the structural invariants in the inner constructor so every
    # construction path (the `Pair...` outer constructor, equality round-trips,
    # `update` value- and node-edits, and direct struct calls) is checked,
    # rather than silently building a malformed node whose failure only surfaces
    # later as a confusing `DomainError` from `Categorical` inside `as_mixture`.
    #
    # The bounds (each prob in `[0, 1]`) and structure (at least two outcomes;
    # names, delays and branch_probs of equal length) hold on every path,
    # including the DynamicPPL extension that rebuilds a `Resolve` from branch
    # probabilities sampled independently from priors. Those sampled probs are
    # in `[0, 1]` but need not sum to one (the AD-safe `_one_of_logmix`
    # scorer handles an unnormalised weight set), so the sum-to-one requirement
    # is enforced at the user-facing `Pair...` constructor and at `as_mixture`
    # (which does need a normalised `Categorical`), not here.
    function Resolve(names::C, delays::D, branch_probs::P,
            branch_prob_prior::S) where {C <: Tuple, D <: Tuple, P <: Tuple, S}
        length(names) >= 2 ||
            throw(ArgumentError("Resolve needs at least two outcomes"))
        (length(names) == length(delays) == length(branch_probs)) ||
            throw(ArgumentError(
                "Resolve names, delays and branch_probs must have equal " *
                "length; got $(length(names)), $(length(delays)), " *
                "$(length(branch_probs))"))
        allunique(names) ||
            throw(ArgumentError("Resolve outcome names must be unique"))
        _validate_branch_prob_bounds(branch_probs)
        _validate_branch_prob_prior(branch_prob_prior, length(names))
        return new{C, D, P, S}(names, delays, branch_probs, branch_prob_prior)
    end
end

# A `Resolve` with no attached prior is fixed structure; this three-argument
# form keeps every existing construction path (the `Pair...` constructor,
# `_rebuild`, `prune`, equality round-trips) building a fixed node unchanged.
function Resolve(names::Tuple, delays::Tuple, branch_probs::Tuple)
    return Resolve(names, delays, branch_probs, nothing)
end

# A fixed node has no branch-probability prior. An attached prior must be a
# `Distributions.Dirichlet` over the `k` outcomes (one weight per outcome); it
# is decomposed into K-1 stick-breaking `Beta`s by the codec.
_validate_branch_prob_prior(::Nothing, ::Int) = nothing
function _validate_branch_prob_prior(prior, k::Int)
    prior isa Distributions.Dirichlet || throw(ArgumentError(
        "the branch-probability prior must be a `Dirichlet` over the $k " *
        "outcomes; got a $(typeof(prior))"))
    length(prior) == k || throw(ArgumentError(
        "the branch-probability `Dirichlet` prior must have one weight per " *
        "outcome (length $k); got length $(length(prior))"))
    return nothing
end

# --- stick-breaking codec for an uncertain branch-probability simplex --------
#
# A `Dirichlet(α)` over the K outcomes is estimated through its stick-breaking
# reparameterisation: K-1 coordinates `v_k ∈ (0, 1)`, independently
# `v_k ~ Beta(α_k, Σ_{j>k} α_j)`, mapped to the K-simplex by `p_1 = v_1`,
# `p_k = v_k ∏_{j<k}(1 - v_j)`, `p_K = ∏_j (1 - v_j)`. The product of those
# `Beta`s over `v` equals the `Dirichlet` over `p` exactly, so scoring each
# `v_k` with its `Beta` (the codec's per-row univariate scoring) reproduces the
# `Dirichlet` with no separate Jacobian. Every `v ∈ (0, 1)^{K-1}` maps to a
# valid simplex, so the map is smooth and always in-support (AD-safe on every
# backend) and needs only K-1 free dimensions (the simplex dimension).

# The stick-coordinate parameter name for coordinate `k` (`:stick_k`), the label
# a fitted chain and `params_table` carry for the estimated branch-prob simplex.
_stick_name(k::Int) = Symbol(:stick_, k)

# The K-1 stick-coordinate names for a `k`-outcome node.
_stick_param_names(k::Int) = ntuple(_stick_name, k - 1)

# The K-1 stick-breaking `Beta`s of a `Dirichlet` prior, in outcome order.
function _dirichlet_stick_betas(prior::Distributions.Dirichlet)
    a = prior.alpha
    K = length(a)
    return ntuple(K - 1) do k
        Distributions.Beta(a[k], sum(@view a[(k + 1):K]))
    end
end

# Stick-breaking map: K-1 coordinates `v` in (0, 1) -> a K-simplex, returned as a
# tuple. Preserves the element type (an AD `Dual` flows through), so the
# reconstructed probabilities differentiate w.r.t. the estimated coordinates.
function _stick_to_simplex(v)
    M = length(v)                      # K - 1
    T = eltype(v)
    rem = Vector{T}(undef, M + 1)      # rem[k] = ∏_{j<k}(1 - v_j)
    rem[1] = one(T)
    @inbounds for k in 1:M
        rem[k + 1] = rem[k] * (one(T) - v[k])
    end
    return ntuple(k -> k <= M ? v[k] * rem[k] : rem[M + 1], M + 1)
end

# Inverse stick-breaking: a K-simplex `p` -> its K-1 coordinates `v`. Used for
# the current probabilities' `value` column; not on a differentiated path.
function _simplex_to_stick(p)
    K = length(p)
    T = float(eltype(p))
    v = Vector{T}(undef, K - 1)
    remaining = one(T)
    @inbounds for k in 1:(K - 1)
        v[k] = p[k] / remaining
        remaining -= p[k]
    end
    return v
end

@doc "

Build a [`Resolve`](@ref) node from `name => (delay, branch_prob)` outcomes.

Each outcome is `name => (delay, branch_prob)`: the outcome name (a `Symbol`),
its delay distribution, and the probability that this outcome occurs. The branch
probabilities must each lie in ``[0, 1]`` and sum to one, and at least two
outcomes are required.

# Examples
```@example
using ComposedDistributions, Distributions

cfr = 0.3
node = Resolve(:death => (Gamma(1.5, 1.0), cfr),
    :disch => (Gamma(2.0, 1.5), 1 - cfr))
mean(node)
```

# See also
- [`Resolve`](@ref): the composer type
- [`as_mixture`](@ref): the `MixtureModel` lowering
"
function Resolve(outcomes::Pair...)
    length(outcomes) >= 2 ||
        throw(ArgumentError("Resolve needs at least two outcomes"))
    names = Tuple(o.first for o in outcomes)
    payloads = Tuple(o.second for o in outcomes)
    all(n -> n isa Symbol, names) ||
        throw(ArgumentError("each one_of outcome name must be a Symbol"))
    delays = Tuple(_one_of_delay(p) for p in payloads)
    branch_probs = Tuple(_one_of_prob(p) for p in payloads)
    # The inner constructor validates the bounds and structure; the user-facing
    # constructor additionally requires the probabilities to sum to one.
    _validate_branch_probs_sum(branch_probs)
    return Resolve(names, delays, branch_probs)
end

@doc "

Build a fixed-probability [`Resolve`](@ref) node from
`name => (delay, branch_prob)` outcomes: exactly one outcome RESOLVES, with cause
independent of timing.

Each outcome is `name => (delay, branch_prob)`; the branch probabilities must
each lie in ``[0, 1]`` and sum to one, and at least two outcomes are required.

The LAST outcome's probability may be OMITTED (a bare `name => delay`): it then
takes the residual `1 - sum(of the others)`, so a probability that is fully
determined by the rest need not be written out (and cannot disagree with them).
The leading probabilities must sum to at most one. Omitting any outcome but the
last, or more than one, is rejected. To omit EVERY probability (a racing-hazard
node where the winning probability is derived from the hazards) use
[`compete`](@ref) instead.

# Arguments
- `outcomes`: two or more `name => (delay, branch_prob)` pairs, each giving the
  outcome name (a `Symbol`), its delay distribution, and the probability that
  the outcome occurs. The last pair's probability may be omitted (a bare
  `name => delay`), taking the residual `1 - sum(of the others)`.

# Examples
```@example
using ComposedDistributions, Distributions

cfr = 0.3
node = resolve(:death => (Gamma(1.5, 1.0), cfr),
    :disch => (Gamma(2.0, 1.5), 1 - cfr))
mean(node)
```

```@example
using ComposedDistributions, Distributions

# The discharge probability is the residual `1 - cfr`, so it is omitted.
cfr = 0.3
node = resolve(:death => (Gamma(1.5, 1.0), cfr),
    :disch => Gamma(2.0, 1.5))
mean(node)
```

# See also
- [`Resolve`](@ref): the composer type
- [`compete`](@ref): the racing-hazard sibling constructor (bare delays)
- [`as_mixture`](@ref): the `MixtureModel` lowering
- [`compose`](@ref): the front-end that nests a `Resolve` as a branch
- [`Sequential`](@ref), [`Parallel`](@ref): the sibling composers
"
function resolve(outcomes::Pair...)
    length(outcomes) >= 2 ||
        throw(ArgumentError("resolve needs at least two outcomes"))
    payloads = Tuple(o.second for o in outcomes)
    # `resolve` builds the fixed-probability mixture `Resolve` (cause and timing
    # independent): every outcome carries a `(delay, branch_prob)` pair, or every
    # outcome but the last does and the last is a bare delay taking the residual
    # `1 - sum(of the others)`. all-bare delays are the racing-hazard form and
    # belong to `compete`; a misplaced omission is rejected as an unclear mix.
    if all(_is_prob_payload, payloads)
        return Resolve(outcomes...)
    elseif _is_residual_shape(payloads)
        return Resolve(_fill_residual_outcome(outcomes)...)
    elseif all(_is_bare_payload, payloads)
        throw(ArgumentError(
            "`resolve` builds the fixed-probability split and needs a branch " *
            "probability on each outcome (`name => (delay, prob)`); every " *
            "outcome here is a bare delay. For the racing-hazard node (winning " *
            "probability derived from the hazards) use `compete` instead"))
    end
    throw(ArgumentError(
        "`resolve` outcomes must ALL carry a branch probability " *
        "(`name => (delay, prob)`), or carry one on every outcome BUT the " *
        "last (`name => delay`), which then takes the residual " *
        "`1 - sum(others)`; the given mix is unclear. For an all-bare-delay " *
        "racing-hazard node use `compete`"))
end

@doc "

Build a racing-hazard [`Compete`](@ref) node from bare `name => delay`
outcomes: the cause-specific delays RACE, the first wins, and the winning
probability of each cause is DERIVED from the hazards (cause coupled to timing).

Each outcome is `name => delay` (a bare delay, NO branch probability). At least
two outcomes are required. To give an explicit fixed probability per outcome (a
mixture where cause is independent of timing) use [`resolve`](@ref) instead.

# Arguments
- `outcomes`: two or more bare `name => delay` pairs, each giving the outcome
  name (a `Symbol`) and its cause-specific delay distribution (no branch
  probability).

# Examples
```@example
using ComposedDistributions, Distributions

node = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
probs(node)
```

# See also
- [`Compete`](@ref): the composer type
- [`resolve`](@ref): the fixed-probability sibling constructor (`(delay, prob)`)
- `Distributions.probs`: the derived per-cause winning probabilities
- [`compose`](@ref): the front-end that nests the node as a branch
"
function compete(outcomes::Pair...)
    length(outcomes) >= 2 ||
        throw(ArgumentError("compete needs at least two outcomes"))
    payloads = Tuple(o.second for o in outcomes)
    # `compete` builds the racing-hazard `Compete`: every outcome is a
    # bare delay (no branch probability), the winning probability being derived
    # from the hazards. A `(delay, prob)` pair anywhere is the fixed-probability
    # mixture and belongs to `resolve`.
    if all(_is_bare_payload, payloads)
        return Compete(outcomes...)
    end
    throw(ArgumentError(
        "`compete` builds the racing-hazard node and needs a bare delay per " *
        "outcome (`name => delay`, no branch probability); a `(delay, prob)` " *
        "pair was given. For the fixed-probability split (an explicit per-" *
        "outcome probability) use `resolve` instead"))
end

# The residual mixture shape: every outcome but the last carries a `(delay,
# prob)` pair and the last is a bare delay, so the last outcome's probability is
# the residual `1 - sum(of the others)`. A bare delay anywhere but the last (or
# more than one bare delay) is not the residual form: it falls through to the
# ambiguous-mix error so a misplaced omission is rejected clearly rather than
# silently treated as a hazard or a residual. At least two outcomes hold by the
# caller's guard, so the leading prob-payload set is non-empty.
function _is_residual_shape(payloads::Tuple)
    n = length(payloads)
    all(_is_prob_payload, Base.front(payloads)) &&
        _is_bare_payload(payloads[n])
end

# Resolve a residual-shape outcome list into the all-explicit `(delay, prob)`
# form: keep every leading `(delay, prob)` outcome and give the last (bare
# delay) outcome the residual probability `1 - sum(of the others)`. The residual
# rides the leading probabilities' element type, so a `logistic(Xβ)`/sampled
# leading prob keeps the last outcome a differentiable function of the others
# (the residual flows the same `Dual`/tracked type). The leading probabilities
# are bounds- and sum-checked (each in `[0, 1]`, summing to `<= 1`) so the
# residual is a valid probability; an over-one leading set errors clearly here
# rather than yielding a negative residual the downstream bounds check catches.
function _fill_residual_outcome(outcomes::Tuple)
    n = length(outcomes)
    leading = Base.front(outcomes)
    last_pair = outcomes[n]
    leading_probs = Tuple(_one_of_prob(o.second) for o in leading)
    _validate_branch_prob_bounds(leading_probs)
    total = sum(leading_probs)
    (total <= 1 + 1e-6) || throw(ArgumentError(
        "the leading one_of branch probabilities sum to $total, " *
        "exceeding one, so the residual last-outcome probability `1 - sum` " *
        "is negative; the probabilities must leave non-negative residual mass"))
    residual = one(total) - total
    filled_last = last_pair.first => (last_pair.second, residual)
    return (leading..., filled_last)
end

# A `(delay, branch_prob)` mixture payload vs a bare-delay hazard payload. A
# one_of outcome delay may be a plain univariate leaf or a composer subtree
# (`Sequential` / `Parallel` / `Choose` / nested `Resolve`, the non-terminal
# branch of #466 Feature 3); `_is_one_of_branch` (defined in `nesting.jl`, once
# those types exist) is the runtime admit-check, so the predicates stay value-based
# rather than referencing the later-loaded composer types in their signatures. A
# `NoEvent` marker is admitted only in the mixture (it carries the no-event mass
# `q`); a bare `NoEvent` in a hazard node has no hazard and is rejected by the
# `Compete` constructor.
_is_prob_payload(p::Tuple{Any, <:Real}) = _is_one_of_branch(p[1])
_is_prob_payload(::Any) = false
_is_bare_payload(x) = _is_one_of_branch(x)

function _one_of_delay(payload::Tuple{Any, <:Real})
    _is_one_of_branch(payload[1]) || throw(ArgumentError(
        "each one_of outcome payload must be a `(delay, branch_prob)` tuple " *
        "whose delay is a univariate distribution or a composer subtree; got " *
        "$(typeof(payload[1]))"))
    return payload[1]
end
function _one_of_delay(payload)
    throw(ArgumentError(
        "each one_of outcome payload must be a `(delay, branch_prob)` " *
        "tuple; got $(typeof(payload))"))
end

_one_of_prob(payload::Tuple{Any, <:Real}) = payload[2]

# Each branch probability must lie in `[0, 1]`. The bounds carry a small
# tolerance so a saturating covariate prob (`logistic(Xβ)` evaluating to a hair
# past 0 or 1 under AD/sampling) is accepted rather than spuriously rejected.
# Comparisons are value-based, so an AD `Dual`/tracked `branch_probs` is compared
# on its value without being stripped of its derivative information.
function _validate_branch_prob_bounds(branch_probs::Tuple)
    tol = 1e-6
    for p in branch_probs
        (p >= -tol && p <= 1 + tol) ||
            throw(ArgumentError(
                "each branch probability must lie in [0, 1]; got $p"))
    end
    return nothing
end

# The branch probabilities must additionally sum to one. Applied on the
# user-facing `Pair...` constructor and at `as_mixture` (which lowers to a
# `Categorical` and so needs a normalised weight set), but not in the inner
# constructor, where a prior-sampled (unnormalised) weight set is legitimate.
# The `isapprox` sum check is value-based, so an AD `Dual` is not stripped.
function _validate_branch_probs_sum(branch_probs::Tuple)
    total = sum(branch_probs)
    isapprox(total, 1; atol = 1e-6) ||
        throw(ArgumentError(
            "one_of branch probabilities sum to $total, not one"))
    return nothing
end

# ---------------------------------------------------------------------------
# Shared self-dispatch scoring
# ---------------------------------------------------------------------------
#
# The Turing-free arithmetic of the `Resolve` self-dispatch (decision 2),
# factored here so both the top-level `composed_distribution_model(d::Resolve,
# row)` (the DynamicPPL extension) and the nested tree scorer use one
# implementation rather than two parallel copies. Each helper consumes already-
# resolved inputs (the observed-outcome index or `nothing`, the gap from the
# node's anchor, and the per-record branch probabilities) and returns a plain
# log density, so the extension supplies the row plumbing and these supply the
# scoring. The probabilities keep their (possibly AD `Dual`) element type so a
# covariate CFR `logistic(Xβ)` differentiates through the node.

# Per-record branch probabilities must each lie in `[0, 1]` and sum to one.
# Delegates to the same bounds and sum validators the stored `branch_probs` use
# (`_validate_branch_prob_bounds` then `_validate_branch_probs_sum`) so the
# per-record override and the node share one validation path; the tolerance and
# AD-`Dual`-preserving, value-based comparisons live in those helpers.
function _validate_record_probs(probs)
    _validate_branch_prob_bounds(probs)
    _validate_branch_probs_sum(probs)
    return nothing
end

# Coerce a per-record branch-probability override to the node's outcome order. A
# `NamedTuple` must name exactly the outcomes; a scalar is the first outcome's
# probability of a two-outcome node (`(p, 1 - p)`). The element type is preserved
# so a `logistic(Xβ)` `Dual` flows through.
function _coerce_branch_probs(c::Resolve, bp::NamedTuple)
    Set(keys(bp)) == Set(c.names) || throw(ArgumentError(
        "per-record branch_probs must name exactly the outcomes " *
        "$(collect(c.names)); got $(collect(keys(bp)))"))
    probs = map(n -> bp[n], c.names)
    _validate_record_probs(probs)
    return probs
end

function _coerce_branch_probs(c::Resolve, p::Real)
    length(c.names) == 2 || throw(ArgumentError(
        "a scalar per-record branch_probs is only defined for a two-outcome " *
        "Resolve (the first outcome's probability); node has " *
        "$(length(c.names)) outcomes, pass a NamedTuple instead"))
    probs = (p, one(p) - p)
    _validate_record_probs(probs)
    return probs
end

# Condition on the observed outcome `i`: `log(p[i]) + logpdf(delay[i], gap)`,
# the observed branch's own (censored) logpdf at its gap from the node's anchor.
# `delay` is optionally pre-censored/truncated by the caller (the same delay the
# top-level path scores), so this is purely the conditioned-branch arithmetic.
function _one_of_condition_logpdf(probs, delay, gap, i::Int)
    return log(probs[i]) + logpdf(delay, gap)
end

# Marginalise an unknown outcome at a resolution time `t`: the branch-prob-
# weighted mixture log-density `log Σ_i p_i f_i(t)` via the log-sum-exp of
# `log p_i + logpdf(delay_i, t)`. `delays` may be pre-censored/truncated by the
# caller (matching the conditioned path's per-record horizon), else the node's
# delays.
#
# Computed directly, not via `MixtureModel(delays, float.(probs))`: `float.`
# strips an AD `Dual`/tracked type from the probabilities, breaking the gradient
# through a sampled / `logistic(Xβ)` branch probability on the marginalised path.
# The explicit reduction keeps the probabilities' element type, so a `Dual`
# propagates exactly as it does on the conditioned path. A zero probability
# contributes no term (its `log` is `-Inf`); an all-zero set returns `-Inf`. A
# no-event branch carries no density at a finite observed time `t` (its mass is
# the survival, not a density term), so it contributes `-Inf` and is skipped
# here, leaving the marginal as the sum over the real outcomes.
function _one_of_logmix(probs, delays, t)
    n = length(probs)
    terms = ntuple(
        i -> _is_no_event(delays[i]) ?
             oftype(log(probs[i]), -Inf) :
             log(probs[i]) + logpdf(delays[i], t), n)
    m = maximum(terms)
    isfinite(m) || return m
    s = zero(m)
    @inbounds for term in terms
        s += exp(term - m)
    end
    return m + log(s)
end

@doc "

Lower a [`Resolve`](@ref) node to a `Distributions.MixtureModel`.

Returns the `MixtureModel` over the outcome delays weighted by the branch
probabilities, the marginal time-to-resolution regardless of which outcome
occurs.

# Examples
```@example
using ComposedDistributions, Distributions

node = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
    :disch => (Gamma(2.0, 1.5), 0.7))
as_mixture(node)
```

# See also
- [`Resolve`](@ref): the composer type
"
function as_mixture(c::Resolve)
    # A non-terminal node (a composer-valued outcome) is multivariate: no single
    # marginal time-to-resolution exists, so the scalar lowering is rejected.
    _is_nonterminal(c) && _nonterminal_marginal_error("as_mixture")
    # A no-event branch makes the observed-time mass `< 1` (a defective marginal),
    # so there is no proper `MixtureModel` over the observed delays: the node is
    # multivariate / sub-stochastic and is scored only through the event-vector
    # path. Reject the scalar lowering with a clear message.
    _has_no_event(c) && _no_event_marginal_error("as_mixture")
    # The `Categorical` inside the `MixtureModel` needs a normalised weight set,
    # so reject an unnormalised `Resolve` (e.g. one built directly with
    # branch probabilities that do not sum to one) with a clear error rather
    # than the confusing `DomainError` `Categorical` would otherwise throw.
    _validate_branch_probs_sum(c.branch_probs)
    return MixtureModel(
        collect(c.delays), collect(float.(c.branch_probs)))
end

# A defective-marginal (no-event) one_of node has no scalar `logpdf` / `mean`
# / `as_mixture`: its observed-time mass is `< 1`, so it is multivariate and
# scored only through the event-vector path. Errors with a clear message.
function _no_event_marginal_error(what::AbstractString)
    throw(ArgumentError(
        "a Resolve node with a no-event branch is a defective marginal " *
        "(its observed-time mass is < 1) and has no scalar `$(what)`; score it " *
        "through the event-vector path (its observed-outcome / no-event record)"))
end

# A non-terminal (composer-outcome) one_of node is multivariate: an outcome's
# subtree spans several event slots, so there is no single marginal
# time-to-resolution and no scalar `logpdf` / `mean` / `as_mixture`. It is scored
# only through the event-vector path (its outcome subtree's slice), so the scalar
# methods error with a clear message pointing at the event-vector / NamedTuple
# path. (#466 Feature 3.) Defined after `as_mixture` so the `@doc` block above
# `as_mixture` attaches to it (an intervening function definition would steal the
# docstring and leave `as_mixture` undocumented, an Aqua failure).
function _nonterminal_marginal_error(what::AbstractString)
    throw(ArgumentError(
        "a non-terminal Resolve node (an outcome whose payload is a composer " *
        "subtree) is multivariate and has no scalar `$(what)`; score it through " *
        "the event-vector path (nest it in a `compose(...)` tree and pass the " *
        "outcome subtree's event slots, or use `event_names` / NamedTuple I/O)"))
end

params(c::Resolve) = (map(params, c.delays), c.branch_probs)

# The univariate interface delegates to the mixture lowering, so a `Resolve`
# behaves as the marginal time-to-resolution wherever a distribution is needed.
Base.minimum(c::Resolve) = minimum(as_mixture(c))
Base.maximum(c::Resolve) = maximum(as_mixture(c))
insupport(c::Resolve, x::Real) = insupport(as_mixture(c), x)
mean(c::Resolve) = mean(as_mixture(c))
var(c::Resolve) = var(as_mixture(c))

@doc "

Log probability density of the one_of-outcome marginal at `x`.

Routed through the AD-safe `_one_of_logmix` reduction rather than
`logpdf(as_mixture(c), x)`: `as_mixture` does `float.(branch_probs)`, which
strips an AD `Dual`/tracked type from the branch probabilities, breaking the
gradient w.r.t. a covariate case-fatality term (`logistic(Xβ)`) when a
`Resolve` is scored as a leaf of a plain (non-censored) `compose(...)` tree.
The explicit log-sum-exp keeps the probabilities' element type, so a `Dual`
propagates exactly as on the censored-tree scorer.

See also: [`as_mixture`](@ref)
"
function logpdf(c::Resolve, x::Real)
    _is_nonterminal(c) && _nonterminal_marginal_error("logpdf")
    _has_no_event(c) && _no_event_marginal_error("logpdf")
    return _one_of_logmix(c.branch_probs, c.delays, x)
end

@doc "

Probability density of the one_of-outcome marginal at `x`.

`exp` of the AD-safe [`logpdf`](@ref), so branch-prob gradients survive (see the
`logpdf` note on why `as_mixture` is avoided on a differentiated path).

See also: [`logpdf`](@ref)
"
pdf(c::Resolve, x::Real) = exp(logpdf(c, x))

@doc "

Cumulative distribution function of the one_of-outcome marginal at `x`.

The branch-prob-weighted mixture cdf `Σ_i p_i F_i(x)`, summed directly so the
probabilities keep their (possibly AD `Dual`) element type rather than being
stripped by `as_mixture`'s `float.(branch_probs)`. This keeps the cdf AD-safe on
a differentiated path (e.g. a censored-survival term), matching `logpdf`.

See also: [`as_mixture`](@ref)
"
function cdf(c::Resolve, x::Real)
    _is_nonterminal(c) && _nonterminal_marginal_error("cdf")
    _has_no_event(c) && _no_event_marginal_error("cdf")
    return sum(ntuple(
        i -> c.branch_probs[i] * cdf(c.delays[i], x), length(c.branch_probs)))
end

@doc "

Sample a [`Resolve`](@ref) node, returning the full named event record of the
outcome that fired.

The draw resolves to a single outcome (sampled from the branch probabilities)
and the result is a `NamedTuple` keyed by [`event_names`](@ref): a positional
origin slot then one slot per outcome, with the fired outcome's time present and
the others `missing`. This is the same self-describing record the in-tree path
produces (a `Resolve` nested in a `compose(...)` tree), so a standalone draw
identifies which outcome won and feeds straight back into [`logpdf`](@ref).

To recover the marginal time-to-resolution alone (the mixture over outcomes,
discarding which fired) sample [`as_mixture`](@ref)`(c)` instead.

See also: [`event_names`](@ref), [`as_mixture`](@ref), [`rand_outcome`](@ref)
"
Base.rand(rng::AbstractRNG, c::Resolve) = _one_of_event_record(rng, c)
Base.rand(c::Resolve) = rand(default_rng(), c)

# The scalar marginal draw of a terminal Resolve (its branch-prob-weighted
# mixture time-to-resolution, discarding which outcome fired). Used by the plain
# flat value path (`child_rand!`), where a Resolve child is one value slot, and
# wherever the marginal time alone is wanted.
_one_of_marginal_rand(rng::AbstractRNG, c::Resolve) = rand(rng, as_mixture(c))

@doc "

Sample a composed-distribution outcome AND its time, returning `(name, time)`.

`rand_outcome` retains WHICH outcome/cause occurred, unlike the univariate
[`rand`](@ref) (the marginal time only, which discards which outcome/cause
won). Dispatches on the composer:

- a [`Resolve`](@ref) node (below): draws the resolved outcome from the branch
  probabilities and the time from that outcome's own delay;
- a [`Compete`](@ref) node (below, in `hazard_one_of.jl`): draws a
  racing-hazard outcome, a latent time per cause with the `argmin` cause and
  its `min` time returned.

# `rand_outcome(rng, c::Resolve)`

Used by the full-path tree simulation, where a `Resolve` node resolves to a
single named outcome, so the chosen outcome is retained rather than discarded.

## Arguments
- `rng`: random number generator (the no-`rng` method uses the global default).
- `c`: the [`Resolve`](@ref) node to sample an outcome from.

## Examples
```@example
using ComposedDistributions, Distributions, Random

node = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
    :disch => (Gamma(2.0, 1.5), 0.7))
name, time = rand_outcome(MersenneTwister(1), node)
```

# `rand_outcome(rng, c::Compete)`

This is the generative dual of the [`logpdf`](@ref) (`f_j ∏_{k≠j} S_k`) and of
the forward `convolve_series` stream: the Monte Carlo winning-cause
frequencies match the derived `Distributions.probs` split and the forward
per-outcome stream masses.

## Arguments
- `rng`: random number generator (the no-`rng` method uses the global default).
- `c`: the [`Compete`](@ref) node to sample a winning cause from.

## Examples
```@example
using ComposedDistributions, Distributions, Random

node = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
name, time = rand_outcome(MersenneTwister(1), node)
```

# See also
- [`Resolve`](@ref), [`Compete`](@ref): the composer nodes
- [`rand`](@ref): the marginal time-only draw
- `Distributions.probs`
"
function rand_outcome end

function rand_outcome(rng::AbstractRNG, c::Resolve)
    i = _sample_branch(rng, c.branch_probs)
    # A no-event win yields `missing` (no event time recorded); a real outcome
    # draws its own delay.
    _is_no_event(c.delays[i]) && return c.names[i], missing
    return c.names[i], rand(rng, c.delays[i])
end

rand_outcome(c::Resolve) = rand_outcome(default_rng(), c)

# ----------------------------------------------------------------------------
# Standalone one_of event records: the shape a bare `rand(::Resolve)` /
# `rand(::Compete)` returns, and its round-trip scorer (#639)
# ----------------------------------------------------------------------------
#
# A standalone one_of node (sampled on its own, not nested in a `compose(...)`
# tree) draws the full named event record of the outcome that fired: a
# `NamedTuple` keyed by `_flat_event_names(c) = (:event_1, c.names...)`, a
# positional origin slot (anchored at zero) then one slot per outcome, with the
# fired outcome's time present and the others `missing`. This is the same
# self-describing record the in-tree path produces, so a standalone draw
# identifies which outcome won and round-trips through `logpdf(c, rand(c))`. A
# non-terminal node (a composer-valued outcome) has no standalone record layout
# (its subtree spans several event slots, addressable only once nested), so it
# errors with the same guidance the scalar marginal methods give.
function _one_of_event_record(rng::AbstractRNG, c::AbstractOneOf)
    _is_nonterminal(c) && _nonterminal_marginal_error("rand")
    name, time = rand_outcome(rng, c)
    T = float(_one_of_record_eltype(c))
    out = Vector{Union{Missing, T}}(missing, _event_child_nleaves(c) + 1)
    out[1] = zero(T)
    # A no-event win records no time (every outcome slot stays `missing`); a
    # real outcome fills its slot (outcome `i` at slot `i + 1`; origin at 1).
    if time !== missing
        i = something(findfirst(==(name), c.names))
        out[i + 1] = convert(T, time)
    end
    return NamedTuple{_flat_event_names(c)}(Tuple(out))
end

# Independent count draw of a standalone one_of node: a vector of records (the
# univariate count form Distributions cannot build, since a record is a
# `NamedTuple`, not the scalar `eltype`).
function Base.rand(rng::AbstractRNG, c::AbstractOneOf, n::Int)
    return [rand(rng, c) for _ in 1:n]
end

# The promoted float element type of a terminal one_of node's sampled event
# times: each outcome delay's own `eltype` promoted together (a no-event marker
# carries none and is skipped). head/tail recursion, not `mapreduce`, to keep
# the promotion inferable over the heterogeneous outcome tuple (the same reason
# `_one_of_outcome_nleaves` recurses).
_one_of_record_eltype(c::AbstractOneOf) = _promote_outcome_eltype(c.delays)
_promote_outcome_eltype(::Tuple{}) = Float64
function _promote_outcome_eltype(delays::Tuple)
    rest = _promote_outcome_eltype(Base.tail(delays))
    d = first(delays)
    return _is_no_event(d) ? rest : promote_type(eltype(d), rest)
end

# The observed outcome index of a standalone (terminal) one_of record: the
# single outcome slot carrying a time (slots `2 … k + 1`, one per outcome). `0`
# when none is observed (a no-event / latent non-occurrence record). At most one
# outcome may be observed. The standalone record is always terminal (a
# non-terminal node is rejected upstream), so every outcome is one slot at
# `i + 1`.
function _observed_one_of_outcome(c::AbstractOneOf, events)
    obs_i = 0
    @inbounds for i in 1:_n_branches(c)
        events[i + 1] === missing && continue
        obs_i == 0 || throw(ArgumentError(
            "a standalone one_of record may observe at most one outcome; got " *
            "outcomes $(c.names[obs_i]) and $(c.names[i])"))
        obs_i = i
    end
    return obs_i
end

# Score a table / vector of standalone one_of records: the sum of each record's
# single-record log density. A standalone one_of node carries no per-record
# routing, so the table is scored by summing the per-row scorer directly.
function logpdf(c::AbstractOneOf, rows::AbstractVector{<:NamedTuple})
    return sum(logpdf(c, r) for r in rows)
end

function _one_of_table_logpdf(c::AbstractOneOf, table)
    return sum(logpdf(c, r) for r in Tables.namedtupleiterator(table))
end

@doc "

Score a standalone [`Resolve`](@ref) outcome record (the shape a bare `rand(c)`
returns): `log p_i + logpdf(delay_i, t)` for the fired outcome `i` at time `t`,
so `logpdf(c, rand(c))` round-trips. A column table (a `NamedTuple` of vectors)
is a multi-record source, summed per row.

See also: [`rand`](@ref), [`event_names`](@ref)
"
function logpdf(c::Resolve, x::NamedTuple)
    Tables.istable(x) && return _one_of_table_logpdf(c, x)
    _is_nonterminal(c) && _nonterminal_marginal_error("logpdf")
    events = _row_event_vector_by_name(_flat_event_names(c), x)
    obs_i = _observed_one_of_outcome(c, events)
    obs_i == 0 && return 0.0
    # An observed non-occurrence scores the no-event mass `log q` alone.
    _is_no_event(c.delays[obs_i]) && return log(c.branch_probs[obs_i])
    y = events[obs_i + 1]
    o = events[1]
    # `obs_i > 0` guarantees the outcome slot and the origin are observed; the
    # guard narrows the `Union{Missing, Float64}` cells to `Float64` for the
    # scorer (the branch is unreachable for a `rand`-shaped record).
    (y === missing || o === missing) && return 0.0
    gap = y - o
    return _one_of_condition_logpdf(c.branch_probs, c.delays[obs_i], gap, obs_i)
end

@doc "

Print a [`Resolve`](@ref) node as a recursive indented tree, labelling each
outcome with its name and branch probability and descending into any nested
composer outcome so the whole structure is shown at once.

See also: [`Resolve`](@ref)
"
function Base.show(io::IO, ::MIME"text/plain", c::Resolve)
    _show_composer_tree(io, c)
    return nothing
end

function Base.show(io::IO, c::Resolve)
    parts = ["$(c.names[k])@$(c.branch_probs[k])" for k in 1:_n_branches(c)]
    print(io, "Resolve(", join(parts, " | "), ")")
    return nothing
end

# ----------------------------------------------------------------------------
# Fixed-probability outcome probabilities (the Compete duals live in
# hazard_one_of.jl)
# ----------------------------------------------------------------------------

@doc "

The per-outcome probabilities of a fixed-probability [`Resolve`](@ref) node:
its declared branch probabilities (the no-event branch's mass is the
non-occurrence probability), returned as a `NamedTuple` keyed by the outcome
names.

This is the [`Resolve`](@ref) method of `Distributions.probs`, the standard
mixture-weight reader: a `Resolve` lowers to a `MixtureModel` (see
[`as_mixture`](@ref)), so its weights ARE the declared branch probabilities.
The racing-hazard [`Compete`](@ref) sibling derives the same split from the
hazards instead.

# Arguments
- `c`: the [`Resolve`](@ref) node whose declared branch probabilities to read.

# Examples
```@example
using ComposedDistributions, Distributions

node = resolve(:death => (Gamma(1.5, 1.0), 0.3),
    :disch => (Gamma(2.0, 1.5), 0.7))
probs(node)
```

See also: [`occurrence_probability`](@ref)
"
function probs(c::Resolve)
    return NamedTuple{c.names}(c.branch_probs)
end

@doc "

The probability that ANY (non-no-event) outcome occurs for a fixed-probability
[`Resolve`](@ref) node: one minus the no-event branch mass.

See also: `Distributions.probs`
"
function occurrence_probability(c::Resolve)
    total = zero(float(eltype(c.branch_probs)))
    @inbounds for k in 1:_n_branches(c)
        _is_no_event(c.delays[k]) && continue
        total += c.branch_probs[k]
    end
    return total
end
