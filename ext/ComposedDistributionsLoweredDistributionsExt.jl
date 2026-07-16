"""
    ComposedDistributionsLoweredDistributionsExt

Lower a composed distribution to a backend-agnostic dynamical-systems
representation (a `LoweredDistributions.AbstractLowering`), so
`lower(compose(...))` yields a phase-type or continuous-time Markov chain for
the whole composed delay structure that a downstream Catalyst / ODE / Petri /
Jump backend can consume.

Each source package owns its own bridge to `LoweredDistributions` (the ecosystem
ownership split), so the composer methods of `lower` live here rather than in
`LoweredDistributions`. Composition of phase-types is closed under the algebra,
so the scalar composers lower exactly:

  - `lower(::Sequential)` convolves the step lowerings into one series phase-type
    (the steps run back to back, so phase `i`'s exit feeds phase `i + 1`).
  - `lower(::Resolve)` mixes the outcome lowerings into a hyper-phase-type
    weighted by the branch probabilities (cause and timing independent).
  - `lower(::Compete)` races the cause-specific lowerings through a
    competing-risks generator (the Kronecker sum, so absorption is the first
    exit, i.e. the minimum).
  - `lower(::Shared)` is transparent: a tie is a parameter-sharing annotation
    with no effect on the dynamics, so it lowers its wrapped leaf.

The two vector-valued composers have no single scalar time to absorption, so
they lower to a `CTMC` over a joint state space rather than a phase-type:

  - `lower(::Parallel)` runs the branches jointly and independently, so its
    generator is the Kronecker sum of the branch generators over the product
    state space.
  - `lower(::Choose)` activates exactly one alternative, chosen externally by
    the selector, so its generator is the block-diagonal union of the
    alternative generators.

A composer whose branches are themselves `Parallel` or `Choose` cannot be folded
into a phase-type (those lower to a joint `CTMC` over several states, whose
vector or conditional meaning is not a scalar delay), so nesting one inside a
scalar composer raises a clear error rather than a silent misrepresentation.
"""
module ComposedDistributionsLoweredDistributionsExt

using ComposedDistributions: Sequential, Parallel, Resolve, Compete, Choose,
                             Shared, NoEvent, component_names
import LoweredDistributions: lower
using LoweredDistributions: AbstractLowering, AbstractChainTrick, PhaseType,
                            CTMC

# --- Canonical (α, S) view of a lowered component ---------------------------

# Every scalar composition works on the canonical phase-type pair (α, S): an
# initial distribution over transient phases and a sub-generator. A component
# is lowered first (a leaf through LoweredDistributions, a nested composer
# through the methods below) and then canonicalised.
_canonical(x) = _pair(_phasetype(lower(x)))

_phasetype(p::PhaseType) = p
_phasetype(c::AbstractChainTrick) = PhaseType(c)

# A CTMC folds into a phase-type only in the two-state Exponential fast path
# (one transient `on` phase to one absorbing state), the single CTMC that
# `LoweredDistributions.lower` produces from a leaf. A larger CTMC is a joint
# lowering from Parallel or Choose, whose vector or conditional meaning has no
# scalar phase-type form, so it stops here rather than being silently recast as
# the time to the joint absorbing state.
function _phasetype(m::CTMC)
    n = length(m.states)
    n == 2 || throw(ArgumentError(
        "only the two-state Exponential CTMC folds into a phase-type; a " *
        "$(n)-state CTMC from Parallel or Choose has no scalar phase-type " *
        "form, so it cannot nest inside a scalar composition"))
    return PhaseType([1.0], reshape([float(m.Q[1, 1])], 1, 1))
end

_pair(p::PhaseType) = (collect(float.(p.α)), Matrix{Float64}(float.(p.S)))

# The per-phase exit rate to absorption is the row shortfall of the
# sub-generator.
_exit(S) = [-sum(@view S[i, :]) for i in axes(S, 1)]

# --- Sequential: series convolution -----------------------------------------

"""
    lower(d::Sequential)

Lower a chain of steps to the series phase-type that convolves the step
lowerings. Phase `i`'s exit feeds the start of phase `i + 1`, so the absorbing
time is the sum of the step delays.
"""
function lower(d::Sequential)
    pair = _canonical(first(d.components))
    for c in Base.tail(d.components)
        pair = _series(pair, _canonical(c))
    end
    return PhaseType(pair...)
end

function _series((αa, Sa), (αb, Sb))
    ka, kb = length(αa), length(αb)
    sa = _exit(Sa)
    α = vcat(αa, zeros(kb))
    S = zeros(ka + kb, ka + kb)
    S[1:ka, 1:ka] .= Sa
    # a's exit feeds the start of b (the outer product of the exit rates and αb).
    S[1:ka, (ka + 1):end] .= sa .* reshape(αb, 1, kb)
    S[(ka + 1):end, (ka + 1):end] .= Sb
    return (α, S)
end

# --- Resolve: hyper-phase-type mixture --------------------------------------

"""
    lower(d::Resolve)

Lower a fixed-probability one_of node to the hyper-phase-type that mixes the
outcome lowerings, weighting each outcome's initial distribution by its branch
probability.
"""
function lower(d::Resolve)
    any(x -> x isa NoEvent, d.delays) && throw(ArgumentError(
        "a Resolve with a no-event branch is defective (it places mass at no " *
        "event) and has no phase-type lowering; its initial distribution " *
        "cannot sum to one"))
    weights = collect(float.(d.branch_probs))
    pairs = map(_canonical, d.delays)
    return PhaseType(_mixture(weights, pairs)...)
end

function _mixture(weights, pairs)
    α = Float64[]
    for (w, (αi, _)) in zip(weights, pairs)
        append!(α, w .* αi)
    end
    return (α, _blockdiag([S for (_, S) in pairs]))
end

# --- Compete: competing-risks minimum ---------------------------------------

"""
    lower(d::Compete)

Lower a racing-hazard one_of node to the competing-risks phase-type. The
generator is the Kronecker sum of the cause-specific lowerings, so absorption is
the first exit (the minimum of the racing delays).
"""
function lower(d::Compete)
    pair = _canonical(first(d.delays))
    for c in Base.tail(d.delays)
        pair = _min(pair, _canonical(c))
    end
    return PhaseType(pair...)
end

function _min((αa, Sa), (αb, Sb))
    ka, kb = length(αa), length(αb)
    α = _kron(αa, αb)
    S = zeros(ka * kb, ka * kb)
    # (i, j) -> row (i - 1) * kb + j; exit when either sub-chain exits.
    for i in 1:ka
        for j in 1:kb
            r = (i - 1) * kb + j
            for ip in 1:ka
                S[r, (ip - 1) * kb + j] += Sa[i, ip]
            end
            for jp in 1:kb
                S[r, (i - 1) * kb + jp] += Sb[j, jp]
            end
        end
    end
    return (α, S)
end

# --- Shared: transparent tie ------------------------------------------------

"""
    lower(d::Shared)

Lower a shared-parameter leaf by lowering its wrapped distribution. The tie is a
parameter-sharing annotation with no effect on the dynamics.
"""
lower(d::Shared) = lower(d.dist)

# --- Parallel: independent joint CTMC ---------------------------------------

"""
    lower(d::Parallel)

Lower independent branches to the joint continuous-time Markov chain whose
generator is the Kronecker sum of the branch generators over the product state
space. The branches evolve independently, each reaching its own absorbing state.
"""
function lower(d::Parallel)
    blocks = map(_full_block, d.components, component_names(d))
    return _kron_sum_ctmc(blocks)
end

# --- Choose: selector-switched block-diagonal CTMC --------------------------

"""
    lower(d::Choose)

Lower a selector switch to the block-diagonal continuous-time Markov chain that
unions the alternative generators. Exactly one alternative is active, chosen
externally by the selector, so the alternatives share no transitions.
"""
function lower(d::Choose)
    blocks = map(_full_block, d.alternatives, component_names(d))
    return _blockdiag_ctmc(blocks)
end

# A component's full generator (transient phases plus one absorbing state) and
# its state names, tagged with the branch name so the joint states stay unique.
function _full_block(x, name::Symbol)
    α, S = _canonical(x)
    k = length(α)
    s = _exit(S)
    G = zeros(k + 1, k + 1)
    G[1:k, 1:k] .= S
    for i in 1:k
        G[i, k + 1] = s[i]
    end
    names = Symbol[Symbol(name, :_, j) for j in 1:k]
    push!(names, Symbol(name, :_absorbed))
    return (names, G)
end

function _kron_sum_ctmc(blocks)
    names, G = first(blocks)
    for (nb, Gb) in Base.tail(blocks)
        names, G = _kron_sum((names, G), (nb, Gb))
    end
    return CTMC(Tuple(names), G)
end

function _kron_sum((na, Ga), (nb, Gb))
    ka, kb = length(na), length(nb)
    names = Symbol[Symbol(a, :__, b) for a in na for b in nb]
    G = zeros(ka * kb, ka * kb)
    for i in 1:ka
        for j in 1:kb
            r = (i - 1) * kb + j
            for ip in 1:ka
                G[r, (ip - 1) * kb + j] += Ga[i, ip]
            end
            for jp in 1:kb
                G[r, (i - 1) * kb + jp] += Gb[j, jp]
            end
        end
    end
    return (names, G)
end

function _blockdiag_ctmc(blocks)
    names = reduce(vcat, first.(blocks))
    total = length(names)
    G = zeros(total, total)
    offset = 0
    for (nb, Gb) in blocks
        k = length(nb)
        G[(offset + 1):(offset + k), (offset + 1):(offset + k)] .= Gb
        offset += k
    end
    return CTMC(Tuple(names), G)
end

# --- small Base-only array helpers ------------------------------------------

_kron(a::AbstractVector, b::AbstractVector) = [x * y for x in a for y in b]

function _blockdiag(mats)
    total = sum(size(m, 1) for m in mats)
    out = zeros(total, total)
    offset = 0
    for m in mats
        k = size(m, 1)
        out[(offset + 1):(offset + k), (offset + 1):(offset + k)] .= m
        offset += k
    end
    return out
end

end # module ComposedDistributionsLoweredDistributionsExt
