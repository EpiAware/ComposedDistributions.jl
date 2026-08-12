# Numeric `Distributions.quantile` for the one_of composers (`Resolve` /
# `Compete`): their marginal -- a branch-weighted mixture, resp. a
# racing-hazard `min` -- has no closed-form inverse cdf in general, so the
# quantile is found by inverting `cdf` numerically.
#
# The inversion is a BRACKETED ROOT FIND (bisection on the monotone cdf), not
# a derivative-free optimisation. This used to live in a weak-dependency
# extension that called `ConvolvedDistributions.quantile_by_optimization`, a
# Nelder-Mead solve over the squared log-odds residual, and #356 was a CI hang
# inside exactly that solve: the Windows Julia 1.x cell went silent the moment
# the extension loaded and ran to the 60-minute job timeout, while every other
# cell (including Windows on the Julia pre release, same solver versions)
# passed in minutes.
#
# A wall-clock bound on the solve could not fix that (#358 tried): the bound
# is forwarded to Optim's `time_limit`, which is only checked BETWEEN
# iterations, so it cannot interrupt one slow objective evaluation. Bisection
# removes the failure mode instead of bounding it. A cdf is monotone
# non-decreasing, so its inverse is a root find, and a bracketed root find
# terminates in a step count fixed by the bracket and the floating-point
# resolution -- by construction, whatever the node. It is also more accurate
# here (machine precision rather than the solver's tolerance) and drops the
# Optimization.jl / OptimizationOptimJL.jl stack from this package entirely.

# Bracket expansion is geometric (each step doubles), so this cap is reached
# only after growing the bracket by 2^200 -- far beyond any finite Float64
# scale. It exists so the search is bounded by construction rather than by
# the cdf eventually behaving.
const _QUANTILE_BRACKET_STEPS = 200

# Halving a bracket 100 times takes any finite Float64 interval below the gap
# between adjacent floats, so the loop's own "midpoint is an endpoint" exit
# fires long before this cap. Same reason as above: a hard bound, not a
# tolerance.
const _QUANTILE_BISECT_STEPS = 100

# A finite bracket `[a, b]` with `cdf(d, a) <= p <= cdf(d, b)`.
#
# Finite support ends are used directly. An unbounded end is grown from a
# finite base point by doubling until the cdf brackets `p`. Deliberately no
# `mean(d)`-based starting point: a `Compete` mean is a panelled quadrature
# that itself calls `quantile` on every cause (expensive, and recursive for a
# nested node), and the mean of a heavy-tailed cause is `NaN` or `Inf`
# anyway (#344) -- the geometric expansion finds the scale in a handful of
# cdf evaluations without needing either.
function _one_of_quantile_bracket(d, p::Real)
    support_lo = float(minimum(d))
    support_hi = float(maximum(d))
    base = isfinite(support_lo) ? support_lo :
           (isfinite(support_hi) ? support_hi : zero(support_lo))

    a = support_lo
    if !isfinite(a)
        step = one(base)
        a = base - step
        for _ in 1:_QUANTILE_BRACKET_STEPS
            cdf(d, a) <= p && break
            step *= 2
            a = base - step
        end
    end

    b = support_hi
    if !isfinite(b)
        step = one(base)
        b = base + step
        for _ in 1:_QUANTILE_BRACKET_STEPS
            cdf(d, b) >= p && break
            step *= 2
            b = base + step
        end
    end

    return a, b
end

# Bisect the monotone cdf on `[a, b]`, returning the midpoint of the final
# bracket. The loop exits as soon as the midpoint lands on an endpoint, i.e.
# once the bracket is two adjacent floats, so the answer is at the
# floating-point resolution of the quantile itself.
function _one_of_quantile_bisect(d, p::Real, a::Real, b::Real)
    lo = float(a)
    hi = float(b)
    for _ in 1:_QUANTILE_BISECT_STEPS
        mid = lo + (hi - lo) / 2
        (mid <= lo || mid >= hi) && break
        cdf(d, mid) < p ? (lo = mid) : (hi = mid)
    end
    return lo + (hi - lo) / 2
end

function _one_of_quantile(d, p::Real)
    if isnan(p) || p < 0 || p > 1
        throw(ArgumentError("p must be in [0, 1], got $p"))
    end
    # The support ends are the exact answers at the boundaries.
    p == 0 && return minimum(d)
    p == 1 && return maximum(d)

    a, b = _one_of_quantile_bracket(d, p)
    # `p` above the reachable cdf leaves the quantile at the support ceiling:
    # a racing node whose causes need not all fire has a sub-stochastic cdf
    # that never reaches `p`, and the honest answer is the ceiling rather than
    # whatever the bracket's last point happens to be. Same argument at the
    # floor.
    cdf(d, b) < p && return maximum(d)
    cdf(d, a) > p && return minimum(d)
    return _one_of_quantile_bisect(d, p, a, b)
end

@doc "

Compute the quantile (inverse cdf) of a one_of node's marginal: the
fixed-probability mixture of a [`Resolve`](@ref), or the racing-hazard
any-event time of a [`Compete`](@ref), the minimum over its causes.

Neither marginal has a closed-form inverse in general, so [`cdf`](@ref) is
inverted numerically by bisection over a bracket grown from the node's
support. A cdf is monotone, so this is a root find with a step count bounded
by construction, and it resolves the quantile to floating-point precision.

`p = 0` and `p = 1` return the support ends exactly. A racing node whose
causes need not all fire has a cdf that tops out below one, so any `p` above
that ceiling returns `maximum(d)`.

# Arguments
- `d`: the one_of node ([`Resolve`](@ref) or [`Compete`](@ref)) to invert.
- `p`: the probability to invert at, in `[0, 1]`.

# Examples
```@example
using ComposedDistributions, Distributions

node = resolve(:death => (Gamma(1.5, 1.0), 0.3),
    :disch => (Gamma(2.0, 1.5), 0.7))
quantile(node, 0.5)
```

```@example
using ComposedDistributions, Distributions

node = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
quantile(node, 0.9)
```

See also: [`Resolve`](@ref), [`Compete`](@ref), [`cdf`](@ref)
"
quantile(d::AbstractOneOf, p::Real) = _one_of_quantile(d, p)
