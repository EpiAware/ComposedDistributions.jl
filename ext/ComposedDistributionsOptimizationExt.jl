"""
    ComposedDistributionsOptimizationExt

Numeric `Distributions.quantile` for the one_of composers ([`Resolve`](@ref) /
[`Compete`](@ref)): their marginal (a branch-weighted mixture, resp. a
racing-hazard `min`) has no closed-form inverse CDF in general, so this
extension inverts [`cdf`](@ref) numerically via
`ConvolvedDistributions.quantile_by_optimization` -- the shared Nelder-Mead
solve ConvolvedDistributions' own `Convolved`/`Difference`/`Product` quantile
methods use.

A weak dependency, mirroring ConvolvedDistributions' own
`ConvolvedDistributionsOptimizationExt`: `cdf`/`pdf`/`logpdf` never need a
solver, so the base package stays dependency-light and only a caller who
needs the quantile (`truncated` sampling, or a `Compete` cause whose window /
panel construction wants a tight quantile ladder rather than the moment-based
fallback, see `src/composers/Compete.jl`) pulls in Optimization.jl +
OptimizationOptimJL.jl.
"""
module ComposedDistributionsOptimizationExt

using ComposedDistributions: ComposedDistributions, Resolve, Compete
import Distributions
using Distributions: mean
using ConvolvedDistributions: quantile_by_optimization

# A cheap starting point for the Nelder-Mead inversion: the node's own mean,
# clamped into its support. Exact for a degenerate (zero-variance) node and a
# reasonable neighbourhood otherwise -- the solve only needs a value close
# enough for Nelder-Mead to converge, not a tight guess.
#
# `mean(d)` is not always finite: a heavy-tailed cause with no finite mean
# (e.g. Cauchy) returns `NaN`, and a cause with unbounded mean (e.g. a
# shape<=1 Pareto) returns `Inf`; clamping either into `[lo, hi]` still
# yields a non-finite guess. Handing Nelder-Mead a non-finite `x0` breaks
# the solve rather than erroring cleanly -- on this package's pinned
# Optim.jl it asserts during simplex construction, but nothing here rules
# out a version/platform where the same non-finite state spins without
# ever satisfying a NaN-valued convergence check instead (#344, #356
# diagnosed a Windows CI hang traced to exactly this NaN guess). Fall back
# to a finite, support-based point rather than ever handing the solver a
# NaN/Inf.
function _quantile_guess(d, p::Real)
    lo = float(minimum(d))
    hi = float(maximum(d))
    m = float(mean(d))
    isfinite(m) && return [clamp(m, lo, hi)]
    isfinite(lo) && isfinite(hi) && return [(lo + hi) / 2]
    isfinite(lo) && return [lo + one(lo)]
    isfinite(hi) && return [hi - one(hi)]
    return [zero(lo)]
end

# A hard wall-clock cutoff on the Nelder-Mead solve itself, independent of
# the initial-guess fix above: defence in depth against any other input
# (not just a non-finite guess) that makes the solve slow to converge or
# fail to terminate, so a single `quantile` call can never itself hang CI
# to the job timeout (#356). Generous relative to a normal solve (a
# fraction of a second, see the package's own quantile tests) but far
# below CI's own timeout.
const _QUANTILE_MAXTIME = 10.0

@doc "

Compute the quantile (inverse CDF) of the fixed-probability mixture marginal.

No closed form exists for a general [`Resolve`](@ref) mixture, so the
quantile is found by numerically inverting [`cdf`](@ref) with a Nelder-Mead
solve (`ConvolvedDistributions.quantile_by_optimization`), started from the
node's own mean clamped into its support.

Requires Optimization.jl and OptimizationOptimJL.jl to be loaded (this method
lives in the `ComposedDistributionsOptimizationExt` extension).

# Examples
```@example
using ComposedDistributions, Distributions
using Optimization, OptimizationOptimJL

node = resolve(:death => (Gamma(1.5, 1.0), 0.3), :disch => (Gamma(2.0, 1.5), 0.7))
quantile(node, 0.5)
```

See also: [`Resolve`](@ref), [`cdf`](@ref)
"
function Distributions.quantile(d::Resolve, p::Real)
    return quantile_by_optimization(
        d, p, _quantile_guess(d, p); maxtime = _QUANTILE_MAXTIME)
end

@doc "

Compute the quantile (inverse CDF) of the racing-hazard marginal any-event
time.

No closed form exists for a general [`Compete`](@ref) node (the marginal
`T = min_k D_k`), so the quantile is found by numerically inverting
[`cdf`](@ref) with a Nelder-Mead solve
(`ConvolvedDistributions.quantile_by_optimization`), started from the node's
own mean clamped into its support.

Requires Optimization.jl and OptimizationOptimJL.jl to be loaded (this method
lives in the `ComposedDistributionsOptimizationExt` extension).

# Examples
```@example
using ComposedDistributions, Distributions
using Optimization, OptimizationOptimJL

node = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
quantile(node, 0.5)
```

See also: [`Compete`](@ref), [`cdf`](@ref)
"
function Distributions.quantile(d::Compete, p::Real)
    return quantile_by_optimization(
        d, p, _quantile_guess(d, p); maxtime = _QUANTILE_MAXTIME)
end

end # module
