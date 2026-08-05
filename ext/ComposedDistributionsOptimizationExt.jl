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
function _quantile_guess(d, p::Real)
    lo = float(minimum(d))
    hi = float(maximum(d))
    return [clamp(float(mean(d)), lo, hi)]
end

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
    return quantile_by_optimization(d, p, _quantile_guess(d, p))
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
    return quantile_by_optimization(d, p, _quantile_guess(d, p))
end

end # module
