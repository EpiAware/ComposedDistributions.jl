"""
    ComposedDistributionsLogDensityProblemsExt

Expose a `ComposedLogDensity` as a `LogDensityProblems` problem, so a composed
distribution's posterior over its estimated parameters is sampleable by any
LogDensityProblems consumer (AdvancedHMC, DynamicHMC, Pathfinder, Turing's
`externalsampler`) with gradients supplied by LogDensityProblemsAD.

The density is the one `as_logdensity` assembles: the sum of the uncertain
specs' log-densities plus the data log-likelihood of the tree reconstructed at
the estimated parameters. It is evaluated on the constrained flat parameter
vector, in `params_table` row order restricted to the estimated rows, so the
dimension is `flat_dimension(prob.dist)`. Sampling on the unconstrained scale
composes `to_constrained` (the Bijectors extension) for the prior-driven
transform, whose log-Jacobian is added to this density.

Only the zeroth-order interface is provided (`LogDensityOrder{0}`); a gradient
comes from wrapping the problem with LogDensityProblemsAD and an AD backend the
codec already differentiates under.
"""
module ComposedDistributionsLogDensityProblemsExt

using ComposedDistributions: ComposedLogDensity
import ComposedDistributions
import LogDensityProblems

# The codec supplies the log-density itself; a gradient is delegated to
# LogDensityProblemsAD, so only the zeroth-order capability is claimed.
function LogDensityProblems.capabilities(::Type{<:ComposedLogDensity})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(prob::ComposedLogDensity)
    ComposedDistributions.flat_dimension(prob.dist)
end

# Qualified call: both packages export `logdensity`, so the composed codec's
# evaluator is named through its module rather than imported.
function LogDensityProblems.logdensity(prob::ComposedLogDensity, x::AbstractVector)
    ComposedDistributions.logdensity(prob, x)
end

end # module ComposedDistributionsLogDensityProblemsExt
