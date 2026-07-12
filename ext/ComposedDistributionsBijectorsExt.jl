# ComposedDistributions x Bijectors: the prior-driven unconstrained <->
# constrained transform for the PPL-neutral codec. Each ESTIMATED flat row's
# constraint is carried by its prior itself, so `bijector(prior)` per row
# gives the flat transform with no bespoke domain table. A centred-pooled row
# (see `pool`) carries a `CentredPoolPrior` marker rather than a fixed prior
# distribution (its population is hyperparameter-dependent), so that row's
# transform is read off the population's family instead — the family (and so
# the bijector) does not depend on the current hyperparameter values, only the
# member's own logdensity term does. Loaded only when Bijectors is available.
module ComposedDistributionsBijectorsExt

using ComposedDistributions: ComposedDistributions, ComposedLogDensity,
                             CentredPoolPrior, _population_template
using Bijectors: Bijectors, bijector, inverse, with_logabsdet_jacobian

# The bijector for one ESTIMATED row's prior: the ordinary transform for a
# fixed prior distribution, or (for a centred-pooled row) the transform fixed
# by its population's family.
_row_bijector(prior) = bijector(prior)
function _row_bijector(prior::CentredPoolPrior)
    return bijector(_population_template(prior.pool.population))
end

# The per-row inverse bijectors (unconstrained -> constrained), one per
# ESTIMATED flat parameter, in table-row order. `ComposedLogDensity` already
# carries `flat_priors` (flattened once at construction), so no table walk is
# needed here.
function _inverse_bijectors(prob::ComposedLogDensity)
    return map(inverse ∘ _row_bijector, prob.flat_priors)
end

# `to_constrained(prob, z)`: push each unconstrained coordinate through its
# row's inverse bijector, accumulating the log-Jacobian. Every transform here
# is univariate (one scalar prior per row — an ordinary prior, a
# stick-breaking `Beta`, a pooled hyperparameter, or a centred-pooled member
# read off its population), so the estimated dimension is unchanged and the
# map is element-wise; the total log-Jacobian is the sum of the per-row terms.
function ComposedDistributions.to_constrained(
        prob::ComposedLogDensity, z::AbstractVector)
    binvs = _inverse_bijectors(prob)
    length(z) == length(binvs) || throw(DimensionMismatch(
        "unconstrained vector has length $(length(z)) but $(prob.dist) has " *
        "$(length(binvs)) estimated parameters"))
    xs_and_logj = map((b, zi) -> with_logabsdet_jacobian(b, zi), binvs, z)
    x = [xi for (xi, _) in xs_and_logj]
    logjac = isempty(xs_and_logj) ? zero(eltype(z)) : sum(last, xs_and_logj)
    return x, logjac
end

end # module
