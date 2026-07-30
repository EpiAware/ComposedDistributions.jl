# Test-only delay discretisation for the vector-convolution tests.
#
# ConvolvedDistributions 0.2 shipped `discretise_pmf`; 0.3 dropped it, because
# discretising a continuous delay is a censoring choice CensoredDistributions.jl
# owns (Convolved#68). These tests only need *some* PMF to feed
# `convolve_series(pmf, series)`, so they build the same CDF-difference masses
# here rather than this package growing a public discretiser — where that
# belongs is still open in #108.

@testsnippet DelayMasses begin
    using Distributions: UnivariateDistribution, cdf

    # Masses on `[0, 1), [1, 2), ..., [maxlag, maxlag + 1)`: raw CDF
    # differences, clamped at zero, never renormalised.
    function delay_masses(d::UnivariateDistribution, maxlag::Integer)
        return map(0:maxlag) do k
            mass = cdf(d, float(k + 1)) - cdf(d, float(k))
            max(mass, zero(mass))
        end
    end
end
