# The recurrent-operator ('recur') seam end to end from this package's side
# (#82): a time-varying chain resolves per step against a Context, each resolved
# chain collapses to its observed total, discretises to a delay PMF, and drives
# the vector convolution surface. That per-step-kernel sequence is exactly what
# a renewal generator consumes (I[t] = R_t * sum_s g_s I[t-s]). The renewal
# layer itself is handed to a future RenewalDistributions.jl, so here we pin
# only that the seam it stands on holds. Refs #82.

@testitem "recur seam: per-step kernels drive the convolution surface" begin
    using Distributions

    # A chain whose onset->admit delay lengthens with calendar time, so the
    # observed onset->death kernel drifts step to step.
    chain = sequential(:onset_admit => varying(t -> Gamma(2.0, 1.0 + 0.1 * t)),
        :admit_death => LogNormal(0.5, 0.4))
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
    maxlag = length(series) - 1
    times = (0.0, 5.0, 10.0)

    # Per step: instantiate at the context, collapse to the observed total,
    # discretise it to the delay PMF the convolution surface consumes.
    ods = map(t -> observed_distribution(instantiate(chain, Context(time = t))), times)
    kernels = map(od -> discretise_pmf(od, maxlag), ods)

    # The per-step kernels genuinely differ across the context: the resolved
    # total's mean rises with t, shifting mass to longer lags.
    @test mean(ods[1]) < mean(ods[2]) < mean(ods[3])
    @test kernels[1].masses != kernels[2].masses
    @test kernels[2].masses != kernels[3].masses

    # Each kernel drives the vector convolution identically to rebuilding it
    # straight from the chain resolved at that step: the seam is transparent.
    for (t, k) in zip(times, kernels)
        direct = discretise_pmf(
            observed_distribution(instantiate(chain, Context(time = t))), maxlag)
        @test convolve_series(k, series) == convolve_series(direct, series)
    end

    # One hand-rolled renewal step off the last kernel: I_next = sum_s g_s I[t-s]
    # (the modulator-free renewal recurrence a future RenewalDistributions.jl
    # owns). It must be finite, positive, and match the explicit per-lag sum.
    g = kernels[end].masses
    incidence = [1.0, 2.0, 4.0, 7.0, 9.0]
    window = reverse(incidence)
    n = min(length(g), length(incidence))
    I_next = sum(g[1:n] .* window[1:n])
    manual = sum(g[s] * incidence[end - s + 1] for s in 1:n)
    @test I_next ≈ manual
    @test isfinite(I_next)
    @test I_next > 0
end
