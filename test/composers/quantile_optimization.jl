# Numeric `Distributions.quantile` for the composers with no closed form
# (`Sequential`, `Resolve`, `Compete`), backed by
# `ConvolvedDistributions.quantile_by_optimization`. `Resolve`/`Compete`'s own
# methods live in the `ComposedDistributionsOptimizationExt` extension
# (Optimization.jl + OptimizationOptimJL.jl); every testitem below that
# exercises them loads both to trigger it, mirroring
# ConvolvedDistributions' own `test/distributions/quantile.jl`.

@testitem "Resolve/Compete quantile: absent without the Optimization ext" begin
    # A fresh subprocess, not a check in this (shared, single-process)
    # TestItemRunner session: testitems do not run in file/textual order (an
    # earlier `using Optimization, OptimizationOptimJL` elsewhere in this
    # file would leave the extension loaded for the rest of the process
    # regardless of where this testitem sits), so only an isolated process
    # gives a reliable "extension not loaded" reading.
    # Without the extension, `quantile` on a `Resolve`/`Compete` still falls
    # through to `Statistics.quantile(itr, p)` (the same generic `quantile`
    # name, extended rather than shadowed by `Distributions`) since neither
    # type has a specific method yet, so the failure surfaces as a
    # `MethodError` on `iterate`, not a missing `quantile` method itself;
    # `hasmethod` on `quantile` alone cannot tell the two states apart.
    script = """
    using ComposedDistributions, Distributions
    r = resolve(:death => (Gamma(1.5, 1.0), 0.3), :disch => (Gamma(2.0, 1.5), 0.7))
    c = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    for d in (r, c)
        try
            quantile(d, 0.5)
            error("expected quantile without the Optimization ext to throw")
        catch e
            e isa MethodError || rethrow()
        end
    end
    """
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`
    @test success(cmd)
end

@testitem "Resolve quantile inverts cdf" begin
    using Distributions
    using Optimization, OptimizationOptimJL

    # Independent brute-force reference: plain bisection on cdf, not the
    # package's own Nelder-Mead solve, so this checks the result rather than
    # the same machinery agreeing with itself. Gamma/LogNormal outcomes have
    # unbounded support, so the upper bracket is a wide moment-based window
    # rather than `maximum(d)` (which is `Inf`).
    function bisect_quantile(d, p::Real; tol = 1e-10)
        flo = float(minimum(d))
        fhi = float(mean(d) + 50 * std(d))
        for _ in 1:100
            (fhi - flo) < tol && break
            mid = (flo + fhi) / 2
            cdf(d, mid) < p ? (flo = mid) : (fhi = mid)
        end
        return (flo + fhi) / 2
    end

    r = resolve(:death => (Gamma(1.5, 1.0), 0.3), :disch => (Gamma(2.0, 1.5), 0.7))
    for p in (0.1, 0.25, 0.5, 0.75, 0.9)
        q = quantile(r, p)
        @test cdf(r, q) ≈ p atol = 1e-3
        @test q ≈ bisect_quantile(r, p) atol = 1e-3
    end

    # Boundary shortcuts return the support ends exactly.
    @test quantile(r, 0.0) == minimum(r)
    @test quantile(r, 1.0) == maximum(r)
    @test_throws ArgumentError quantile(r, -0.1)
    @test_throws ArgumentError quantile(r, 1.1)
end

@testitem "Compete quantile inverts cdf" begin
    using Distributions
    using Optimization, OptimizationOptimJL

    function bisect_quantile(d, p::Real; tol = 1e-10)
        flo = float(minimum(d))
        fhi = float(mean(d) + 50 * std(d))
        for _ in 1:100
            (fhi - flo) < tol && break
            mid = (flo + fhi) / 2
            cdf(d, mid) < p ? (flo = mid) : (fhi = mid)
        end
        return (flo + fhi) / 2
    end

    c = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    for p in (0.1, 0.25, 0.5, 0.75, 0.9)
        q = quantile(c, p)
        @test cdf(c, q) ≈ p atol = 1e-3
        @test q ≈ bisect_quantile(c, p) atol = 1e-3
    end

    @test quantile(c, 0.0) == minimum(c)
    @test quantile(c, 1.0) == maximum(c)
    @test_throws ArgumentError quantile(c, -0.1)
    @test_throws ArgumentError quantile(c, 1.1)
end

@testitem "Sequential quantile: one-step exact, multi-step via Convolved" begin
    using Distributions
    using Optimization, OptimizationOptimJL

    # A one-step chain collapses to its bare leaf (`observed_distribution`
    # returns it unwrapped), so this is exact with no solver involved.
    one_step = sequential(:onset => Gamma(2.0, 1.0))
    @test quantile(one_step, 0.3) == quantile(Gamma(2.0, 1.0), 0.3)

    # A multi-step chain collapses to a `Convolved`; the chain's quantile
    # delegates to it, so it needs the extension only through that.
    chain = sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_death => LogNormal(0.5, 0.4))
    total = observed_distribution(chain)
    for p in (0.1, 0.5, 0.9)
        q = quantile(chain, p)
        @test q ≈ quantile(total, p) atol = 1e-8
        @test cdf(total, q) ≈ p atol = 1e-3
    end
end

@testitem "Compete: window/panel breaks use the quantile once the ext loads" begin
    using Distributions
    using Optimization, OptimizationOptimJL
    using ConvolvedDistributions: GaussLegendre, integrate

    # Same composite-cause fixture as the moment-fallback test in
    # composers.jl ("Compete: quadrature window survives a composite/heavy
    # cause"), which pins the *fallback* path's probs to 1e-6 against a
    # 65536-node reference. With the extension loaded here, `composite` (a
    # bare `Convolved`, no `quantile` of its own before this PR) now answers
    # `quantile`, so `_component_quad_window`/`_cause_panel_breaks` take the
    # quantile-ladder branch instead of the `mean + 10*std` fallback that
    # sibling test exercises; the split stays at least as accurate.
    composite = observed_distribution(sequential(Gamma(2.0, 1.0),
        LogNormal(0.5, 0.4)))
    bad = compete(:composite => composite, :b => Gamma(3.0, 2.0))

    # The composite cause now answers quantile directly (a MethodError
    # before this extension loaded), so the panel-break ladder no longer
    # falls back to the moment-based window for it.
    @test isfinite(quantile(composite, 0.9999))

    p_quantile_path = probs(bad)

    function reference_probs(c, hi; n = 65536)
        rule = GaussLegendre(; n = n)
        names = ComposedDistributions.component_names(c)
        vals = ntuple(length(names)) do j
            integrate(rule,
                t -> exp(ComposedDistributions._hazard_cause_logpdf(c, j, t)),
                0.0, hi)
        end
        return NamedTuple{names}(vals)
    end
    ref = reference_probs(bad, ComposedDistributions._hazard_quad_window(bad))

    # The same 1e-6 tolerance the sibling non-ext test pins the moment-
    # window fallback's own accuracy to, against the same independent
    # reference: the quantile-driven path agrees with it for sanity.
    @test p_quantile_path.composite ≈ ref.composite atol = 1e-6
    @test p_quantile_path.b ≈ ref.b atol = 1e-6
end
