# Numeric `Distributions.quantile` for the composers with no closed form.
#
# `Resolve` / `Compete` invert their own marginal `cdf` by bracketed bisection
# in `src/composers/one_of_quantile.jl` -- no solver, no extension (#356). A
# `Sequential` chain instead delegates to the `Convolved` total it collapses
# to, whose quantile is ConvolvedDistributions' Nelder-Mead solve, so only the
# testitems that exercise that delegation load Optimization.jl +
# OptimizationOptimJL.jl.

@testitem "Resolve/Compete quantile needs no solver stack" begin
    # A fresh subprocess, not a check in this (shared, single-process)
    # TestItemRunner session: another testitem's `using Optimization,
    # OptimizationOptimJL` leaves those loaded for the rest of the process
    # regardless of ordering, so only an isolated process can show that the
    # one_of quantile answers with nothing but ComposedDistributions and
    # Distributions present. This is the regression guard for #356: the
    # inversion must not reach for a solver.
    script = """
    using ComposedDistributions, Distributions
    r = resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    c = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    for d in (r, c)
        q = quantile(d, 0.5)
        isfinite(q) || error("expected a finite quantile, got \$q")
        abs(cdf(d, q) - 0.5) < 1e-8 || error("quantile did not invert cdf")
    end
    isdefined(Main, :Optimization) && error("the solver stack was loaded")
    """
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`
    @test success(cmd)
end

@testitem "Resolve quantile inverts cdf" begin
    using Distributions

    # Independent brute-force reference: plain bisection written out here, so
    # this checks the result rather than the package's own machinery agreeing
    # with itself. Gamma/LogNormal outcomes have unbounded support, so the
    # upper bracket is a wide moment-based window rather than `maximum(d)`
    # (which is `Inf`).
    function bisect_quantile(d, p::Real; tol = 1e-12)
        flo = float(minimum(d))
        fhi = float(mean(d) + 50 * std(d))
        for _ in 1:200
            (fhi - flo) < tol && break
            mid = (flo + fhi) / 2
            cdf(d, mid) < p ? (flo = mid) : (fhi = mid)
        end
        return (flo + fhi) / 2
    end

    r = resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    for p in (0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999)
        q = quantile(r, p)
        # The bisection resolves to floating-point precision, so both the
        # round trip and the agreement with the reference are pinned far
        # tighter than the 1e-3 the solver-backed inversion could hold.
        @test cdf(r, q) ≈ p atol = 1e-10
        @test q ≈ bisect_quantile(r, p) rtol = 1e-8
    end

    # Boundary shortcuts return the support ends exactly.
    @test quantile(r, 0.0) == minimum(r)
    @test quantile(r, 1.0) == maximum(r)
    @test_throws ArgumentError quantile(r, -0.1)
    @test_throws ArgumentError quantile(r, 1.1)
    @test_throws ArgumentError quantile(r, NaN)
end

@testitem "Compete quantile inverts cdf" begin
    using Distributions

    function bisect_quantile(d, p::Real; tol = 1e-12)
        flo = float(minimum(d))
        fhi = float(mean(d) + 50 * std(d))
        for _ in 1:200
            (fhi - flo) < tol && break
            mid = (flo + fhi) / 2
            cdf(d, mid) < p ? (flo = mid) : (fhi = mid)
        end
        return (flo + fhi) / 2
    end

    c = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    for p in (0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999)
        q = quantile(c, p)
        @test cdf(c, q) ≈ p atol = 1e-10
        @test q ≈ bisect_quantile(c, p) rtol = 1e-8
    end

    @test quantile(c, 0.0) == minimum(c)
    @test quantile(c, 1.0) == maximum(c)
    @test_throws ArgumentError quantile(c, -0.1)
    @test_throws ArgumentError quantile(c, 1.1)
    @test_throws ArgumentError quantile(c, NaN)
end

@testitem "quantile brackets a cause with no finite mean (#344)" begin
    using Distributions

    # `mean(Cauchy())` is `NaN` (no finite mean) and a shape<=1 `Pareto`'s is
    # `Inf`, so the old mean-seeded solve handed its optimiser a non-finite
    # starting point (#344, diagnosed as the mechanism behind the #356 CI
    # hang). The bracketed inversion never asks for a mean: it grows a finite
    # bracket from the support by doubling.
    r = resolve(:death => (Cauchy(0.0, 1.0), 0.5),
        :disch => (Gamma(2.0, 1.5), 0.5))
    c = compete(:death => Cauchy(0.0, 1.0), :recover => Gamma(2.0, 1.5))
    heavy = compete(:heavy => Pareto(0.5, 1.0), :b => Gamma(3.0, 2.0))

    @test isnan(mean(Cauchy(0.0, 1.0)))
    @test !isfinite(mean(Pareto(0.5, 1.0)))

    for d in (r, c, heavy)
        # The support is doubly unbounded for the Cauchy nodes, so the
        # bracket has to be grown at both ends.
        lo, hi = ComposedDistributions._one_of_quantile_bracket(d, 0.5)
        @test isfinite(lo) && isfinite(hi)
        @test lo < hi

        for p in (0.01, 0.1, 0.5, 0.9, 0.99)
            q = quantile(d, p)
            @test isfinite(q)
            @test cdf(d, q) ≈ p atol = 1e-9
        end
    end
end

@testitem "quantile: bounded support, defective node, monotonicity" begin
    using Distributions

    # A bounded support needs no bracket growth at all: both ends come
    # straight from the node.
    b = resolve(:a => (Uniform(0.0, 1.0), 0.5), :b => (Uniform(0.5, 2.0), 0.5))
    @test minimum(b) == 0.0
    @test maximum(b) == 2.0
    for p in (0.1, 0.5, 0.9)
        @test cdf(b, quantile(b, p)) ≈ p atol = 1e-10
    end

    # A defective race: each cause fires with probability 1/2 and never
    # otherwise, so the joint survival flattens at 1/4 and the marginal cdf
    # tops out at 3/4. Any `p` above that ceiling has no finite solution, and
    # the answer is the support ceiling rather than whatever the last bracket
    # point happened to be.
    struct HalfFiring <: ContinuousUnivariateDistribution end
    Base.minimum(::HalfFiring) = 0.0
    Base.maximum(::HalfFiring) = Inf
    Distributions.logccdf(::HalfFiring, t::Real) = t <= 0 ? 0.0 :
                                                   log(0.5 + 0.5 * exp(-t))
    Distributions.ccdf(d::HalfFiring, t::Real) = exp(logccdf(d, t))
    Distributions.cdf(d::HalfFiring, t::Real) = -expm1(logccdf(d, t))

    defective = compete(:a => HalfFiring(), :b => HalfFiring())
    @test cdf(defective, 1.0e6) ≈ 0.75
    @test cdf(defective, quantile(defective, 0.5)) ≈ 0.5 atol = 1e-10
    @test quantile(defective, 0.9) == maximum(defective)

    # The inverse of a monotone cdf is monotone: a solver could return a
    # locally-converged point that breaks this, a bracketed root find cannot.
    c = compete(:death => Gamma(2.0, 3.0), :recover => LogNormal(0.5, 0.4))
    ps = 0.05:0.05:0.95
    qs = [quantile(c, p) for p in ps]
    @test issorted(qs)
    @test all(isfinite, qs)
end

@testitem "Sequential quantile: one-step exact, multi-step via Convolved" begin
    using Distributions
    using Optimization, OptimizationOptimJL

    # A one-step chain collapses to its bare leaf (`observed_distribution`
    # returns it unwrapped), so this is exact with no solver involved.
    one_step = sequential(:onset => Gamma(2.0, 1.0))
    @test quantile(one_step, 0.3) == quantile(Gamma(2.0, 1.0), 0.3)

    # A multi-step chain collapses to a `Convolved`, whose own quantile is
    # ConvolvedDistributions' Nelder-Mead solve, so this arm still needs the
    # solver stack loaded (see EpiAware/ConvolvedDistributions.jl: the same
    # bracketed root find belongs there).
    chain = sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_death => LogNormal(0.5, 0.4))
    total = observed_distribution(chain)
    for p in (0.1, 0.5, 0.9)
        q = quantile(chain, p)
        @test q ≈ quantile(total, p) atol = 1e-8
        @test cdf(total, q) ≈ p atol = 1e-3
    end
end

@testitem "Compete: window/panel breaks use a composite cause's quantile" begin
    using Distributions
    using Optimization, OptimizationOptimJL
    using ConvolvedDistributions: GaussLegendre, integrate

    # Same composite-cause fixture as the moment-fallback test in
    # composers.jl ("Compete: quadrature window survives a composite/heavy
    # cause"), which pins the *fallback* path's probs to 1e-6 against a
    # 65536-node reference. With the solver stack loaded here, `composite` (a
    # bare `Convolved`) answers `quantile`, so
    # `_component_quad_window`/`_cause_panel_breaks` take the quantile-ladder
    # branch instead of the `mean + 10*std` fallback that sibling test
    # exercises; the split stays at least as accurate.
    composite = observed_distribution(sequential(Gamma(2.0, 1.0),
        LogNormal(0.5, 0.4)))
    bad = compete(:composite => composite, :b => Gamma(3.0, 2.0))

    # The composite cause answers quantile directly once the solver stack is
    # loaded (a MethodError without it), so the panel-break ladder no longer
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

    # The same 1e-6 tolerance the sibling non-solver test pins the moment-
    # window fallback's own accuracy to, against the same independent
    # reference: the quantile-driven path agrees with it for sanity.
    @test p_quantile_path.composite ≈ ref.composite atol = 1e-6
    @test p_quantile_path.b ≈ ref.b atol = 1e-6
end
