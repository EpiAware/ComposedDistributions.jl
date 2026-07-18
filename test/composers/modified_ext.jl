# ComposedDistributions × ModifiedDistributions integration: a modified leaf
# (`Affine` / `Weighted` / `Transformed` / `Modified`) peels correctly inside a
# composed tree. The full leaf-protocol implementation for these types now
# lives in ModifiedDistributions' `ModifiedDistributionsComposedDistributionsExt`
# (#170 step 2: this package no longer carries a reverse extension of its
# own), so these tests exercise a representative subset of that behaviour
# through the public leaf protocol, confirming the two packages still
# interoperate end to end. The exhaustive extension-behaviour coverage lives
# in ModifiedDistributions' own test suite.

@testitem "Modified extension: free_leaf / rewrap_leaf / shared_tag" begin
    using Distributions
    using ModifiedDistributions
    using ModifiedDistributions: affine, weight, thin, modify, get_dist

    # The extension loads once ModifiedDistributions is present.
    @test Base.get_extension(ModifiedDistributions,
        :ModifiedDistributionsComposedDistributionsExt) !== nothing

    inner = Gamma(2.0, 1.0)
    aff = affine(inner; scale = 2.0, shift = 1.0)
    mod = modify(inner, 0.5)

    # free_leaf reaches the inner free delay through each modifier.
    @test ComposedDistributions.free_leaf(aff) == inner
    @test ComposedDistributions.free_leaf(mod) == inner

    # rewrap_leaf rebuilds the modifier around a new inner delay.
    new_inner = Gamma(3.0, 1.5)
    raff = ComposedDistributions.rewrap_leaf(aff, new_inner)
    @test ComposedDistributions.free_leaf(raff) == new_inner
    @test get_dist(raff) == new_inner

    # A shared tag is visible through a modifier wrapper.
    tagged = ComposedDistributions.shared(:inc, inner)
    @test ComposedDistributions.shared_tag(tagged) == :inc
    @test get_dist(tagged) == inner
end

@testitem "Modified extension: params_table peels a modified leaf" begin
    using Distributions
    using ModifiedDistributions: affine, modify

    # A composed tree with an affine-modified leaf reports only the inner free
    # delay's parameters (the affine scale/shift are fixed structure).
    tree = compose((onset_admit = affine(Gamma(2.0, 1.0); scale = 2.0),
        admit_death = LogNormal(0.5, 0.4)))
    tbl = params_table(tree)
    @test tbl.param == [:shape, :scale, :mu, :sigma]

    # Likewise a hazard-modified leaf (the effect/link are fixed structure).
    mtree = compose((onset_admit = modify(Gamma(2.0, 1.0), 0.5),
        admit_death = LogNormal(0.5, 0.4)))
    mtbl = params_table(mtree)
    @test mtbl.param == [:shape, :scale, :mu, :sigma]
end

@testitem "Modified extension: thin factor round-trips through update" begin
    using Distributions
    using ModifiedDistributions
    using ModifiedDistributions: thin, affine, get_dist
    import ComposedDistributions: set_extra_leaf_params

    tree = compose((cases = thin(LogNormal(1.5, 0.4), 0.3),))

    # update re-routes a new thin weight into the ThinOp, keeping the inner
    # delay params, and the new weight surfaces back through params_table.
    updated = update(tree, (cases = (mu = 1.0, sigma = 0.5, thin = 0.6),))
    leaf = event(updated, :cases)
    @test leaf isa ModifiedDistributions.Transformed
    @test leaf.op isa ModifiedDistributions.ThinOp
    @test leaf.op.factor == 0.6
    @test get_dist(leaf) == LogNormal(1.0, 0.5)

    rt = params_table(updated)
    @test rt.value[findfirst(==(:thin), rt.param)] == 0.6

    # A thinned leaf wrapped in a fixed-structure modifier (affine) still
    # peels through to the same ThinOp, so set_extra_leaf_params reaches it.
    wrapped = affine(thin(Gamma(2.0, 1.0), 0.4); scale = 2.0)
    reset = set_extra_leaf_params(wrapped, (thin = 0.9,))
    @test reset isa ModifiedDistributions.Affine
    @test reset.dist.op.factor == 0.9
end

@testitem "Modified extension: Affine moments honour scale/shift" begin
    using Distributions, Statistics, Random
    using ModifiedDistributions: affine

    # A chain with an affine step: the overall mean/var must honour the affine
    # scale/shift, matching the samples `rand` draws.
    seq = Sequential(affine(Gamma(2.0, 1.0); scale = 2.0, shift = 1.0),
        Gamma(3.0, 1.0))
    # Analytic honoured totals: leaf affine mean = 2*2 + 1 = 5, plus Gamma(3,1)
    # mean 3 → 8; leaf affine var = 2^2 * 2 = 8, plus Gamma(3,1) var 3 → 11.
    @test mean(seq) ≈ 8.0
    @test var(seq) ≈ 11.0

    # Monte-Carlo cross-check the analytic totals against the samples.
    rng = Xoshiro(7)
    tot = [sum(values(rand(rng, seq))) for _ in 1:200_000]
    @test mean(tot) ≈ 8.0 rtol = 0.02
    @test var(tot) ≈ 11.0 rtol = 0.02
end

@testitem "Modified extension: a modifier leaf inside each composer" begin
    using Distributions, Random
    using ModifiedDistributions: affine

    aff = affine(Gamma(2.0, 1.0); scale = 2.0, shift = 1.0)
    aff_mean = 2.0 * mean(Gamma(2.0, 1.0)) + 1.0   # 5.0
    aff_var = 2.0^2 * var(Gamma(2.0, 1.0))         # 8.0

    # Sequential: the affine step's honoured moment adds into the chain total.
    seq = sequential(:onset_admit => aff, :admit_death => Gamma(3.0, 1.0))
    @test mean(seq) ≈ aff_mean + 3.0
    @test params_table(seq).param == [:shape, :scale, :shape, :scale]
    @test all(isfinite, values(rand(Xoshiro(1), seq)))

    # Parallel: the affine branch's honoured moment is its endpoint moment.
    par = parallel(:admit => aff, :notif => Gamma(3.0, 1.0))
    @test mean(par).admit ≈ aff_mean
    @test params_table(par).param == [:shape, :scale, :shape, :scale]

    # Resolve: the affine outcome's honoured moment feeds the mixture moment.
    res = resolve(:recover => (aff, 0.6), :die => (Gamma(3.0, 1.0), 0.4))
    @test mean(res) ≈ 0.6 * aff_mean + 0.4 * 3.0

    # Compete: the racing-hazard marginal honours the affine through its own
    # cdf; the moment is finite and the affine step peels in params_table.
    cmp = compete(:recover => aff, :die => Gamma(3.0, 1.0))
    @test isfinite(mean(cmp))
    @test params_table(cmp).param == [:shape, :scale, :shape, :scale]

    # Choose: a whole-tree moment is ill-defined, so take the chosen
    # alternative's moment; it honours the affine.
    chz = choose(:a => aff, :b => Gamma(3.0, 1.0))
    @test_throws ArgumentError mean(chz)
    @test mean(event(chz, :a)) ≈ aff_mean
end

@testitem "Modified extension: AD through an affine modifier" begin
    using Distributions
    using ModifiedDistributions: affine
    using ForwardDiff

    # logpdf of a chain whose first step is an affine-modified Gamma, as a
    # function of the inner Gamma's (shape, scale). The affine scale/shift are
    # fixed structure, so the gradient flows through the inner delay's own
    # logpdf via the change of variables the affine applies.
    x = [3.2, 1.1]
    f = θ -> logpdf(
        sequential(:a => affine(Gamma(θ[1], θ[2]); scale = 2.0, shift = 1.0),
            :b => LogNormal(0.5, 0.4)), x)
    θ0 = [2.0, 1.0]
    g = ForwardDiff.gradient(f, θ0)
    @test all(isfinite, g)

    # Matches central finite differences.
    h = 1e-6
    fd = map(eachindex(θ0)) do i
        e = zeros(length(θ0))
        e[i] = h
        (f(θ0 .+ e) - f(θ0 .- e)) / (2h)
    end
    @test g ≈ fd rtol = 1e-4
end

@testitem "Modified extension: convolve_series with an affine chain step" begin
    using Distributions
    using ModifiedDistributions: affine

    aff = affine(Gamma(2.0, 1.0); scale = 2.0, shift = 1.0)
    chain = Sequential(aff, Gamma(3.0, 1.0))
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]

    # observed_distribution keeps the affine step (it is the observed delay,
    # not a free parameter), so convolving the chain honours it. Under
    # ConvolvedDistributions 0.2 the bare-distribution convolve_series is
    # discrete-only, so the chain path discretises the observed total first.
    out = convolve_series(chain, series)
    maxlag = length(series) - 1
    @test out ==
          convolve_series(discretise_pmf(observed_distribution(chain), maxlag),
        series)
    @test length(out) == length(series)
end

@testitem "Modified extension: a modifier wrapping a Varying leaf resolves" begin
    using Distributions
    using ModifiedDistributions
    using ModifiedDistributions: modify

    # Without the instantiate/has_varying descent this would silently score
    # against the reference (the footgun has_varying guards against).
    v = varying(t -> LogNormal(1.0 + 0.1 * t, 0.5);
        covariate = :time, reference = LogNormal(1.0, 0.5))
    ctx = Context(time = 3.0)

    @test has_varying(modify(v, 0.5))
    @test !has_varying(modify(LogNormal(1.0, 0.5), 0.5))

    rm = instantiate(modify(v, 0.5), ctx)
    @test rm isa ModifiedDistributions.Modified
    @test !has_varying(rm)

    # End to end: a Sequential chain with a modified varying step.
    chain = sequential(:step => modify(v, 0.5), :tail => Gamma(3.0, 1.0))
    @test has_varying(chain)
    @test !has_varying(instantiate(chain, ctx))
end

@testitem "Modified extension: extra param survives an unsupplied round-trip" begin
    using Distributions, Random
    using ModifiedDistributions: thin, get_dist
    import ComposedDistributions: extra_leaf_params

    # The BoundsError crux (#170): a thinned leaf carries an extra `:thin`
    # parameter appended after its native params, but `params(free_leaf(leaf))`
    # holds only the native values. A round-trip that re-pins or draws a
    # NATIVE parameter without supplying `:thin` must fall its `:thin` slot
    # back to the extra map, not index past the native tuple.
    th = thin(Gamma(2.0, 1.0), 0.3)

    u = uncertain(th; shape = LogNormal(log(2.0), 0.2), scale = 1.5)
    @test extra_leaf_params(u) == (thin = (value = 0.3, support = (0.0, 1.0)),)
    tbl = params_table(compose((cases = u,)))
    @test tbl.param == [:shape, :scale, :thin]

    tree = compose((cases = u,))
    @test all(isfinite, values(rand(Xoshiro(1), tree)))

    merged = update(tree, (cases = (shape = 3.0, scale = LogNormal(0.0, 1.0)),))
    leaf = event(merged, :cases)
    @test extra_leaf_params(leaf) == (thin = (value = 0.3, support = (0.0, 1.0)),)
    @test ComposedDistributions.free_leaf(leaf) == Gamma(3.0, 1.5)

    collapsed = update(tree, (cases = (shape = 2.0, scale = 1.0, thin = 0.7),))
    cleaf = event(collapsed, :cases)
    @test extra_leaf_params(cleaf) == (thin = (value = 0.7, support = (0.0, 1.0)),)
    @test get_dist(cleaf) == Gamma(2.0, 1.0)
end

@testitem "Leaf protocol: published names reachable, thin alias gone (#170)" begin
    using Distributions

    # The generalised extra-parameter hook is public and empty for a plain leaf.
    @test ComposedDistributions.extra_leaf_params(Gamma(2.0, 1.0)) == (;)
    @test ComposedDistributions.set_extra_leaf_params(Gamma(2.0, 1.0), (;)) ==
          Gamma(2.0, 1.0)

    # The published leaf-protocol names resolve (public, reached qualified).
    for name in (:uncertain_specs, :shared_tag, :leaf_mean, :leaf_var,
        :leaf_param_names, :leaf_detail_lines, :extra_leaf_params,
        :set_extra_leaf_params)
        @test isdefined(ComposedDistributions, name)
    end

    # The scalar thin-factor hook is fully replaced, not aliased.
    @test !isdefined(ComposedDistributions, :_thin_factor)
    @test !isdefined(ComposedDistributions, :_set_thin_factor)
end
