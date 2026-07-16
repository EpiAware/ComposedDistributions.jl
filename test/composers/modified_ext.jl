# ComposedDistributions × ModifiedDistributions extension: a modified leaf peels
# correctly inside a composed tree (free_leaf reaches the inner delay,
# rewrap_leaf rebuilds the modifier, _shared_tag sees through it), and a
# thin(...) reporting probability surfaces as a free `:thin` parameter through
# the core's extra_leaf_params / set_extra_leaf_params hooks.

@testitem "Modified extension: free_leaf / rewrap_leaf / _shared_tag" begin
    using Distributions
    using ModifiedDistributions
    using ModifiedDistributions: affine, weight, thin, modify, get_dist

    # The extension loads once ModifiedDistributions is present.
    @test Base.get_extension(ComposedDistributions,
        :ComposedDistributionsModifiedDistributionsExt) !== nothing

    inner = Gamma(2.0, 1.0)
    aff = affine(inner; scale = 2.0, shift = 1.0)
    wt = weight(inner, 0.5)
    th = thin(inner, 0.3)
    mod = modify(inner, 0.5)

    # free_leaf reaches the inner free delay through each modifier.
    @test ComposedDistributions.free_leaf(aff) == inner
    @test ComposedDistributions.free_leaf(wt) == inner
    @test ComposedDistributions.free_leaf(th) == inner
    @test ComposedDistributions.free_leaf(mod) == inner

    # rewrap_leaf rebuilds the modifier around a new inner delay.
    new_inner = Gamma(3.0, 1.5)
    raff = ComposedDistributions.rewrap_leaf(aff, new_inner)
    @test ComposedDistributions.free_leaf(raff) == new_inner
    @test get_dist(raff) == new_inner
    rmod = ComposedDistributions.rewrap_leaf(mod, new_inner)
    @test rmod isa ModifiedDistributions.Modified
    @test ComposedDistributions.free_leaf(rmod) == new_inner
    @test rmod.effect == mod.effect
    @test rmod.link == mod.link

    # A shared tag is visible through a modifier wrapper. (`Shared`'s declared
    # value-support is generic `ValueSupport`, so it cannot itself be wrapped
    # in a `Modified` — that constructor requires a `Continuous`-typed inner
    # distribution — but `_shared_tag`/`free_leaf` peel through `Modified`
    # just as they do the other modifiers, tested above via `mod`.)
    tagged = ComposedDistributions.shared(:inc, inner)
    @test ComposedDistributions._shared_tag(tagged) == :inc
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

@testitem "Modified extension: thin factor surfaces through params_table" begin
    using Distributions
    using ModifiedDistributions: thin
    import ComposedDistributions: extra_leaf_params

    # thin(leaf, p) is a Transformed carrying a ThinOp; the core's extra-param
    # hooks now surface its reporting probability as a free `:thin` row,
    # mirroring CensoredDistributions' forward_transform.jl precedent.
    tree = compose((inc = Gamma(2.0, 1.0), cases = thin(LogNormal(1.5, 0.4), 0.3)))
    tbl = params_table(tree)
    @test tbl.param == [:shape, :scale, :mu, :sigma, :thin]

    thin_row = only(filter(i -> tbl.param[i] == :thin, eachindex(tbl.param)))
    @test tbl.edge[thin_row] === :cases
    @test tbl.value[thin_row] == 0.3
    @test tbl.support[thin_row] == (0.0, 1.0)

    leaf = event(tree, :cases)
    @test extra_leaf_params(leaf) ==
          (thin = (value = 0.3, support = (0.0, 1.0)),)
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

@testitem "Modified extension: uncertain specs seen through a modifier" begin
    using Distributions, Random
    using ModifiedDistributions: affine, modify

    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
    au = affine(u; scale = 2.0, shift = 0.0)

    # The spec protocol sees through the modifier, so the uncertainty is
    # visible to the routing predicate and the prior column.
    @test ComposedDistributions._uncertain_specs(au) == u.specs
    @test has_uncertain(au)

    # Likewise through a hazard-modified leaf.
    mu = modify(u, 0.5)
    @test ComposedDistributions._uncertain_specs(mu) == u.specs
    @test has_uncertain(mu)

    # The marginal `rand` draws through the modifier (a fresh parameter each
    # call), and `update` collapses the wrapped uncertainty, keeping the
    # modifier's fixed structure.
    tree = compose((onset_admit = au,))
    @test all(isfinite, values(rand(Xoshiro(1), tree)))
    collapsed = update(tree, (onset_admit = (shape = 3.0, scale = 1.0),))
    leaf = event(collapsed, :onset_admit)
    @test !has_uncertain(leaf)
    @test ComposedDistributions.free_leaf(leaf) == Gamma(3.0, 1.0)
end

@testitem "Modified extension: a Modified leaf reports no extra params" begin
    using Distributions
    using ModifiedDistributions: modify
    import ComposedDistributions: extra_leaf_params, set_extra_leaf_params

    # `Modified` can only wrap a `Continuous`-typed inner distribution (its own
    # constructor's restriction), and `Transformed`/`Weighted` declare a
    # generic `ValueSupport` rather than propagating their wrapped delay's
    # concrete support, so a `Modified` can never itself sit around a thinned
    # delay via the public constructors today. The peel-through methods added
    # here are future-proofing (symmetry with Affine/Weighted) for when that
    # changes; meanwhile a bare Modified leaf reports no extra params, same as
    # any other non-thinned leaf.
    mod = modify(Gamma(2.0, 1.0), 0.5)
    @test extra_leaf_params(mod) == (;)

    # Setting no extras is the identity.
    reset = set_extra_leaf_params(mod, (;))
    @test reset isa typeof(mod)
    @test extra_leaf_params(reset) == (;)
    @test reset.dist == mod.dist
    @test reset.effect == mod.effect
end

@testitem "Modified extension: Affine moments honour scale/shift" begin
    using Distributions, Statistics, Random
    using ModifiedDistributions: affine, weight, thin

    # A chain with an affine step: the overall mean/var must honour the affine
    # scale/shift, matching the samples `rand` draws. The default `_leaf_mean`
    # peels the affine off (`mean(free_leaf(leaf))`), understating both moments;
    # the ext's `_leaf_mean(::Affine)` / `_leaf_var(::Affine)` fix that.
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

    # Weighted / Transformed delegate their moments straight to the inner delay,
    # so their free-leaf moment already agrees — no scale/shift to honour.
    wt = weight(Gamma(2.0, 1.0), 0.5)
    th = thin(Gamma(2.0, 1.0), 0.3)
    @test mean(Sequential(wt, Gamma(3.0, 1.0))) ≈ mean(Gamma(2.0, 1.0)) + 3.0
    @test mean(Sequential(th, Gamma(3.0, 1.0))) ≈ mean(Gamma(2.0, 1.0)) + 3.0
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
    @test var(seq) ≈ aff_var + 3.0
    @test params_table(seq).param == [:shape, :scale, :shape, :scale]
    @test all(isfinite, values(rand(Xoshiro(1), seq)))

    # Parallel: the affine branch's honoured moment is its endpoint moment.
    par = parallel(:admit => aff, :notif => Gamma(3.0, 1.0))
    @test mean(par).admit ≈ aff_mean
    @test var(par).admit ≈ aff_var
    @test mean(par).notif ≈ 3.0
    @test params_table(par).param == [:shape, :scale, :shape, :scale]

    # Resolve: the affine outcome's honoured moment feeds the mixture moment.
    res = resolve(:recover => (aff, 0.6), :die => (Gamma(3.0, 1.0), 0.4))
    @test mean(res) ≈ 0.6 * aff_mean + 0.4 * 3.0
    # The affine outcome peels to its inner Gamma params; the branch-probability
    # simplex adds its own two entries.
    @test params_table(res).param ==
          [:shape, :scale, :shape, :scale, :recover, :die]

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
    @test params_table(chz).param == [:shape, :scale, :shape, :scale]
end

@testitem "Modified extension: a pooled spec seen through a modifier" begin
    using Distributions
    using ModifiedDistributions: modify

    # A pool spec attached to an uncertain inner delay, under a hazard modifier:
    # the spec protocol sees the Pool through the Modified wrapper, so routing
    # and codec treat the modified leaf as partially pooled.
    u = uncertain(Gamma(2.0, 1.0); shape = pool(:g))
    md = modify(u, -log(2.0); link = log)
    @test ComposedDistributions._uncertain_specs(md) == u.specs
    @test u.specs.shape isa Pool
    @test has_uncertain(md)
end

@testitem "Modified extension: Modified leaf moment is MD#44-blocked" begin
    using Distributions
    using ModifiedDistributions: modify

    md = modify(LogNormal(0.5, 0.4), -log(2.0); link = log)
    seq = Sequential(md, Gamma(3.0, 1.0))

    # A Modified has no analytic mean/var yet (blocked on
    # ModifiedDistributions#44's numeric cumulative-hazard path). The ext errors
    # informatively rather than silently returning the UNMODIFIED free-leaf
    # moment (which peeling to `mean(free_leaf(md))` would give). Revisit this
    # contract once #44 lands a numeric moment.
    @test_throws ArgumentError mean(seq)
    @test_throws ArgumentError var(seq)

    # The structural surface still works: the leaf peels and the tree scores.
    @test ComposedDistributions.free_leaf(seq.components[1]) ==
          LogNormal(0.5, 0.4)
    @test logpdf(seq, [1.5, 2.0]) isa Real
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

    # observed_distribution keeps the affine step (it is the observed delay, not
    # a free parameter), so convolving the chain honours it. Under
    # ConvolvedDistributions 0.2 the bare-distribution convolve_series is
    # discrete-only, so the chain path discretises the observed total first; the
    # hand-built equivalents discretise the same total before convolving.
    out = convolve_series(chain, series)
    maxlag = length(series) - 1
    @test out ==
          convolve_series(discretise_pmf(observed_distribution(chain), maxlag),
        series)
    @test out ==
          convolve_series(discretise_pmf(convolved(aff, Gamma(3.0, 1.0)), maxlag),
        series)
    @test length(out) == length(series)
end

@testitem "Modified extension: a Varying leaf mapping to affine resolves" begin
    using Distributions
    using ModifiedDistributions: affine

    # A time-varying leaf whose map yields an affine-modified delay. instantiate
    # resolves the Varying to a concrete affine leaf, which then peels through
    # the modifier extension exactly like a plain affine leaf — the peel composes
    # through both wrappers.
    v = varying(
        t -> affine(Gamma(2.0, 1.0 + 0.1 * t); scale = 2.0, shift = 1.0);
        covariate = :time,
        reference = affine(Gamma(2.0, 1.0); scale = 2.0, shift = 1.0))
    tree = sequential(:step => v, :tail => Gamma(3.0, 1.0))
    @test has_varying(tree)

    resolved = instantiate(tree, Context(time = 5.0))
    @test !has_varying(resolved)

    # The resolved step is the affine at t = 5, peeled to its inner Gamma params.
    @test params_table(resolved).param == [:shape, :scale, :shape, :scale]
    @test ComposedDistributions.free_leaf(event(resolved, :step)) ==
          Gamma(2.0, 1.5)
    # The overall chain moment honours the resolved affine (mean 2*3 + 1 = 7).
    @test mean(resolved) ≈ (2.0 * mean(Gamma(2.0, 1.5)) + 1.0) + 3.0
end

@testitem "Modified extension: a modifier wrapping a Varying leaf resolves" begin
    using Distributions
    using ModifiedDistributions
    using ModifiedDistributions: affine, weight, thin, modify

    # The OUTER form: a modifier wraps a Varying leaf directly. Without the
    # instantiate/has_varying descent this silently scores against the
    # reference (the footgun has_varying guards against).
    v = varying(t -> LogNormal(1.0 + 0.1 * t, 0.5);
        covariate = :time, reference = LogNormal(1.0, 0.5))
    ctx = Context(time = 3.0)
    resolved_inner = instantiate(v, ctx)

    @test has_varying(affine(v; scale = 2.0))
    @test has_varying(weight(v, 0.5))
    @test has_varying(thin(v, 0.3))
    @test has_varying(modify(v, 0.5))
    @test !has_varying(modify(LogNormal(1.0, 0.5), 0.5))

    @test instantiate(affine(v; scale = 2.0), ctx) ==
          affine(resolved_inner; scale = 2.0)
    @test instantiate(weight(v, 0.5), ctx) == weight(resolved_inner, 0.5)
    @test instantiate(thin(v, 0.3), ctx) == thin(resolved_inner, 0.3)

    rm = instantiate(modify(v, 0.5), ctx)
    @test rm isa ModifiedDistributions.Modified
    @test rm.dist == resolved_inner
    @test !has_varying(rm)

    # End to end: a Sequential chain with a modified varying step.
    chain = sequential(:step => modify(v, 0.5), :tail => Gamma(3.0, 1.0))
    @test has_varying(chain)
    @test !has_varying(instantiate(chain, ctx))
end

@testitem "Modified extension: modified leaves drive convolve_series (#117)" begin
    using Distributions
    using ModifiedDistributions: affine, weight, thin

    # ModifiedDistributions#40: a Modified-wrapped chain step must participate in
    # the observed-total collapse that drives a count series, not be silently
    # dropped. Each modifier family (affine / weight / thin) wraps the first step
    # of a Sequential; the chain lowers through observed_distribution and
    # convolve_series identically to the hand-built Convolved of the same wrapped
    # leaf. Under ConvolvedDistributions 0.2 the bare convolve_series is
    # discrete-only, so the observed total is discretised first. One matrix cell
    # per modifier.
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
    maxlag = length(series) - 1
    tail = Gamma(3.0, 1.0)

    for mod in (affine(Gamma(2.0, 1.0); scale = 2.0, shift = 1.0),
        weight(Gamma(2.0, 1.0), 0.5),
        thin(Gamma(2.0, 1.0), 0.3))
        chain = Sequential(mod, tail)

        # The collapse accepts the wrapped leaf: the modifier rides into the
        # observed Convolved total as the first component.
        od = observed_distribution(chain)
        @test od isa Convolved
        @test od.components[1] == mod

        # The convolved output matches discretising the observed total by hand,
        # and equals the direct Convolved of the wrapped leaf and the tail.
        out = convolve_series(chain, series)
        @test out == convolve_series(discretise_pmf(od, maxlag), series)
        @test out ==
              convolve_series(discretise_pmf(convolved(mod, tail), maxlag),
            series)
        @test length(out) == length(series)
    end
end

@testitem "Modified extension: extra param survives an unsupplied round-trip" begin
    using Distributions, Random
    using ModifiedDistributions: thin, get_dist
    import ComposedDistributions: extra_leaf_params

    # The BoundsError crux (#170): a thinned leaf carries an extra `:thin`
    # parameter appended after its native params, but `params(free_leaf(leaf))`
    # holds only the native values. A round-trip that re-pins or draws a NATIVE
    # parameter without supplying `:thin` must fall its `:thin` slot back to the
    # extra map, not index past the native tuple. Before the fix each of the
    # three fallback sites (`uncertain`, `_uncertain_leaf`, `_merge_leaf`) threw.
    th = thin(Gamma(2.0, 1.0), 0.3)

    # `uncertain` with a spec on one native param and a Real pin on the other:
    # exercises the pinning `ntuple` fallback (the `:thin` slot reads the extra
    # value, not `tvals[3]`).
    u = uncertain(th; shape = LogNormal(log(2.0), 0.2), scale = 1.5)
    @test extra_leaf_params(u) == (thin = (value = 0.3, support = (0.0, 1.0)),)
    tbl = params_table(compose((cases = u,)))
    @test tbl.param == [:shape, :scale, :thin]
    @test tbl.value[findfirst(==(:thin), tbl.param)] == 0.3

    # The marginal `rand` draws the spec'd native param and rebuilds the leaf,
    # keeping `:thin`: exercises `_uncertain_leaf`.
    tree = compose((cases = u,))
    @test all(isfinite, values(rand(Xoshiro(1), tree)))

    # `update` in merge mode (a distribution value) re-pins one native param and
    # makes another uncertain while `:thin` is untouched: exercises
    # `_merge_leaf`. The thin factor survives the fallback.
    merged = update(tree, (cases = (shape = 3.0, scale = LogNormal(0.0, 1.0)),))
    leaf = event(merged, :cases)
    @test extra_leaf_params(leaf) == (thin = (value = 0.3, support = (0.0, 1.0)),)
    @test ComposedDistributions.free_leaf(leaf) == Gamma(3.0, 1.5)

    # A strict-mode collapse supplying every coordinate (native + `:thin`) also
    # round-trips, updating the thin factor.
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
