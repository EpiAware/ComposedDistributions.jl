# Context-indexed leaves (`Varying`) and the `instantiate` resolution seam:
# a leaf varies with a covariate, and resolving a tree against a `Context`
# yields a concrete stationary tree that scores/samples/convolves unchanged.

@testitem "Varying: constructor and reference delegation" begin
    using Distributions

    f = t -> Gamma(2.0, 1.0 + 0.1 * t)

    # Default reference is the map at the origin (t = 0).
    d = varying(f)
    @test d isa ComposedDistributions.Varying
    @test d.covariate === :time
    @test d.reference == Gamma(2.0, 1.0)

    # A Varying leaf delegates every scalar query to its reference.
    @test mean(d) == mean(Gamma(2.0, 1.0))
    @test var(d) == var(Gamma(2.0, 1.0))
    @test logpdf(d, 1.5) == logpdf(Gamma(2.0, 1.0), 1.5)
    @test cdf(d, 2.0) == cdf(Gamma(2.0, 1.0), 2.0)
    @test quantile(d, 0.5) == quantile(Gamma(2.0, 1.0), 0.5)
    @test params(d) == params(Gamma(2.0, 1.0))
    @test minimum(d) == minimum(Gamma(2.0, 1.0))
    @test insupport(d, 1.0) == insupport(Gamma(2.0, 1.0), 1.0)

    # An explicit reference wins (needed for a non-numeric covariate where
    # `f(0.0)` is not meaningful).
    ref = Gamma(3.0, 2.0)
    d2 = varying(k -> k === :a ? Gamma(1.0, 1.0) : Gamma(5.0, 1.0);
        covariate = :region, reference = ref)
    @test d2.covariate === :region
    @test d2.reference == ref
    @test mean(d2) == mean(ref)

    # A non-distribution reference is rejected.
    @test_throws ArgumentError varying(t -> Gamma(2.0, 1.0); reference = 3.0)
end

@testitem "instantiate: leaf resolution and identity defaults" begin
    using Distributions

    d = varying(t -> Gamma(2.0, 1.0 + 0.1 * t))

    # A Varying leaf resolves to its map at the context covariate.
    @test instantiate(d, Context(time = 5.0)) == Gamma(2.0, 1.5)
    @test instantiate(d, Context(time = 0.0)) == Gamma(2.0, 1.0)

    # A fixed leaf is the identity under any context.
    fixed = LogNormal(0.5, 0.4)
    @test instantiate(fixed, Context(time = 5.0)) === fixed

    # `nothing` context is always a no-op.
    @test instantiate(d, nothing) === d
    @test instantiate(fixed, nothing) === fixed

    # A strata-style categorical covariate resolves by lookup.
    strat = varying(k -> k === :north ? Gamma(2.0, 1.0) : Gamma(4.0, 1.0);
        covariate = :region, reference = Gamma(2.0, 1.0))
    @test instantiate(strat, Context(region = :north)) == Gamma(2.0, 1.0)
    @test instantiate(strat, Context(region = :south)) == Gamma(4.0, 1.0)

    # A missing covariate is a clear error.
    @test_throws ArgumentError instantiate(d, Context(region = :north))
end

@testitem "instantiate: resolves a tree to a concrete stationary tree" begin
    using Distributions

    # A two-step chain with a time-varying first step.
    tree = compose((onset_admit = varying(t -> Gamma(2.0, 1.0 + 0.1 * t)),
        admit_death = LogNormal(0.5, 0.4)))

    resolved = instantiate(tree, Context(time = 5.0))

    # The resolved tree equals the hand-built stationary tree at t = 5, so it
    # scores / samples / convolves identically. Structure and names are kept.
    expected = compose((onset_admit = Gamma(2.0, 1.5),
        admit_death = LogNormal(0.5, 0.4)))
    @test resolved == expected
    @test event_names(resolved) == event_names(tree)

    # Resolving at the origin reproduces the reference tree.
    @test instantiate(tree, Context(time = 0.0)) ==
          compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))

    # A stationary tree is untouched.
    stat = compose((a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4)))
    @test instantiate(stat, Context(time = 5.0)) == stat
end

@testitem "instantiate: through Sequential / Parallel / Resolve / Choose" begin
    using Distributions

    v(a) = varying(t -> Gamma(a, 1.0 + 0.1 * t))

    seq = Sequential(v(2.0), LogNormal(0.5, 0.4))
    @test instantiate(seq, Context(time = 10.0)) ==
          Sequential(Gamma(2.0, 2.0), LogNormal(0.5, 0.4))

    par = Parallel((v(1.0), v(3.0)), (:a, :b))
    @test instantiate(par, Context(time = 10.0)) ==
          Parallel((Gamma(1.0, 2.0), Gamma(3.0, 2.0)), (:a, :b))

    # A Resolve node: its varying delays resolve, branch probs preserved.
    res = resolve(:death => (v(1.5), 0.3), :disch => Gamma(2.0, 1.5))
    rres = instantiate(res, Context(time = 10.0))
    @test rres isa ComposedDistributions.Resolve
    @test rres == resolve(:death => (Gamma(1.5, 2.0), 0.3),
        :disch => Gamma(2.0, 1.5))

    # A Choose node: the selected alternative resolves.
    ch = choose(:index => v(2.0), :sourced => Gamma(4.0, 1.5))
    rch = instantiate(ch, Context(time = 10.0))
    @test logpdf(rch, 3.0; kind = :index) == logpdf(Gamma(2.0, 2.0), 3.0)
end

@testitem "instantiate: the convolution kernel varies with the context" begin
    using Distributions

    # A Sequential chain collapses to a Convolved kernel; with a varying step
    # that kernel is context-dependent — resolve, then convolve. (A `compose`
    # NamedTuple would build a Parallel, which has no single observed scalar.)
    chain = sequential(:onset_admit => varying(t -> Gamma(2.0, 1.0 + 0.1 * t)),
        :admit_death => LogNormal(0.5, 0.4))

    k0 = observed_distribution(instantiate(chain, Context(time = 0.0)))
    k5 = observed_distribution(instantiate(chain, Context(time = 5.0)))

    # Different contexts give different kernels; each equals the stationary
    # collapse of the chain resolved at that context.
    @test k0 == observed_distribution(sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_death => LogNormal(0.5, 0.4)))
    @test k5 == observed_distribution(sequential(:onset_admit => Gamma(2.0, 1.5),
        :admit_death => LogNormal(0.5, 0.4)))
    @test mean(k5) > mean(k0)
end

@testitem "Varying: introspection is transparent to the reference" begin
    using Distributions

    # The varying map is fixed structure; params_table shows the reference's
    # free parameters and a Varying leaf peels/rewraps like any wrapper.
    inner = Gamma(2.0, 1.0)
    d = varying(t -> Gamma(2.0, 1.0 + 0.1 * t))
    @test ComposedDistributions.free_leaf(d) == inner

    rebuilt = ComposedDistributions.rewrap_leaf(d, Gamma(3.0, 1.5))
    @test rebuilt isa ComposedDistributions.Varying
    @test rebuilt.reference == Gamma(3.0, 1.5)
    @test rebuilt.covariate === d.covariate

    # A shared tag is visible through a Varying leaf.
    tagged = varying(t -> Gamma(2.0, 1.0 + 0.1 * t);
        reference = ComposedDistributions.shared(:inc, Gamma(2.0, 1.0)))
    @test ComposedDistributions._shared_tag(tagged) == :inc

    # params_table over a tree with a varying leaf matches the same tree built
    # from the leaf's reference: the varying map is fixed structure, so only the
    # reference's free parameters are inventoried. Compare the columns (a
    # ParamsTable has no value-`==`), reached via the forwarded property access.
    tbl = params_table(compose((onset_admit = d, admit_death = LogNormal(0.5, 0.4))))
    ref = params_table(compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4))))
    @test tbl.edge == ref.edge
    @test tbl.param == ref.param
    @test tbl.value == ref.value
    @test tbl.support == ref.support
end
