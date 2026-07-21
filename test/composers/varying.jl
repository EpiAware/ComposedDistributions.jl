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

    # A Compete racing-hazard node: its varying delays resolve in place.
    cmp = compete(:death => v(1.5), :recover => Gamma(3.0, 2.0))
    rcmp = instantiate(cmp, Context(time = 10.0))
    @test rcmp isa ComposedDistributions.Compete
    @test rcmp == compete(:death => Gamma(1.5, 2.0), :recover => Gamma(3.0, 2.0))
end

@testitem "has_varying: flags an un-instantiated tree" begin
    using Distributions

    chain = sequential(:a => varying(t -> Gamma(2.0, 1.0 + 0.1 * t)),
        :b => LogNormal(0.5, 0.4))

    # A raw tree with a Varying leaf is flagged; the resolved tree is clean.
    @test has_varying(chain)
    @test !has_varying(instantiate(chain, Context(time = 3.0)))

    # A stationary tree and a plain leaf are never flagged; a bare Varying is.
    @test !has_varying(sequential(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4)))
    @test !has_varying(Gamma(2.0, 1.0))
    @test has_varying(varying(t -> Gamma(2.0, 1.0 + 0.1 * t)))

    # It sees a Varying nested inside a one_of node.
    @test has_varying(resolve(:death => (varying(t -> Gamma(1.5, 1.0 + 0.1 * t)), 0.3),
        :disch => Gamma(2.0, 1.5)))
end

@testitem "required_covariates / required_parameters / missing_covariates (#266)" begin
    using Distributions

    # A stationary, fully-concrete tree needs neither covariates nor parameters.
    stationary = sequential(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4))
    @test isempty(required_covariates(stationary))
    @test isempty(required_parameters(stationary))
    @test isempty(missing_covariates(stationary, Context()))

    # A single Varying leaf: the covariate is keyed to its edge path.
    tree = sequential(:onset => varying(t -> Gamma(2.0, 1.0 + 0.1 * t)),
        :admit => LogNormal(0.5, 0.4))
    rc = required_covariates(tree)
    @test rc == Dict(:time => [:onset])
    @test missing_covariates(tree, Context(region = "a")) == [:time]
    @test isempty(missing_covariates(tree, Context(time = 4.0)))

    # Two leaves sharing the same covariate name both land under one key.
    both = sequential(:onset => varying(t -> Gamma(2.0, 1.0 + 0.1 * t)),
        :admit => varying(t -> LogNormal(0.5, 0.4 + 0.01 * t)))
    @test Set(required_covariates(both)[:time]) == Set([:onset, :admit])

    # A Varying leaf nested inside a Resolve outcome is still found.
    nested = resolve(:death => (varying(t -> Gamma(1.5, 1.0 + 0.1 * t)), 0.3),
        :disch => Gamma(2.0, 1.5))
    @test required_covariates(nested) == Dict(:time => [:death])

    # A data-selected `Choose` reads its own `selector` covariate, labelled
    # with a `:selector` suffix (mirroring how `params_table` labels a
    # `Resolve`'s own `branch_probs` row), in addition to whatever its
    # alternatives read.
    ch = choose(:index => Gamma(2.0, 1.0), :sourced => Gamma(4.0, 1.5))
    @test required_covariates(ch) == Dict(:kind => [:selector])

    mixed = sequential(:br => choose(
        :a => varying(t -> Gamma(2.0, 1.0 + 0.1 * t)), :b => Gamma(1.0, 1.0)))
    rc_mixed = required_covariates(mixed)
    @test rc_mixed[:kind] == [Symbol("br.selector")]
    @test rc_mixed[:time] == [Symbol("br.a")]

    # An uncertain leaf's estimated parameter is reported by
    # `required_parameters`; a fixed leaf contributes nothing.
    utree = sequential(:onset => uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(0.0, 0.3)),
        :admit => LogNormal(0.5, 0.4))
    @test required_parameters(utree) == [(edge = :onset, param = :shape)]
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

@testitem "instantiate: node-level variation (a time-varying Resolve CFR)" begin
    using Distributions

    # A whole Resolve node varies with the context: the death-branch probability
    # (the CFR) rises with calendar time. A univariate node (Resolve) IS a
    # UnivariateDistribution, so it wraps in a Varying and rides the same seam —
    # node-level variation falls out of the leaf seam for univariate nodes.
    cfr(t) = 0.2 + 0.02 * t
    node = varying(t -> resolve(:death => (Gamma(1.5, 1.0), cfr(t)),
        :disch => Gamma(2.0, 1.5)))

    # The reference is the node at t = 0 (CFR 0.2, residual 0.8).
    @test node.reference == resolve(:death => (Gamma(1.5, 1.0), 0.2),
        :disch => Gamma(2.0, 1.5))

    # Resolving at t = 10 gives the concrete Resolve with CFR 0.4.
    @test instantiate(node, Context(time = 10.0)) ==
          resolve(:death => (Gamma(1.5, 1.0), 0.4), :disch => Gamma(2.0, 1.5))

    # It nests as a univariate child of a chain and resolves in place.
    chain = Sequential(Gamma(2.0, 1.0), node)
    @test instantiate(chain, Context(time = 10.0)) ==
          Sequential(Gamma(2.0, 1.0),
        resolve(:death => (Gamma(1.5, 1.0), 0.4), :disch => Gamma(2.0, 1.5)))
end

@testitem "instantiate: a latent (sampled) parameter is just a covariate" begin
    using Distributions

    # An uncertain leaf: its shape is a parameter the sampler draws, named as a
    # covariate. The same Varying/Context seam resolves it — the index is LATENT
    # (filled by the sampler) rather than OBSERVED (filled by the data). This is
    # the integration point for the uncertain-distributions work.
    leaf = varying(θ -> Gamma(θ, 1.0); covariate = :inc_shape,
        reference = Gamma(2.0, 1.0))
    @test instantiate(leaf, Context(inc_shape = 3.0)) == Gamma(3.0, 1.0)

    # `with_covariates` threads a sampled parameter onto an observed context, so
    # observed covariates (time) and latent parameters (inc_shape) share one bag.
    obs = Context(time = 4.0)
    full = with_covariates(obs; inc_shape = 2.5)
    @test full.covariates.time == 4.0
    @test full.covariates.inc_shape == 2.5
    @test instantiate(leaf, full) == Gamma(2.5, 1.0)

    # Later keys win over earlier ones.
    @test with_covariates(Context(a = 1); a = 9).covariates.a == 9
end

@testitem "instantiate: Choose selects on the seam when the selector is present" begin
    using Distributions

    ch = choose(:index => varying(t -> Gamma(2.0, 1.0 + 0.1 * t)),
        :sourced => Gamma(4.0, 1.5); selector = :kind)

    # No selector in the context: every alternative is resolved, the Choose kept
    # (the forward-simulation form, where no data has named the branch).
    resolved_all = instantiate(ch, Context(time = 10.0))
    @test resolved_all isa ComposedDistributions.Choose
    @test logpdf(resolved_all, 3.0; kind = :index) == logpdf(Gamma(2.0, 2.0), 3.0)

    # Selector present: collapse to the chosen alternative, resolved at the
    # context — the categorical selection unified with covariate indexing.
    @test instantiate(ch, Context(kind = :index, time = 10.0)) == Gamma(2.0, 2.0)

    # A stationary alternative selects with no other covariate needed.
    @test instantiate(ch, Context(kind = :sourced)) == Gamma(4.0, 1.5)
end
