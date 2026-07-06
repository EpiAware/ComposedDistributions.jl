# Interop matrix: ConvolvedDistributions verbs driven by composed trees, and
# `Convolved` / `Difference` nodes used as leaves inside composed trees. Each
# testitem is one matrix cell (works / errors informatively / fixed structure).
# Censoring-free: this package composes any `UnivariateDistribution`.

@testitem "convolve_distributions(chain, series): vector convolution" begin
    using Distributions

    chain = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]

    out = convolve_distributions(chain, series)
    # Identical to collapsing the chain to its observed total by hand.
    obs = observed_distribution(chain)
    @test out == convolve_distributions(obs, series)
    @test length(out) == length(series)
    # First step ties directly to the observed-total CDF over the first grid
    # bin (the lag-0 interval mass times the first series value).
    lag0 = cdf(obs, 1.0) - cdf(obs, 0.0)
    @test out[1] ≈ lag0 * series[1]
    # A nested chain collapses through to the same flat total.
    nested = Sequential(Sequential(Gamma(2.0, 1.0), Gamma(1.0, 1.0)),
        LogNormal(0.5, 0.4))
    @test convolve_distributions(nested, series) ==
          convolve_distributions(observed_distribution(nested), series)
end

@testitem "convolve_distributions: univariate one_of marginal drives a series" begin
    using Distributions

    # A Resolve / Compete marginal is univariate, so it already hits the base
    # univariate timeseries method (no bridge needed); the result matches
    # convolving its marginal delay.
    r = resolve(:recover => (Gamma(2.0, 1.0), 0.7),
        :die => (Gamma(1.5, 2.0), 0.3))
    series = [0.0, 1.0, 2.0, 4.0, 3.0]
    @test convolve_distributions(r, series) ==
          convolve_distributions(observed_distribution(r), series)
    @test length(convolve_distributions(r, series)) == length(series)
end

@testitem "convolve_distributions: Parallel / Choose error informatively" begin
    using Distributions

    p = parallel(:admit => Gamma(2.0, 1.0), :notif => LogNormal(1.0, 0.5))
    series = [0.0, 1.0, 2.0]
    @test_throws ArgumentError convolve_distributions(p, series)
    @test_throws ArgumentError observed_distribution(p)

    ch = choose(:a => Gamma(2.0, 1.0), :b => Gamma(1.0, 2.0))
    @test_throws ArgumentError convolve_distributions(ch, series)
    @test_throws ArgumentError observed_distribution(ch)
end

@testitem "difference(chain, chain): difference of observed totals" begin
    using Distributions

    onset = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    report = Sequential(Gamma(1.5, 1.0), Gamma(1.0, 2.0))

    d = difference(onset, report)
    @test d isa Difference
    @test mean(d) ≈ mean(observed_distribution(onset)) -
                    mean(observed_distribution(report))
    # Mixed operands: a chain against a bare distribution, either order.
    g = Gamma(3.0, 1.0)
    @test mean(difference(onset, g)) ≈
          mean(observed_distribution(onset)) - mean(g)
    @test mean(difference(g, onset)) ≈
          mean(g) - mean(observed_distribution(onset))
end

@testitem "Convolved leaf in a tree: rand / logpdf / moments" begin
    using Distributions
    using Random

    conv = convolve_distributions(Gamma(2.0, 1.0), Gamma(1.0, 1.0))
    seq = sequential(:total => conv, :report => LogNormal(0.5, 0.4))

    # logpdf flows through the flat-vector machinery: a Convolved is a plain
    # univariate leaf (one flat slot).
    x = [2.3, 0.9]
    @test logpdf(seq, x) ≈ logpdf(conv, 2.3) + logpdf(LogNormal(0.5, 0.4), 0.9)
    @test pdf(seq, x) ≈ exp(logpdf(seq, x))

    # rand yields one value per flat leaf.
    r = rand(Random.MersenneTwister(1), seq)
    @test length(r) == 2
    @test all(isfinite, r)

    # Overall moments see the Convolved leaf's additive mean/var.
    @test mean(seq) ≈ mean(conv) + mean(LogNormal(0.5, 0.4))
    @test var(seq) ≈ var(conv) + var(LogNormal(0.5, 0.4))
end

@testitem "Difference leaf in a tree: logpdf flows through" begin
    using Distributions

    diff = difference(Gamma(2.0, 1.0), Gamma(1.5, 2.0))
    par = parallel(:gap => diff, :other => LogNormal(0.5, 0.4))
    x = [0.5, 1.2]
    @test logpdf(par, x) ≈ logpdf(diff, 0.5) + logpdf(LogNormal(0.5, 0.4), 1.2)
end

@testitem "free_leaf / rewrap_leaf: a composite leaf is its own free leaf" begin
    using Distributions
    using ComposedDistributions: free_leaf, rewrap_leaf

    conv = convolve_distributions(Gamma(2.0, 1.0), Gamma(1.0, 1.0))
    # A Convolved has no outer wrapper to peel: it free-leafs to itself, and
    # rewrapping replaces it wholesale.
    @test free_leaf(conv) === conv
    @test rewrap_leaf(conv, Gamma(3.0, 1.5)) == Gamma(3.0, 1.5)
end

@testitem "Convolved leaf is fixed structure for params_table / build_priors" begin
    using Distributions
    using ComposedDistributions: Tables

    conv = convolve_distributions(Gamma(2.0, 1.0), Gamma(1.0, 1.0))
    seq = sequential(:total => conv, :report => LogNormal(0.5, 0.4))

    tbl = params_table(seq)
    rows = collect(Tables.rows(tbl))
    # No rows for the composite leaf; only the LogNormal's two scalar params.
    @test length(rows) == 2
    @test all(r -> r.value isa Real, rows)
    @test Set(r.param for r in rows) == Set((:mu, :sigma))
    @test all(r -> r.edge == :report, rows)

    # build_priors consumes the well-formed table (no composite prior).
    pr = build_priors(tbl)
    @test pr isa NamedTuple
    @test haskey(pr, :report)
    @test !haskey(pr, :total)
end

@testitem "update leaves a Convolved leaf unchanged (fixed structure)" begin
    using Distributions

    conv = convolve_distributions(Gamma(2.0, 1.0), Gamma(1.0, 1.0))
    seq = sequential(:total => conv, :report => LogNormal(0.5, 0.4))

    seq2 = update(seq, (report = (mu = 0.8, sigma = 0.6),))
    # The composite leaf is untouched; the fittable leaf is updated.
    @test mean(event(seq2, :total)) ≈ mean(conv)
    @test event(seq2, :report) == LogNormal(0.8, 0.6)
end

@testitem "structural edits around a Convolved leaf: prune / splice" begin
    using Distributions

    conv = convolve_distributions(Gamma(2.0, 1.0), Gamma(1.0, 1.0))
    seq = sequential(:total => conv, :report => LogNormal(0.5, 0.4),
        :tail => Gamma(1.2, 1.0))

    # Prune the Convolved leaf itself: the rest of the tree survives.
    pruned = prune(seq, :total)
    @test ComposedDistributions.component_names(pruned) == (:report, :tail)

    # Prune a sibling: the Convolved leaf survives intact.
    pruned2 = prune(seq, :tail)
    @test ComposedDistributions.component_names(pruned2) == (:total, :report)
    @test mean(event(pruned2, :total)) ≈ mean(conv)

    # Splice a follow-up step after the Convolved leaf: the edited tree still
    # scores (the Convolved leaf sits happily inside the new shape).
    spliced = splice(seq, :total; after = :extra => Gamma(1.0, 1.0))
    @test logpdf(spliced, rand(spliced)) isa Real
end
