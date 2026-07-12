# Interop matrix: ConvolvedDistributions verbs driven by composed trees, and
# `Convolved` / `Difference` nodes used as leaves inside composed trees. Each
# testitem is one matrix cell (works / errors informatively / see-through
# component fitting). Censoring-free: this package composes any
# `UnivariateDistribution`.

@testitem "convolve_series(chain, series): vector convolution" begin
    using Distributions

    chain = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]

    out = convolve_series(chain, series)
    # ConvolvedDistributions 0.2 is discrete-only, so the chain collapses to its
    # continuous observed total and discretises it explicitly; the composed path
    # is identical to discretising the total by hand and convolving the PMF (and
    # so reproduces the pre-0.2 continuous output exactly).
    obs = observed_distribution(chain)
    @test out == convolve_series(discretise_pmf(obs, length(series) - 1), series)
    # The bare continuous total is now rejected: it must be discretised first.
    @test_throws ArgumentError convolve_series(obs, series)
    @test length(out) == length(series)
    # First step ties directly to the observed-total CDF over the first grid
    # bin (the lag-0 interval mass times the first series value).
    lag0 = cdf(obs, 1.0) - cdf(obs, 0.0)
    @test out[1] ≈ lag0 * series[1]
    # A nested chain collapses through to the same flat total.
    nested = Sequential(Sequential(Gamma(2.0, 1.0), Gamma(1.0, 1.0)),
        LogNormal(0.5, 0.4))
    @test convolve_series(nested, series) == convolve_series(
        discretise_pmf(observed_distribution(nested), length(series) - 1),
        series)
end

@testitem "convolve_series(chain, series; events): per-event series" begin
    using Distributions

    g1 = Gamma(2.0, 1.0)
    g2 = LogNormal(0.5, 0.4)
    g3 = Gamma(1.5, 1.0)
    chain = sequential(:onset_admit => g1, :admit_death => g2,
        :death_report => g3)
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]

    # An interim event's series is the series convolved through the cumulative
    # delay of the prefix leading to it (collapse the prefix by hand).
    admit = convolve_series(chain, series; events = :admit)
    @test admit ==
          convolve_series(discretise_pmf(g1, length(series) - 1), series)
    death = convolve_series(chain, series; events = :death)
    @test death == convolve_series(
        discretise_pmf(convolved([g1, g2]), length(series) - 1), series)

    # A tuple of names returns a NamedTuple keyed by the names; a vector too.
    nt = convolve_series(chain, series; events = (:admit, :report))
    @test nt isa NamedTuple{(:admit, :report)}
    @test nt.admit == admit
    @test nt.report == convolve_series(
        discretise_pmf(convolved([g1, g2, g3]), length(series) - 1), series)
    vt = convolve_series(chain, series; events = [:admit, :death])
    @test vt.admit == admit && vt.death == death
end

@testitem "convolve_series(chain, series; events): endpoint == whole" begin
    using Distributions

    chain = sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_death => LogNormal(0.5, 0.4))
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]

    # Selecting the terminal event reproduces the plain whole-chain result.
    @test convolve_series(chain, series; events = :death) ==
          convolve_series(chain, series)
    # A positional-default chain names its events :event_i; the endpoint matches.
    pos = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test convolve_series(pos, series; events = event_names(pos)[end]) ==
          convolve_series(pos, series)
end

@testitem "convolve_series(chain, series; events): errors" begin
    using Distributions

    chain = sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_death => LogNormal(0.5, 0.4))
    series = [0.0, 1.0, 2.0]

    # An unknown event name lists the valid events.
    err = try
        convolve_series(chain, series; events = :nope)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("valid events", err.msg)
    @test occursin("admit", err.msg) && occursin("death", err.msg)

    # The origin has no elapsed delay, so it is not a convolvable event.
    @test_throws ArgumentError convolve_series(
        chain, series; events = :onset)

    # A branching step (a Parallel inside the chain) is rejected: its flat
    # events do not line up one-to-one with delay steps.
    branched = sequential(:onset_admit => Gamma(2.0, 1.0),
        :split => parallel(:a => Gamma(1.0, 1.0), :b => Gamma(2.0, 1.0)))
    @test_throws ArgumentError convolve_series(
        branched, series; events = :admit)
end

@testitem "convolve_series: univariate one_of marginal drives a series" begin
    using Distributions

    # A Resolve / Compete marginal is a continuous univariate delay. The base
    # ConvolvedDistributions 0.2 `convolve_series` is discrete-only, so the
    # one_of bridge discretises the marginal for it; the result matches
    # discretising and convolving that marginal delay directly.
    r = resolve(:recover => (Gamma(2.0, 1.0), 0.7),
        :die => (Gamma(1.5, 2.0), 0.3))
    series = [0.0, 1.0, 2.0, 4.0, 3.0]
    @test convolve_series(r, series) == convolve_series(
        discretise_pmf(observed_distribution(r), length(series) - 1), series)
    @test length(convolve_series(r, series)) == length(series)
end

@testitem "convolve_series: Compete marginal drives a series" begin
    using Distributions

    series = [0.0, 1.0, 2.0, 4.0, 3.0]

    # A Compete's observed quantity is its marginal any-event (first-event) time,
    # a continuous univariate delay, so the one_of bridge discretises it before
    # convolving; `observed_distribution` returns it unchanged.
    c = Compete(:recover => Gamma(2.0, 1.0), :die => Gamma(1.5, 2.0))
    out = convolve_series(c, series)
    @test observed_distribution(c) === c
    @test out == convolve_series(
        discretise_pmf(observed_distribution(c), length(series) - 1), series)
    @test length(out) == length(series)
    @test all(>=(0), out)
    # A finite window recovers only the mass that lands within it: the delayed
    # counts never exceed the input mass pushed through them.
    @test sum(out) <= sum(series)

    # Staggered per-cause support floors (truncated causes) push the any-event
    # time later, so fewer events land inside the window than the unfloored
    # racing pair — a cross-check on the Compete marginal's discretised PMF. The
    # exact per-lag values depend on `minimum(::Compete)` (the support floor the
    # quadrature window keys off), which the concurrent Compete correctness fix
    # revises; if that changes these outputs, reconcile at the update-branch.
    cf = Compete(:recover => truncated(Gamma(2.0, 1.0); lower = 1.0),
        :die => truncated(Gamma(1.5, 2.0); lower = 2.0))
    outf = convolve_series(cf, series)
    @test outf == convolve_series(
        discretise_pmf(observed_distribution(cf), length(series) - 1), series)
    @test length(outf) == length(series)
    @test all(>=(0), outf)
    @test sum(outf) < sum(out)
end

@testitem "convolve_series: Parallel / Choose error informatively" begin
    using Distributions

    p = parallel(:admit => Gamma(2.0, 1.0), :notif => LogNormal(1.0, 0.5))
    series = [0.0, 1.0, 2.0]
    @test_throws ArgumentError convolve_series(p, series)
    @test_throws ArgumentError observed_distribution(p)

    ch = choose(:a => Gamma(2.0, 1.0), :b => Gamma(1.0, 2.0))
    @test_throws ArgumentError convolve_series(ch, series)
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

@testitem "product / Product reachable via ComposedDistributions (#139)" begin
    using Distributions

    # The Mellin product family (`Z = X * Y`) is reachable through
    # ComposedDistributions alone, so a downstream sitting on this package sees
    # the whole convolution surface. `product` is exported; `Product` stays
    # unexported (Distributions clash) but is reachable module-qualified. Bare
    # re-export only; composing a `Product` leaf into a tree is out of scope.
    @test isdefined(ComposedDistributions, :product)
    @test isdefined(ComposedDistributions, :Product)
    d = product(Gamma(3.0, 1.0), LogNormal(0.0, 0.3))
    @test d isa ComposedDistributions.Product
    @test mean(d) ≈ 3.0 * exp(0.3^2 / 2)
end

@testitem "Convolved leaf in a tree: rand / logpdf / moments" begin
    using Distributions
    using Random

    conv = convolved(Gamma(2.0, 1.0), Gamma(1.0, 1.0))
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

    conv = convolved(Gamma(2.0, 1.0), Gamma(1.0, 1.0))
    # A Convolved has no outer wrapper to peel: it free-leafs to itself, and
    # rewrapping replaces it wholesale.
    @test free_leaf(conv) === conv
    @test rewrap_leaf(conv, Gamma(3.0, 1.5)) == Gamma(3.0, 1.5)
end

@testitem "Convolved leaf: params_table sees through to component params" begin
    using Distributions
    using ComposedDistributions: Tables

    conv = convolved(Gamma(2.0, 1.0), Gamma(1.0, 1.5))
    seq = sequential(:total => conv, :report => LogNormal(0.5, 0.4))

    tbl = params_table(seq)
    rows = collect(Tables.rows(tbl))
    # One row per scalar component parameter (two Gammas) plus the LogNormal's.
    @test length(rows) == 6
    @test all(r -> r.value isa Real, rows)
    # The composite's components are namespaced `total.component_i`, each with
    # the leaf's own scalar params and value.
    comp_rows = filter(r -> r.edge != :report, rows)
    @test Set(r.edge for r in comp_rows) ==
          Set((Symbol("total.component_1"), Symbol("total.component_2")))
    c1 = filter(r -> r.edge == Symbol("total.component_1"), comp_rows)
    @test [(r.param, r.value) for r in c1] == [(:shape, 2.0), (:scale, 1.0)]
    c2 = filter(r -> r.edge == Symbol("total.component_2"), comp_rows)
    @test [(r.param, r.value) for r in c2] == [(:shape, 1.0), (:scale, 1.5)]

    # build_priors assembles the nested prior tree down to the components.
    pr = build_priors(tbl)
    @test pr isa NamedTuple
    @test haskey(pr, :report)
    @test haskey(pr.total.component_1, :shape)
    @test haskey(pr.total.component_2, :scale)
end

@testitem "Difference leaf: params_table sees through to (x, y) params" begin
    using Distributions
    using ComposedDistributions: Tables

    diff = difference(Gamma(2.0, 1.0), Normal(1.0, 0.5))
    par = parallel(:gap => diff, :other => LogNormal(0.5, 0.4))

    tbl = params_table(par)
    rows = collect(Tables.rows(tbl))
    comp_rows = filter(r -> r.edge != :other, rows)
    @test Set(r.edge for r in comp_rows) ==
          Set((Symbol("gap.component_1"), Symbol("gap.component_2")))
    # component_1 is the minuend (Gamma), component_2 the subtrahend (Normal).
    @test Set((r.edge, r.param) for r in comp_rows) == Set([
        (Symbol("gap.component_1"), :shape), (Symbol("gap.component_1"), :scale),
        (Symbol("gap.component_2"), :mu), (Symbol("gap.component_2"), :sigma)])
end

@testitem "update round-trips a Convolved leaf's component params" begin
    using Distributions

    conv = convolved(Gamma(2.0, 1.0), Gamma(1.0, 1.5))
    seq = sequential(:total => conv, :report => LogNormal(0.5, 0.4))

    # A concrete update replaces every component parameter, rebuilding the
    # composite from the pinned components (the solver method is preserved).
    seq2 = update(seq,
        (
            total = (component_1 = (shape = 3.0, scale = 2.0),
                component_2 = (shape = 1.2, scale = 0.8)),
            report = (mu = 0.8, sigma = 0.6)))
    total2 = event(seq2, :total)
    @test total2 isa Convolved
    @test total2.components[1] == Gamma(3.0, 2.0)
    @test total2.components[2] == Gamma(1.2, 0.8)
    @test event(seq2, :report) == LogNormal(0.8, 0.6)
    # The composite is still one flat scored slot.
    @test length(seq2) == 2
end

@testitem "update makes a Convolved component uncertain (partial merge)" begin
    using Distributions

    conv = convolved(Gamma(2.0, 1.0), Gamma(1.0, 1.5))
    seq = sequential(:total => conv, :report => LogNormal(0.5, 0.4))

    # A distribution in one component slot makes just that parameter uncertain;
    # untouched components and siblings keep their values.
    est = update(seq,
        (total = (component_1 = (shape = LogNormal(log(2.0), 0.2),),),))
    @test has_uncertain(est)
    @test has_uncertain(event(est, :total))
    total_est = event(est, :total)
    @test total_est.components[2] == Gamma(1.0, 1.5)
    @test event(est, :report) == LogNormal(0.5, 0.4)
end

@testitem "update round-trips a Difference leaf's (x, y) params" begin
    using Distributions

    diff = difference(Gamma(2.0, 1.0), Normal(1.0, 0.5))
    par = parallel(:gap => diff, :other => LogNormal(0.5, 0.4))

    par2 = update(par,
        (
            gap = (component_1 = (shape = 3.0, scale = 2.0),
                component_2 = (mu = 0.5, sigma = 1.0)),
            other = (mu = 0.8, sigma = 0.6)))
    g2 = event(par2, :gap)
    @test g2 isa Difference
    @test g2.x == Gamma(3.0, 2.0)
    @test g2.y == Normal(0.5, 1.0)
    @test g2.method === diff.method
end

@testitem "deferred-leaf see-through: a Convolved with a varying component" begin
    using Distributions

    # A composite rides the shared deferred-leaf walk, so a Varying COMPONENT is
    # visible to has_varying and resolved in place by instantiate (mirroring the
    # has_uncertain see-through), while a plain composite stays inert.
    vc = convolved(
        varying(t -> Gamma(2.0, 1.0 + 0.1t);
            covariate = :time, reference = Gamma(2.0, 1.0)),
        Gamma(1.0, 1.5))
    seq = sequential(:total => vc, :report => LogNormal(0.5, 0.4))

    @test has_varying(vc)
    @test has_varying(seq)
    resolved = instantiate(seq, Context(time = 10.0))
    @test !has_varying(resolved)
    total = event(resolved, :total)
    @test total isa Convolved
    @test total.components[1] == Gamma(2.0, 1.0 + 0.1 * 10)
    @test total.components[2] == Gamma(1.0, 1.5)

    # A plain composite has nothing to resolve: no varying, instantiate is the
    # identity up to reconstruction (== the same component-identical composite).
    conv = convolved(Gamma(2.0, 1.0), Gamma(1.0, 1.5))
    @test !has_varying(conv)
    @test instantiate(conv, Context(time = 3.0)) == conv
end

@testitem "codec: a spec'd Convolved component counts one estimated dim" begin
    using Distributions
    using ComposedDistributions: flat_dimension, unflatten, flatten

    conv = convolved(Gamma(2.0, 1.0), Gamma(1.0, 1.5))
    seq = sequential(:total => conv, :report => LogNormal(0.5, 0.4))
    est = update(seq,
        (total = (component_1 = (shape = LogNormal(log(2.0), 0.2),),),))

    # Exactly one estimated parameter: total.component_1.shape.
    @test flat_dimension(est) == 1
    # unflatten places the draw at the spec'd component parameter and holds the
    # rest at the template; update then collapses the composite to concrete.
    nt = unflatten(est, [3.0])
    @test nt.total.component_1.shape == 3.0
    @test nt.total.component_2.shape == 1.0
    collapsed = update(est, nt)
    @test !has_uncertain(collapsed)
    @test event(collapsed, :total).components[1] == Gamma(3.0, 1.0)
    # flatten is the inverse on the spec'd rows.
    @test flatten(est, nt) == [3.0]
end

@testitem "structural edits around a Convolved leaf: prune / splice" begin
    using Distributions

    conv = convolved(Gamma(2.0, 1.0), Gamma(1.0, 1.0))
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

@testitem "Convolved leaf under the codec: fixed components add no estimated dim" begin
    using Distributions
    using ComposedDistributions: flat_dimension, flatten, unflatten,
                                 as_logdensity, logdensity

    conv = convolved(Gamma(2.0, 1.0), Gamma(1.0, 1.0))
    u = uncertain(LogNormal(0.5, 0.4); mu = Normal(0.5, 0.2))
    seq = sequential(:total => conv, :report => u)

    # The Convolved leaf's components ARE inventoried (see-through), but none is
    # spec'd, so they add no ESTIMATED dimension: the estimation boundary is
    # exactly the uncertain leaf's one spec'd parameter (`report.mu`).
    @test flat_dimension(seq) == 1

    nt = unflatten(seq, [0.9])
    @test nt.report.mu == 0.9 && nt.report.sigma == 0.4
    # The fixed component parameters ride the full NamedTuple at their template
    # values (see-through), unlike the old fixed-composite contract where the
    # composite had no key at all. (Read by field, not by NamedTuple equality:
    # `unflatten` assembles the nested tuple key order from a Dict.)
    @test nt.total.component_1.shape == 2.0 && nt.total.component_1.scale == 1.0
    @test nt.total.component_2.shape == 1.0 && nt.total.component_2.scale == 1.0
    @test flatten(seq, nt) == [0.9]
    @test flatten(seq, unflatten(seq, [1.3])) == [1.3]

    # `update` rebuilds the component-identical composite (a fresh object, so
    # `==` not `===`) and collapses the uncertain leaf at the draw.
    collapsed = update(seq, nt)
    @test event(collapsed, :total) == conv
    @test !has_uncertain(collapsed)

    data = [[2.3, 0.9], [1.8, 1.1]]
    prob = as_logdensity(seq, data)
    expected = logpdf(Normal(0.5, 0.2), 0.9) +
               sum(record -> logpdf(collapsed, record), data)
    @test logdensity(prob, [0.9]) ≈ expected
end

@testitem "uncertain(...) wrapping a Convolved/Difference template errors informatively" begin
    using Distributions

    conv = convolved(Gamma(2.0, 1.0), Gamma(1.0, 1.0))
    # A Convolved's `params` are its components' own parameter tuples (a
    # nested, non-scalar structure), not the flat scalar list `uncertain`
    # attaches priors to. Wrapping one is refused eagerly at construction
    # (not deep inside a later `rand`, where it would otherwise surface as a
    # confusing low-level `MethodError` from rebuilding the leaf).
    @test_throws ArgumentError uncertain(conv; param_1 = Normal(0.0, 1.0))

    diff = difference(Gamma(2.0, 1.0), Gamma(1.5, 2.0))
    @test_throws ArgumentError uncertain(diff; param_1 = Normal(0.0, 1.0))
end

@testitem "Varying leaf mapping to Convolved distributions: instantiate then fixed" begin
    using Distributions

    conv_early = convolved(Gamma(2.0, 1.0), Gamma(1.0, 1.0))
    conv_late = convolved(Gamma(3.0, 1.0), Gamma(2.0, 1.0))
    v = varying(t -> t < 5 ? conv_early : conv_late;
        covariate = :time, reference = conv_early)
    seq = sequential(:total => v, :report => LogNormal(0.5, 0.4))

    @test has_varying(seq)
    resolved = instantiate(seq, Context(time = 10.0))
    @test !has_varying(resolved)
    @test event(resolved, :total) == conv_late

    # Resolved, the Convolved leaf is seen through like a plain Convolved leaf:
    # its two components' four scalar params are inventoried (four rows) plus the
    # LogNormal's two, and none is spec'd, so the estimated dimension is zero.
    tbl = params_table(resolved)
    @test length(collect(ComposedDistributions.Tables.rows(tbl))) == 6
    @test ComposedDistributions.flat_dimension(resolved) == 0
    x = [2.3, 0.9]
    @test logpdf(resolved, x) ≈
          logpdf(conv_late, 2.3) + logpdf(LogNormal(0.5, 0.4), 0.9)
end

@testitem "update at a composite leaf's own level errors informatively" begin
    using Distributions

    conv = convolved(Gamma(2.0, 1.0), Gamma(1.0, 1.0))
    seq = sequential(:total => conv, :report => LogNormal(0.5, 0.4))

    # Under see-through, a value aimed at the composite's OWN level (no
    # `component_i` segment) does not silently vanish: it hits the shared
    # unexpected-key check and errors, listing the component keys. This holds
    # for a distribution (a mis-aimed bid to make it uncertain) ...
    err = try
        update(seq, (total = (shape = Normal(1.0, 2.0),),
            report = (mu = 0.1, sigma = 0.2)))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("component_1", err.msg) && occursin("component_2", err.msg)

    # ... and for a `Real` re-pin (no no-op survives: a Real at a valid
    # component path pins that component, see the round-trip testitem, but a
    # Real at the composite level with no component segment errors just the same).
    @test_throws ArgumentError update(seq,
        (total = (shape = 9.0,), report = (mu = 0.1, sigma = 0.2)))

    # Same contract for a Difference composite.
    diff = difference(Gamma(2.0, 1.0), Gamma(1.5, 2.0))
    par = parallel(:gap => diff, :other => LogNormal(0.5, 0.4))
    @test_throws ArgumentError update(par,
        (gap = (shape = Normal(1.0, 2.0),), other = (mu = 0.1, sigma = 0.2)))
end
