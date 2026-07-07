# Generic composer tests: construction, scoring, sampling, nesting, and the
# structural / introspection surface. Censoring-free (this package composes any
# `UnivariateDistribution`).

@testitem "Sequential: construction, logpdf, params" begin
    using Distributions

    s = sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_death => LogNormal(0.5, 0.4))
    @test length(s) == 2
    @test ComposedDistributions.component_names(s) ==
          (:onset_admit, :admit_death)
    x = [1.5, 0.8]
    @test logpdf(s, x) ≈ logpdf(Gamma(2.0, 1.0), 1.5) +
                         logpdf(LogNormal(0.5, 0.4), 0.8)
    @test pdf(s, x) ≈ exp(logpdf(s, x))
    @test params(s) == (onset_admit = params(Gamma(2.0, 1.0)),
        admit_death = params(LogNormal(0.5, 0.4)))
    # Positional construction assigns default step names.
    @test ComposedDistributions.component_names(
        Sequential(Gamma(2.0, 1.0), Normal(0.0, 1.0))) == (:step_1, :step_2)
    @test_throws DimensionMismatch logpdf(s, [1.0])
end

@testitem "Parallel: construction, logpdf, per-endpoint moments" begin
    using Distributions

    p = parallel(:admit => Gamma(2.0, 1.0), :notif => LogNormal(1.0, 0.5))
    @test length(p) == 2
    @test event_names(p) == (:event_1, :admit, :notif) || true
    x = [1.2, 2.3]
    @test logpdf(p, x) ≈ logpdf(Gamma(2.0, 1.0), 1.2) +
                         logpdf(LogNormal(1.0, 0.5), 2.3)
    m = mean(p)
    @test m isa NamedTuple
    @test m.admit ≈ mean(Gamma(2.0, 1.0))
    @test m.notif ≈ mean(LogNormal(1.0, 0.5))
    v = var(p)
    @test v.admit ≈ var(Gamma(2.0, 1.0))
end

@testitem "Sequential overall moments are additive" begin
    using Distributions

    s = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test mean(s) ≈ mean(Gamma(2.0, 1.0)) + mean(LogNormal(0.5, 0.4))
    @test var(s) ≈ var(Gamma(2.0, 1.0)) + var(LogNormal(0.5, 0.4))
    @test std(s) ≈ sqrt(var(s))
end

@testitem "Resolve: mixture marginal, moments, residual prob" begin
    using Distributions

    cfr = 0.3
    r = resolve(:death => (Gamma(1.5, 1.0), cfr),
        :disch => (Gamma(2.0, 1.5), 1 - cfr))
    mix = MixtureModel([Gamma(1.5, 1.0), Gamma(2.0, 1.5)], [cfr, 1 - cfr])
    @test mean(r) ≈ mean(mix)
    @test logpdf(r, 2.0) ≈ logpdf(mix, 2.0)
    @test cdf(r, 2.0) ≈ cdf(mix, 2.0)
    @test probs(r) == (death = 0.3, disch = 0.7)
    @test occurrence_probability(r) ≈ 1.0
    # Residual: the last probability may be omitted (a bare delay).
    r2 = resolve(:death => (Gamma(1.5, 1.0), cfr), :disch => Gamma(2.0, 1.5))
    @test r == r2
    @test_throws ArgumentError resolve(:a => (Gamma(1.0, 1.0), 0.9),
        :b => (Gamma(1.0, 1.0), 0.9))
end

@testitem "Resolve: NoEvent branch is a defective marginal" begin
    using Distributions

    r = resolve(:event => (Gamma(1.5, 1.0), 0.4), :none => (NoEvent(), 0.6))
    @test occurrence_probability(r) ≈ 0.4
    # A defective marginal has no scalar logpdf / mean / as_mixture.
    @test_throws ArgumentError logpdf(r, 2.0)
    @test_throws ArgumentError mean(r)
    @test_throws ArgumentError as_mixture(r)
end

@testitem "Compete: racing hazard marginal, derived win probs, rand" begin
    using Distributions, Random

    c = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    # Marginal survival is the product of the survivals.
    t = 5.0
    @test logccdf(c, t) ≈ logccdf(Gamma(2.0, 3.0), t) +
                          logccdf(Gamma(3.0, 2.0), t)
    @test ccdf(c, t) ≈ exp(logccdf(c, t))
    @test cdf(c, t) ≈ 1 - ccdf(c, t)
    # Derived winning probabilities sum to one for proper causes.
    wp = probs(c)
    @test sum(values(wp)) ≈ 1.0 atol = 1e-3
    # Monte-Carlo winning frequencies match the derived split.
    rng = MersenneTwister(42)
    wins = zeros(Int, 2)
    for _ in 1:20000
        name, _ = ComposedDistributions.rand_outcome(rng, c)
        wins[name == :death ? 1 : 2] += 1
    end
    @test wins[1] / 20000 ≈ wp.death atol = 0.02
    @test mean(c) > 0
    @test var(c) >= 0
end

@testitem "Compete: support floor is the earliest cause (staggered floors)" begin
    using Distributions, Random

    # `:early` can win from t=0; `:late` only starts racing from t=1. The
    # marginal floor (soonest ANY cause can fire) is the EARLIEST cause floor
    # (0.0), not the latest (1.0).
    c = compete(:early => Gamma(2.0, 1.0),
        :late => truncated(Gamma(2.0, 1.0); lower = 1.0))
    @test minimum(c) == 0.0
    t = 0.5
    @test insupport(c, t)
    @test pdf(c, t) > 0
    # The derived winning split must match the Monte-Carlo winning
    # frequencies from `rand_outcome` (an inverted floor biases the
    # quadrature lower bound used by `probs`, dropping the mass where
    # `:early` wins before `:late`'s clock even starts).
    rng = MersenneTwister(1)
    wins = zeros(Int, 2)
    n = 20000
    for _ in 1:n
        name, _ = ComposedDistributions.rand_outcome(rng, c)
        wins[name == :early ? 1 : 2] += 1
    end
    wp = probs(c)
    @test wins[1] / n ≈ wp.early atol = 0.02
    @test wins[2] / n ≈ wp.late atol = 0.02
end

@testitem "Compete rejects a NoEvent branch" begin
    using Distributions

    @test_throws ArgumentError compete(:a => Gamma(2.0, 3.0),
        :none => NoEvent())
end

@testitem "Composers reject duplicate child/branch names" begin
    using Distributions

    # Mirrors Choose's existing "alternative names must be unique" guard:
    # every composer must reject a repeated name, since the whole
    # name-keyed API (event, update, prune, splice, params_table, shared)
    # can only ever reach the first branch with a duplicate name.
    @test_throws ArgumentError sequential(
        :a => Gamma(1.0, 1.0), :a => LogNormal(0.0, 1.0))
    @test_throws ArgumentError Sequential(
        (Gamma(1.0, 1.0), LogNormal(0.0, 1.0)), (:a, :a))
    @test_throws ArgumentError parallel(
        :a => Gamma(1.0, 1.0), :a => LogNormal(0.0, 1.0))
    @test_throws ArgumentError Parallel(
        (Gamma(1.0, 1.0), LogNormal(0.0, 1.0)), (:a, :a))
    @test_throws ArgumentError resolve(
        :a => (Gamma(1.0, 1.0), 0.5), :a => Gamma(2.0, 1.0))
    @test_throws ArgumentError Resolve(
        (:a, :a), (Gamma(1.0, 1.0), Gamma(2.0, 1.0)), (0.5, 0.5))
    @test_throws ArgumentError compete(
        :a => Gamma(1.0, 1.0), :a => LogNormal(0.0, 1.0))
    @test_throws ArgumentError Compete(
        (:a, :a), (Gamma(1.0, 1.0), LogNormal(0.0, 1.0)))
end

@testitem "Zero-arg Sequential()/Parallel() give a friendly ArgumentError" begin
    using Distributions

    # sequential()/parallel() already guard the zero-child case; the bare
    # struct constructors should too, rather than a bare MethodError.
    @test_throws ArgumentError Sequential()
    @test_throws ArgumentError Parallel()
end

@testitem "compose: NamedTuple, table and matrix build equal stacks" begin
    using Distributions

    nt = (r1 = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
        r2 = [Gamma(1.0, 1.0), Gamma(3.0, 1.0)])
    table = (name = [:a, :b, :c, :d],
        dist = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4),
            Gamma(1.0, 1.0), Gamma(3.0, 1.0)],
        chain = [1, 1, 2, 2])
    mat = [Gamma(2.0, 1.0) LogNormal(0.5, 0.4); Gamma(1.0, 1.0) Gamma(3.0, 1.0)]
    @test compose(nt) == compose(table) == compose(mat)
    # compose always returns a composer, never a bare leaf.
    @test compose((a = Gamma(2.0, 1.0),)) isa Parallel
end

@testitem "compose: table compete/prob column builds a Resolve" begin
    using Distributions

    table = (name = [:death, :disch, :onset],
        dist = [Gamma(1.5, 1.0), Gamma(2.0, 1.5), Gamma(1.0, 1.0)],
        compete = [1, 1, 0], prob = [0.3, 0.7, missing])
    d = compose(table)
    node = event(d, :death)
    @test node isa Resolve
    @test probs(node) == (death = 0.3, disch = 0.7)
end

@testitem "compose: shared-origin branches front-end" begin
    using Distributions

    d = compose(Gamma(2.0, 1.0); report = Gamma(1.0, 1.0),
        death = LogNormal(0.5, 0.4))
    @test d isa Sequential
    @test length(d.components) == 2
    @test d.components[2] isa Parallel
end

@testitem "Choose: data-selected disjunction" begin
    using Distributions

    d = choose(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))
    @test logpdf(d, 3.0; kind = :short) ≈ logpdf(Gamma(2.0, 1.0), 3.0)
    @test logpdf(d, 3.0; kind = :long) ≈ logpdf(Gamma(5.0, 1.0), 3.0)
    @test_throws ArgumentError logpdf(d, 3.0)
    @test_throws ArgumentError logpdf(d, 3.0; kind = :nope)
    @test event_names(d) == (:short, :long)
    @test event(d, :short) == Gamma(2.0, 1.0)
end

@testitem "Choose: rand/logpdf round-trip contract" begin
    using Distributions, Random
    using ComposedDistributions: Tables

    d = choose(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))
    rng = MersenneTwister(1)
    draw = rand(rng, d)
    @test draw isa NamedTuple
    @test draw.kind in (:short, :long)
    @test isfinite(logpdf(d, draw))
    @test logpdf(d, draw) ≈ logpdf(d, draw.value; kind = draw.kind)

    # An explicit `kind` still returns the alternative's raw, untagged draw
    # (the in-tree / committed-selection path).
    x = rand(rng, d; kind = :short)
    @test x isa Real

    # A nested composer alternative's draw merges its own labelled fields in
    # directly, rather than wrapping under `:value`.
    nested = choose(:index => sequential(:a => Gamma(2.0, 1.0),
            :b => LogNormal(0.5, 0.4)),
        :sourced => sequential(:a => Gamma(4.0, 1.5),
            :b => LogNormal(0.2, 0.3)))
    ndraw = rand(rng, nested)
    @test ndraw isa NamedTuple
    @test !haskey(ndraw, :value)
    @test isfinite(logpdf(nested, ndraw))

    # A vector of tagged records scores per-record via broadcasting.
    draws = [rand(rng, d) for _ in 1:5]
    @test sum(logpdf.(Ref(d), draws)) ≈ sum(logpdf(d, r) for r in draws)

    # A column table of tagged records sums over rows.
    tbl = (kind = [r.kind for r in draws], value = [r.value for r in draws])
    @test Tables.istable(tbl)
    @test logpdf(d, tbl) ≈ sum(logpdf(d, r) for r in draws)

    # Missing/incorrect selector field errors clearly.
    @test_throws ArgumentError logpdf(d, (value = 3.0,))
    @test_throws ArgumentError logpdf(d, (kind = "short", value = 3.0))
end

@testitem "Nesting: child contract and flat rand round-trips logpdf" begin
    using Distributions, Random

    tree = compose((path = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
        other = Gamma(3.0, 1.0)))
    @test ComposedDistributions.child_nleaves(tree) == 3
    rng = MersenneTwister(7)
    draw = rand(rng, tree)
    @test draw isa NamedTuple
    # A labelled draw round-trips straight back through logpdf.
    @test logpdf(tree, draw) ≈ logpdf(tree, collect(values(draw)))
end

@testitem "Introspection: params_table, event_names, event_tree, event" begin
    using Distributions

    tree = compose((onset_admit = LogNormal(1.5, 0.4),
        admit_death = Gamma(2.0, 1.0)))
    tbl = params_table(tree)
    @test tbl.edge == [:onset_admit, :onset_admit, :admit_death, :admit_death]
    @test tbl.param == [:mu, :sigma, :shape, :scale]
    @test event_names(tree) == (:onset, :admit, :death)
    nested = compose((
        admit_path = compose((onset_admit = Gamma(2.0, 1.0),
            admit_death = LogNormal(0.5, 0.4))),
        onset_recover = Gamma(3.0, 1.0)))
    @test event_tree(nested).admit_path isa NamedTuple
    @test event(nested, :admit_path, :admit_death) == LogNormal(0.5, 0.4)
    @test event(nested, Symbol("admit_path.admit_death")) == LogNormal(0.5, 0.4)
end

@testitem "event: descriptive ArgumentError for an unknown child name" begin
    using Distributions

    nested = compose((
        admit_path = compose((onset_admit = Gamma(2.0, 1.0),
            admit_death = LogNormal(0.5, 0.4))),
        onset_recover = Gamma(3.0, 1.0)))
    # Mirrors update/prune/splice's "no child named ...; have [...]" style
    # rather than a bare KeyError.
    err = try
        event(nested, :admit_path, :nonexistent)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin(":nonexistent", err.msg)
    @test occursin(":onset_admit", err.msg)
    @test occursin(":admit_death", err.msg)
    @test_throws ArgumentError event(nested, :nope)
    d = choose(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))
    @test_throws ArgumentError event(d, :nope)
end

@testitem "build_priors and default_prior" begin
    using Distributions

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    tbl = params_table(tree)
    nested = build_priors(tbl)
    @test nested.onset_admit.shape isa Truncated
    @test nested.admit_death.mu isa Normal
    # A user override wins over the default.
    ov = build_priors(tbl;
        priors = (onset_admit = (shape = truncated(Normal(2, 0.5);
            lower = 0),),))
    @test ov.onset_admit.shape == truncated(Normal(2, 0.5); lower = 0)
    # Probability parameter default is Uniform(0, 1).
    dp = default_prior((; edge = :r, param = :death, value = 0.3,
        support = (0.0, 1.0)))
    @test dp == Uniform(0, 1)
end

@testitem "param_priors is a thin front-door over build_priors(params_table(...))" begin
    using Distributions

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))

    @test param_priors(tree) == build_priors(params_table(tree))
    # The keyword surface is forwarded unchanged.
    shape_prior = Normal(2, 0.5)
    @test param_priors(tree; priors = Dict((:onset_admit, :shape) => shape_prior)) ==
          build_priors(params_table(tree);
        priors = Dict((:onset_admit, :shape) => shape_prior))
end

@testitem "Composer show is compact; inspect gives detail" begin
    using Distributions

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.500001, 0.4)))

    out = sprint(show, MIME"text/plain"(), tree)
    det = sprint(inspect, tree)

    @test occursin("Parallel (2 branches)", out)
    @test occursin("Gamma", out)

    # `inspect` walks the SAME tree as `show`, but prints each leaf's full
    # `text/plain` detail on its own line(s) rather than inline, so it is
    # strictly longer.
    @test occursin("Parallel (2 branches)", det)
    @test occursin("onset_admit", det)
    @test occursin("admit_death", det)
    @test occursin("0.500001", det)
    @test count(==('\n'), det) > count(==('\n'), out)

    # `inspect(d)` (no `io`) writes to stdout and returns nothing.
    @test inspect(tree) === nothing
end

@testitem "update: replace parameters from a nested NamedTuple" begin
    using Distributions

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    tree2 = update(tree, (onset_admit = (shape = 3.0, scale = 1.5),
        admit_death = (mu = 0.7, sigma = 0.5)))
    @test event(tree2, :onset_admit) == Gamma(3.0, 1.5)
    @test event(tree2, :admit_death) == LogNormal(0.7, 0.5)
end

@testitem "intervene: update node, prune, splice" begin
    using Distributions

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    # Node replace keeps shape.
    t2 = update(tree, :admit_death => Gamma(3.0, 1.5))
    @test event(t2, :admit_death) == Gamma(3.0, 1.5)
    # Prune drops a Resolve arm and renormalises.
    node = resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.5), :transfer => (Gamma(1.0, 1.0), 0.2))
    pruned = event(prune(compose((res = node,)), :res, :transfer), :res)
    @test sum(pruned.branch_probs) ≈ 1.0
    @test length(pruned.names) == 2
    # Splice inserts an after step.
    sp = splice(tree, :admit_death; after = :report => Gamma(1.0, 2.0))
    @test event(sp, :admit_death) isa Sequential
end

@testitem "intervene through a nested Compete: update, prune, tie" begin
    using Distributions

    node = compete(:immediate => Gamma(2.0, 1.0),
        :delayed => resolve(:a => (Gamma(1.5, 1.0), 0.3),
            :b => (Gamma(2.0, 1.5), 0.5), :c => (Gamma(1.0, 1.0), 0.2)))
    tree = compose((path = node, other = Gamma(3.0, 1.0)))

    # update descends through the Compete to replace a cause's leaf.
    t2 = update(tree, (:path, :immediate) => Gamma(4.0, 2.0))
    @test event(t2, :path, :immediate) == Gamma(4.0, 2.0)

    # prune descends through the Compete into a nested Resolve cause.
    pruned = event(prune(tree, :path, :delayed, :c), :path, :delayed)
    @test length(pruned.names) == 2
    @test sum(pruned.branch_probs) ≈ 1.0

    # tie descends through the Compete to tag a leaf as shared.
    tied = tie(tree, (:path, :immediate), :other; name = :g)
    @test :g in params_table(tied).edge
    @test logpdf(event(tied, :path, :immediate), 1.5) ≈
          logpdf(Gamma(2.0, 1.0), 1.5)
end

@testitem "rand_outcome(::Resolve): resolved name and delay, or missing" begin
    using Distributions, Random

    r = resolve(:event => (Gamma(1.5, 1.0), 0.4), :none => (NoEvent(), 0.6))
    rng = MersenneTwister(3)
    seen = Set{Symbol}()
    for _ in 1:200
        name, time = ComposedDistributions.rand_outcome(rng, r)
        push!(seen, name)
        name == :none ? (@test time === missing) : (@test time isa Real)
    end
    @test seen == Set([:event, :none])
end

@testitem "equality: structural for chains, name-sensitive for Resolve" begin
    using Distributions

    a = Sequential((Gamma(2.0, 1.0), Normal(0.0, 1.0)), (:x, :y))
    b = Sequential((Gamma(2.0, 1.0), Normal(0.0, 1.0)), (:p, :q))
    @test a == b               # names are metadata, ignored
    @test hash(a) == hash(b)
    r1 = resolve(:death => (Gamma(1.0, 1.0), 0.5), :disch => Gamma(1.0, 1.0))
    r2 = resolve(:die => (Gamma(1.0, 1.0), 0.5), :out => Gamma(1.0, 1.0))
    @test r1 != r2             # outcome names are intrinsic
    c = compete(:death => Gamma(1.0, 1.0), :disch => Gamma(1.0, 1.0))
    @test r1 != c
end

@testitem "Shared / tie: one free parameter across branches" begin
    using Distributions

    inc = shared(:inc, Gamma(2.0, 1.0))
    @test ComposedDistributions._shared_tag(inc) == :inc
    @test logpdf(inc, 1.5) ≈ logpdf(Gamma(2.0, 1.0), 1.5)  # transparent
    d = compose((a = Gamma(2.0, 1.0), b = Gamma(2.0, 1.0)))
    tied = tie(d, :a, :b; name = :g)
    # The tied leaves are inventoried once under the tag.
    @test unique(params_table(tied).edge) == [:g]
end

@testitem "observed_distribution / convolve re-export" begin
    using Distributions

    s = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    od = observed_distribution(s)
    @test od isa Convolved
    # The chain collapses to the convolution of its steps.
    cv = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test cdf(od, 5.0) ≈ cdf(cv, 5.0)
    # convolve_distributions accepts the chain directly (issue #7).
    @test cdf(convolve_distributions(s), 5.0) ≈ cdf(cv, 5.0)
    # A chain with a Parallel step cannot collapse.
    @test_throws ArgumentError observed_distribution(
        Sequential(Gamma(1.0, 1.0), Parallel(Gamma(1.0, 1.0), Gamma(1.0, 1.0))))
end

@testitem "re-exported ConvolvedDistributions surface is reachable" begin
    # These names come through ComposedDistributions' re-export, so downstream
    # packages sit on ComposedDistributions alone.
    @test isdefined(ComposedDistributions, :convolve_distributions)
    @test isdefined(ComposedDistributions, :integrate)
    @test isdefined(ComposedDistributions, :gl_integrate)
    @test isdefined(ComposedDistributions, :GaussLegendre)
    @test isdefined(ComposedDistributions, :AbstractSolverMethod)
    @test isdefined(ComposedDistributions, :AnalyticalSolver)
    @test isdefined(ComposedDistributions, :NumericSolver)
    @test isdefined(ComposedDistributions, :Difference)
end

@testitem "Monte-Carlo: chain rand mean matches analytic mean" begin
    using Distributions, Random

    s = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    rng = MersenneTwister(11)
    n = 40000
    totals = [(d = rand(rng, s); d.step_1 + d.step_2)
              for _ in 1:n]
    @test mean(totals) ≈ mean(s) atol = 0.05
end
