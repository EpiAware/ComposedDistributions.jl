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
    @test event_names(p) == (:event_1, :event_2, :event_3)
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

@testitem "Parallel minimum/maximum are per-endpoint NamedTuples" begin
    using Distributions

    p = parallel(:admit => Gamma(2.0, 1.0), :notif => LogNormal(1.0, 0.5))
    lo = minimum(p)
    hi = maximum(p)
    @test lo isa NamedTuple
    @test hi isa NamedTuple
    @test lo == (admit = 0.0, notif = 0.0)
    @test hi == (admit = Inf, notif = Inf)
end

@testitem "minimum/maximum on a collapsible composer error clearly" begin
    using Distributions

    # A `Sequential` (and a `Choose`) have no whole-node support bound; the
    # generic must raise an ArgumentError, not the opaque MethodError about
    # `iterate` the `Distributions` fallback used to surface.
    s = sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_death => LogNormal(0.5, 0.4))
    @test_throws ArgumentError minimum(s)
    @test_throws ArgumentError maximum(s)
    d = choose(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))
    @test_throws ArgumentError minimum(d)
    @test_throws ArgumentError maximum(d)
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
    # A defective marginal has no scalar logpdf / as_mixture: the observed
    # non-occurrence mass is scored through the event-vector path instead.
    @test_throws ArgumentError logpdf(r, 2.0)
    @test_throws ArgumentError as_mixture(r)
end

@testitem "Resolve: defective marginal survival (#254)" begin
    using Distributions

    r = resolve(:event => (Gamma(2.0, 1.0), 0.6), :none => (NoEvent(), 0.4))

    # `cdf` sums only the occurring branches, so it rises to
    # `occurrence_probability` rather than one.
    @test cdf(r, 3.0) ≈ 0.6 * cdf(Gamma(2.0, 1.0), 3.0)
    @test cdf(r, Inf) ≈ occurrence_probability(r)

    # `ccdf` (the generic `1 - cdf` fallback) flattens at the no-event mass
    # instead of decaying to zero.
    @test ccdf(r, 3.0) ≈ 0.4 + 0.6 * ccdf(Gamma(2.0, 1.0), 3.0)
    @test ccdf(r, Inf) ≈ 0.4

    # `mean` is the conditional-on-occurrence mean of the observed branches,
    # renormalised by `occurrence_probability` (a single occurring branch
    # here, so it equals that branch's own mean).
    @test mean(r) ≈ mean(Gamma(2.0, 1.0))

    # Two occurring branches: the conditional mean is their probability-
    # weighted average, renormalised by the occurrence probability.
    r2 = resolve(:a => (Gamma(2.0, 1.0), 0.3), :b => (Gamma(5.0, 1.0), 0.3),
        :none => (NoEvent(), 0.4))
    @test mean(r2) ≈
          (0.3 * mean(Gamma(2.0, 1.0)) + 0.3 * mean(Gamma(5.0, 1.0))) / 0.6

    # A proper node (no no-event branch) is unaffected: `mean`/`cdf` still
    # match the ordinary mixture lowering.
    proper = resolve(:a => (Gamma(2.0, 1.0), 0.4), :b => (Gamma(1.5, 1.0), 0.6))
    @test mean(proper) ≈ mean(as_mixture(proper))
    @test cdf(proper, 1.0) ≈ cdf(as_mixture(proper), 1.0)

    # A non-terminal node (a composer-valued outcome) stays multivariate:
    # `mean`/`cdf` still reject it, no-event branch or not.
    inner = resolve(:a => (Gamma(2.0, 1.0), 0.5), :b => (Gamma(1.5, 1.0), 0.5))
    nonterminal = resolve(:sub => (inner, 0.5), :c => (Gamma(1.0, 1.0), 0.5))
    @test_throws ArgumentError mean(nonterminal)
    @test_throws ArgumentError cdf(nonterminal, 1.0)
end

@testitem "Choose: whole-tree mean/var/std are ill-defined" begin
    using Distributions

    ch = choose(:a => Gamma(1.5, 1.0), :b => Gamma(2.0, 1.0))
    # A Choose has no single layout (data-selects the active alternative), so
    # a whole-tree moment is ill-defined; only the chosen alternative's own
    # moment is available.
    @test_throws "mean(::Choose) needs a selection" mean(ch)
    @test_throws "var(::Choose) needs a selection" var(ch)
    @test_throws "std(::Choose) needs a selection" std(ch)
    @test mean(event(ch, :a)) ≈ mean(Gamma(1.5, 1.0))
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
        name, _ = rand(rng, c; outcome = true)
        wins[name == :death ? 1 : 2] += 1
    end
    @test wins[1] / 20000 ≈ wp.death atol = 0.02
    @test mean(c) > 0
    @test var(c) >= 0
end

@testitem "Compete: winning probabilities sum to exactly one (#115)" begin
    using Distributions

    # Two proper competing risks: exactly one must win, so the derived winning
    # probabilities sum to one. The per-cause Gauss-Legendre quadrature can
    # overshoot (sum ≈ 1.0000322 before normalisation); `probs` rescales the
    # split so the vector is a valid probability vector.
    c = compete(:death => Gamma(1.5, 1.0), :recover => Gamma(2.0, 1.5))
    p = probs(c)
    @test sum(values(p)) ≈ 1.0
    @test sum(values(p)) <= 1.0
    @test occurrence_probability(c) ≈ 1.0
    @test occurrence_probability(c) <= 1.0
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
    # frequencies from `rand(c; outcome = true)` (an inverted floor biases
    # the quadrature lower bound used by `probs`, dropping the mass where
    # `:early` wins before `:late`'s clock even starts).
    rng = MersenneTwister(1)
    wins = zeros(Int, 2)
    n = 20000
    for _ in 1:n
        name, _ = rand(rng, c; outcome = true)
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

@testitem "NamedTuple constructor spelling matches the Pairs spelling" begin
    using Distributions

    # A positional NamedTuple `(a = v, …)` lowers to `:a => v, …` Pairs, so the
    # two spellings build identical nodes (field order preserved).
    cfr = 0.3
    @test resolve((death = (Gamma(1.5, 1.0), cfr),
        disch = (Gamma(2.0, 1.5), 1 - cfr))) ==
          resolve(:death => (Gamma(1.5, 1.0), cfr),
        :disch => (Gamma(2.0, 1.5), 1 - cfr))
    # The residual (last probability omitted) spelling round-trips too.
    @test resolve((death = (Gamma(1.5, 1.0), cfr), disch = Gamma(2.0, 1.5))) ==
          resolve(:death => (Gamma(1.5, 1.0), cfr), :disch => Gamma(2.0, 1.5))
    @test compete((death = Gamma(2.0, 3.0), recover = Gamma(3.0, 2.0))) ==
          compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    # `choose`'s NamedTuple form keeps `selector` a keyword (not an alternative).
    @test choose((index = Gamma(2.0, 1.0), sourced = Gamma(4.0, 1.5));
        selector = :kind) ==
          choose(:index => Gamma(2.0, 1.0), :sourced => Gamma(4.0, 1.5);
        selector = :kind)
    # The multi-child composers accept the same spelling.
    @test sequential((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4))) ==
          sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_death => LogNormal(0.5, 0.4))
    @test parallel((admit = Gamma(2.0, 1.0), notif = LogNormal(1.0, 0.5))) ==
          parallel(:admit => Gamma(2.0, 1.0), :notif => LogNormal(1.0, 0.5))
end

@testitem "Composers reject duplicate child/branch names" begin
    using Distributions

    # Mirrors Choose's existing "alternative names must be unique" guard:
    # every composer must reject a repeated name, since the whole
    # name-keyed API (event, update, prune, splice, params_table, shared)
    # can only ever reach the first branch with a duplicate name.
    # Each composer's own message is pinned, not just the exception type, so
    # a swapped guard or a reworded message is caught (the #217 lesson).
    @test_throws "Sequential step names must be unique" sequential(
        :a => Gamma(1.0, 1.0), :a => LogNormal(0.0, 1.0))
    @test_throws "Sequential step names must be unique" Sequential(
        (Gamma(1.0, 1.0), LogNormal(0.0, 1.0)), (:a, :a))
    @test_throws "Parallel branch names must be unique" parallel(
        :a => Gamma(1.0, 1.0), :a => LogNormal(0.0, 1.0))
    @test_throws "Parallel branch names must be unique" Parallel(
        (Gamma(1.0, 1.0), LogNormal(0.0, 1.0)), (:a, :a))
    @test_throws "Resolve outcome names must be unique" resolve(
        :a => (Gamma(1.0, 1.0), 0.5), :a => Gamma(2.0, 1.0))
    @test_throws "Resolve outcome names must be unique" Resolve(
        (:a, :a), (Gamma(1.0, 1.0), Gamma(2.0, 1.0)), (0.5, 0.5))
    @test_throws "Compete outcome names must be unique" compete(
        :a => Gamma(1.0, 1.0), :a => LogNormal(0.0, 1.0))
    @test_throws "Compete outcome names must be unique" Compete(
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
    using Distributions, Random

    nt = (r1 = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
        r2 = [Gamma(1.0, 1.0), Gamma(3.0, 1.0)])
    table = (name = [:a, :b, :c, :d],
        dist = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4),
            Gamma(1.0, 1.0), Gamma(3.0, 1.0)],
        chain = [1, 1, 2, 2])
    mat = [Gamma(2.0, 1.0) LogNormal(0.5, 0.4); Gamma(1.0, 1.0) Gamma(3.0, 1.0)]
    @test compose(nt) == compose(table) == compose(mat)
    # compose always returns a composer, never a bare leaf; a single-branch
    # Parallel scores, moments and draws like the wrapped leaf alone.
    single = compose((a = Gamma(2.0, 1.0),))
    @test single isa Parallel
    @test logpdf(single, [1.5]) ≈ logpdf(Gamma(2.0, 1.0), 1.5)
    @test mean(single).a ≈ mean(Gamma(2.0, 1.0))
    @test minimum(single).a == minimum(Gamma(2.0, 1.0))
    @test maximum(single).a == maximum(Gamma(2.0, 1.0))
    @test rand(Xoshiro(3), single).a == rand(Xoshiro(3), Gamma(2.0, 1.0))
end

@testitem "compose: varargs-pairs spelling matches the NamedTuple spelling" begin
    using Distributions

    # `compose(:a => d1, ...)` is a thin convenience over the NamedTuple form
    # (CensoredDistributions-migration compat, #145); both spellings round-trip
    # to the identical stack, nested branches included.
    @test compose(:onset_admit => Gamma(2.0, 1.0),
        :admit_death => LogNormal(0.5, 0.4)) ==
          compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    @test compose(:a => [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
        :b => Gamma(1.0, 1.0)) ==
          compose((a = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
        b = Gamma(1.0, 1.0)))
    @test_throws ArgumentError compose()
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

@testitem "Missing-admitting scoring on Sequential/Parallel (#271)" begin
    using Distributions

    # A `missing` value scores the observed prefix and integrates out the
    # unobserved remainder (each unobserved step's own marginal contributes
    # zero log density).
    s = sequential(:first => Gamma(2.0, 1.0), :second => LogNormal(0.5, 0.4))
    @test logpdf(s, (first = 3.2, second = missing)) ≈
          logpdf(Gamma(2.0, 1.0), 3.2)
    @test logpdf(s, (first = missing, second = missing)) == 0.0
    # A fully-observed record still round-trips.
    @test logpdf(s, (first = 3.2, second = 0.6)) ≈
          logpdf(Gamma(2.0, 1.0), 3.2) + logpdf(LogNormal(0.5, 0.4), 0.6)

    p = parallel(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4))
    @test logpdf(p, (a = 3.2, b = missing)) ≈ logpdf(Gamma(2.0, 1.0), 3.2)

    # A `missing` value inside a nested composer step is integrated out at its
    # own leaf, the surrounding steps still scored.
    nested = sequential(:inner => sequential(:x => Gamma(2.0, 1.0),
            :y => Gamma(1.0, 1.0)),
        :outer => LogNormal(0.5, 0.4))
    draw = rand(nested)
    partial = merge(draw, (inner_y = missing,))
    @test logpdf(nested, partial) ≈
          logpdf(Gamma(2.0, 1.0), draw.inner_x) +
          logpdf(LogNormal(0.5, 0.4), draw.outer)

    # A `missing` value at a `Resolve` child (a collapsed value slot within a
    # `Sequential`) is integrated out like any other leaf.
    node = resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    chain = sequential(:onset => Gamma(2.0, 1.0), :resolve_step => node)
    chain_draw = rand(chain)
    chain_partial = merge(chain_draw, (resolve_step = missing,))
    @test logpdf(chain, chain_partial) ≈
          logpdf(Gamma(2.0, 1.0), chain_draw.onset)

    # The flat-vector entry point also admits `missing` directly, and the
    # dimension-mismatch guard still applies.
    @test logpdf(s, [3.2, missing]) ≈ logpdf(Gamma(2.0, 1.0), 3.2)
    @test_throws DimensionMismatch logpdf(s, [1.0, missing, 2.0])
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
    unknown_child = r"(?=.*:nonexistent)(?=.*:onset_admit)(?=.*:admit_death)"
    @test_throws unknown_child event(nested, :admit_path, :nonexistent)
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
    using ComposedDistributions: update
    using Distributions

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    tree2 = update(tree, (onset_admit = (shape = 3.0, scale = 1.5),
        admit_death = (mu = 0.7, sigma = 0.5)))
    @test event(tree2, :onset_admit) == Gamma(3.0, 1.5)
    @test event(tree2, :admit_death) == LogNormal(0.7, 0.5)
end

@testitem "structural edits: update node, prune, splice" begin
    using ComposedDistributions: update
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
    @test length(ComposedDistributions.component_names(pruned)) == 2
    # Splice inserts an after step.
    sp = splice(tree, :admit_death; after = :report => Gamma(1.0, 2.0))
    @test event(sp, :admit_death) isa Sequential
end

@testitem "structural edits: prune/splice boundary guards" begin
    using Distributions

    # A Sequential/Parallel can't be pruned down to zero children.
    single = compose((only = Gamma(2.0, 1.0),))
    @test_throws "Parallel needs at least one remaining child" prune(
        single, :only)

    # A Resolve/Choose can't be pruned below two remaining outcomes.
    two_arm = resolve(:a => (Gamma(1.5, 1.0), 0.4), :b => (Gamma(2.0, 1.0), 0.6))
    @test_throws "Resolve needs at least two remaining outcomes" prune(
        compose((res = two_arm,)), :res, :a)
    two_alt = choose(:a => Gamma(1.5, 1.0), :b => Gamma(2.0, 1.0))
    @test_throws "Choose needs at least two remaining alternatives" prune(
        compose((ch = two_alt,)), :ch, :a)

    # Pruning a path that bottoms out at a leaf (not a composer) has no child
    # to drop.
    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    @test_throws "has no child to drop" prune(tree, :onset_admit, :bogus)

    # prune/splice both reject an empty path.
    @test_throws "prune needs a non-empty path" prune(tree, ())
    @test_throws "splice needs a non-empty path" splice(
        tree, (); after = :x => Gamma(1.0, 1.0))

    # splice needs at least one of before/after.
    @test_throws "splice needs a `before` and/or `after` step" splice(
        tree, :admit_death)
end

@testitem "structural edits through a nested Compete: update, prune, tie" begin
    using ComposedDistributions: update
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
    @test length(ComposedDistributions.component_names(pruned)) == 2
    @test sum(pruned.branch_probs) ≈ 1.0

    # tie descends through the Compete to tag a leaf as shared.
    tied = tie(tree, (:path, :immediate), :other; name = :g)
    @test :g in params_table(tied).edge
    @test logpdf(event(tied, :path, :immediate), 1.5) ≈
          logpdf(Gamma(2.0, 1.0), 1.5)
end

@testitem "rand(::Resolve; outcome): resolved name and delay, or missing" begin
    using Distributions, Random

    r = resolve(:event => (Gamma(1.5, 1.0), 0.4), :none => (NoEvent(), 0.6))
    rng = MersenneTwister(3)
    seen = Set{Symbol}()
    for _ in 1:200
        name, time = rand(rng, r; outcome = true)
        push!(seen, name)
        name == :none ? (@test time === missing) : (@test time isa Real)
    end
    @test seen == Set([:event, :none])
    # The keyword-free draw still returns the full named event record, not the
    # compact pair.
    rec = rand(rng, r)
    @test rec isa NamedTuple
    @test keys(rec) == (:event_1, :event, :none)
    # The default-RNG keyword form returns the same `(name, time)` pair shape.
    pair = rand(r; outcome = true)
    @test pair isa Tuple && pair[1] isa Symbol
    @test pair[2] isa Real || pair[2] === missing
end

@testitem "one_of rand returns a named event record (#639)" begin
    using Distributions, Random
    import ComposedDistributions: _one_of_marginal_rand, event_names, as_mixture

    # Resolve: a bare rand draws the fired outcome's event record, keyed by the
    # flat event names, origin at slot 1, the fired outcome present, the rest
    # missing; it round-trips as `log p_i + logpdf(delay_i, t)`.
    r = resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    rng = MersenneTwister(1)
    rec = rand(rng, r)
    @test rec isa NamedTuple
    @test keys(rec) == event_names(r) == (:event_1, :death, :disch)
    @test rec.event_1 == 0.0
    present = filter(n -> rec[n] !== missing, (:death, :disch))
    @test length(present) == 1
    i = present[1] == :death ? 1 : 2
    gap = rec[present[1]]
    @test logpdf(r, rec) ≈ log(probs(r)[present[1]]) +
                           logpdf(r.delays[i], gap)
    # The scalar marginal accessor and `as_mixture` still give a bare time.
    @test _one_of_marginal_rand(MersenneTwister(2), r) isa Real
    @test rand(MersenneTwister(2), as_mixture(r)) isa Real
    # The count form draws a vector of records that scores as the row sum.
    recs = rand(MersenneTwister(4), r, 6)
    @test recs isa Vector && length(recs) == 6
    @test logpdf(r, recs) ≈ sum(logpdf(r, x) for x in recs)

    # Compete: the record names the winning cause; it round-trips as the
    # cause-resolved sub-density (no branch-prob term).
    c = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    crec = rand(MersenneTwister(5), c)
    @test keys(crec) == (:event_1, :death, :recover)
    @test crec.event_1 == 0.0
    win = filter(n -> crec[n] !== missing, (:death, :recover))[1]
    j = win == :death ? 1 : 2
    t = crec[win]
    @test logpdf(c, crec) ≈
          logpdf(c.delays[j], t) +
          sum(logccdf(c.delays[k], t) for k in 1:2 if k != j)
    @test _one_of_marginal_rand(MersenneTwister(6), c) isa Real
end

@testitem "one_of nested in a chain stays a scalar value slot" begin
    using Distributions, Random

    # A nested one_of child occupies one scalar value slot (its marginal), not
    # the standalone event record, so the flat value path round-trips.
    seq = sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_out => resolve(:death => (Gamma(1.5, 1.0), 0.4),
            :disch => Gamma(2.0, 1.5)))
    draw = rand(MersenneTwister(1), seq)
    @test draw isa NamedTuple
    @test draw.admit_out isa Real
    @test isfinite(logpdf(seq, draw))
end

@testitem "params_table is a 5-column superset with a thin hook (#96)" begin
    using Distributions
    import ComposedDistributions: extra_leaf_params, leaf_param_names

    d = compose((onset = Gamma(2.0, 1.0), report = LogNormal(0.5, 0.4)))
    tbl = params_table(d)
    @test Tuple(propertynames(tbl)) == (:edge, :param, :value, :support, :prior)
    # No modifier owns an extra parameter here, so the extra-parameter hook is
    # empty and no `:thin` row appears (the table matches the plain per-param
    # inventory).
    leaf = Gamma(2.0, 1.0)
    @test extra_leaf_params(leaf) == (;)
    @test :thin ∉ leaf_param_names(leaf)
    @test :thin ∉ tbl.param
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

@testitem "type-parameterised names: accessors and runtime-Vector names" begin
    using Distributions
    import ComposedDistributions: component_names, shared_tag, pool_group,
                                  pool_noncentred, rewrap_leaf

    # Names built from a runtime `Vector{Symbol}` (not a tuple literal, e.g. a
    # data-driven column list) instantiate the same type-parameterised struct
    # as the literal-tuple form, so the two are `==` and share the accessor.
    names_vec = [:onset, :admit]
    s_vec = Sequential(
        (Gamma(2.0, 1.0), LogNormal(0.5, 0.4)), Tuple(names_vec))
    s_lit = Sequential(
        (Gamma(2.0, 1.0), LogNormal(0.5, 0.4)), (:onset, :admit))
    @test component_names(s_vec) == (:onset, :admit)
    @test s_vec == s_lit
    @test typeof(s_vec) == typeof(s_lit)

    p_vec = Parallel((Gamma(2.0, 1.0), Gamma(1.0, 1.0)), Tuple([:a, :b]))
    @test component_names(p_vec) == (:a, :b)

    ch_vec = Choose(Tuple([:i, :s]), (Gamma(2.0, 1.0), Gamma(1.0, 1.0)),
        :kind)
    @test component_names(ch_vec) == (:i, :s)

    r_vec = Resolve(Tuple([:death, :disch]),
        (Gamma(1.5, 1.0), Gamma(2.0, 1.5)), (0.3, 0.7))
    @test component_names(r_vec) == (:death, :disch)

    c_vec = Compete(Tuple([:death, :recover]),
        (Gamma(2.0, 3.0), Gamma(3.0, 2.0)))
    @test component_names(c_vec) == (:death, :recover)

    # The Shared tag and Pool group/noncentred accessors round-trip through
    # the type parameter the same way, including a runtime (non-literal)
    # Symbol value.
    tag = Symbol("inc_", 1)
    tagged = shared(tag, Gamma(2.0, 1.0))
    @test shared_tag(tagged) == tag
    @test shared_tag(rewrap_leaf(tagged, Gamma(3.0, 1.0))) == tag

    group = Symbol("district_", 1)
    spec = pool(group)
    @test pool_group(spec) == group
    @test pool_noncentred(spec)
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

@testitem "observed_distribution / convolve interop" begin
    using Distributions
    using ConvolvedDistributions: ConvolvedDistributions, convolved, convolve_series,
                                  discretise_pmf, DelayPMF, Difference,
                                  difference, product, Product, Convolved,
                                  AnalyticalSolver, NumericSolver, GaussLegendre,
                                  integrate, gl_integrate, AbstractSolverMethod

    s = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    od = observed_distribution(s)
    @test od isa Convolved
    # The chain collapses to the convolution of its steps.
    cv = convolved(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test cdf(od, 5.0) ≈ cdf(cv, 5.0)
    # convolved accepts the chain directly (issue #7).
    @test cdf(convolved(s), 5.0) ≈ cdf(cv, 5.0)
    # A chain with a Parallel step cannot collapse.
    @test_throws ArgumentError observed_distribution(
        Sequential(Gamma(1.0, 1.0), Parallel(Gamma(1.0, 1.0), Gamma(1.0, 1.0))))
end

@testitem "ConvolvedDistributions surface is no longer re-exported (#228)" begin
    # #139 re-exported these names so a downstream sat on ComposedDistributions
    # alone; #228 dropped the re-export (own package boundary, own `using`).
    # `Base.isexported` (not `isdefined`) is the right check: `convolve_series`/
    # `Difference`/`Convolved`/`convolved`/`GaussLegendre`/`integrate` stay
    # defined internally (this package extends/constructs them), just no
    # longer exported, which is what a downstream `using ComposedDistributions`
    # actually sees.
    @test !Base.isexported(ComposedDistributions, :convolved)
    @test !Base.isexported(ComposedDistributions, :convolve_series)
    @test !Base.isexported(ComposedDistributions, :integrate)
    @test !Base.isexported(ComposedDistributions, :gl_integrate)
    @test !Base.isexported(ComposedDistributions, :GaussLegendre)
    @test !Base.isexported(ComposedDistributions, :AbstractSolverMethod)
    @test !Base.isexported(ComposedDistributions, :AnalyticalSolver)
    @test !Base.isexported(ComposedDistributions, :NumericSolver)
    @test !Base.isexported(ComposedDistributions, :Difference)
    # Genuinely gone (not merely unexported): dropped from the internal
    # import entirely since CD never constructs/extends them itself.
    @test !isdefined(ComposedDistributions, :gl_integrate)
    @test !isdefined(ComposedDistributions, :AbstractSolverMethod)
    @test !isdefined(ComposedDistributions, :AnalyticalSolver)
    @test !isdefined(ComposedDistributions, :NumericSolver)
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

@testitem "leaf_ctor: a leaf whose params are not its native ctor args" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions

    # A moment-parameterised leaf: it reports a mean and a standard deviation as
    # its parameters and evaluates through the LogNormal those moments imply. Its
    # family is a type parameter, so the bare `MomentLeaf` UnionAll cannot be
    # called positionally — this is exactly the leaf shape that the hard-coded
    # `Base.typename(typeof(inner)).wrapper` reconstruction could not rebuild.
    struct MomentLeaf{D} <: ContinuousUnivariateDistribution
        vals::Tuple{Float64, Float64}
    end

    function native(d::MomentLeaf{LogNormal})
        mean, sd = d.vals
        s2 = log1p((sd / mean)^2)
        return LogNormal(log(mean) - s2 / 2, sqrt(s2))
    end

    Distributions.params(d::MomentLeaf) = d.vals
    Distributions.logpdf(d::MomentLeaf, x::Real) = logpdf(native(d), x)
    Distributions.cdf(d::MomentLeaf, x::Real) = cdf(native(d), x)
    Distributions.quantile(d::MomentLeaf, q::Real) = quantile(native(d), q)
    Base.minimum(::MomentLeaf) = 0.0
    Base.maximum(::MomentLeaf) = Inf

    # The two coordinate hooks: the moments are the free parameters, and the
    # rebuild closes over the family the value tuple does not carry.
    ComposedDistributions.param_names(::MomentLeaf) = (:mean, :sd)
    function ComposedDistributions.leaf_ctor(::MomentLeaf{D}) where {D}
        return (vals...) -> MomentLeaf{D}((vals[1], vals[2]))
    end

    # The default hook is unchanged for a native family.
    @test ComposedDistributions.leaf_ctor(Gamma(2.0, 1.0)) === Gamma

    # Why the hook is needed: the UnionAll is not positionally callable.
    @test_throws MethodError MomentLeaf(8.0, 2.0)

    leaf = MomentLeaf{LogNormal}((8.0, 2.0))
    tree = sequential(:onset_admit => leaf, :admit_death => Gamma(2.0, 1.0))

    # params_table reports the moments, not the LogNormal's native (mu, sigma).
    tbl = params_table(tree)
    @test :mean in tbl.param
    @test :sd in tbl.param
    @test :mu ∉ tbl.param
    @test :sigma ∉ tbl.param

    # Reconstruction round-trips through the hook (this MethodError'd before):
    # the moment leaf is rebuilt from moment coordinates, the native leaf from
    # its own.
    bumped = update(tree, (onset_admit = (mean = 10.0, sd = 3.0),
        admit_death = (shape = 2.0, scale = 1.0)))
    @test params(bumped).onset_admit == (10.0, 3.0)
    @test logpdf(bumped, [2.0, 1.5]) ≈
          logpdf(MomentLeaf{LogNormal}((10.0, 3.0)), 2.0) +
          logpdf(Gamma(2.0, 1.0), 1.5)

    # A prior can be placed on a moment, which is the whole point.
    u = uncertain(leaf; mean = LogNormal(2.0, 0.2))
    @test keys(u.specs) == (:mean,)
    # And a native parameter of the implied LogNormal is not a parameter here.
    @test_throws ArgumentError uncertain(leaf; sigma = LogNormal(2.0, 0.2))

    # The hook must be transparent through a wrapper, or the override is
    # bypassed for exactly the leaves that matter: an `uncertain` leaf carrying
    # the prior, and a truncated one. `free_leaf` peels to the moment leaf, so
    # `leaf_ctor` must recurse rather than read the peeled type directly.
    for wrapped in (truncated(leaf; upper = 30.0), u, shared(:m, leaf))
        @test ComposedDistributions.leaf_ctor(wrapped) ===
              ComposedDistributions.leaf_ctor(leaf)
    end

    # And reconstruction really works through those wrappers.
    trunc_tree = sequential(:onset_admit => truncated(leaf; upper = 30.0),
        :admit_death => Gamma(2.0, 1.0))
    bumped_trunc = update(trunc_tree,
        (onset_admit = (mean = 10.0, sd = 3.0),
            admit_death = (shape = 2.0, scale = 1.0)))
    # A truncated leaf reports its inner params followed by its bounds: the
    # moments were rebuilt, and the truncation was re-applied around them.
    @test params(bumped_trunc).onset_admit == (10.0, 3.0, nothing, 30.0)
    @test params(ComposedDistributions.free_leaf(
        event(bumped_trunc, :onset_admit))) == (10.0, 3.0)

    # Collapsing the uncertain leaf to a concrete one goes through the same
    # rebuild, in moment coordinates.
    u_tree = sequential(:onset_admit => u, :admit_death => Gamma(2.0, 1.0))
    collapsed = update(u_tree, (onset_admit = (mean = 9.0, sd = 2.5),
        admit_death = (shape = 2.0, scale = 1.0)))
    @test params(collapsed).onset_admit == (9.0, 2.5)
end
