@testsnippet RoundTripFixtures begin
    # A snippet gets no default imports, so the package is named explicitly.
    using ComposedDistributions
    using ComposedDistributions: update, component_names
    using Distributions, ConvolvedDistributions
    using ConvolvedDistributions: convolved, difference, NumericSolver

    # One fixture per tree shape the table has to carry, named so a failure
    # says which shape broke rather than which index.
    function round_trip_fixtures()
        inc = shared(:inc, Gamma(2.0, 1.0))
        pair = resolve(:mild => (Gamma(2.0, 1.0), 0.7),
            :severe => (LogNormal(0.5, 0.4), 0.3))
        triple = resolve(:a => (Gamma(2.0, 1.0), 0.5),
            :b => (LogNormal(0.5, 0.4), 0.2), :c => (Gamma(1.0, 2.0), 0.3))
        return [
            :sequential => sequential(:a => Gamma(2.0, 1.0),
                :b => LogNormal(0.5, 0.4)),
            :parallel => parallel(:a => Gamma(2.0, 1.0),
                :b => LogNormal(0.5, 0.4)),
            :nested => compose((a = Gamma(2.0, 1.0),
                c = compose((x = Gamma(3.0, 1.0), y = LogNormal(0.1, 0.5))))),
            :deeply_nested => compose((a = compose((b = compose((
                c = Gamma(2.0, 1.0), d = LogNormal(0.2, 0.3))),)),)),
            :truncated => compose((a = truncated(Gamma(2.0, 1.0);
                upper = 10.0),)),
            :censored => compose((a = censored(Gamma(2.0, 1.0); upper = 5.0),)),
            :choose => choose(:index => Gamma(2.0, 1.0),
                :sourced => LogNormal(0.5, 0.4)),
            :compete => compete(:a => Gamma(2.0, 1.0),
                :b => LogNormal(0.5, 0.4)),
            :shared => choose(:index => inc,
                :sourced => compose((src = LogNormal(0.5, 0.4), inc = inc))),
            :resolve_fixed => pair,
            # An uncertain Resolve's `:param` rows are stick coordinates, so
            # its probabilities have to survive by another route.
            :resolve_dirichlet => update(pair,
                (branch_probs = Dirichlet([1.0, 1.0]),)),
            :resolve_no_prior => update(pair, (branch_probs = no_prior(),)),
            :resolve_three_outcomes => update(triple,
                (branch_probs = Dirichlet(ones(3)),)),
            :no_event => resolve(:event => (Gamma(1.5, 1.0), 0.4),
                :none => (NoEvent(), 0.6)),
            :uncertain => compose((
                u = uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.5, 0.2)),
                v = update(Gamma(2.0, 1.0), (theta = no_prior(),)))),
            :uncertain_under_wrapper => compose((a = truncated(
                uncertain(Gamma(2.0, 1.0); alpha = Normal(2.0, 0.5));
                upper = 9.0),)),
            :varying => compose((v = varying(x -> Gamma(2.0, x);
                covariate = :temp, reference = Gamma(2.0, 1.0)),)),
            :convolved => compose((
                c = convolved(Gamma(2.0, 1.0), LogNormal(0.5, 0.4)),)),
            # The solver is fixed structure no other row records.
            :convolved_numeric => compose((c = convolved(Gamma(2.0, 1.0),
                LogNormal(0.5, 0.4); method = NumericSolver()),)),
            :difference => compose((
                d = difference(Gamma(3.0, 1.0), Gamma(2.0, 1.0)),)),
            :pool_noncentred => compose((
                w = uncertain(Gamma(2.0, 1.0); alpha = pool(:d)),)),
            :pool_centred => compose((w = uncertain(Gamma(2.0, 1.0);
                alpha = pool(:d, LogNormal(0.0, 1.0); noncentred = false)),)),
            :pool_two_groups => compose((
                north = uncertain(Gamma(2.0, 1.0); alpha = pool(:region_a)),
                south = uncertain(Gamma(2.0, 1.0); alpha = pool(:region_a)),
                east = uncertain(Gamma(3.0, 1.0); alpha = pool(:region_b)))),
            :shared_and_pooled => compose((
                a = shared(:g, uncertain(Gamma(2.0, 1.0); alpha = pool(:d))),
                b = shared(:g, uncertain(Gamma(2.0, 1.0); alpha = pool(:d)))))
        ]
    end
end

@testitem "compose(table): the round trip is an identity on every shape" setup=[RoundTripFixtures] begin
    for (name, tree) in round_trip_fixtures()
        @testset "$(name)" begin
            rebuilt = compose(composed_to_table(tree))
            @test rebuilt == tree
            # `==` on a Sequential/Parallel deliberately ignores its names
            # (equality.jl), so the names are asserted separately or a names
            # bug would pass the equality check above unnoticed.
            @test component_names(rebuilt) == component_names(tree)
            @test typeof(rebuilt) == typeof(tree)
        end
    end
end

@testitem "compose(table): the round trip goes through any Tables.jl source" setup=[RoundTripFixtures] begin
    using Tables

    # `composed_to_table` returns a column table; a row table (what a
    # `Vector{<:NamedTuple}` or an iterated DataFrame gives) must read back the
    # same, since the in-memory DataFrame round trip is the documented one.
    for (name, tree) in round_trip_fixtures()
        rows = Tables.rowtable(composed_to_table(tree))
        @testset "$(name)" begin
            @test compose(rows) == tree
        end
    end
    # And the tree is itself a Tables.jl source over its own full table.
    tree = compose((a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4)))
    @test compose(Tables.columntable(tree)) == tree
end

@testitem "compose(table): an edited table composes to the edited tree" begin
    using ComposedDistributions: update
    using Distributions, Tables

    # The spreadsheet workflow: read a tree out, edit a cell, compose it back.
    tree = compose((onset = Gamma(2.0, 1.0), admit = LogNormal(0.5, 0.4)))
    tbl = composed_to_table(tree)
    rows = [NamedTuple(r) for r in Tables.rows(tbl)]
    edited = map(rows) do r
        r.role == :param && r.edge == :onset && r.param == :alpha ?
        merge(r, (value = 5.0,)) : r
    end
    rebuilt = compose(edited)
    @test event(rebuilt, :onset) == Gamma(5.0, 1.0)
    @test event(rebuilt, :admit) == LogNormal(0.5, 0.4)

    # Writing a prior into the `prior` cell makes that parameter uncertain,
    # exactly as it does through `update(tree, table)`.
    with_prior = map(rows) do r
        r.role == :param && r.edge == :admit && r.param == :mu ?
        merge(r, (prior = Normal(0.5, 0.1),)) : r
    end
    @test has_uncertain(compose(with_prior))
end

@testitem "compose(table): a leaf and a well-shaped node cost zero methods" begin
    using ComposedDistributions: AbstractComposedDistribution
    import ComposedDistributions: node_children, node_rebuild, component_names
    using Distributions

    # A third-party node built the documented way -- `T(children, names)`, as
    # `Sequential`/`Parallel` and the `Both` example in the extending guide are
    # -- round-trips on the node contract's three methods alone. No
    # `from_table` here: the default covers it, so adding a node does not
    # get more expensive.
    struct Both{names, C <: Tuple} <:
           AbstractComposedDistribution{Multivariate, Continuous}
        children::C
        Both{names}(children::C) where {names, C <: Tuple} = new{names, C}(
            children)
    end
    function Both(children::C, names::NTuple{N, Symbol}) where {N, C <: Tuple}
        return Both{names}(children)
    end
    node_children(d::Both) = d.children
    node_rebuild(d::Both, children::Tuple) = Both(children, component_names(d))
    component_names(::Both{names}) where {names} = names

    both = Both((Gamma(2.0, 1.0), LogNormal(0.5, 0.4)), (:first, :second))
    @test compose(composed_to_table(both)) == both
    @test component_names(compose(composed_to_table(both))) ==
          (:first, :second)

    # And nested under a built-in, both directions of the same contract.
    seq = Sequential((both, Gamma(1.0, 1.0)), (:branch, :tail))
    @test compose(composed_to_table(seq)) == seq

    # A Distributions.jl family is a leaf with no methods of its own, so the
    # default `from_table` rebuilds it from its parameter values.
    @test ComposedDistributions.from_table(
        Gamma, (alpha = 2.0, theta = 1.0)) == Gamma(2.0, 1.0)
    exotic = compose((a = Weibull(2.0, 1.0), b = Beta(2.0, 3.0),
        c = Uniform(0.0, 4.0)))
    @test compose(composed_to_table(exotic)) == exotic
end

@testitem "compose(table): the two table shapes are told apart by their columns" begin
    using Distributions

    # The authoring shape is unchanged: a whole distribution per `dist` cell.
    authored = compose((name = [:a, :b], dist = [Gamma(2.0, 1.0),
        LogNormal(0.5, 0.4)]))
    @test authored == compose((a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4)))
    # ... and it is still not the inverse of `composed_to_table`; the
    # round-trip shape is.
    @test compose(composed_to_table(authored)) == authored

    # A table carrying both column sets is ambiguous and is refused rather
    # than guessed at.
    @test_throws r"both the authoring columns" compose([(
        name = :a, dist = Gamma(2.0, 1.0), edge = :a, param = :alpha,
        role = :param, node = Gamma, value = 2.0)])
end

@testitem "compose(table): an unusable table is refused, not half-built" begin
    using ComposedDistributions: _param_rows
    using Distributions

    tree = compose((a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4)))

    # The parameter-only projection is the shape a reader most plausibly
    # reaches for and the one that cannot work: no `:node`/`:attribute` rows,
    # so it records the parameters but not the structure.
    @test_throws r"parameter-only projection" compose(_param_rows(tree))

    # Neither column set.
    @test_throws r"either .name. and .dist." compose([(foo = 1, bar = 2)])

    # A partial table with no root row is not a whole tree.
    @test_throws r"no root" compose([(edge = :a, param = Symbol(""),
        node = Gamma, role = :node, value = nothing)])
end

@testitem "compose(table): an unsupported type errors naming its missing method" begin
    import ComposedDistributions: inner_dist, rewrap_leaf, node_attributes,
                                  param_names, rebuild_leaf
    using Distributions, Random

    # A leaf-wrapper layer has no default rebuild -- its fixed structure is its
    # own -- so it must say so rather than drop the layer silently.
    struct Odd{D} <: ContinuousUnivariateDistribution
        d::D
        k::Float64
    end
    inner_dist(x::Odd) = x.d
    rewrap_leaf(x::Odd, inner) = Odd(inner, x.k)
    node_attributes(x::Odd) = (; k = x.k)
    Distributions.logpdf(x::Odd, v) = logpdf(x.d, v)
    Base.minimum(x::Odd) = minimum(x.d)
    Base.maximum(x::Odd) = maximum(x.d)
    Base.rand(rng::AbstractRNG, x::Odd) = rand(rng, x.d)

    wrapped = compose((a = Odd(Gamma(2.0, 1.0), 3.0),))
    @test_throws r"from_table" compose(composed_to_table(wrapped))

    # A leaf whose free parameters are NOT its constructor arguments is the
    # silent-wrongness case: `MomGamma(2.0, 1.0)` constructs happily and is
    # simply a different distribution. The default `from_table` post-checks
    # that the rebuilt leaf reports back the coordinates it was handed.
    struct MomGamma <: ContinuousUnivariateDistribution
        alpha::Float64
        theta::Float64
    end
    Distributions.params(d::MomGamma) = (d.alpha * d.theta,
        sqrt(d.alpha) * d.theta)
    param_names(::MomGamma) = (:m, :s)
    rebuild_leaf(::MomGamma, v::Tuple) = MomGamma((v[1] / v[2])^2,
        v[2]^2 / v[1])
    Distributions.logpdf(d::MomGamma, x) = logpdf(Gamma(d.alpha, d.theta), x)
    Base.minimum(::MomGamma) = 0.0
    Base.maximum(::MomGamma) = Inf
    Base.rand(rng::AbstractRNG, d::MomGamma) = rand(rng,
        Gamma(d.alpha, d.theta))

    moment = compose((a = MomGamma(4.0, 0.5),))
    @test_throws r"from_table" compose(composed_to_table(moment))

    # The one method it needs, and it round-trips.
    ComposedDistributions.from_table(::Type{MomGamma}, v::NamedTuple) = MomGamma(
        (v.m /
         v.s)^2, v.s^2 /
                 v.m)
    @test compose(composed_to_table(moment)) == moment
end

@testitem "compose(table): a shared tag colliding with a tree edge is refused" begin
    using ComposedDistributions: validate_tree_names
    using Distributions

    # `validate_tree_names` gates this on a BUILT tree; reading a table there
    # is no tree yet, so the rows are checked directly and refused with the
    # same advice rather than mis-parsed (the tag edge and the real edge would
    # otherwise carry two different leaves' rows under one name).
    colliding = compose((inc = LogNormal(1.0, 1.0),
        b = shared(:inc, LogNormal(1.5, 0.4))))
    @test_throws r"shared tag" compose(composed_to_table(colliding))
    @test_throws ArgumentError validate_tree_names(colliding)
end

@testitem "compose(table): structure a parameter-only view cannot carry" begin
    using ComposedDistributions: shared_tag
    using Distributions

    # The point of the round trip: these pairs have identical `:param` rows and
    # differ only in structure, so each must come back as itself.
    seq = sequential(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4))
    par = parallel(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4))
    @test compose(composed_to_table(seq)) isa Sequential
    @test compose(composed_to_table(par)) isa Parallel

    # A wrapper stack, in order, with its bounds.
    stacked = compose((a = truncated(Gamma(2.0, 1.0); lower = 0.5,
        upper = 10.0),))
    rebuilt = event(compose(composed_to_table(stacked)), :a)
    @test rebuilt isa Truncated
    @test (rebuilt.lower, rebuilt.upper) == (0.5, 10.0)

    # A Choose's selector.
    picked = choose(:index => Gamma(2.0, 1.0), :sourced => LogNormal(0.5, 0.4);
        selector = :severity)
    @test compose(composed_to_table(picked)).selector == :severity

    # A shared tie: one leaf object reached from two places, still tied.
    inc = shared(:inc, Gamma(2.0, 1.0))
    tied = compose((a = inc, b = compose((c = inc, d = LogNormal(0.5, 0.4)))))
    back = compose(composed_to_table(tied))
    @test shared_tag(event(back, :a)) == :inc
    @test shared_tag(event(back, Symbol("b.c"))) == :inc
end
