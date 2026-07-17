# `@events` declares an event-tree topology (names + composition structure, no
# distributions); `update(skeleton; name = dist, ...)` fills the holes and builds
# the concrete composed tree through the existing verbs. The `|` node becomes a
# `Resolve` or `Compete` decided by the fill value type. The fill is
# type-agnostic: a plain distribution, an `uncertain(...)` leaf and a
# ModifiedDistributions modifier leaf all compose without any MD-specific code in
# `@events`.

@testitem "@events: a → chain fills to a Sequential" begin
    using Distributions

    skel = @events begin
        a → b → c
    end
    @test skel isa EventSkeleton

    da, db, dc = Gamma(2.0, 1.0), LogNormal(0.5, 0.4), Gamma(1.0, 1.5)
    tree = update(skel; a = da, b = db, c = dc)

    # Equal to the direct sequential build, same event names.
    @test tree == sequential(:a => da, :b => db, :c => dc)
    @test tree isa Sequential
    @test event_names(tree) == event_names(sequential(:a => da, :b => db,
        :c => dc))
    # A nested → chain flattens into ONE sequential of all three operands.
    @test length(tree.components) == 3
end

@testitem "@events: a bare expression (no begin block) works" begin
    using Distributions

    skel = @events a → b
    tree = update(skel; a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4))
    @test tree == sequential(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4))
end

@testitem "@events: a | node fills to Resolve on (dist, prob) tuples" begin
    using Distributions

    skel = @events begin
        onset → admission → (death | discharge)
    end
    tree = update(skel;
        onset = Gamma(2.0, 1.0),
        admission = LogNormal(0.5, 0.4),
        death = (Gamma(1.5, 1.0), 0.3),
        discharge = (Gamma(2.0, 1.5), 0.7))

    @test tree isa Sequential
    last_step = tree.components[end]
    @test last_step isa Resolve
    @test probs(last_step) == (death = 0.3, discharge = 0.7)
    # The nested one_of step is auto-named from its branches.
    @test ComposedDistributions.component_names(tree) ==
          (:onset, :admission, :death_or_discharge)
end

@testitem "@events: the residual last branch is a Resolve" begin
    using Distributions

    # The headline form: death carries a probability, discharge omits it and
    # takes the residual (1 - 0.3), so the node is a fixed-probability Resolve.
    skel = @events begin
        onset → admission → (death | discharge)
    end
    tree = update(skel;
        onset = Gamma(2.0, 1.0),
        admission = LogNormal(0.5, 0.4),
        death = (Gamma(1.5, 1.0), 0.3),
        discharge = Gamma(2.0, 1.5))

    node = tree.components[end]
    @test node isa Resolve
    @test probs(node).death == 0.3
    @test probs(node).discharge ≈ 0.7
end

@testitem "@events: a | node fills to Compete on bare distributions" begin
    using Distributions

    skel = @events begin
        onset → (death | discharge)
    end
    tree = update(skel;
        onset = Gamma(2.0, 1.0),
        death = Gamma(1.5, 1.0),
        discharge = Gamma(2.0, 1.5))

    node = tree.components[end]
    @test node isa Compete
    @test Set(keys(probs(node))) == Set((:death, :discharge))
end

@testitem "@events: a mixed | fill throws" begin
    using Distributions

    skel = @events begin
        onset → (a | b | c)
    end
    # A bare-then-prob mix is not the residual shape and is rejected.
    @test_throws ArgumentError update(skel;
        onset = Gamma(2.0, 1.0),
        a = Gamma(1.5, 1.0),
        b = (Gamma(2.0, 1.5), 0.4),
        c = Gamma(2.0, 1.5))
end

@testitem "@events: a & node fills to Parallel" begin
    using Distributions

    skel = @events begin
        onset → (admission & notification)
    end
    tree = update(skel;
        onset = Gamma(2.0, 1.0),
        admission = LogNormal(0.5, 0.4),
        notification = Gamma(1.0, 1.0))

    node = tree.components[end]
    @test node isa Parallel
    @test ComposedDistributions.component_names(node) ==
          (:admission, :notification)
end

@testitem "@events: an unfilled hole throws naming it" begin
    using Distributions

    skel = @events begin
        a → b → c
    end
    err = try
        update(skel; a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("c", sprint(showerror, err))
end

@testitem "@events: an unknown fill key throws" begin
    using Distributions

    skel = @events begin
        a → b
    end
    @test_throws ArgumentError update(skel;
        a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4), z = Gamma(1.0, 1.0))
end

@testitem "@events: a reused event name is rejected" begin
    @test_throws ArgumentError (@events begin
        a → a
    end)
end

@testitem "@events: ModifiedDistributions leaf fills compose (design test)" begin
    using Distributions
    using ModifiedDistributions
    using ModifiedDistributions: affine, get_dist

    skel = @events begin
        onset → admission → (death | discharge)
    end
    # An affine-wrapped Gamma fills the onset hole. No MD-specific code in
    # @events: the modifier leaf is a UnivariateDistribution, so the verbs admit
    # it and the existing MD extension peels it.
    onset_leaf = affine(Gamma(2.0, 1.0); scale = 2.0)
    tree = update(skel;
        onset = onset_leaf,
        admission = LogNormal(0.5, 0.4),
        death = (Gamma(1.5, 1.0), 0.3),
        discharge = (Gamma(2.0, 1.5), 0.7))

    # The onset leaf is the affine-wrapped Gamma: free_leaf peels to the Gamma
    # and get_dist sees the modifier's inner delay.
    built = tree.components[1]
    @test ComposedDistributions.free_leaf(built) == Gamma(2.0, 1.0)
    @test get_dist(built) == Gamma(2.0, 1.0)
    # params_table sees through the modifier to the inner free parameters.
    tbl = params_table(tree)
    @test :shape in tbl.param
    @test :scale in tbl.param
end

@testitem "@events: an @uncertain leaf fill stays uncertain" begin
    using Distributions

    skel = @events begin
        onset → admission
    end
    tree = update(skel;
        onset = (@uncertain Gamma(Normal(0.7, 0.2), 1.0)),
        admission = LogNormal(0.5, 0.4))

    @test has_uncertain(tree)
    @test tree.components[1] isa Uncertain
end

@testitem "@events: the realistic tree builds, rands and scores" begin
    using Distributions
    using Random

    skel = @events begin
        onset → admission → (death | discharge)
    end
    tree = update(skel;
        onset = Gamma(2.0, 1.0),
        admission = LogNormal(0.5, 0.4),
        death = (Gamma(1.5, 1.0), 0.3),
        discharge = Gamma(2.0, 1.5))

    @test event_names(tree) ==
          event_names(sequential(:onset => Gamma(2.0, 1.0),
        :admission => LogNormal(0.5, 0.4),
        :death_or_discharge => resolve(:death => (Gamma(1.5, 1.0), 0.3),
            :discharge => Gamma(2.0, 1.5))))

    draw = rand(MersenneTwister(1), tree)
    @test draw isa NamedTuple
    @test isfinite(logpdf(tree, draw))
end
