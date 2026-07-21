# `event_times` converts a drawn record of per-step increments into absolute
# positions measured from the tree's origin; `event_increments` is the inverse
# back to the per-step representation the scorer consumes. Chain steps
# accumulate, parallel branches anchor on their shared parent (the origin), and
# a resolved/racing node stamps only the outcome that fired. See #269.

@testitem "event_times: pure chain accumulates" begin
    using ComposedDistributions: event_times, event_increments,
                                 compose, sequential
    using Distributions

    tree = sequential(:a => LogNormal(0.5, 0.4),
        :b => Gamma(2.0, 1.0),
        :c => Gamma(1.5, 1.0))
    rec = (a = 1.0, b = 2.0, c = 0.5)
    @test event_times(tree, rec) == (a = 1.0, b = 3.0, c = 3.5)
    # round trip back to per-step increments
    @test event_increments(tree, event_times(tree, rec)) == rec
end

@testitem "event_times: parallel branches anchor at the origin" begin
    using ComposedDistributions: event_times, event_increments,
                                 compose, sequential
    using Distributions

    tree = compose((x = Gamma(2.0, 1.0), y = LogNormal(0.5, 0.4)))
    rec = (x = 2.0, y = 1.5)
    # each branch is measured from the shared origin, not from each other
    @test event_times(tree, rec) == (x = 2.0, y = 1.5)
    @test event_increments(tree, event_times(tree, rec)) == rec
end

@testitem "event_times: chain on one branch, leaf on another" begin
    using ComposedDistributions: event_times, event_increments,
                                 compose, sequential
    using Distributions

    tree = compose((
        path = sequential(:step_a => LogNormal(0.5, 0.4),
            :step_b => Gamma(2.0, 1.0)),
        side = Gamma(1.5, 1.0)))
    rec = (path_step_a = 1.0, path_step_b = 2.0, side = 3.0)
    # path accumulates (1, 1+2); side anchors at the origin (3)
    @test event_times(tree, rec) ==
          (path_step_a = 1.0, path_step_b = 3.0, side = 3.0)
    @test event_increments(tree, event_times(tree, rec)) == rec
end

@testitem "event_times: round trip on a random draw" begin
    using ComposedDistributions: event_times, event_increments,
                                 compose, sequential
    using Distributions, Random

    tree = compose((
        path = sequential(:step_a => LogNormal(0.5, 0.4),
            :step_b => Gamma(2.0, 1.0)),
        side = Gamma(1.5, 1.0)))
    rec = rand(Xoshiro(1), tree)
    abs = event_times(tree, rec)
    @test collect(values(event_increments(tree, abs))) ≈
          collect(values(rec))
end

@testitem "event_times: batch table round trips" begin
    using ComposedDistributions: event_times, event_increments,
                                 compose, sequential
    using Distributions, Random

    tree = compose((
        path = sequential(:step_a => LogNormal(0.5, 0.4),
            :step_b => Gamma(2.0, 1.0)),
        side = Gamma(1.5, 1.0)))
    table = [rand(Xoshiro(i), tree) for i in 1:8]
    abs = event_times(tree, table)
    back = event_increments(tree, abs)
    @test length(back) == length(table)
    @test all(collect(values(back[i])) ≈ collect(values(table[i]))
    for i in eachindex(table))
end

@testitem "event_times: chain resumes past a nested parallel" begin
    using ComposedDistributions: event_times, event_increments,
                                 sequential, parallel
    using Distributions

    # A chain whose middle step is itself a parallel split, then a further
    # chain step. The step after the split resumes from the PREVIOUS chain
    # step (step1), not from either branch endpoint, mirroring the tree's
    # terminal-name rule.
    tree = sequential(:step1 => LogNormal(0.5, 0.4),
        :mid => parallel(:x => Gamma(2.0, 1.0), :y => Gamma(1.5, 1.0)),
        :step2 => Gamma(2.0, 1.0))
    rec = (step1 = 1.0, mid_x = 2.0, mid_y = 3.0, step2 = 4.0)
    # step1 = 1; branches anchor at step1: mid_x = 1+2 = 3, mid_y = 1+3 = 4;
    # step2 resumes from step1 (1), not the branches: step2 = 1+4 = 5.
    @test event_times(tree, rec) ==
          (step1 = 1.0, mid_x = 3.0, mid_y = 4.0, step2 = 5.0)
    @test event_increments(tree, event_times(tree, rec)) == rec
end

@testitem "event_times: a bare leaf or one_of errors clearly" begin
    using ComposedDistributions: event_times, event_increments, resolve
    using Distributions

    @test_throws ArgumentError event_times(Gamma(2.0, 1.0), (a = 1.0,))
    @test_throws ArgumentError event_increments(Gamma(2.0, 1.0), (a = 1.0,))
    bare = resolve(:left => (Gamma(1.5, 1.0), 0.3), :right => Gamma(2.0, 1.5))
    @test_throws ArgumentError event_times(bare, (event_1 = 1.0,))
end
