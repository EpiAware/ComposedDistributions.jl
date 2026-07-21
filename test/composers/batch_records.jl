# `rand(d, n)` draws a Tables.jl column table of value-name records that scores
# straight back through `logpdf(d, table)`, closing the simulate/score round
# trip for the multi-child composers. The batch `logpdf` dispatch must not
# shadow the flat single-record method for scalar-valued trees. See #276.

@testitem "rand(d, n): column table round-trips through logpdf" begin
    using ComposedDistributions: compose, sequential
    using Distributions, Random
    import Tables

    trees = (
        compose((onset_fever = Gamma(2.0, 1.0),
            onset_rash = LogNormal(0.5, 0.4))),
        sequential(:a => Gamma(2.0, 1.0), :b => Gamma(1.5, 1.0)),
        compose((path = sequential(:s1 => LogNormal(0.5, 0.4),
                :s2 => Gamma(2.0, 1.0)),
            side = Gamma(1.5, 1.0))))
    for tree in trees
        tbl = rand(Xoshiro(1), tree, 6)
        @test Tables.istable(tbl)
        @test length(Tables.getcolumn(tbl, 1)) == 6
        rows = collect(Tables.namedtupleiterator(tbl))
        # the round trip: scoring the table equals summing per-row scores
        @test logpdf(tree, tbl) ≈ sum(logpdf(tree, r) for r in rows)
        # a plain Vector of records scores identically
        @test logpdf(tree, rows) ≈ logpdf(tree, tbl)
    end
end

@testitem "rand(d, n): value-name columns, rng reproducible" begin
    using ComposedDistributions: compose, sequential, _value_names
    using Distributions, Random
    import Tables

    tree = compose((path = sequential(:s1 => LogNormal(0.5, 0.4),
            :s2 => Gamma(2.0, 1.0)),
        side = Gamma(1.5, 1.0)))
    tbl = rand(Xoshiro(7), tree, 4)
    @test Tuple(Tables.columnnames(tbl)) == _value_names(tree)
    @test Tables.getcolumn(tbl, :side) ==
          rand(Xoshiro(7), tree, 4).side          # same seed, same draws
end

@testitem "rand(d, n): a nested resolve scores back too" begin
    using ComposedDistributions: compose, sequential, resolve
    using Distributions, Random
    import Tables

    tree = compose((
        path = sequential(:step_a => LogNormal(0.5, 0.4),
            :split => resolve(:left => (Gamma(1.5, 1.0), 0.3),
                :right => Gamma(2.0, 1.5))),
        side = Gamma(1.5, 1.0)))
    tbl = rand(Xoshiro(2), tree, 5)
    rows = collect(Tables.namedtupleiterator(tbl))
    @test isfinite(logpdf(tree, tbl))
    @test logpdf(tree, tbl) ≈ sum(logpdf(tree, r) for r in rows)
end

@testitem "batch logpdf does not shadow the flat single-record method" begin
    using ComposedDistributions: sequential
    using Distributions

    tree = sequential(:a => Gamma(2.0, 1.0), :b => Gamma(1.5, 1.0))
    # Direction 1: a flat single record (Vector of Reals) stays on the flat
    # path — NOT read as a 2-row batch.
    @test logpdf(tree, [1.0, 2.0]) ≈ logpdf(tree, (a = 1.0, b = 2.0))
    # Direction 2: a Vector of records is a batch — NOT the flat method.
    batch = [(a = 1.0, b = 2.0), (a = 1.5, b = 2.5)]
    @test logpdf(tree, batch) ≈
          logpdf(tree, (a = 1.0, b = 2.0)) + logpdf(tree, (a = 1.5, b = 2.5))
    # a one-row batch equals the single record
    @test logpdf(tree, [(a = 1.0, b = 2.0)]) ≈ logpdf(tree, (a = 1.0, b = 2.0))
end
