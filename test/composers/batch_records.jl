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

@testitem "rand(d, n): the rng-less form threads the default RNG" begin
    using ComposedDistributions: compose, sequential, _value_names
    using Distributions, Random
    import Tables

    # `rand(d, n)` without an explicit rng must build the same column table as
    # the seeded form — same layout, same width, same round trip — differing
    # only in which RNG feeds it. Both the Parallel (`compose`) and Sequential
    # arms of the method's Union take this path.
    trees = (
        compose((onset = Gamma(2.0, 1.0), rash = LogNormal(0.5, 0.4))),
        sequential(:a => Gamma(2.0, 1.0), :b => Gamma(1.5, 1.0)))
    for tree in trees
        tbl = rand(tree, 5)
        @test Tables.istable(tbl)
        @test Tuple(Tables.columnnames(tbl)) == _value_names(tree)
        @test length(Tables.getcolumn(tbl, 1)) == 5
        rows = collect(Tables.namedtupleiterator(tbl))
        @test logpdf(tree, tbl) ≈ sum(logpdf(tree, r) for r in rows)
    end

    # Threading the default RNG (rather than a fresh one per call) makes the
    # rng-less form reproducible under `Random.seed!`.
    tree = compose((onset = Gamma(2.0, 1.0), rash = LogNormal(0.5, 0.4)))
    Random.seed!(20260726)
    a = rand(tree, 4)
    Random.seed!(20260726)
    b = rand(tree, 4)
    @test Tables.getcolumn(a, :onset) == Tables.getcolumn(b, :onset)
    @test Tables.getcolumn(a, :rash) == Tables.getcolumn(b, :rash)
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

@testitem "rand(d, n): Choose batches from kind, or tags kind-less draws" begin
    using ComposedDistributions: choose
    using Distributions, Random

    d = choose(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))

    # With `kind`, the batch draws directly from that alternative — the
    # alternative's own `rand(rng, dist, n)` — mirroring the single-draw
    # committed-selection path (no selector tag needed).
    xs = rand(Xoshiro(1), d, 5; kind = :short)
    @test xs isa Vector{Float64}
    @test length(xs) == 5
    @test all(isfinite, logpdf(d, x; kind = :short) for x in xs)

    # Without `kind`, each draw is its own self-describing tagged record
    # (mirroring the single-draw forward-simulation path), and each
    # round-trips through `logpdf` with no extra argument.
    records = rand(Xoshiro(2), d, 6)
    @test records isa Vector
    @test length(records) == 6
    @test all(r -> r.kind in (:short, :long), records)
    @test all(r -> isfinite(logpdf(d, r)), records)

    # The rng-less form threads the default RNG, both with and without `kind`.
    Random.seed!(20260803)
    a = rand(d, 4; kind = :long)
    Random.seed!(20260803)
    b = rand(d, 4; kind = :long)
    @test a == b

    Random.seed!(20260803)
    c = rand(d, 4)
    Random.seed!(20260803)
    e = rand(d, 4)
    @test c == e
end

@testitem "rand(p, n): Pool batches from the population" begin
    using ComposedDistributions: pool
    using Distributions, Random

    p = pool(:district)
    xs = rand(Xoshiro(3), p, 5)
    @test xs isa Vector{Float64}
    @test length(xs) == 5
    @test all(x -> isfinite(logpdf(p.population, x)), xs)

    # A fixed (non-uncertain) population batches too.
    q = pool(:g, LogNormal(0.5, 0.3))
    ys = rand(Xoshiro(4), q, 4)
    @test length(ys) == 4
    @test all(y -> isfinite(logpdf(q.population, y)), ys)

    # Reproducible under a seeded rng.
    @test rand(Xoshiro(5), p, 3) == rand(Xoshiro(5), p, 3)
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
