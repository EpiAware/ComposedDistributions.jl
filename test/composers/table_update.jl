# `update(d, table)`: the bulk Tables.jl route, plus the dispatch-safety fix
# that keeps it from colliding with the flat-vector arm or a
# DistributionsInference-shaped (dotted-`name`) row table (CD#195/DI#20
# design pass).

@testitem "update(tree, table): params_table round-trips and edits" begin
    using Distributions, Tables

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))

    # A no-op round trip: every `value` re-applied.
    tbl = params_table(tree)
    @test update(tree, tbl) == tree

    # Editing a `value` and bulk-writing it back is a concrete change.
    rows = Tables.rowtable(tbl)
    edited = map(rows) do row
        row.edge == :onset_admit && row.param == :shape ?
        merge(row, (; value = 3.5)) : row
    end
    written = update(tree, edited)
    @test event(written, :onset_admit) == Gamma(3.5, 1.0)

    # A hand-built minimal table (no `support`/`prior` columns) still works:
    # only `edge`/`param`/`value` are required for a concrete write.
    minimal = [(edge = :onset_admit, param = :shape, value = 4.0),
        (edge = :onset_admit, param = :scale, value = 1.0),
        (edge = :admit_death, param = :mu, value = 0.5),
        (edge = :admit_death, param = :sigma, value = 0.4)]
    @test event(update(tree, minimal), :onset_admit) == Gamma(4.0, 1.0)
end

@testitem "update(tree, table): a `prior` column promotes to uncertain" begin
    using Distributions, Tables

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))

    rows = [
        (edge = :onset_admit, param = :shape, value = 2.0,
            prior = LogNormal(log(2.0), 0.2)),
        (edge = :onset_admit, param = :scale, value = 1.0, prior = nothing)]
    promoted = update(tree, rows)
    @test has_uncertain(promoted)
    @test promoted == update(tree,
        (onset_admit = (shape = LogNormal(log(2.0), 0.2),),))
end

@testitem "update(tree, table): dispatch safety against Real vectors and DI rows" begin
    using Distributions

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = LogNormal(0.5, 0.4)))

    # The flat-vector arm is now restricted to AbstractVector{<:Real}, so it
    # still works for a genuine flat vector...
    @test event(update(tree, [3.0]), :onset_admit) == Gamma(3.0, 1.0)

    # ...but a `Vector{<:NamedTuple}` no longer matches it (it would
    # previously have been silently handed to `unflatten`, which expects
    # `Real` elements): it now routes to the table arm instead, and errors
    # NAMING the columns it found rather than the elements' types, because a
    # `Vector{<:NamedTuple}` is `Tables.istable` under BOTH this package's
    # `edge`/`param` convention and DistributionsInference's dotted-`name`
    # `parameter_rows` convention (DI#20) — this checks the shapes are told
    # apart rather than one silently misread as the other.
    di_shaped_rows = [(name = :onset_admit_shape, value = 2.0,
        prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf))]
    err = try
        update(tree, di_shaped_rows)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("edge", err.msg) && occursin("param", err.msg)

    # A non-table, non-vector, non-NamedTuple second argument still errors
    # clearly (not a bare, unhelpful MethodError).
    @test_throws ArgumentError update(tree, "not a table")
end
