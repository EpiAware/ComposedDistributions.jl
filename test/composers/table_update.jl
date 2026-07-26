# `update(d, table)`: the bulk Tables.jl route, plus the dispatch-safety fix
# that keeps it from colliding with the flat-vector arm or a
# DistributionsInference-shaped (dotted-`name`) row table (CD#195/DI#20
# design pass).

@testitem "update(tree, table): params_table round-trips and edits" begin
    using ComposedDistributions: update
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
    using ComposedDistributions: update
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

@testitem "update(tree, table): round-trips a fixed Resolve past an unrelated uncertain leaf (#219)" begin
    using ComposedDistributions: update
    using Distributions

    # An uncertain leaf (`onset_admit.shape`) and an unrelated fixed-probability
    # `Resolve` (`admit_resolve`) in the same tree: the uncertain leaf's row
    # carries a distribution prior, which used to flip the WHOLE update into
    # merge mode, and merge mode then rejected the `Resolve`'s plain-float
    # `branch_probs` (demanding a `Dirichlet`), even though that `Resolve` has
    # nothing to do with the leaf that triggered merge mode.
    tree = @uncertain compose((
        clinical = sequential(
            :onset_admit => Gamma(Normal(1.2, 0.2), 3.0),
            :admit_resolve => resolve(
                :death => (Gamma(2.0, 3.5), 0.3),
                :discharge => (Gamma(1.0, 8.0), 0.7))),
        community = Gamma(2.0, 4.0)))

    # The round trip the `update` docstring shows first must not throw and must
    # reproduce the tree exactly: the uncertain leaf keeps its prior, the fixed
    # `Resolve` keeps its concrete probabilities.
    @test update(tree, params_table(tree)) == tree
    @test has_uncertain(update(tree, params_table(tree)))
    @test probs(event(update(tree, params_table(tree)),
        :clinical, :admit_resolve)) == (death = 0.3, discharge = 0.7)
end

@testitem "update: a pinned branch_probs must sum to one; readback still passes (#219)" begin
    using ComposedDistributions: update, flat_dimension
    using Distributions

    tree = compose((resolution = resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7)),))

    # Pinning fixed branch_probs that do NOT sum to one is now rejected. The
    # pin only runs the collapse path (via the merge-mode route a distribution
    # elsewhere flips on), and the inner Resolve constructor deliberately skips
    # the sum-to-one check (a prior-sampled weight set is legitimately
    # unnormalised) — so without this guard `(0.9, 0.9)` built a node whose
    # `logpdf` silently scored an unnormalised mixture through `_one_of_logmix`.
    @test_throws ArgumentError update(tree,
        (resolution = (
            death = (shape = Normal(1.5, 0.2),),
            branch_probs = (death = 0.9, disch = 0.9)),))

    # A pin that does sum to one still works and collapses the node to fixed.
    ok = update(tree,
        (resolution = (
            death = (shape = Normal(1.5, 0.2),),
            branch_probs = (death = 0.4, disch = 0.6)),))
    @test probs(event(ok, :resolution)) == (death = 0.4, disch = 0.6)

    # The guard must NOT break the readback / reconstruction path: a
    # stick-reconstructed simplex always sums to one, so folding a flat draw
    # back through `update` still passes. (This is the path the inner
    # constructor's skip exists for.)
    promoted = update(tree, param_priors(tree))
    @test has_uncertain(promoted)
    back = update(promoted, fill(0.5, flat_dimension(promoted)))
    @test sum(event(back, :resolution).branch_probs) ≈ 1
end

@testitem "update(tree, table): dispatch safety against Real vectors and DI rows" begin
    using ComposedDistributions: update
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
    # naming the columns it found rather than the elements' types, because a
    # `Vector{<:NamedTuple}` is `Tables.istable` under both this package's
    # `edge`/`param` convention and DistributionsInference's dotted-`name`
    # `parameter_rows` convention (DI#20) — this checks the shapes are told
    # apart rather than one silently misread as the other.
    di_shaped_rows = [(name = :onset_admit_shape, value = 2.0,
        prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf))]
    @test_throws r"(?=.*edge)(?=.*param)" update(tree, di_shaped_rows)

    # A non-table, non-vector, non-NamedTuple second argument still errors
    # clearly (not a bare, unhelpful MethodError).
    @test_throws ArgumentError update(tree, "not a table")
end
